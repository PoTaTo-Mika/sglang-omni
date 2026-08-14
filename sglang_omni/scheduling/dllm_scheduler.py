# SPDX-License-Identifier: Apache-2.0
"""DllmScheduler — stage-facing scheduler for Diffusion LLM stages.

Provides the same public contract (inbox, outbox, start, stop, abort)
as OmniScheduler so it is interchangeable from the Stage's perspective.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import threading
import time
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


def _release_kv_once(req, tree_cache):
    if getattr(req, "kv_committed_freed", False):
        return
    release_kv_cache(req, tree_cache)

    req.kv_committed_freed = True


class DllmScheduler:
    """Stage-facing scheduler for Diffusion LLM stages.

    Public contract (used by Stage):
        ``inbox``, ``outbox``, ``start()``, ``stop()``, ``abort(request_id)``
    """

    def __init__(
        self,
        tp_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        dllm_config: Any,
        *,
        request_builder: Callable,
        result_adapter: Callable,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        self._request_builder = request_builder
        self._result_adapter = result_adapter

        self.tp_worker = tp_worker
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.server_args = server_args
        self.model_config = model_config
        self.dllm_config = dllm_config
        self._chunked_prefill_size = (
            getattr(dllm_config, "block_size", None) or server_args.chunked_prefill_size
        )

        self._running = False
        self._abort_lock = threading.Lock()
        self._aborted_request_ids: set[str] = set()
        self._rid_to_req_data: dict[str, Any] = {}
        self._waiting_queue: list[Req] = []
        self._staging_queue: list[Req] = []

        # CFG tracking: cond_rid <-> uncond_rid(s)
        self._cond_to_unconds: dict[str, list[str]] = {}
        self._uncond_to_cond: dict[str, str] = {}
        self._uncond_rids: set[str] = set()
        self._orphaned_uncond_rids: set[str] = set()

    def start(self) -> None:
        self._running = True
        self._event_loop()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted_request_ids.add(request_id)

    def _event_loop(self) -> None:
        while self._running:
            self._drain_and_purge()
            batch = self._schedule_next_batch()

            if batch is None:
                time.sleep(0.001)
                continue

            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.tp_worker.model_runner
            )
            forward_batch.reqs = batch.reqs
            self._apply_cfg_padding_metadata(forward_batch, batch)
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)

            batch.output_ids = batch_result.next_token_ids
            self._apply_results(batch, batch_result)
            self._post_step(batch)

    def _drain_and_purge(self) -> None:
        with self._abort_lock:
            aborted = self._aborted_request_ids
            self._aborted_request_ids = set()
        # Aborting any member of a CFG group must purge the whole group.
        aborted_groups: set[str] = set()
        for rid in aborted:
            cond_rid = self._uncond_to_cond.get(rid, rid)
            aborted_groups.add(cond_rid)
            aborted_groups.update(self._cond_to_unconds.get(cond_rid, ()))
        aborted = aborted_groups

        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break

            if msg.request_id in aborted:
                continue

            if msg.type == "new_request":
                req_data = self._request_builder(msg.data)
                req = req_data.req
                self._rid_to_req_data[req.rid] = req_data
                self._waiting_queue.append(req)

                # CFG: create companion uncond Reqs (text guidance always; image
                # guidance only when cfg_image_scale > 0).
                uncond_ids = getattr(req, "_uncond_input_ids", None)
                if uncond_ids is not None:
                    self._create_uncond_companion(
                        req,
                        uncond_ids,
                        getattr(req, "_uncond_left_pad_len", 0),
                        "-uncond",
                        mark_img=False,
                    )
                    uncond_img_ids = getattr(req, "_uncond_img_input_ids", None)
                    if uncond_img_ids is not None:
                        self._create_uncond_companion(
                            req,
                            uncond_img_ids,
                            getattr(req, "_uncond_img_left_pad_len", 0),
                            "-uncond-img",
                            mark_img=True,
                        )

                companion_rids = self._cond_to_unconds.get(req.rid, ())
                reqs_by_rid = {queued.rid: queued for queued in self._waiting_queue}
                request_group = [reqs_by_rid[rid] for rid in (req.rid, *companion_rids)]
                try:
                    self._validate_request_group_capacity(request_group)
                except RuntimeError as exc:
                    logger.warning(
                        "DllmScheduler: rejecting request %s: %s",
                        req.rid,
                        exc,
                    )
                    self._reject_waiting_request_group(req.rid, str(exc))
            else:
                logger.warning(
                    "DllmScheduler: unhandled message type %r for request %s",
                    msg.type,
                    msg.request_id,
                )

        # Purge aborted requests
        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in aborted and not r.finished()
        ]
        new_staging = []
        for req in self._staging_queue:
            if req.rid in aborted:
                _release_kv_once(req, self.tree_cache)
            elif not req.finished():
                new_staging.append(req)
        self._staging_queue = new_staging

        for rid in aborted:
            self._rid_to_req_data.pop(rid, None)
            for uncond_rid in self._cond_to_unconds.pop(rid, []):
                self._uncond_to_cond.pop(uncond_rid, None)
            cond_rid = self._uncond_to_cond.pop(rid, None)
            if cond_rid is not None and cond_rid in self._cond_to_unconds:
                companions = self._cond_to_unconds[cond_rid]
                if rid in companions:
                    companions.remove(rid)
            self._uncond_rids.discard(rid)
            self._orphaned_uncond_rids.discard(rid)

    def _create_uncond_companion(
        self,
        cond_req: Req,
        uncond_input_ids: list[int],
        left_pad_len: int,
        rid_suffix: str,
        mark_img: bool,
    ) -> None:
        """Create a companion uncond Req for CFG and add to waiting queue."""
        from sglang.srt.sampling.sampling_params import SamplingParams

        uncond_rid = f"{cond_req.rid}{rid_suffix}"
        uncond_input_ids = list(uncond_input_ids)
        if len(uncond_input_ids) != len(cond_req.origin_input_ids):
            raise ValueError(
                "CFG companion input must be physically aligned with the "
                f"conditional input: cond={len(cond_req.origin_input_ids)}, "
                f"companion={len(uncond_input_ids)}"
            )
        uncond_sampling_params = SamplingParams(
            max_new_tokens=cond_req.sampling_params.max_new_tokens,
            temperature=0.0,
        )
        # Populate stop_strs / stop_str_max_len / stop_regex_* so
        # check_finished() has normalized sampling state.
        uncond_sampling_params.normalize(None)
        uncond_req = Req(
            rid=uncond_rid,
            origin_input_text="",
            origin_input_ids=uncond_input_ids,
            sampling_params=uncond_sampling_params,
            vocab_size=cond_req.vocab_size,
            eos_token_ids=cond_req.eos_token_ids,
        )
        uncond_req._is_uncond = True
        uncond_req._dllm_left_pad_len = int(left_pad_len)
        uncond_req._cfg_scale = getattr(cond_req, "_cfg_scale", 4.0)
        uncond_req._cfg_rescale = getattr(cond_req, "_cfg_rescale", 0.7)
        cond_req._cfg_group_rid = cond_req.rid
        uncond_req._cfg_group_rid = cond_req.rid
        if mark_img:
            uncond_req._is_uncond_img = True

        # Copy dllm_config from cond
        if hasattr(cond_req, "dllm_config"):
            uncond_req.dllm_config = cond_req.dllm_config

        self._waiting_queue.append(uncond_req)
        self._cond_to_unconds.setdefault(cond_req.rid, []).append(uncond_rid)
        self._uncond_to_cond[uncond_rid] = cond_req.rid
        self._uncond_rids.add(uncond_rid)

    def _reject_waiting_request_group(
        self,
        cond_rid: str,
        error: str,
    ) -> None:
        companion_rids = self._cond_to_unconds.pop(cond_rid, [])
        group_rids = {cond_rid, *companion_rids}
        self._waiting_queue = [
            req for req in self._waiting_queue if req.rid not in group_rids
        ]
        self._rid_to_req_data.pop(cond_rid, None)
        for companion_rid in companion_rids:
            self._uncond_to_cond.pop(companion_rid, None)
            self._uncond_rids.discard(companion_rid)
            self._orphaned_uncond_rids.discard(companion_rid)
        self.outbox.put(
            OutgoingMessage(
                request_id=cond_rid,
                type="error",
                data=error,
            )
        )

    def _apply_cfg_padding_metadata(self, forward_batch, batch) -> None:
        """Apply mask and position metadata for mask-padded CFG branches."""
        left_pad_lengths = [
            int(getattr(req, "_dllm_left_pad_len", 0)) for req in batch.reqs
        ]
        if not any(left_pad_lengths):
            return
        if any(left_pad_length < 0 for left_pad_length in left_pad_lengths):
            raise RuntimeError("CFG left-pad lengths must be non-negative")
        if not forward_batch.forward_mode.is_extend():
            raise RuntimeError("CFG left-pad metadata requires an extend batch")

        extend_sequence_lengths = list(forward_batch.extend_seq_lens_cpu)
        if len(extend_sequence_lengths) != len(left_pad_lengths):
            raise RuntimeError(
                f"CFG pad metadata batch mismatch: {len(left_pad_lengths)} vs "
                f"{len(extend_sequence_lengths)}"
            )

        position_start = 0
        for left_pad_length, extend_sequence_length in zip(
            left_pad_lengths, extend_sequence_lengths
        ):
            position_end = position_start + int(extend_sequence_length)
            if left_pad_length:
                request_positions = forward_batch.positions[position_start:position_end]
                request_positions.sub_(left_pad_length).clamp_min_(0)
            position_start = position_end
        if position_start != forward_batch.positions.numel():
            raise RuntimeError(
                f"CFG position span {position_start} != "
                f"{forward_batch.positions.numel()}"
            )

        forward_batch.dllm_left_pad_lens = torch.tensor(
            left_pad_lengths,
            dtype=forward_batch.seq_lens.dtype,
            device=forward_batch.seq_lens.device,
        )

    def _synchronize_cfg_phases(self, reqs: list[Req]) -> None:
        """Keep CFG companions in the conditional request's DLLM phase."""
        if len(reqs) < 2:
            return
        cond_req = next(
            (req for req in reqs if not getattr(req, "_is_uncond", False)), None
        )
        if cond_req is None:
            return
        for req in reqs:
            if getattr(req, "_is_uncond", False):
                req.dllm_phase = cond_req.dllm_phase

    def _get_request_group(self, queue: list[Req]) -> list[Req]:
        """Return the complete logical request group at the head of a queue."""
        if not queue:
            return []

        first_req = queue[0]
        cond_rid = self._uncond_to_cond.get(first_req.rid, first_req.rid)
        expected_rids = [
            cond_rid,
            *self._cond_to_unconds.get(cond_rid, ()),
        ]
        reqs_by_rid = {req.rid: req for req in queue}
        missing_rids = [rid for rid in expected_rids if rid not in reqs_by_rid]
        if missing_rids:
            raise RuntimeError(
                f"Incomplete CFG request group {cond_rid}: missing {missing_rids}"
            )
        return [reqs_by_rid[rid] for rid in expected_rids]

    def _validate_request_group_capacity(self, reqs: list[Req]) -> None:
        if len(reqs) <= 1:
            return

        max_running_requests = getattr(
            self.dllm_config,
            "max_running_requests",
            None,
        )
        if max_running_requests is not None and max_running_requests < len(reqs):
            raise RuntimeError(
                "CFG request group requires "
                f"{len(reqs)} running requests, but max_running_requests="
                f"{max_running_requests}"
            )

        block_size = getattr(self.dllm_config, "block_size", None)
        max_prefill_tokens = getattr(
            self.server_args,
            "max_prefill_tokens",
            None,
        )
        page_size = max(int(getattr(self.server_args, "page_size", 1)), 1)
        if block_size is not None:
            block_size = int(block_size)
            block_charge = (block_size + page_size - 1) // page_size * page_size
            largest_initial_extend = max(
                len(req.origin_input_ids) + block_size for req in reqs
            )
            largest_initial_extend = (
                (largest_initial_extend + page_size - 1) // page_size * page_size
            )
            # In SGLang 0.5.12, every request after the first must have
            # real_input_tokens strictly below the remaining prefill budget.
            required_prefill_tokens = (
                largest_initial_extend + (len(reqs) - 1) * block_charge + 1
            )
        else:
            required_prefill_tokens = None
        if (
            required_prefill_tokens is not None
            and max_prefill_tokens is not None
            and max_prefill_tokens < required_prefill_tokens
        ):
            raise RuntimeError(
                "CFG request group requires at least "
                f"{required_prefill_tokens} max_prefill_tokens, but configured "
                f"value is {max_prefill_tokens}"
            )

    def _rollback_partial_admission(
        self,
        admitted_reqs: list[Req],
        *,
        from_staging: bool,
        request_snapshots: list[tuple[Req, dict[str, Any]]],
    ) -> None:
        """Undo request and cache mutations from an incomplete group probe."""
        if not from_staging:
            for req in admitted_reqs:
                self.tree_cache.dec_lock_ref(req.last_node)
        for req, state in request_snapshots:
            req.__dict__.clear()
            req.__dict__.update(state)

    def _schedule_next_batch(self) -> ScheduleBatch | None:
        if not self._waiting_queue and not self._staging_queue:
            return None

        source_queue = (
            self._staging_queue if self._staging_queue else self._waiting_queue
        )
        request_group = self._get_request_group(source_queue)
        self._validate_request_group_capacity(request_group)
        request_snapshots = [(req, req.__dict__.copy()) for req in request_group]

        adder = PrefillAdder(
            self.server_args.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            None,  # running_batch
            0.5,  # new_token_ratio
            self.server_args.max_prefill_tokens,
            self._chunked_prefill_size,
            prefill_max_requests=len(request_group),
            dllm_config=self.dllm_config,
        )

        from_staging = source_queue is self._staging_queue
        if from_staging:
            for req in request_group:
                req.init_next_round_input()
                result = adder.add_dllm_staging_req(req)
                if result == AddReqResult.NO_TOKEN:
                    break
        else:
            for req in request_group:
                req.init_next_round_input(self.tree_cache)
                result = adder.add_one_req(
                    req,
                    has_chunked_req=False,
                    truncation_align_size=None,
                )
                if result != AddReqResult.CONTINUE:
                    break

        expected_rids = [req.rid for req in request_group]
        scheduled_rids = [req.rid for req in adder.can_run_list]
        if scheduled_rids != expected_rids:
            self._rollback_partial_admission(
                adder.can_run_list,
                from_staging=from_staging,
                request_snapshots=request_snapshots,
            )
            return None

        self._synchronize_cfg_phases(adder.can_run_list)

        # Diffusion requests need to be rescheduled until they finish. Keep each
        # scheduled request in our stage-local staging queue; current SGLang no
        # longer exposes the old dllm_staging_reqs helper on PrefillAdder.
        staging_rids = {r.rid for r in self._staging_queue}
        for req in adder.can_run_list:
            if req.rid not in staging_rids:
                self._staging_queue.append(req)
                staging_rids.add(req.rid)
        self._waiting_queue = [
            r for r in self._waiting_queue if r.rid not in staging_rids
        ]

        for req in adder.can_run_list:
            req.is_chunked += 1

        new_batch = ScheduleBatch.init_new(
            reqs=adder.can_run_list,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()

        # SGLang 0.5.12 uses one DLLM forward mode for both phases. The
        # prefill/decode distinction lives on Req.dllm_phase.
        prefill_flags = {req.is_dllm_prefill() for req in new_batch.reqs}
        if len(prefill_flags) != 1:
            request_phases = [
                getattr(req, "dllm_phase", None) for req in new_batch.reqs
            ]
            raise RuntimeError(f"Cannot mix DLLM phases in one batch: {request_phases}")
        new_batch.forward_mode = ForwardMode.DLLM_EXTEND
        new_batch.decoding_reqs = None
        return new_batch

    def _apply_results(self, batch: Any, batch_result: Any) -> None:
        # DLLM prefill builds prompt KV only and must not emit algorithm output.
        if batch.reqs and all(req.is_dllm_prefill() for req in batch.reqs):
            return

        next_token_ids = batch_result.next_token_ids
        if next_token_ids is None:
            return

        token_ids = (
            next_token_ids.tolist()
            if hasattr(next_token_ids, "tolist")
            else next_token_ids
        )
        # Normalize flat list into per-request shape for batch=1.
        # For batch=2 (CFG), the algorithm returns per-request lists.
        if len(batch.reqs) == 1 and (not token_ids or isinstance(token_ids[0], int)):
            token_ids_per_req = [token_ids]
        else:
            token_ids_per_req = token_ids

        for req, req_token_ids in zip(batch.reqs, token_ids_per_req):
            # Keep the CFG companion synchronized for the next dLLM block, but
            # do not expose it as a user-visible result.
            if req.rid in self._uncond_rids:
                req.output_ids.extend(int(token_id) for token_id in req_token_ids)
                continue

            for token_id in req_token_ids:
                req.output_ids.append(int(token_id))
                req.check_finished()
                if req.finished():
                    break

            if req.finished():
                # Companion(s) have no user-visible output; retire in post-step.
                for uncond_rid in self._cond_to_unconds.pop(req.rid, []):
                    self._uncond_to_cond.pop(uncond_rid, None)
                    self._orphaned_uncond_rids.add(uncond_rid)

                req_data = self._rid_to_req_data.pop(req.rid, None)
                if req_data is None:
                    continue
                req_data.output_ids = list(req.output_ids_through_stop)
                finished_reason = req.finished_reason
                req_data.finish_reason = (
                    finished_reason.to_json().get("type")
                    if finished_reason is not None
                    else None
                )
                try:
                    result = self._result_adapter(req_data)
                except Exception as exc:
                    logger.warning(
                        "DllmScheduler: result adapter failed for request %s: %s",
                        req.rid,
                        exc,
                    )
                    message_type = "error"
                    result = str(exc)
                else:
                    message_type = "result"
                self.outbox.put(
                    OutgoingMessage(
                        request_id=req.rid,
                        type=message_type,
                        data=result,
                    )
                )

    def _post_step(self, batch: Any) -> None:
        orphaned_uncond_rids = set(self._orphaned_uncond_rids)
        exclude = set()
        for req in batch.reqs:
            if req.finished() or req.rid in orphaned_uncond_rids:
                _release_kv_once(req, self.tree_cache)
                exclude.add(req)

        self._waiting_queue = [
            req for req in self._waiting_queue if req.rid not in orphaned_uncond_rids
        ]

        new_staging = []
        for req in self._staging_queue:
            exclude.add(req)
            if req.rid in orphaned_uncond_rids:
                _release_kv_once(req, self.tree_cache)
                continue
            if req.finished():
                continue
            self.tree_cache.cache_unfinished_req(req, chunked=True)
            if req.req_pool_idx is not None:
                self.req_to_token_pool.free(req)
            new_staging.append(req)
        self._staging_queue = new_staging

        self._uncond_rids.difference_update(orphaned_uncond_rids)
        self._orphaned_uncond_rids.clear()

        batch.filter_batch(chunked_req_to_exclude=list(exclude))
