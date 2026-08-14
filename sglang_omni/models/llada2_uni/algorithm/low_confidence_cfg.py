# SPDX-License-Identifier: Apache-2.0
"""LowConfidence with Classifier-Free Guidance for dLLM mask diffusion.

Supports three modes:
- **batch=1 (no CFG)**: identical to upstream LowConfidence.
- **batch=2 (CFG)**: cond + uncond Reqs; ``guided = uncond + cfg_scale * (cond - uncond)``.
- **batch=3 (editing CFG)**: cond + no-text + no-image Reqs; three-way guidance
  ``guided = no_text + cfg_text*(full - no_text) + cfg_image*(no_text - no_img)``.

The uncond Req is created by SGLangDLLMBatchPlanner and shares the
same ScheduleBatch.  No manual KV cache management is needed.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _should_force_image_only(req: object) -> bool:
    task_kind = getattr(req, "_task_kind", "chat")
    if task_kind == "interleaved":
        return getattr(req, "_interleaved_phase", None) == "image"

    is_thinking_phase1 = getattr(req, "_is_thinking_phase1", False)
    return task_kind in ("t2i", "edit") and not is_thinking_phase1


def _allowed_image_stop_token_ids(
    req: object | None,
    *,
    image_token_offset: int,
) -> tuple[int, ...]:
    """Return text-vocabulary stop tokens allowed during image generation."""
    if (
        req is None
        or getattr(req, "_task_kind", None) != "interleaved"
        or getattr(req, "_interleaved_phase", None) != "image"
    ):
        return ()
    return tuple(
        sorted(
            int(token_id)
            for token_id in getattr(req, "eos_token_ids", ())
            if 0 <= int(token_id) < image_token_offset
        )
    )


def _get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    """Compute per-step minimum transfer count schedule."""
    steps = min(max(steps, 1), block_length)
    base = block_length // steps
    remainder = block_length % steps
    schedule = torch.full((steps,), base, dtype=torch.int64)
    schedule[:remainder] += 1
    return schedule


def _slice_cfg_output_ids(
    ids: torch.Tensor,
    start_list: list[int],
    cond_idx: int,
    *,
    is_dllm_prefill: bool,
) -> list[torch.Tensor]:
    """Return physically aligned generated tokens for both CFG branches."""
    if is_dllm_prefill:
        return [ids[i, :0] for i in range(ids.shape[0])]

    # Left padding uses mask-token IDs, so counting masks in the unconditional
    # branch would move its output start into the prompt. CFG generation tokens
    # are physically aligned with the conditional branch after padding.
    generation_start = start_list[cond_idx]
    return [ids[i, generation_start:] for i in range(ids.shape[0])]


def _argmax_confidence_from_logits(
    logits: torch.Tensor,
    *,
    use_triton: bool = False,
    force_image_only: bool = False,
    image_token_offset: int = 0,
    allowed_token_ids: tuple[int, ...] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return argmax token ids and their softmax probabilities.

    The Triton path fuses the large-vocab argmax and winning-token confidence
    computation and avoids materializing the full softmax tensor.
    """
    if use_triton and logits.is_cuda:
        try:
            from sglang_omni.models.llada2_uni.algorithm.triton_decode import (
                argmax_confidence_triton,
            )

            return argmax_confidence_triton(
                logits,
                force_image_only=force_image_only,
                image_token_offset=image_token_offset,
                allowed_token_ids=allowed_token_ids,
            )
        except Exception:  # pragma: no cover - serving safety fallback
            logger.exception(
                "Triton LowConfidence argmax/confidence failed; falling back"
            )

    work = logits
    if force_image_only:
        work = logits.clone()
        work[:, :image_token_offset] = float("-inf")
        if allowed_token_ids:
            allowed = torch.tensor(allowed_token_ids, device=logits.device)
            work[:, allowed] = logits[:, allowed]
    token_ids = torch.argmax(work, dim=-1)
    probs = torch.gather(
        F.softmax(work, dim=-1), dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_ids, probs


def _select_keep_by_confidence(
    conf: torch.Tensor,
    threshold: float,
    num_to_transfer: int,
) -> torch.Tensor:
    """Implement LowConfidence keep selection without a high_conf.sum().item sync."""
    k = min(num_to_transfer, conf.numel())
    _, idx = torch.topk(conf, k=k)
    topk_keep = torch.zeros_like(conf, dtype=torch.bool)
    topk_keep.scatter_(0, idx, True)

    high_conf = conf > threshold
    kth_conf = torch.gather(conf, 0, idx[-1:]).squeeze(0)
    return torch.where(kth_conf > threshold, high_conf, topk_keep)


class LowConfidenceCFG(DllmAlgorithm):
    """LowConfidence unmasking with per-step Classifier-Free Guidance."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.image_token_offset = config.algorithm_config.get(
            "image_token_offset", 157184
        )
        self.use_triton_decode = True

    # ------------------------------------------------------------------
    # Standard (no-CFG) run — identical to upstream LowConfidence
    # ------------------------------------------------------------------

    def _run_standard(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        remaining_masked = 0
        remaining_masked_by_block = []
        for block_id in range(batch_size):
            s = block_id * self.block_size
            e = s + self.block_size
            blk = forward_batch.input_ids[s:e]
            start = self.block_size - int((blk == self.mask_id).sum().item())
            start_list.append(start)
            block_remaining = self.block_size - start
            remaining_masked_by_block.append(block_remaining)
            remaining_masked += block_remaining

        # Determine steps, schedule, and task kind
        reqs = getattr(forward_batch, "reqs", None)
        dllm_steps = self.block_size
        force_image_only = False
        allowed_token_ids: tuple[int, ...] = ()
        if reqs:
            dllm_steps = getattr(reqs[0], "_dllm_steps", self.block_size)
            force_image_only = _should_force_image_only(reqs[0])
            allowed_token_ids = _allowed_image_stop_token_ids(
                reqs[0], image_token_offset=self.image_token_offset
            )
        schedule = _get_num_transfer_tokens(self.block_size, dllm_steps)

        for num_to_transfer_tensor in schedule:
            if remaining_masked <= 0:
                break
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, _can_run_cuda_graph = out.logits_output, out.can_run_graph
            num_to_transfer = num_to_transfer_tensor.item()
            for bid in range(batch_size):
                if remaining_masked_by_block[bid] <= 0:
                    continue
                cs = bid * self.block_size
                ce = cs + self.block_size
                blk_ids = forward_batch.input_ids[cs:ce]
                blk_mask = blk_ids == self.mask_id
                logits = logits_output.full_logits[cs:ce]
                block_tpf = 0
                if (
                    self.use_triton_decode
                    and logits.is_cuda
                    and logits.dtype == torch.float32
                ):
                    try:
                        from sglang_omni.models.llada2_uni.algorithm.triton_decode import (
                            low_confidence_update_triton,
                        )

                        block_tpf = low_confidence_update_triton(
                            logits,
                            forward_batch.input_ids,
                            input_start=cs,
                            mask_id=self.mask_id,
                            threshold=self.threshold,
                            num_to_transfer=num_to_transfer,
                            force_image_only=force_image_only,
                            image_token_offset=self.image_token_offset,
                            allowed_token_ids=allowed_token_ids,
                        )
                    except Exception:  # pragma: no cover - serving safety fallback
                        logger.exception(
                            "Triton LowConfidence update failed; falling back"
                        )
                        block_tpf = 0

                if block_tpf == 0:
                    x, p = _argmax_confidence_from_logits(
                        logits,
                        use_triton=self.use_triton_decode,
                        force_image_only=force_image_only,
                        image_token_offset=self.image_token_offset,
                        allowed_token_ids=allowed_token_ids,
                    )
                    x = torch.where(blk_mask, x, blk_ids)
                    conf = torch.where(blk_mask, p, -np.inf)
                    keep = (
                        _select_keep_by_confidence(
                            conf, self.threshold, num_to_transfer
                        )
                        & blk_mask
                    )
                    blk_ids[keep] = x[keep]
                    block_tpf = int(keep.sum().item())

                remaining_masked_by_block[bid] -= block_tpf
                remaining_masked -= block_tpf

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        return (
            out.logits_output,
            [ids[i, start_list[i] :] for i in range(batch_size)],
            out.can_run_graph,
        )

    # ------------------------------------------------------------------
    # Batch=2 CFG run
    # ------------------------------------------------------------------

    def _run_cfg_batch2(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        cond_idx: int,
        uncond_idx: int,
        cfg_scale: float,
        cfg_rescale: float,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        bs = self.block_size
        reqs = getattr(forward_batch, "reqs", None)

        # CFG left pads are prompt padding, so prefill only builds KV state.
        if reqs and all(req.is_dllm_prefill() for req in reqs):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        mask_index = forward_batch.input_ids == self.mask_id
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # Compute start positions for all Reqs
        start_list = []
        for bid in range(batch_size):
            s = bid * bs
            e = s + bs
            blk = forward_batch.input_ids[s:e]
            start_list.append(bs - int((blk == self.mask_id).sum().item()))

        # Cond block boundaries
        cs = cond_idx * bs
        ce = cs + bs
        # Uncond block boundaries
        us = uncond_idx * bs
        ue = us + bs

        # Determine steps, schedule, and task kind from cond Req
        dllm_steps = bs
        force_image_only = False
        allowed_token_ids: tuple[int, ...] = ()
        if reqs and len(reqs) > cond_idx:
            dllm_steps = getattr(reqs[cond_idx], "_dllm_steps", bs)
            force_image_only = _should_force_image_only(reqs[cond_idx])
            allowed_token_ids = _allowed_image_stop_token_ids(
                reqs[cond_idx], image_token_offset=self.image_token_offset
            )
        schedule = _get_num_transfer_tokens(bs, dllm_steps)

        for num_to_transfer_tensor in schedule:
            cond_mask = forward_batch.input_ids[cs:ce] == self.mask_id
            num_masked_tokens = int(cond_mask.sum().item())
            if num_masked_tokens == 0:
                break

            # Single forward (batch=2)
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            full_logits = out.logits_output.full_logits

            # Split logits
            cond_logits = full_logits[cs:ce]
            uncond_logits = full_logits[us:ue]

            # CFG formula
            guided = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

            # CFG rescale (variance normalization)
            if cfg_rescale > 0:
                std_c = cond_logits.std(dim=-1, keepdim=True)
                std_g = guided.std(dim=-1, keepdim=True)
                rescaled = guided * (std_c / (std_g + 1e-6))
                guided = cfg_rescale * rescaled + (1.0 - cfg_rescale) * guided

            # Confidence-based unmasking with the fixed transfer schedule.
            blk_ids = forward_batch.input_ids[cs:ce]
            x, p = _argmax_confidence_from_logits(
                guided,
                use_triton=self.use_triton_decode,
                force_image_only=force_image_only,
                image_token_offset=self.image_token_offset,
                allowed_token_ids=allowed_token_ids,
            )
            x = torch.where(cond_mask, x, blk_ids)
            conf = torch.where(cond_mask, p, -np.inf)

            num_to_transfer = num_to_transfer_tensor.item()
            keep = (
                _select_keep_by_confidence(conf, self.threshold, num_to_transfer)
                & cond_mask
            )

            # Write to cond block and mirror selected tokens to uncond block.
            blk_ids[keep] = x[keep]
            forward_batch.input_ids[us:ue][keep] = x[keep]

        # Final forward
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        # Prefill only builds prompt KV and must not emit prompt/padding tokens.
        ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        is_dllm_prefill = bool(reqs) and all(req.is_dllm_prefill() for req in reqs)
        next_token_ids_list = _slice_cfg_output_ids(
            ids,
            start_list,
            cond_idx,
            is_dllm_prefill=is_dllm_prefill,
        )

        return (
            out.logits_output,
            next_token_ids_list,
            out.can_run_graph,
        )

    # ------------------------------------------------------------------
    # Batch=3 CFG run (editing: full / no-text / no-image)
    # ------------------------------------------------------------------

    def _run_cfg_batch3(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        cond_idx: int,
        no_text_idx: int,
        no_img_idx: int,
        cfg_text_scale: float,
        cfg_image_scale: float,
        cfg_rescale: float,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        bs = self.block_size
        reqs = getattr(forward_batch, "reqs", None)

        # CFG left pads are prompt padding, so prefill only builds KV state.
        if reqs and all(req.is_dllm_prefill() for req in reqs):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        mask_index = forward_batch.input_ids == self.mask_id
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # Compute start positions for all Reqs
        start_list = []
        for bid in range(batch_size):
            s = bid * bs
            e = s + bs
            blk = forward_batch.input_ids[s:e]
            start_list.append(bs - int((blk == self.mask_id).sum().item()))

        cs = cond_idx * bs
        ce = cs + bs
        nt_s = no_text_idx * bs
        nt_e = nt_s + bs
        ni_s = no_img_idx * bs
        ni_e = ni_s + bs

        # Determine steps, schedule, and task kind from cond Req
        dllm_steps = bs
        force_image_only = False
        allowed_token_ids: tuple[int, ...] = ()
        if reqs and len(reqs) > cond_idx:
            dllm_steps = getattr(reqs[cond_idx], "_dllm_steps", bs)
            force_image_only = _should_force_image_only(reqs[cond_idx])
            allowed_token_ids = _allowed_image_stop_token_ids(
                reqs[cond_idx], image_token_offset=self.image_token_offset
            )
        schedule = _get_num_transfer_tokens(bs, dllm_steps)

        for num_to_transfer_tensor in schedule:
            cond_mask = forward_batch.input_ids[cs:ce] == self.mask_id
            num_masked_tokens = int(cond_mask.sum().item())
            if num_masked_tokens == 0:
                break

            # Single forward (batch=3)
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            full_logits = out.logits_output.full_logits

            cond_logits = full_logits[cs:ce]
            no_text_logits = full_logits[nt_s:nt_e]
            no_img_logits = full_logits[ni_s:ni_e]

            # Three-way editing CFG:
            # logits = no_text + cfg_text*(full - no_text) + cfg_image*(no_text - no_img)
            guided = (
                no_text_logits
                + cfg_text_scale * (cond_logits - no_text_logits)
                + cfg_image_scale * (no_text_logits - no_img_logits)
            )

            # CFG rescale (variance normalization)
            if cfg_rescale > 0:
                std_c = cond_logits.std(dim=-1, keepdim=True)
                std_g = guided.std(dim=-1, keepdim=True)
                rescaled = guided * (std_c / (std_g + 1e-6))
                guided = cfg_rescale * rescaled + (1.0 - cfg_rescale) * guided

            # Confidence-based unmasking with the fixed transfer schedule.
            blk_ids = forward_batch.input_ids[cs:ce]
            x, p = _argmax_confidence_from_logits(
                guided,
                use_triton=self.use_triton_decode,
                force_image_only=force_image_only,
                image_token_offset=self.image_token_offset,
                allowed_token_ids=allowed_token_ids,
            )
            x = torch.where(cond_mask, x, blk_ids)
            conf = torch.where(cond_mask, p, -np.inf)

            num_to_transfer = num_to_transfer_tensor.item()
            keep = (
                _select_keep_by_confidence(conf, self.threshold, num_to_transfer)
                & cond_mask
            )

            # Write to cond block and mirror selected tokens to uncond block.
            blk_ids[keep] = x[keep]
            forward_batch.input_ids[nt_s:nt_e][keep] = x[keep]
            forward_batch.input_ids[ni_s:ni_e][keep] = x[keep]

        # Final forward
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        is_dllm_prefill = bool(reqs) and all(req.is_dllm_prefill() for req in reqs)
        next_token_ids_list = _slice_cfg_output_ids(
            ids,
            start_list,
            cond_idx,
            is_dllm_prefill=is_dllm_prefill,
        )

        return (
            out.logits_output,
            next_token_ids_list,
            out.can_run_graph,
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        left_pad_lens = getattr(forward_batch, "dllm_left_pad_lens_cpu", None)
        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        graph_runner = model_runner.graph_runner
        if (
            graph_runner is not None
            and left_pad_lens is not None
            and prefix_lens is not None
            and any(
                pad_len > prefix_len
                for pad_len, prefix_len in zip(left_pad_lens, prefix_lens)
            )
        ):
            # The query-padding path uses a custom attention branch that is not
            # part of the captured graph. Keep this block eager until all CFG
            # padding has moved into the cached prefix.
            model_runner.graph_runner = None

        try:
            return self._run(model_runner, forward_batch)
        finally:
            model_runner.graph_runner = graph_runner

    def _run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        reqs = getattr(forward_batch, "reqs", None)
        batch_size = forward_batch.batch_size

        cfg_marked = bool(reqs) and any(
            getattr(req, "_is_uncond", False)
            or getattr(req, "_cfg_group_rid", None) is not None
            for req in reqs
        )
        if cfg_marked:
            cond_indices = [
                i for i, req in enumerate(reqs) if not getattr(req, "_is_uncond", False)
            ]
            uncond_text_indices = [
                i
                for i, req in enumerate(reqs)
                if getattr(req, "_is_uncond", False)
                and not getattr(req, "_is_uncond_img", False)
            ]
            uncond_img_indices = [
                i for i, req in enumerate(reqs) if getattr(req, "_is_uncond_img", False)
            ]
            valid_roles = (
                len(reqs) == batch_size
                and len(cond_indices) == 1
                and len(uncond_text_indices) == 1
                and (
                    (batch_size == 2 and not uncond_img_indices)
                    or (batch_size == 3 and len(uncond_img_indices) == 1)
                )
            )
            cond_idx = cond_indices[0] if len(cond_indices) == 1 else None
            cond_rid = reqs[cond_idx].rid if cond_idx is not None else None
            valid_group = cond_rid is not None and all(
                getattr(req, "_cfg_group_rid", None) == cond_rid for req in reqs
            )
            if not valid_roles or not valid_group:
                raise RuntimeError(
                    "Malformed CFG batch: expected one conditional request, "
                    "one no-text companion, and for batch=3 one no-image "
                    "companion from the same request group"
                )

            uncond_text_idx = uncond_text_indices[0]
            cfg_text_scale = getattr(reqs[cond_idx], "_cfg_scale", 4.0)
            cfg_rescale = getattr(reqs[cond_idx], "_cfg_rescale", 0.7)
            if batch_size == 3:
                uncond_img_idx = uncond_img_indices[0]
                cfg_image_scale = getattr(reqs[cond_idx], "_cfg_image_scale", 0.0)
                return self._run_cfg_batch3(
                    model_runner,
                    forward_batch,
                    cond_idx,
                    uncond_text_idx,
                    uncond_img_idx,
                    cfg_text_scale,
                    cfg_image_scale,
                    cfg_rescale,
                )
            return self._run_cfg_batch2(
                model_runner,
                forward_batch,
                cond_idx,
                uncond_text_idx,
                cfg_text_scale,
                cfg_rescale,
            )

        # Standard path (no CFG)
        return self._run_standard(model_runner, forward_batch)


Algorithm = LowConfidenceCFG
