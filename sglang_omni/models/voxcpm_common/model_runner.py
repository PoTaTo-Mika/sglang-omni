# SPDX-License-Identifier: Apache-2.0
"""SGLang-native recurrent runner for VoxCPM continuous acoustic latents."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.utils.audio_payload import audio_waveform_payload

_MASK64 = (1 << 64) - 1


def derive_step_seed(seed: int, step: int) -> int:
    value = (int(seed) & _MASK64) ^ ((int(step) + 0x9E3779B97F4A7C15) & _MASK64)
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


@dataclass
class _RequestRuntime:
    row: int
    generated: list[torch.Tensor] = field(default_factory=list)
    audio: list[torch.Tensor] = field(default_factory=list)
    stream_pending: list[torch.Tensor] = field(default_factory=list)
    pending_vae_latents: list[torch.Tensor] = field(default_factory=list)
    decoder_initialized: bool = False
    launched_steps: int = 0


@dataclass
class _AsyncStep:
    host_buffer: torch.Tensor
    latent_shape: tuple[int, int]
    audio_width: int
    step_counts: list[int]


class VoxCPMModelRunner(ModelRunner):
    """Runs batched CFM samples and configurable incremental VAE decode groups."""

    def __init__(self, tp_worker: Any, output_processor: Any, *, variant: str):
        super().__init__(tp_worker, output_processor)
        self.variant = variant
        self._states: dict[str, _RequestRuntime] = {}
        self._outbox: Any | None = None
        self._collect_device_staging: torch.Tensor | None = None
        self._latent_collect_staging: torch.Tensor | None = None
        self._vae_decode_every = max(
            1, int(os.getenv("SGLANG_VOXCPM_VAE_DECODE_EVERY", "1"))
        )
        self._vae = self.model.model.vae
        self._decoder = self.model.model.vae_streaming_decoder
        if self._vae is None or self._decoder is None:
            raise RuntimeError("VoxCPM runner requires an attached streaming AudioVAE")

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def reset_request(self, request_id: str) -> None:
        runtime = self._states.pop(request_id, None)
        if self._decoder is not None:
            self._decoder.release(request_id)
        self.model.reset_request(request_id)
        del runtime

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        rows = []
        temperatures = []
        cfg_values = []
        seeds = []
        for sched_req in requests:
            if sched_req.request_id not in self._states:
                row = int(self.model.acquire_row(sched_req.request_id))
                self._states[sched_req.request_id] = _RequestRuntime(row=row)
            row = self._states[sched_req.request_id].row
            sched_req.data.row_id = row
            values = sched_req.data.state.generation_kwargs
            rows.append(row)
            temperatures.append(float(values["temperature"]))
            cfg_values.append(float(values["cfg_value"]))
            seed = values.get("seed")
            seeds.append(-1 if seed is None else int(seed))
        if rows:
            pool = self.model.model.state_pool
            pool.configure(
                rows,
                temperature=torch.tensor(
                    temperatures, device=pool.feedback.device, dtype=pool.feedback.dtype
                ),
                cfg_value=torch.tensor(
                    cfg_values, device=pool.feedback.device, dtype=pool.feedback.dtype
                ),
                seed=torch.tensor(seeds, device=pool.feedback.device, dtype=torch.long),
            )

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch
        embeddings: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        embedding = self.model.get_input_embeddings()
        weight = next(embedding.parameters())

        for sched_req in requests:
            data = sched_req.data
            req = data.req
            prefix = len(req.prefix_indices)
            extend = int(req.extend_input_len)
            stop = prefix + extend
            runtime = self._states[sched_req.request_id]

            prompt_ids = data.input_ids
            generated_count = len(runtime.generated)
            generated_ids = torch.full(
                (generated_count,),
                int(data.continue_token_id),
                dtype=torch.long,
            )
            all_ids = torch.cat((prompt_ids.cpu(), generated_ids), 0)
            prompt_feat = data.feat
            if runtime.generated:
                generated_feat = torch.stack(runtime.generated, 0)
                all_feat = torch.cat((prompt_feat, generated_feat), 0)
                all_mask = torch.cat(
                    (
                        data.feat_mask,
                        torch.ones(generated_count, dtype=torch.bool),
                    ),
                    0,
                )
            else:
                all_feat, all_mask = prompt_feat, data.feat_mask
            if stop > all_ids.shape[0]:
                raise RuntimeError(
                    f"VoxCPM re-prefill range exceeds request history for {sched_req.request_id}"
                )
            ids = all_ids[prefix:stop].to(weight.device)
            embeddings.append(embedding(ids).to(dtype=weight.dtype))
            features.append(all_feat[prefix:stop].to(weight.device, weight.dtype))
            masks.append(all_mask[prefix:stop].to(weight.device))

        input_embeds = torch.cat(embeddings, 0)
        feat = torch.cat(features, 0)
        feat_mask = torch.cat(masks, 0)
        row_ids = torch.tensor(
            [self._states[req.request_id].row for req in requests],
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        return self._forward(
            forward_batch,
            input_embeds=input_embeds,
            feat=feat,
            feat_mask=feat_mask,
            row_ids=row_ids,
        )

    def _forward(
        self,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> Any:
        from sglang.srt.managers.scheduler import GenerationBatchResult

        backend = self.tp_worker.model_runner.attn_backend
        backend.init_forward_metadata(forward_batch)
        positions = (
            forward_batch.mrope_positions
            if forward_batch.mrope_positions is not None
            else forward_batch.positions
        )
        output = self.model(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            feat=feat,
            feat_mask=feat_mask,
            row_ids=row_ids,
        )
        return GenerationBatchResult(logits_output=output, can_run_cuda_graph=False)

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch
        rows = torch.tensor(
            [self._states[req.request_id].row for req in requests],
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        if is_lookahead and rows.numel():
            pool = self.model.model.state_pool
            done = pool.done.index_select(0, rows)
            rows = torch.where(
                done,
                torch.full_like(rows, int(pool.padding_row)),
                rows,
            )
        forward_batch.input_ids[: len(requests)].copy_(rows)
        forward_batch.input_ids = forward_batch.input_ids[: len(requests)]
        forward_batch.voxcpm_rows = rows

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        self._collect_step(result, forward_batch, schedule_batch, requests)

    def post_decode_launch(
        self, result: Any, forward_batch: Any, requests: list
    ) -> _AsyncStep | None:
        if not requests:
            return None
        generated, step_counts = self._generate_step(
            result,
            requests,
            rows=getattr(forward_batch, "voxcpm_rows", None),
        )
        staging, latent_shape, audio_width = self._decode_pack_gpu(
            latents=generated["latents"],
            stop_flags=generated["stop_flag"],
            requests=requests,
        )
        host_buffer = self._next_host_staging(staging.shape, staging.dtype)
        host_buffer[: len(requests)].copy_(staging, non_blocking=True)
        return _AsyncStep(
            host_buffer=host_buffer,
            latent_shape=latent_shape,
            audio_width=audio_width,
            step_counts=step_counts,
        )

    def post_decode_resolve(
        self,
        launch_buf: _AsyncStep | None,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        if launch_buf is None:
            return
        active = [
            index
            for index, req in enumerate(requests)
            if not req.data.req.finished()
            and not bool(getattr(req.data.req, "is_retracted", False))
        ]
        if not active:
            return
        self._collect_host(
            packed_cpu=launch_buf.host_buffer,
            latent_shape=launch_buf.latent_shape,
            audio_width=launch_buf.audio_width,
            step_counts=[launch_buf.step_counts[index] for index in active],
            requests=[requests[index] for index in active],
            source_indices=active,
        )

    def lookahead_eligible(self, batch: Any) -> bool:
        if self._vae_decode_every > 1:
            return False
        reqs = getattr(batch, "reqs", None) or []
        if not reqs or len(reqs) > int(
            getattr(self.model, "diffusion_graph_max_bs", 0)
        ):
            return False
        default_steps = int(self.model.model.feat_decoder.inference_timesteps)
        for req in reqs:
            data = getattr(req, "_omni_data", None)
            if data is None:
                continue
            steps = int(data.state.generation_kwargs["inference_timesteps"])
            if steps != default_steps:
                return False
        return super().lookahead_eligible(batch)

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        batch = getattr(scheduler_output, "batch_data", None)
        if bool(getattr(batch, "is_prefill_only", False)):
            return {req.request_id for req in scheduler_output.requests}
        return set()

    def _collect_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not requests:
            return
        generated, step_counts = self._generate_step(
            result,
            requests,
            rows=getattr(forward_batch, "voxcpm_rows", None),
        )
        self._decode_and_collect(
            latents=generated["latents"],
            stop_flags=generated["stop_flag"],
            step_counts=step_counts,
            requests=requests,
        )
        schedule_batch.output_ids = result.next_token_ids

    def _generate_step(
        self,
        result: Any,
        requests: list,
        *,
        rows: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        hidden = result.logits_output.hidden_states
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        hidden = hidden[: len(requests)]
        device, dtype = hidden.device, hidden.dtype
        if rows is None:
            rows = torch.tensor(
                [self._states[req.request_id].row for req in requests],
                dtype=torch.long,
                device=device,
            )
        pool = self.model.model.state_pool
        temperatures = pool.temperature.index_select(0, rows)
        cfg_values = pool.cfg_value.index_select(0, rows)
        noise = torch.empty(
            len(requests),
            int(self.model.model.feat_dim),
            int(self.model.model.patch_size),
            dtype=dtype,
            device=device,
        )
        unseeded = []
        step_counts = []
        for index, sched_req in enumerate(requests):
            values = sched_req.data.state.generation_kwargs
            runtime = self._states[sched_req.request_id]
            step_count = (
                max(int(runtime.launched_steps), int(sched_req.data.generation_steps))
                + 1
            )
            runtime.launched_steps = step_count
            step_counts.append(step_count)
            seed = values.get("seed")
            if seed is None or int(seed) < 0:
                unseeded.append(index)
                continue
            generator = torch.Generator(device=device)
            generator.manual_seed(derive_step_seed(int(seed), step_count - 1))
            noise[index].copy_(
                torch.randn(
                    noise[index].shape,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
            )
        if unseeded:
            indices = torch.tensor(unseeded, dtype=torch.long, device=device)
            noise.index_copy_(
                0,
                indices,
                torch.randn(
                    len(unseeded),
                    noise.shape[1],
                    noise.shape[2],
                    dtype=dtype,
                    device=device,
                ),
            )
        generated = self._generate_grouped(
            hidden=hidden,
            rows=rows,
            temperatures=temperatures,
            cfg_values=cfg_values,
            noise=noise,
            requests=requests,
        )
        raw_stop = generated["stop_flag"].bool()
        min_lengths = torch.tensor(
            [int(req.data.state.generation_kwargs["min_len"]) for req in requests],
            device=device,
            dtype=torch.long,
        )
        max_lengths = torch.tensor(
            [
                int(req.data.state.generation_kwargs["max_new_tokens"])
                for req in requests
            ],
            device=device,
            dtype=torch.long,
        )
        steps = torch.tensor(step_counts, device=device, dtype=torch.long)
        finished = (raw_stop & (steps >= min_lengths)) | (steps >= max_lengths)
        finished |= rows == int(pool.padding_row)
        pool.done.index_copy_(0, rows, finished)
        stop_ids = torch.tensor(
            [int(req.data.stop_token_id) for req in requests],
            device=device,
            dtype=torch.long,
        )
        continue_ids = torch.tensor(
            [int(req.data.continue_token_id) for req in requests],
            device=device,
            dtype=torch.long,
        )
        result.next_token_ids = torch.where(finished, stop_ids, continue_ids)
        return generated, step_counts

    def _decode_and_collect(
        self,
        *,
        latents: torch.Tensor,
        stop_flags: torch.Tensor,
        step_counts: list[int],
        requests: list,
    ) -> None:
        if not requests:
            return
        if self._vae_decode_every > 1:
            self._decode_and_collect_buffered(
                latents=latents,
                stop_flags=stop_flags,
                step_counts=step_counts,
                requests=requests,
            )
            return
        staging, latent_shape, audio_width = self._decode_pack_gpu(
            latents=latents,
            stop_flags=stop_flags,
            requests=requests,
        )
        packed_cpu = staging.to("cpu").contiguous()
        self._collect_host(
            packed_cpu=packed_cpu,
            latent_shape=latent_shape,
            audio_width=audio_width,
            step_counts=step_counts,
            requests=requests,
            clone_rows=False,
        )

    def _decode_and_collect_buffered(
        self,
        *,
        latents: torch.Tensor,
        stop_flags: torch.Tensor,
        step_counts: list[int],
        requests: list,
    ) -> None:
        device = latents.device
        latent_shape = (int(latents.shape[1]), int(latents.shape[2]))
        latent_width = latent_shape[0] * latent_shape[1]
        capacity = int(self.model.model.state_pool.capacity)
        staging = self._latent_collect_staging
        if (
            staging is None
            or staging.device != device
            or staging.shape != (capacity, latent_width + 1)
        ):
            staging = torch.empty(
                capacity,
                latent_width + 1,
                device=device,
                dtype=torch.float32,
            )
            self._latent_collect_staging = staging
        active = staging[: len(requests)]
        torch.cat(
            (
                latents.detach().float().reshape(len(requests), latent_width),
                stop_flags.detach().float().reshape(-1, 1),
            ),
            dim=1,
            out=active,
        )
        packed_cpu = active.to("cpu").contiguous()

        ready_groups: dict[int, list[tuple[Any, _RequestRuntime, bool, bool]]] = {}
        for index, (sched_req, step_count) in enumerate(zip(requests, step_counts)):
            data = sched_req.data
            runtime = self._states[sched_req.request_id]
            latent = packed_cpu[index, :latent_width].reshape(latent_shape)
            runtime.generated.append(latent)
            data.generated_latents.append(latent)
            runtime.pending_vae_latents.append(latents[index].detach())

            values = data.state.generation_kwargs
            stopped = bool(packed_cpu[index, -1]) and step_count >= int(
                values["min_len"]
            )
            at_limit = step_count >= int(values["max_new_tokens"])
            if stopped:
                data.stop_step = step_count - 1
                data.finish_kind = "stop"
            elif at_limit:
                data.finish_kind = "length"

            pending_count = len(runtime.pending_vae_latents)
            if pending_count >= self._vae_decode_every or stopped or at_limit:
                ready_groups.setdefault(pending_count, []).append(
                    (sched_req, runtime, stopped, at_limit)
                )

        for group in ready_groups.values():
            vae_input = torch.stack(
                [
                    torch.cat(runtime.pending_vae_latents, dim=0)
                    for _, runtime, _, _ in group
                ],
                dim=0,
            ).float()
            initial_contexts = []
            for sched_req, runtime, _, _ in group:
                context = sched_req.data.initial_decode_context
                initial_contexts.append(
                    None
                    if runtime.decoder_initialized or context is None
                    else context.to(device=device, dtype=torch.float32)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                runtime.decoder_initialized = True
            audio_device = self._decoder.decode_chunks(
                vae_input.transpose(1, 2),
                [sched_req.request_id for sched_req, _, _, _ in group],
                initial_contexts,
            )
            audio_cpu = audio_device.detach().float().to("cpu").contiguous()
            for index, (sched_req, runtime, stopped, at_limit) in enumerate(group):
                runtime.pending_vae_latents.clear()
                audio = audio_cpu[index]
                runtime.audio.append(audio)
                sched_req.data.generated_audio.append(audio)
                if sched_req.data.is_streaming:
                    runtime.stream_pending.append(audio)
                    prefix = int(
                        sched_req.data.state.generation_kwargs.get(
                            "streaming_prefix_len", 1
                        )
                    )
                    decoded_patches = len(runtime.generated)
                    if decoded_patches >= prefix or stopped or at_limit:
                        self._emit_audio(sched_req.request_id, runtime.stream_pending)
                        runtime.stream_pending.clear()

    def _decode_pack_gpu(
        self,
        *,
        latents: torch.Tensor,
        stop_flags: torch.Tensor,
        requests: list,
    ) -> tuple[torch.Tensor, tuple[int, int], int]:
        if not requests:
            raise ValueError("cannot pack an empty VoxCPM decode batch")
        device = latents.device
        initial_contexts = []
        for sched_req in requests:
            runtime = self._states[sched_req.request_id]
            context = sched_req.data.initial_decode_context
            initial_contexts.append(
                None
                if runtime.decoder_initialized or context is None
                else context.to(device=device, dtype=torch.float32)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            runtime.decoder_initialized = True
        audio_device = self._decoder.decode_chunks(
            latents.float().transpose(1, 2),
            [req.request_id for req in requests],
            initial_contexts,
        )
        latent_shape = (int(latents.shape[1]), int(latents.shape[2]))
        latent_width = latent_shape[0] * latent_shape[1]
        audio_width = audio_device[0].numel()
        packed_width = latent_width + audio_width + 1
        capacity = int(self.model.model.state_pool.capacity)
        staging = self._collect_device_staging
        if (
            staging is None
            or staging.device != device
            or staging.shape != (capacity, packed_width)
        ):
            staging = torch.empty(
                capacity,
                packed_width,
                device=device,
                dtype=torch.float32,
            )
            self._collect_device_staging = staging
        active = staging[: len(requests)]
        torch.cat(
            (
                latents.detach().float().reshape(len(requests), latent_width),
                audio_device.detach().float().reshape(len(requests), audio_width),
                stop_flags.detach().float().reshape(-1, 1),
            ),
            dim=1,
            out=active,
        )
        return active, latent_shape, int(audio_width)

    def _collect_host(
        self,
        *,
        packed_cpu: torch.Tensor,
        latent_shape: tuple[int, int],
        audio_width: int,
        step_counts: list[int],
        requests: list,
        source_indices: list[int] | None = None,
        clone_rows: bool = True,
    ) -> None:
        latent_width = latent_shape[0] * latent_shape[1]
        if source_indices is None:
            source_indices = list(range(len(requests)))
        for index, (source_index, sched_req) in enumerate(
            zip(source_indices, requests)
        ):
            data = sched_req.data
            runtime = self._states[sched_req.request_id]
            row = packed_cpu[source_index]
            latent = row[:latent_width].reshape(latent_shape)
            audio = row[latent_width : latent_width + audio_width]
            if clone_rows:
                latent = latent.clone()
                audio = audio.clone()
            stop_flag = bool(row[-1])
            runtime.generated.append(latent)
            runtime.audio.append(audio)
            data.generated_latents.append(latent)
            data.generated_audio.append(audio)
            step_count = step_counts[index]
            values = data.state.generation_kwargs
            stopped = stop_flag and step_count >= int(values["min_len"])
            at_limit = step_count >= int(values["max_new_tokens"])
            if stopped:
                data.stop_step = step_count - 1
                data.finish_kind = "stop"
            elif at_limit:
                data.finish_kind = "length"
            if data.is_streaming:
                runtime.stream_pending.append(audio)
                prefix = int(values.get("streaming_prefix_len", 1))
                if len(runtime.audio) >= prefix or stopped or at_limit:
                    self._emit_audio(sched_req.request_id, runtime.stream_pending)
                    runtime.stream_pending.clear()

    def _generate_grouped(
        self,
        *,
        hidden: torch.Tensor,
        rows: torch.Tensor,
        temperatures: torch.Tensor,
        cfg_values: torch.Tensor,
        noise: torch.Tensor,
        requests: list,
    ) -> dict[str, torch.Tensor]:
        """Honor mixed per-request CFM step counts without sharing mutable state."""
        step_counts = [
            int(req.data.state.generation_kwargs["inference_timesteps"])
            for req in requests
        ]
        core_model = getattr(self.model, "model", self.model)
        feat_decoder = getattr(core_model, "feat_decoder", None)
        default_steps = (
            None if feat_decoder is None else int(feat_decoder.inference_timesteps)
        )
        graph_max_bs = int(getattr(self.model, "diffusion_graph_max_bs", 0))
        if (
            step_counts
            and default_steps is not None
            and all(steps == default_steps for steps in step_counts)
            and len(requests) <= graph_max_bs
        ):
            return self.model.generate_batch_graphed(
                hidden,
                rows,
                temperature=temperatures,
                cfg_value=cfg_values,
                z_noise=noise,
            )
        inference_timesteps = torch.tensor(
            step_counts,
            dtype=torch.long,
            device=hidden.device,
        )
        return self.model.generate_batch(
            hidden,
            rows,
            temperature=temperatures,
            cfg_value=cfg_values,
            z_noise=noise,
            inference_timesteps=inference_timesteps,
        )

    def _emit_audio(self, request_id: str, chunks: list[torch.Tensor]) -> None:
        if self._outbox is None or not chunks:
            return
        sample_rate = int(self._vae.out_sample_rate)
        waveform = torch.cat(chunks, -1)
        self._outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data=audio_waveform_payload(
                    waveform,
                    sample_rate=sample_rate,
                    modality="audio",
                    source_hint="VoxCPM streaming",
                ),
                metadata={"modality": "audio"},
            )
        )


__all__ = ["VoxCPMModelRunner", "derive_step_seed"]
