# SPDX-License-Identifier: Apache-2.0
"""SGLang-native recurrent runner for VoxCPM continuous acoustic latents."""

from __future__ import annotations

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
    decoder_initialized: bool = False


class VoxCPMModelRunner(ModelRunner):
    """Runs one batched CFM sample and one incremental VAE decode per AR step."""

    def __init__(self, tp_worker: Any, output_processor: Any, *, variant: str):
        super().__init__(tp_worker, output_processor)
        self.variant = variant
        self._states: dict[str, _RequestRuntime] = {}
        self._outbox: Any | None = None
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
        for sched_req in requests:
            if sched_req.request_id not in self._states:
                row = int(self.model.acquire_row(sched_req.request_id))
                self._states[sched_req.request_id] = _RequestRuntime(row=row)
            sched_req.data.row_id = self._states[sched_req.request_id].row

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
        if is_lookahead:
            raise RuntimeError("VoxCPM async lookahead is unsupported")
        rows = torch.tensor(
            [self._states[req.request_id].row for req in requests],
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[: len(requests)].copy_(rows)

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
        del forward_batch
        if not requests:
            return
        hidden = result.logits_output.hidden_states
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        device, dtype = hidden.device, hidden.dtype
        rows = torch.tensor(
            [self._states[req.request_id].row for req in requests],
            dtype=torch.long,
            device=device,
        )
        temperatures = torch.tensor(
            [
                float(req.data.state.generation_kwargs["temperature"])
                for req in requests
            ],
            dtype=dtype,
            device=device,
        )
        cfg_values = torch.tensor(
            [float(req.data.state.generation_kwargs["cfg_value"]) for req in requests],
            dtype=dtype,
            device=device,
        )
        noise = torch.empty(
            len(requests),
            int(self.model.model.feat_dim),
            int(self.model.model.patch_size),
            dtype=dtype,
            device=device,
        )
        for index, sched_req in enumerate(requests):
            values = sched_req.data.state.generation_kwargs
            seed = values.get("seed")
            if seed is None or int(seed) < 0:
                noise[index].normal_()
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    derive_step_seed(int(seed), int(sched_req.data.generation_steps))
                )
                noise[index].copy_(
                    torch.randn(
                        noise[index].shape,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                    )
                )
        generated = self._generate_grouped(
            hidden=hidden,
            rows=rows,
            temperatures=temperatures,
            cfg_values=cfg_values,
            noise=noise,
            requests=requests,
        )
        latents = generated["latents"]
        stop_flags = generated["stop_flag"].detach().cpu().tolist()
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
        audio_batch = (
            self._decoder.decode_chunks(
                latents.float().transpose(1, 2),
                [req.request_id for req in requests],
                initial_contexts,
            )
            .detach()
            .to("cpu", torch.float32)
        )

        next_ids: list[int] = []
        for index, sched_req in enumerate(requests):
            data = sched_req.data
            runtime = self._states[sched_req.request_id]
            runtime.decoder_initialized = True
            latent = latents[index].detach().to("cpu", torch.float32)
            audio = audio_batch[index].reshape(-1).contiguous()
            runtime.generated.append(latent)
            runtime.audio.append(audio)
            data.generated_latents.append(latent)
            data.generated_audio.append(audio)
            step_count = int(data.generation_steps) + 1
            values = data.state.generation_kwargs
            stopped = bool(stop_flags[index]) and step_count >= int(values["min_len"])
            at_limit = step_count >= int(values["max_new_tokens"])
            if stopped:
                data.stop_step = step_count - 1
                data.finish_kind = "stop"
            elif at_limit:
                data.finish_kind = "length"
            next_ids.append(
                int(data.stop_token_id)
                if stopped or at_limit
                else int(data.continue_token_id)
            )
            if data.is_streaming:
                runtime.stream_pending.append(audio)
                prefix = int(values.get("streaming_prefix_len", 1))
                if len(runtime.audio) >= prefix or stopped or at_limit:
                    self._emit_audio(sched_req.request_id, runtime.stream_pending)
                    runtime.stream_pending.clear()

        next_token_ids = torch.tensor(next_ids, dtype=torch.long, device=device)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

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
        inference_timesteps = torch.tensor(
            [
                int(req.data.state.generation_kwargs["inference_timesteps"])
                for req in requests
            ],
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
