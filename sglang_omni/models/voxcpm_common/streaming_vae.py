# SPDX-License-Identifier: Apache-2.0
"""Request-keyed incremental decoder for causal VoxCPM AudioVAEs."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch
from torch import nn


class BatchedStreamingVAEDecoder:
    """Patch causal layers so dynamically batched streams keep separate state."""

    def __init__(
        self,
        vae: nn.Module,
        causal_conv_type: type[nn.Conv1d],
        causal_transpose_conv_type: type[nn.ConvTranspose1d],
    ) -> None:
        self._vae = vae
        self._conv_type = causal_conv_type
        self._transpose_type = causal_transpose_conv_type
        self._states: dict[int, dict[Hashable, torch.Tensor]] = {}
        self._stream_ids: Sequence[Hashable] | None = None
        self._initialized: set[Hashable] = set()
        self._install()

    @torch.inference_mode()
    def decode_chunks(
        self,
        chunks: torch.Tensor,
        stream_ids: Sequence[Hashable],
        initial_contexts: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if chunks.ndim != 3:
            raise ValueError("VAE chunks must have shape [batch, latent_dim, time]")
        if len(stream_ids) != chunks.shape[0] or len(set(stream_ids)) != len(
            stream_ids
        ):
            raise ValueError("stream_ids must be unique and match the VAE batch")
        contexts = initial_contexts or [None] * len(stream_ids)
        if len(contexts) != len(stream_ids):
            raise ValueError("initial_contexts must match stream_ids")

        missing = object()
        snapshot = {
            key: {rid: states.get(rid, missing) for rid in stream_ids}
            for key, states in self._states.items()
        }
        initialized = self._initialized.intersection(stream_ids)
        try:
            for rid, context in zip(stream_ids, contexts):
                if rid in self._initialized:
                    continue
                if context is not None and context.numel():
                    if context.ndim != 3 or context.shape[0] != 1:
                        raise ValueError(
                            "initial VAE context must have shape [1, D, T]"
                        )
                    self._decode(context, [rid])
                self._initialized.add(rid)
            return self._decode(chunks, stream_ids)
        except Exception:
            for key, states in self._states.items():
                for rid, old in snapshot[key].items():
                    if old is missing:
                        states.pop(rid, None)
                    else:
                        states[rid] = old
            self._initialized.difference_update(stream_ids)
            self._initialized.update(initialized)
            raise

    def release(self, stream_id: Hashable) -> None:
        self._initialized.discard(stream_id)
        for states in self._states.values():
            states.pop(stream_id, None)

    def clear(self) -> None:
        self._initialized.clear()
        for states in self._states.values():
            states.clear()

    def _decode(
        self, chunks: torch.Tensor, stream_ids: Sequence[Hashable]
    ) -> torch.Tensor:
        if self._stream_ids is not None:
            raise RuntimeError("streaming VAE decode is not reentrant")
        self._stream_ids = stream_ids
        try:
            return self._vae.decode(chunks)
        finally:
            self._stream_ids = None

    def _ids(self, batch: int) -> Sequence[Hashable]:
        if self._stream_ids is None or len(self._stream_ids) != batch:
            raise RuntimeError("causal VAE layer ran outside decode_chunks")
        return self._stream_ids

    def _install(self) -> None:
        for module in self._vae.decoder.modules():
            if isinstance(module, self._conv_type):
                padding = module._CausalConv1d__padding * 2 - getattr(
                    module, "_CausalConv1d__output_padding", 0
                )
                if padding > 0:
                    self._patch_conv(module, padding)
            elif isinstance(module, self._transpose_type):
                trim = (
                    module._CausalTransposeConv1d__padding * 2
                    - module._CausalTransposeConv1d__output_padding
                )
                context = module.kernel_size[0] // module.stride[0] - 1
                if context > 0:
                    self._patch_transpose(module, context, trim)

    def _patch_conv(self, module: nn.Conv1d, padding: int) -> None:
        states = self._states.setdefault(id(module), {})

        def forward(
            x: torch.Tensor,
            *,
            _module: nn.Conv1d = module,
            _padding: int = padding,
            _states: dict[Hashable, torch.Tensor] = states,
        ) -> torch.Tensor:
            if self._stream_ids is None:
                return type(_module).forward(_module, x)
            contexts = []
            for index, rid in enumerate(self._ids(x.shape[0])):
                state = _states.get(rid)
                if state is None:
                    state = x.new_zeros((x.shape[1], _padding))
                contexts.append(state)
                _states[rid] = torch.cat((state, x[index]), -1)[:, -_padding:].detach()
            return nn.Conv1d.forward(_module, torch.cat((torch.stack(contexts), x), -1))

        module.forward = forward  # type: ignore[method-assign]

    def _patch_transpose(
        self, module: nn.ConvTranspose1d, context: int, trim: int
    ) -> None:
        states = self._states.setdefault(id(module), {})

        def forward(
            x: torch.Tensor,
            *,
            _module: nn.ConvTranspose1d = module,
            _context: int = context,
            _trim: int = trim,
            _states: dict[Hashable, torch.Tensor] = states,
        ) -> torch.Tensor:
            if self._stream_ids is None:
                return type(_module).forward(_module, x)
            contexts = []
            for index, rid in enumerate(self._ids(x.shape[0])):
                state = _states.get(rid)
                if state is None:
                    state = x.new_zeros((x.shape[1], _context))
                contexts.append(state)
                _states[rid] = torch.cat((state, x[index]), -1)[:, -_context:].detach()
            output = nn.ConvTranspose1d.forward(
                _module, torch.cat((torch.stack(contexts), x), -1)
            )
            left = _context * _module.stride[0]
            return output[..., left:-_trim] if _trim else output[..., left:]

        module.forward = forward  # type: ignore[method-assign]


__all__ = ["BatchedStreamingVAEDecoder"]
