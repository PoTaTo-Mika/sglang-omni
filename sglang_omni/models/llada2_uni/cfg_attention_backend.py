# SPDX-License-Identifier: Apache-2.0
"""FlashInfer backend for LLaDA2 image-edit CFG padding."""

from __future__ import annotations

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
    merge_state,
)


class LLaDA2CFGFlashInferAttnBackend(FlashInferAttnBackend):
    """Stock FlashInfer plus opt-in image-edit CFG left-pad masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )

    @staticmethod
    def _clear_stale_ragged_custom_mask(ragged_prefill_wrapper) -> None:
        """Clear a previous edit batch's private FlashInfer custom-mask state."""
        if ragged_prefill_wrapper.is_cuda_graph_enabled:
            return
        for attr in ("_custom_mask_buf", "_mask_indptr_buf"):
            if not hasattr(ragged_prefill_wrapper, attr):
                raise RuntimeError(
                    f"Unsupported FlashInfer ragged wrapper: missing {attr}"
                )
            setattr(ragged_prefill_wrapper, attr, None)

    def init_forward_metadata(self, forward_batch):
        self._cfg_local_left_pad_active = False

        # Clear FlashInfer 0.5.3's stale ragged custom mask before each non-graph batch.
        cfg_prefill_wrapper = self._cfg_prefill_wrapper_ragged
        self._clear_stale_ragged_custom_mask(cfg_prefill_wrapper)

        left_pad_lens = getattr(forward_batch, "dllm_left_pad_lens", None)
        forward_mode = forward_batch.forward_mode
        if (
            left_pad_lens is None
            or not forward_mode.is_dllm_extend()
            or not bool(torch.any(left_pad_lens > 0).item())
        ):
            return super().init_forward_metadata(forward_batch)

        if self.num_wrappers != 1:
            raise RuntimeError(
                "DLLM edit CFG padding currently requires one "
                "FlashInfer attention wrapper"
            )

        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        left_pad_lens = left_pad_lens.to(device=seq_lens.device, dtype=seq_lens.dtype)
        if left_pad_lens.numel() != seq_lens.numel():
            raise RuntimeError(
                f"CFG pad metadata has {left_pad_lens.numel()} entries for "
                f"batch size {seq_lens.numel()}"
            )
        # Split edit padding across the cached prefix and current query span.
        query_lens = seq_lens - prefix_lens
        if bool(torch.any(query_lens <= 0).item()):
            raise RuntimeError("DLLM CFG batch contains an empty active block")
        cached_left_pad_lens = torch.minimum(left_pad_lens, prefix_lens)
        local_left_pad_lens = torch.minimum(
            torch.clamp(left_pad_lens - prefix_lens, min=0), query_lens
        )
        if bool(torch.any(local_left_pad_lens > 0).item()):
            flattened_request_masks = []
            for query_length, local_left_pad_length in zip(
                query_lens.tolist(), local_left_pad_lens.tolist()
            ):
                request_attention_mask = torch.ones(
                    (query_length, query_length),
                    dtype=torch.bool,
                    device=seq_lens.device,
                )
                if local_left_pad_length:
                    request_attention_mask[:, :local_left_pad_length] = False
                    # Give discarded pad queries a diagonal key to avoid invalid softmax.
                    pad_indices = torch.arange(
                        local_left_pad_length, device=seq_lens.device
                    )
                    request_attention_mask[pad_indices, pad_indices] = True
                flattened_request_masks.append(request_attention_mask.flatten())
            custom_mask = torch.cat(flattened_request_masks)
            qo_indptr = torch.zeros(
                seq_lens.numel() + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            qo_indptr[1:] = torch.cumsum(query_lens, dim=0)
            prefill_indices_updater = self.indices_updater_prefill
            # Exclude edit pads from cached-prefix attention without replacing the custom mask.
            paged_kernel_lens = prefix_lens - cached_left_pad_lens
            prefill_indices_updater.call_begin_forward(
                cfg_prefill_wrapper,
                self.prefill_wrappers_paged[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                int(paged_kernel_lens.sum().item()),
                seq_lens,
                prefix_lens,
                cached_left_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                False,
                None,
                fixed_split_size=self.prefill_split_tile_size,
            )
            cfg_prefill_wrapper.begin_forward(
                qo_indptr,
                qo_indptr,
                prefill_indices_updater.num_qo_heads,
                prefill_indices_updater.num_kv_heads,
                prefill_indices_updater.head_dim,
                custom_mask=custom_mask,
                causal=False,
                q_data_type=prefill_indices_updater.q_data_type,
                kv_data_type=prefill_indices_updater.data_type,
                non_blocking=True,
                fixed_split_size=self.prefill_split_tile_size,
            )
            self._cfg_local_left_pad_active = True
            self._cfg_has_cached_prefix = bool(torch.any(paged_kernel_lens > 0).item())
        else:
            self._cfg_local_left_pad_active = False
            paged_kernel_lens = prefix_lens - cached_left_pad_lens
            prefill_indices_updater = self.indices_updater_prefill
            prefill_indices_updater.call_begin_forward(
                prefill_indices_updater.prefill_wrapper_ragged,
                self.prefill_wrappers_paged[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                int(paged_kernel_lens.sum().item()),
                seq_lens,
                prefix_lens,
                cached_left_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                True,
                None,
                fixed_split_size=self.prefill_split_tile_size,
            )

        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_paged,
            use_ragged=True,
            extend_no_prefix=False,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        if not getattr(self, "_cfg_local_left_pad_active", False):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        if k is None or v is None:
            raise RuntimeError("CFG first-block attention requires explicit K/V")

        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_view = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_view = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if getattr(self, "_cfg_has_cached_prefix", False):
            current_output, current_lse = (
                self._cfg_prefill_wrapper_ragged.forward_return_lse(
                    q_view,
                    k_view,
                    v_view,
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )
            )
            cached_output, cached_lse = self.prefill_wrappers_paged[
                0
            ].forward_return_lse(
                q_view,
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )
            attention_output, _ = merge_state(
                current_output, current_lse, cached_output, cached_lse
            )
        else:
            attention_output = self._cfg_prefill_wrapper_ragged.forward(
                q_view,
                k_view,
                v_view,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )
        if save_kv_cache:
            kv_cache_location = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, kv_cache_location, k, v, layer.k_scale, layer.v_scale
            )
        return attention_output.view(-1, layer.tp_q_head_num * layer.head_dim)


def register_llada2_cfg_flashinfer_backend() -> None:
    """Install the LLaDA2-compatible FlashInfer factory before runner init."""

    def _create_backend(runner):
        if runner.use_mla_backend:
            raise ValueError("LLaDA2 CFG attention does not use an MLA backend")
        return LLaDA2CFGFlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )

    # Keep normal validation and delegate non-padded batches to stock FlashInfer.
    ATTENTION_BACKENDS["flashinfer"] = _create_backend
