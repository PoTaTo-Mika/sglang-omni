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

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: torch.Tensor | None = None,
    ) -> None:
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)
        if self.skip_prefill or not self.is_dllm_model:
            return
        if self.num_wrappers != 1:
            raise RuntimeError(
                "DLLM edit CFG padding currently requires one "
                "FlashInfer attention wrapper"
            )

        device = self.workspace_buffer.device
        self._cfg_cuda_graph_prefix_lens = torch.empty(
            max_bs, dtype=torch.int32, device=device
        )
        self._cfg_cuda_graph_cached_left_pad_lens = torch.empty(
            max_bs, dtype=torch.int32, device=device
        )
        self._cfg_cuda_graph_paged_kernel_lens = torch.empty(
            max_bs, dtype=torch.int32, device=device
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

    @staticmethod
    def _get_cfg_attention_lengths(forward_batch):
        """Return attention padding lengths from host metadata."""
        left_pad_lens = getattr(forward_batch, "dllm_left_pad_lens_cpu", None)
        if left_pad_lens is None or not any(left_pad_lens):
            return None

        prefix_lens = forward_batch.extend_prefix_lens_cpu
        query_lens = forward_batch.extend_seq_lens_cpu
        if not (
            len(left_pad_lens)
            == len(prefix_lens)
            == len(query_lens)
            == forward_batch.batch_size
        ):
            raise RuntimeError("CFG padding metadata must match batch size")
        if any(query_len <= 0 for query_len in query_lens):
            raise RuntimeError("DLLM CFG batch contains an empty active block")
        local_left_pad_lens = [
            min(max(left_pad - prefix, 0), query)
            for left_pad, prefix, query in zip(left_pad_lens, prefix_lens, query_lens)
        ]
        paged_kernel_num_tokens = sum(
            max(prefix - left_pad, 0)
            for left_pad, prefix in zip(left_pad_lens, prefix_lens)
        )
        return local_left_pad_lens, paged_kernel_num_tokens

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor | None,
        forward_mode,
        spec_info,
        seq_lens_cpu: torch.Tensor | None,
    ) -> None:
        """Refresh attention metadata for a CUDA-graph-replayed DLLM denoise batch.

        The stock replay path (``FlashInferAttnBackend``) keys the cached-prefix
        extent off ``prefix_lens = seq_lens - block_size`` and never reads
        ``forward_batch.dllm_left_pad_lens`` or the real ``extend_prefix_lens``.
        For an uncond CFG branch left-padded on its cached prefix, that makes
        paged attention read the mask-token padding as real KV and lets its
        logits diverge from the eager path.

        Restore the eager handling for the cached-prefix-pad case (the DLLM
        denoise geometry): drop the pad inside the cached prefix from both the
        paged length and KV start offset, then re-plan the captured paged
        wrapper with the corrected values. The algorithm keeps any pad that
        spills into the current query span on the eager custom-mask path.
        """
        forward_batch = getattr(self, "_replay_forward_batch", None)
        padding_metadata = self._get_cfg_attention_lengths(forward_batch)

        if not forward_mode.is_dllm_extend() or padding_metadata is None:
            return super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        local_left_pad_lens_cpu, paged_kernel_num_tokens_cpu = padding_metadata
        raw_bs = int(forward_batch.batch_size)
        if raw_bs > bs:
            raise RuntimeError(
                f"CFG CUDA Graph raw batch {raw_bs} exceeds capture batch {bs}"
            )
        real_prefix_lens = forward_batch.extend_prefix_lens

        # SGLang pads uncaptured raw batches to the next captured size. Mirror
        # that shape for the LLaDA2-specific CFG metadata: real rows keep their
        # eager prefix/pad geometry, while dummy rows have no cached prefix.
        prefix_lens = self._cfg_cuda_graph_prefix_lens[:bs]
        prefix_lens.zero_()
        prefix_lens[:raw_bs].copy_(
            real_prefix_lens.to(device=prefix_lens.device, dtype=prefix_lens.dtype)
        )
        real_left_pad_lens = forward_batch.dllm_left_pad_lens.to(
            device=prefix_lens.device,
            dtype=prefix_lens.dtype,
        )

        cached_left_pad_lens = self._cfg_cuda_graph_cached_left_pad_lens[:bs]
        cached_left_pad_lens.zero_()
        torch.minimum(
            real_left_pad_lens,
            prefix_lens[:raw_bs],
            out=cached_left_pad_lens[:raw_bs],
        )
        if any(local_left_pad_lens_cpu):
            raise RuntimeError(
                "DLLM CFG CUDA Graph replay reached an in-query left pad; "
                "the thinker must run this block eagerly"
            )

        # Keep forward_extend on the stock (captured-wrapper) path.
        self._cfg_local_left_pad_active = False

        paged_kernel_lens = self._cfg_cuda_graph_paged_kernel_lens[:bs]
        torch.sub(prefix_lens, cached_left_pad_lens, out=paged_kernel_lens)
        prefill_indices_updater = self.indices_updater_prefill
        prefill_indices_updater.call_begin_forward(
            prefill_indices_updater.prefill_wrapper_ragged,
            self.prefill_cuda_graph_metadata[bs][0],
            req_pool_indices[:bs],
            paged_kernel_lens,
            paged_kernel_num_tokens_cpu,
            seq_lens[:bs],
            prefix_lens,
            cached_left_pad_lens,
            prefill_indices_updater.kv_indptr[0],
            prefill_indices_updater.qo_indptr[0],
            not self.use_paged,
            None,
            fixed_split_size=self.prefill_split_tile_size,
        )

    def init_forward_metadata(self, forward_batch):
        self._cfg_local_left_pad_active = False

        # Clear FlashInfer 0.5.3's stale ragged custom mask before each non-graph batch.
        cfg_prefill_wrapper = self._cfg_prefill_wrapper_ragged
        self._clear_stale_ragged_custom_mask(cfg_prefill_wrapper)

        padding_metadata = self._get_cfg_attention_lengths(forward_batch)
        forward_mode = forward_batch.forward_mode
        if padding_metadata is None or not forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)

        local_left_pad_lens_cpu, paged_kernel_num_tokens_cpu = padding_metadata
        query_lens_cpu = forward_batch.extend_seq_lens_cpu

        if self.num_wrappers != 1:
            raise RuntimeError(
                "DLLM edit CFG padding currently requires one "
                "FlashInfer attention wrapper"
            )

        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        left_pad_lens = forward_batch.dllm_left_pad_lens.to(
            device=seq_lens.device, dtype=seq_lens.dtype
        )
        # Split edit padding across the cached prefix and current query span.
        query_lens = seq_lens - prefix_lens
        cached_left_pad_lens = torch.minimum(left_pad_lens, prefix_lens)
        if any(local_left_pad_lens_cpu):
            flattened_request_masks = []
            for query_length, local_left_pad_length in zip(
                query_lens_cpu, local_left_pad_lens_cpu
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
                paged_kernel_num_tokens_cpu,
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
            self._cfg_has_cached_prefix = paged_kernel_num_tokens_cpu > 0
        else:
            self._cfg_local_left_pad_active = False
            paged_kernel_lens = prefix_lens - cached_left_pad_lens
            prefill_indices_updater = self.indices_updater_prefill
            prefill_indices_updater.call_begin_forward(
                prefill_indices_updater.prefill_wrapper_ragged,
                self.prefill_wrappers_paged[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                paged_kernel_num_tokens_cpu,
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
