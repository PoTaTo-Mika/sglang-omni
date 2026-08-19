# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Qwen3-ASR packed audio tower.

Fun-ASR graphs a padded (batch, T) encoder. Qwen3 concatenates miss clips
into one packed frame stream, so buckets are (conv chunks, packed tokens,
attention segments). Python split/pad stays outside the graph; a shape that
fits no bucket, or a capture failure, falls back to eager.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.models.qwen3_omni_moe import _get_feat_extract_output_lengths

from sglang_omni.models.qwen3_asr.audio_lengths import qwen3_asr_audio_token_lengths

logger = logging.getLogger(__name__)

_C_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_S_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_N_BUCKET_STEP = 64
_N_BUCKET_MAX = 8192


def _bucket_pow2(value: int, buckets: tuple[int, ...], cap: int) -> int | None:
    if value < 1 or value > cap:
        return None
    for bucket in buckets:
        if bucket > cap:
            break
        if bucket >= value:
            return bucket
    return cap


def _bucket_conv_chunks(chunk_count: int, max_chunks: int) -> int | None:
    return _bucket_pow2(chunk_count, _C_BUCKETS, max_chunks)


def _bucket_attn_segments(segment_count: int, max_segments: int) -> int | None:
    return _bucket_pow2(segment_count, _S_BUCKETS, max_segments)


def _bucket_packed_tokens(token_count: int) -> int | None:
    if token_count < 1 or token_count > _N_BUCKET_MAX:
        return None
    bucket = ((token_count + _N_BUCKET_STEP - 1) // _N_BUCKET_STEP) * _N_BUCKET_STEP
    return min(max(bucket, _N_BUCKET_STEP), _N_BUCKET_MAX)


def _plan_static_shapes(
    *,
    chunk_count: int,
    token_count: int,
    segment_count: int,
    max_chunks: int,
    max_segments: int,
) -> tuple[int, int, int] | None:
    c_bucket = _bucket_conv_chunks(chunk_count, max_chunks)
    s_bucket = _bucket_attn_segments(segment_count, max_segments)
    if c_bucket is None or s_bucket is None:
        return None
    dummy_segments = s_bucket - segment_count
    n_needed = token_count + dummy_segments
    n_bucket = _bucket_packed_tokens(n_needed)
    if n_bucket is None:
        return None
    # note (guozhihao-224): N rounding can leave token padding with no
    # dummy attention slots; bump S so cu_seqlens can absorb it.
    if n_bucket > token_count and dummy_segments == 0:
        s_bucket = _bucket_attn_segments(segment_count + 1, max_segments)
        if s_bucket is None:
            return None
        dummy_segments = s_bucket - segment_count
        n_bucket = _bucket_packed_tokens(token_count + dummy_segments)
        if n_bucket is None:
            return None
    if token_count + dummy_segments > n_bucket:
        n_bucket = _bucket_packed_tokens(token_count + dummy_segments)
        if n_bucket is None:
            return None
    return c_bucket, n_bucket, s_bucket


def _prepare_conv_chunks(
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    *,
    n_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_width = int(n_window) * 2
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_num = torch.ceil(feature_lens / chunk_width).long()
    chunk_lengths = torch.tensor(
        [chunk_width] * int(chunk_num.sum().item()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_width
    chunk_lengths[chunk_lengths == 0] = chunk_width

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(
        chunk_list, batch_first=True
    ).transpose(1, 2)
    # note (guozhihao-224): pad to n_window*2 so conv T and the dummy gather
    # index stay static even when every tail is shorter than the window.
    if padded_feature.shape[-1] < chunk_width:
        padded_feature = F.pad(
            padded_feature, (0, chunk_width - padded_feature.shape[-1])
        )

    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    full_t_cnn = int(
        _get_feat_extract_output_lengths(
            torch.tensor([chunk_width], device=feature_lens.device, dtype=torch.long)
        ).item()
    )
    idx = torch.arange(full_t_cnn, device=padded_feature.device)
    padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)
    padded_feature = padded_feature.unsqueeze(1)
    return padded_feature, padded_mask_after_cnn, aftercnn_lens


def _build_cu_seqlens(
    aftercnn_lens: torch.Tensor,
    *,
    n_window: int,
    n_window_infer: int,
    chunk_aftercnn_width: int,
) -> torch.Tensor:
    window_aftercnn = chunk_aftercnn_width * (
        int(n_window_infer) // (int(n_window) * 2)
    )
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens.tolist():
        cnn_len = int(cnn_len)
        if window_aftercnn <= 0:
            cu_chunk_lens.append(cnn_len)
            continue
        num_full = cnn_len // window_aftercnn
        remainder = cnn_len % window_aftercnn
        cu_chunk_lens.extend([window_aftercnn] * num_full)
        if remainder:
            cu_chunk_lens.append(remainder)
    return torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
        -1, dtype=torch.int32
    )


def _pad_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    s_bucket: int,
    n_bucket: int,
) -> torch.Tensor:
    n_real = int(cu_seqlens[-1].item())
    s_real = int(cu_seqlens.numel()) - 1
    dummy = s_bucket - s_real
    leftover = n_bucket - n_real
    values = cu_seqlens.tolist()
    cursor = n_real
    if dummy == 0:
        if leftover != 0:
            raise ValueError("token padding requires at least one dummy segment")
        out = torch.tensor(values, dtype=torch.int32, device=cu_seqlens.device)
        return out
    dummy_lens = [1] * dummy
    dummy_lens[-1] += leftover - dummy
    if dummy_lens[-1] < 1:
        raise ValueError("dummy attention segments cannot be empty")
    for length in dummy_lens:
        cursor += int(length)
        values.append(cursor)
    if cursor != n_bucket:
        raise ValueError(f"padded cu_seqlens end at {cursor}, expected {n_bucket}")
    return torch.tensor(values, dtype=torch.int32, device=cu_seqlens.device)


def _capture_cu_seqlens(
    s_bucket: int, n_bucket: int, device: torch.device
) -> torch.Tensor:
    # note (guozhihao-224): FA varlen cannot capture a zero-length segment.
    if s_bucket < 1 or n_bucket < s_bucket:
        raise ValueError(
            f"need n_bucket >= s_bucket >= 1, got N={n_bucket} S={s_bucket}"
        )
    first = n_bucket - (s_bucket - 1)
    base = torch.tensor([0, first], dtype=torch.int32, device=device)
    return _pad_cu_seqlens(base, s_bucket=s_bucket, n_bucket=n_bucket)


def _gather_indices(
    padded_mask_after_cnn: torch.Tensor,
    *,
    n_bucket: int,
    dummy_src: int,
) -> torch.Tensor:
    valid = padded_mask_after_cnn.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    n_real = int(valid.numel())
    idx = valid.new_full((n_bucket,), dummy_src)
    idx[:n_real] = valid
    return idx


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    padded_feature: torch.Tensor
    gather_index: torch.Tensor
    cu_seqlens: torch.Tensor
    output: torch.Tensor


class Qwen3ASREncoderCudaGraphRunner:
    """Capture-once/replay per (C, N, S) packed-encoder bucket."""

    def __init__(
        self,
        audio_tower,
        *,
        max_batch_size: int = 8,
        max_clip_frames: int = 6000,
        min_free_gb: float = 3.0,
        warmup_iters: int = 3,
    ) -> None:
        self._audio_tower = audio_tower
        reference = next(audio_tower.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._n_window = int(audio_tower.n_window)
        self._n_window_infer = int(audio_tower.n_window_infer)
        self._conv_chunksize = int(audio_tower.conv_chunksize)
        chunk_width = self._n_window * 2
        self._chunk_width = chunk_width
        self._max_chunks = max(int(max_batch_size), 1) * (
            (int(max_clip_frames) + chunk_width - 1) // chunk_width
        )
        self._max_segments = max(int(max_batch_size) * 16, 1)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = int(warmup_iters)
        self._graphs: dict[tuple[int, int, int], _GraphEntry] = {}
        self._failed: set[tuple[int, int, int]] = set()
        self._pool = None
        # note (guozhihao-224): serializes capture and replay -- replay
        # mutates the bucket's static buffers, and both the pre-LM worker and
        # the scheduler's inline prefill path can reach get_audio_feature.
        self._lock = threading.Lock()
        self._done_event = torch.cuda.Event()
        self._event_recorded = False

    def _enough_free_vram(self) -> tuple[bool, int]:
        free, _ = torch.cuda.mem_get_info(self._device)
        return free >= self._min_free_bytes, free

    def _apply_encoder_layer(
        self,
        encoder_layer,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # note (guozhihao-224): pass host max_seqlen; VisionAttention .item()
        # would D2H-sync and invalidate capture.
        residual = hidden_states
        hidden_states = encoder_layer.self_attn_layer_norm(hidden_states)
        hidden_states = encoder_layer.self_attn(
            x=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = encoder_layer.final_layer_norm(hidden_states)
        hidden_states, _ = encoder_layer.fc1(hidden_states)
        hidden_states = encoder_layer.activation_fn(hidden_states)
        hidden_states, _ = encoder_layer.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states

    def _forward_from_chunks(
        self,
        padded_feature: torch.Tensor,
        gather_index: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        tower = self._audio_tower
        padded_embed = F.gelu(tower.conv2d1(padded_feature))
        padded_embed = F.gelu(tower.conv2d2(padded_embed))
        padded_embed = F.gelu(tower.conv2d3(padded_embed))
        b, c, f, t = padded_embed.size()
        padded_embed = tower.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )[0]
        positional_embedding = (
            tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        flat = padded_embed.reshape(-1, padded_embed.shape[-1])
        dummy = flat.new_zeros((1, flat.shape[-1]))
        hidden_states = torch.cat([flat, dummy], dim=0).index_select(0, gather_index)
        for encoder_layer in tower.layers:
            hidden_states = self._apply_encoder_layer(
                encoder_layer, hidden_states, cu_seqlens, max_seqlen
            )
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        hidden_states = tower.ln_post(hidden_states)
        hidden_states = tower.proj1(hidden_states)[0]
        hidden_states = tower.act(hidden_states)
        hidden_states = tower.proj2(hidden_states)[0]
        return hidden_states

    def _capture(
        self,
        c_bucket: int,
        n_bucket: int,
        s_bucket: int,
        n_mels: int,
        t_cnn: int,
    ) -> _GraphEntry:
        static_feat = torch.zeros(
            c_bucket,
            1,
            n_mels,
            self._chunk_width,
            device=self._device,
            dtype=self._dtype,
        )
        dummy_src = c_bucket * t_cnn
        static_index = torch.full(
            (n_bucket,), dummy_src, device=self._device, dtype=torch.long
        )
        static_cu = _capture_cu_seqlens(s_bucket, n_bucket, self._device)

        def _run() -> torch.Tensor:
            return self._forward_from_chunks(
                static_feat, static_index, static_cu, max_seqlen=n_bucket
            )

        # note (guozhihao-224): warmup on a fresh stream so allocator state
        # settles before capture.
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                _run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # note (guozhihao-224): thread_local error mode -- the LM scheduler
        # thread keeps launching kernels concurrently and must not poison this
        # thread's capture.
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            static_out = _run()
        logger.info(
            "Captured Qwen3-ASR packed encoder CUDA graph C=%d N=%d S=%d -> out %s "
            "(%d cached)",
            c_bucket,
            n_bucket,
            s_bucket,
            tuple(static_out.shape),
            len(self._graphs) + 1,
        )
        return _GraphEntry(
            graph=graph,
            padded_feature=static_feat,
            gather_index=static_index,
            cu_seqlens=static_cu,
            output=static_out,
        )

    @torch.no_grad()
    def run(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Replay packed-tower forward, or None to fall back to eager."""
        if input_features.dim() != 2:
            return None
        n_mels, packed_t = input_features.shape
        if int(feature_lens.sum().item()) != packed_t:
            return None
        if int(feature_lens.numel()) < 1:
            return None

        padded_feature, padded_mask, aftercnn_lens = _prepare_conv_chunks(
            input_features,
            feature_lens,
            n_window=self._n_window,
        )
        chunk_count, _, _, chunk_t = padded_feature.shape
        if chunk_t != self._chunk_width:
            return None
        if chunk_count > self._conv_chunksize:
            return None
        t_cnn = int(padded_mask.shape[-1])
        token_count = int(qwen3_asr_audio_token_lengths(feature_lens).sum().item())
        cu_seqlens = _build_cu_seqlens(
            aftercnn_lens,
            n_window=self._n_window,
            n_window_infer=self._n_window_infer,
            chunk_aftercnn_width=t_cnn,
        )
        if int(cu_seqlens[-1].item()) != token_count:
            return None
        segment_count = int(cu_seqlens.numel()) - 1
        planned = _plan_static_shapes(
            chunk_count=chunk_count,
            token_count=token_count,
            segment_count=segment_count,
            max_chunks=self._max_chunks,
            max_segments=self._max_segments,
        )
        if planned is None:
            return None
        c_bucket, n_bucket, s_bucket = planned
        key = (c_bucket, n_bucket, s_bucket)
        if key in self._failed:
            return None

        dummy_src = c_bucket * t_cnn
        gather_index = _gather_indices(
            padded_mask, n_bucket=n_bucket, dummy_src=dummy_src
        )
        static_cu = _pad_cu_seqlens(cu_seqlens, s_bucket=s_bucket, n_bucket=n_bucket)

        with self._lock:
            entry = self._graphs.get(key)
            if entry is None:
                enough, free = self._enough_free_vram()
                if not enough:
                    logger.warning(
                        "Qwen3-ASR encoder CUDA graph: free VRAM %.1fGB < %.1fGB "
                        "headroom; running C=%d N=%d S=%d eager",
                        free / 1024**3,
                        self._min_free_bytes / 1024**3,
                        c_bucket,
                        n_bucket,
                        s_bucket,
                    )
                    self._failed.add(key)
                    return None
                try:
                    with torch.cuda.device(self._device):
                        entry = self._capture(
                            c_bucket, n_bucket, s_bucket, n_mels, t_cnn
                        )
                except Exception as exc:
                    logger.warning(
                        "Qwen3-ASR encoder CUDA graph capture failed for "
                        "C=%d N=%d S=%d: %s; using eager for this bucket",
                        c_bucket,
                        n_bucket,
                        s_bucket,
                        exc,
                    )
                    self._failed.add(key)
                    # note (guozhihao-224): a failed capture can leave the
                    # mempool recording; drop it before the next bucket.
                    self._pool = None
                    try:
                        torch.cuda.synchronize(self._device)
                    except Exception as sync_exc:
                        logger.warning(
                            "Qwen3-ASR encoder CUDA graph: cleanup "
                            "synchronize failed after capture error: %s",
                            sync_exc,
                        )
                    return None
                self._graphs[key] = entry

            if entry.padded_feature.shape[2] != n_mels:
                return None
            stream = torch.cuda.current_stream(self._device)
            # note (guozhihao-224): wait for previous caller's output copy
            # on some stream to finish before using shared resource
            if self._event_recorded:
                self._done_event.wait(stream)
            entry.padded_feature.zero_()
            entry.padded_feature[:chunk_count, :, :, :chunk_t].copy_(
                padded_feature, non_blocking=True
            )
            entry.gather_index.copy_(gather_index, non_blocking=True)
            entry.cu_seqlens.copy_(static_cu, non_blocking=True)
            entry.graph.replay()
            # note (guozhihao-224): the next call needs to wait on this
            # event before it touches anything shared to ensure clone finishes
            out = entry.output[:token_count].clone()
            self._done_event.record(stream)
            self._event_recorded = True
            return out


__all__ = ["Qwen3ASREncoderCudaGraphRunner"]
