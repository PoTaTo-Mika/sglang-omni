"""Pre-LM audio-encoder service for MOSS-Transcribe-Diarize.

Encoding inside the LM forward stalls every running request at each prefill;
encoding at request-build time on a dedicated thread/stream lets the
compute-bound encoder overlap the memory-bound decode on the same GPU.
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
from typing import Any

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 4096
_CACHE_MAX_BYTES = 2 * 1024**3
_SHUTDOWN = object()


class BatchedAudioEncoderService:
    ENCODE_TIMEOUT_S = 300.0

    def __init__(self, model: Any) -> None:
        self._model = model
        self._device = next(model.whisper_encoder.parameters()).device
        self._stream = torch.cuda.Stream(device=self._device)
        self._cache = StageOutputCache(
            max_size=_CACHE_MAX_ENTRIES,
            max_bytes=_CACHE_MAX_BYTES,
            cache_device="cpu",
        )
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._batch_count = 0
        self._item_count = 0
        self._thread = threading.Thread(
            target=self._worker, name="moss-td-audio-encode", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the encoder worker after all queued requests finish."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=5)

    def _enqueue(self, item: Any, future: concurrent.futures.Future) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MOSS-TD pre-LM encoder service is closed")
            self._queue.put((item, future))

    def encode_item(self, item: Any) -> None:
        """Blocks until item.precomputed_embeddings is attached."""
        if item.hash is not None:
            cached = self._cache.get(str(item.hash))
        else:
            cached = None
        if cached is not None:
            item.precomputed_embeddings = cached.to(self._device, non_blocking=True)
            item.feature = None
            return
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._enqueue(item, future)
        future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _drain_batch(
        self,
    ) -> tuple[list[tuple[Any, concurrent.futures.Future]], bool]:
        # note (yichi): never wait — a window costs 8~16ms at low concurrency, buys <=5ms at high.
        first = self._queue.get()
        if first is _SHUTDOWN:
            return [], True
        batch = [first]
        shutdown = False
        while True:
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued is _SHUTDOWN:
                shutdown = True
                break
            batch.append(queued)
        return batch, shutdown

    def _worker(self) -> None:
        while True:
            batch, shutdown = self._drain_batch()
            if not batch:
                return
            items = [item for item, _ in batch]
            try:
                self._encode_batch(items)
            except Exception:
                logger.exception(
                    f"MOSS-TD batched audio encode failed for {len(items)} "
                    f"items; retrying per item"
                )
                for item, future in batch:
                    try:
                        self._encode_batch([item])
                        future.set_result(None)
                    except Exception as item_exc:
                        future.set_exception(item_exc)
                if shutdown:
                    return
                continue
            for _, future in batch:
                future.set_result(None)
            self._batch_count += 1
            self._item_count += len(items)
            if self._batch_count % 50 == 1:
                logger.info(
                    f"MOSS-TD pre-LM encoder stage: {self._batch_count} batches, "
                    f"{self._item_count} items (avg "
                    f"{self._item_count / self._batch_count:.2f} items/batch, "
                    f"last batch: {len(items)})"
                )
            if shutdown:
                return

    def _encode_batch(self, items: list[Any]) -> None:
        with torch.cuda.stream(self._stream):
            embedding = self._model._get_audio_feature_uncached(items, None)
            token_counts = [
                int(getattr(item, "audio_feature_lengths").sum()) for item in items
            ]
            if embedding.shape[0] != sum(token_counts):
                raise RuntimeError(
                    f"encoder output rows {embedding.shape[0]} != expected "
                    f"{sum(token_counts)}"
                )
            parts = torch.split(embedding, token_counts, dim=0)
            for item, part in zip(items, parts):
                item.precomputed_embeddings = part.contiguous()
                item.feature = None
        self._stream.synchronize()
        for item in items:
            if item.hash is not None:
                self._cache.put(str(item.hash), item.precomputed_embeddings)
