# SPDX-License-Identifier: Apache-2.0
"""Fixed-address, row-indexed generation state for VoxCPM requests."""

from __future__ import annotations

import threading
from typing import Iterable

import torch


class VoxCPMRequestStatePool:
    """Owns graph-stable tensors used to feed generated audio back to the LM."""

    def __init__(
        self,
        capacity: int,
        hidden_size: int,
        patch_size: int,
        feat_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if min(capacity, hidden_size, patch_size, feat_dim) <= 0:
            raise ValueError("all VoxCPM state-pool dimensions must be positive")
        self.capacity = capacity
        self.num_rows = capacity + 1
        self.padding_row = capacity
        self.feedback = torch.zeros(
            self.num_rows, hidden_size, device=device, dtype=dtype
        )
        self.latents = torch.zeros(
            self.num_rows, patch_size, feat_dim, device=device, dtype=dtype
        )
        self.temperature = torch.ones(self.num_rows, device=device, dtype=dtype)
        self.cfg_value = torch.ones(self.num_rows, device=device, dtype=dtype)
        self.stop_flag = torch.zeros(self.num_rows, device=device, dtype=torch.long)
        self.done = torch.zeros(self.num_rows, device=device, dtype=torch.bool)
        self.steps = torch.zeros(self.num_rows, device=device, dtype=torch.long)
        self.seeds = torch.full((self.num_rows,), -1, device=device, dtype=torch.long)
        self._rid_to_row: dict[str, int] = {}
        self._free = list(range(capacity - 1, -1, -1))
        self._lock = threading.RLock()

    def acquire(self, rid: str) -> int:
        if not rid:
            raise ValueError("request id must be non-empty")
        with self._lock:
            existing = self._rid_to_row.get(rid)
            if existing is not None:
                return existing
            if not self._free:
                raise RuntimeError(
                    f"VoxCPM request-state pool exhausted (capacity={self.capacity})"
                )
            row = self._free.pop()
            self._rid_to_row[rid] = row
            self._zero_rows(torch.tensor([row], device=self.feedback.device))
            return row

    acquire_row = acquire

    def row_for(self, rid: str) -> int | None:
        with self._lock:
            return self._rid_to_row.get(rid)

    def reset(self, rid: str) -> None:
        with self._lock:
            row = self._rid_to_row.pop(rid, None)
            if row is None:
                return
            self._zero_rows(torch.tensor([row], device=self.feedback.device))
            self._free.append(row)

    reset_request = reset
    release_row = reset

    def reset_rows(self, rows: torch.Tensor | Iterable[int]) -> None:
        rows = self._rows_tensor(rows)
        self._check_rows(rows)
        self._zero_rows(rows)

    def configure(
        self,
        rows: torch.Tensor | Iterable[int],
        *,
        temperature: torch.Tensor | float | None = None,
        cfg_value: torch.Tensor | float | None = None,
        seed: torch.Tensor | int | None = None,
    ) -> None:
        rows = self._rows_tensor(rows)
        self._check_rows(rows)
        for target, value in (
            (self.temperature, temperature),
            (self.cfg_value, cfg_value),
            (self.seeds, seed),
        ):
            if value is None:
                continue
            source = torch.as_tensor(value, device=target.device, dtype=target.dtype)
            if source.ndim == 0:
                source = source.expand(rows.numel())
            if source.numel() != rows.numel():
                raise ValueError("state values must be scalar or match row count")
            target.index_copy_(0, rows, source.reshape(-1))

    def feedback_for(self, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.to(device=self.feedback.device, dtype=torch.long)
        self._check_rows(rows)
        return self.feedback.index_select(0, rows)

    def update(
        self,
        rows: torch.Tensor,
        *,
        feedback: torch.Tensor,
        latents: torch.Tensor,
        stop_flag: torch.Tensor,
    ) -> None:
        rows = rows.to(device=self.feedback.device, dtype=torch.long)
        self._check_rows(rows)
        batch = rows.numel()
        if feedback.shape != (batch, self.feedback.shape[1]):
            raise ValueError("feedback has an invalid shape")
        if latents.shape != (batch, self.latents.shape[1], self.latents.shape[2]):
            raise ValueError("latents have an invalid shape")
        if stop_flag.numel() != batch:
            raise ValueError("stop_flag must match row count")
        self.feedback.index_copy_(0, rows, feedback.to(self.feedback))
        self.latents.index_copy_(0, rows, latents.to(self.latents))
        self.stop_flag.index_copy_(0, rows, stop_flag.reshape(-1).to(self.stop_flag))
        self.steps.index_add_(0, rows, torch.ones_like(rows, dtype=self.steps.dtype))

    def _rows_tensor(self, rows: torch.Tensor | Iterable[int]) -> torch.Tensor:
        if isinstance(rows, torch.Tensor):
            return rows.to(device=self.feedback.device, dtype=torch.long).reshape(-1)
        return torch.tensor(list(rows), device=self.feedback.device, dtype=torch.long)

    def _check_rows(self, rows: torch.Tensor) -> None:
        if rows.ndim != 1:
            raise ValueError("rows must be one-dimensional")
        # GPU row tensors are produced by the scheduler/model runner. Avoid
        # value-dependent host reads here: .item() is illegal while SGLang is
        # capturing the decode CUDA graph.
        if (
            rows.device.type == "cpu"
            and rows.numel()
            and (int(rows.min().item()) < 0 or int(rows.max().item()) >= self.num_rows)
        ):
            raise IndexError("VoxCPM state-pool row is out of range")

    def _zero_rows(self, rows: torch.Tensor) -> None:
        self.feedback.index_fill_(0, rows, 0)
        self.latents.index_fill_(0, rows, 0)
        self.temperature.index_fill_(0, rows, 1)
        self.cfg_value.index_fill_(0, rows, 1)
        self.stop_flag.index_fill_(0, rows, 0)
        self.done.index_fill_(0, rows, False)
        self.steps.index_fill_(0, rows, 0)
        self.seeds.index_fill_(0, rows, -1)
