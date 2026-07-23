# SPDX-License-Identifier: Apache-2.0
"""Shared validation for prefill admission coalescing."""

from __future__ import annotations

import math


def validate_prefill_coalesce_args(
    prefill_coalesce_requests: int | None,
    prefill_coalesce_wait_ms: float | None,
) -> tuple[int | None, float | None]:
    """Validate and normalize coalescing arguments from any config entrypoint."""
    requests = (
        None if prefill_coalesce_requests is None else int(prefill_coalesce_requests)
    )
    if requests is not None and requests < 0:
        raise ValueError("prefill_coalesce_requests must be >= 0")

    wait_ms = (
        None if prefill_coalesce_wait_ms is None else float(prefill_coalesce_wait_ms)
    )
    if wait_ms is not None and not (math.isfinite(wait_ms) and wait_ms > 0):
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0")

    return requests, wait_ms
