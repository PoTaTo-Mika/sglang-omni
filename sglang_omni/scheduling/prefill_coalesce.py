# SPDX-License-Identifier: Apache-2.0
"""Shared validation for prefill admission coalescing."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def validate_prefill_coalesce_args(
    prefill_coalesce_requests: int | float | None,
    prefill_coalesce_wait_ms: float | None,
) -> tuple[int | None, float | None]:
    """Validate and normalize coalescing arguments from any config entrypoint.

    Both 0 and 1 leave the admission gate disabled: the gate only engages at
    >= 2, since a batch of one has nothing to coalesce with. 0 is the explicit
    off switch; 1 is accepted but warns, as it is usually a misconfiguration.
    """
    # Reject bools and non-integral counts before int() truncates them: YAML
    # `-0.5` would otherwise land as 0 (silently off), `2.9` as 2, `true` as 1.
    if isinstance(prefill_coalesce_requests, bool) or isinstance(
        prefill_coalesce_wait_ms, bool
    ):
        raise ValueError(
            "prefill_coalesce_requests and prefill_coalesce_wait_ms must be "
            "numbers, not booleans"
        )
    if (
        prefill_coalesce_requests is not None
        and float(prefill_coalesce_requests) != int(prefill_coalesce_requests)
    ):
        raise ValueError(
            "prefill_coalesce_requests must be an integer, got "
            f"{prefill_coalesce_requests!r}"
        )
    requests = (
        None if prefill_coalesce_requests is None else int(prefill_coalesce_requests)
    )
    if requests is not None and requests < 0:
        raise ValueError("prefill_coalesce_requests must be >= 0")
    if requests == 1:
        logger.warning(
            "prefill_coalesce_requests=1 disables coalescing: the admission "
            "gate only engages at >= 2 (a batch of one has nothing to "
            "coalesce with). Use 0 to disable explicitly, or >= 2 to enable."
        )

    wait_ms = (
        None if prefill_coalesce_wait_ms is None else float(prefill_coalesce_wait_ms)
    )
    if wait_ms is not None and not (math.isfinite(wait_ms) and wait_ms > 0):
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0")

    return requests, wait_ms
