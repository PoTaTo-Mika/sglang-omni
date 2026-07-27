# SPDX-License-Identifier: Apache-2.0
"""SchedulingConfig validation: types and ranges are enforced at the schema
layer, before int()/float() truncation can silently change meaning."""

from __future__ import annotations

import pytest

from sglang_omni.config.schema import SchedulingConfig


def test_normalizes_values():
    cfg = SchedulingConfig(prefill_coalesce_requests=32, prefill_coalesce_wait_ms=300)
    assert cfg.prefill_coalesce_requests == 32
    assert cfg.prefill_coalesce_wait_ms == 300.0
    assert SchedulingConfig().prefill_coalesce_requests is None


def test_accepts_integral_floats():
    # `32.0` loses nothing under int() and may come from YAML round-tripping.
    cfg = SchedulingConfig(prefill_coalesce_requests=32.0)
    assert cfg.prefill_coalesce_requests == 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefill_coalesce_requests": -1},
        {"prefill_coalesce_wait_ms": 0.0},
        {"prefill_coalesce_wait_ms": float("nan")},
        {"prefill_coalesce_wait_ms": float("inf")},
        # bools and non-integral counts must fail before truncation:
        # YAML `-0.5` would land as 0 (silently off), `2.9` as 2, `true` as 1
        {"prefill_coalesce_requests": True},
        {"prefill_coalesce_requests": False},
        {"prefill_coalesce_wait_ms": True},
        {"prefill_coalesce_requests": -0.5},
        {"prefill_coalesce_requests": 2.9},
    ],
)
def test_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SchedulingConfig(**kwargs)


def test_warns_on_one(caplog):
    # 1 passes validation but the gate only engages at >= 2, so it must warn
    # that coalescing stays disabled; 0 stays silent (explicit off).
    with caplog.at_level("WARNING", logger="sglang_omni.config.schema"):
        assert SchedulingConfig(prefill_coalesce_requests=1).prefill_coalesce_requests == 1
    assert any("disables coalescing" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="sglang_omni.config.schema"):
        assert SchedulingConfig(prefill_coalesce_requests=0).prefill_coalesce_requests == 0
    assert not caplog.records
