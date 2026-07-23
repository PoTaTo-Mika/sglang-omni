# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.scheduling.prefill_coalesce import validate_prefill_coalesce_args


def test_validate_prefill_coalesce_args_normalizes_values():
    assert validate_prefill_coalesce_args(32, 300) == (32, 300.0)
    assert validate_prefill_coalesce_args(None, None) == (None, None)


@pytest.mark.parametrize(
    ("requests", "wait_ms"),
    [
        (-1, 60.0),
        (0, 0.0),
        (0, float("nan")),
        (0, float("inf")),
    ],
)
def test_validate_prefill_coalesce_args_rejects_invalid_values(requests, wait_ms):
    with pytest.raises(ValueError):
        validate_prefill_coalesce_args(requests, wait_ms)
