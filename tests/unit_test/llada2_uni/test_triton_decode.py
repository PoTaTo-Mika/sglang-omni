# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("sglang")

from sglang_omni.models.llada2_uni.algorithm import triton_decode
from sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg import (
    _argmax_confidence_from_logits,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton decode tests require CUDA"
)


def _make_logits(
    rows: int,
    vocab: int,
    *,
    noncontiguous: bool,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if noncontiguous:
        storage = torch.randn((rows, vocab * 2), device="cuda", dtype=dtype)
        logits = storage[:, ::2]
        assert not logits.is_contiguous()
        return logits
    return torch.randn((rows, vocab), device="cuda", dtype=dtype)


def _reference_argmax_confidence(
    logits: torch.Tensor,
    *,
    force_image_only: bool = False,
    image_token_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    work = logits.float().clone()
    if force_image_only:
        work[:, :image_token_offset] = float("-inf")
    token_ids = work.argmax(dim=-1)
    probs = torch.softmax(work, dim=-1).gather(1, token_ids[:, None]).squeeze(1)
    return token_ids, probs


@pytest.mark.parametrize(
    ("rows", "vocab", "noncontiguous", "dtype"),
    [
        (1, 257, False, torch.float32),
        (17, 2053, True, torch.float32),
        (8, 4097, False, torch.bfloat16),
        (2, 173568, False, torch.float32),
    ],
)
def test_argmax_confidence_matches_pytorch(
    rows: int,
    vocab: int,
    noncontiguous: bool,
    dtype: torch.dtype,
) -> None:
    logits = _make_logits(
        rows,
        vocab,
        noncontiguous=noncontiguous,
        dtype=dtype,
    )

    expected_ids, expected_probs = _reference_argmax_confidence(logits)
    actual_ids, actual_probs = triton_decode.argmax_confidence_triton(logits)

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=3e-5, atol=1e-7)


def test_argmax_confidence_respects_image_token_offset() -> None:
    rows, vocab, image_token_offset = 4, 173568, 157184
    logits = _make_logits(rows, vocab, noncontiguous=False)
    logits[:, 0] = 100.0

    expected_ids, expected_probs = _reference_argmax_confidence(
        logits,
        force_image_only=True,
        image_token_offset=image_token_offset,
    )
    actual_ids, actual_probs = triton_decode.argmax_confidence_triton(
        logits,
        force_image_only=True,
        image_token_offset=image_token_offset,
    )

    assert torch.all(actual_ids >= image_token_offset)
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=3e-5, atol=1e-7)


def test_cfg_argmax_confidence_matches_pytorch() -> None:
    rows, vocab, cfg_scale = 13, 2053, 4.0
    cond = _make_logits(rows, vocab, noncontiguous=True)
    uncond = _make_logits(rows, vocab, noncontiguous=True)
    guided = uncond.float() + cfg_scale * (cond.float() - uncond.float())
    expected_ids, expected_probs = _reference_argmax_confidence(guided)

    actual_ids, actual_probs = triton_decode.cfg_argmax_confidence_triton(
        cond,
        uncond,
        cfg_scale=cfg_scale,
    )

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=3e-5, atol=1e-7)


def _reference_low_confidence_update(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    input_start: int,
    mask_id: int,
    threshold: float,
    num_to_transfer: int,
    force_image_only: bool = False,
    image_token_offset: int = 0,
) -> tuple[torch.Tensor, int]:
    rows = logits.shape[0]
    segment = input_ids[input_start : input_start + rows]
    token_ids, probs = _reference_argmax_confidence(
        logits,
        force_image_only=force_image_only,
        image_token_offset=image_token_offset,
    )
    masked = segment == mask_id
    confidence = torch.where(masked, probs, -torch.inf)
    high_confidence = confidence > threshold
    if int(high_confidence.sum().item()) >= num_to_transfer:
        keep = high_confidence
    else:
        transfer_count = min(num_to_transfer, int(masked.sum().item()))
        indices = torch.topk(confidence, k=transfer_count).indices
        keep = torch.zeros_like(masked)
        keep[indices] = True

    expected = input_ids.clone()
    expected_segment = expected[input_start : input_start + rows]
    expected_segment[keep] = token_ids[keep]
    return expected, int(keep.sum().item())


@pytest.mark.parametrize("threshold", [0.95, 1.1])
def test_low_confidence_update_matches_pytorch(threshold: float) -> None:
    rows, vocab = 32, 2053
    input_start = 3
    mask_id = 156895
    logits = _make_logits(rows, vocab, noncontiguous=True)
    input_ids = torch.full(
        (rows + input_start + 2,), mask_id, device="cuda", dtype=torch.int64
    )
    input_ids[input_start : input_start + rows : 5] = 7
    if threshold < 1.0:
        masked_rows = torch.nonzero(
            input_ids[input_start : input_start + rows] == mask_id
        ).squeeze(1)
        winner_ids = torch.arange(masked_rows.numel(), device="cuda") % vocab
        logits[masked_rows[:8]] = 0.0
        logits[masked_rows[:8], winner_ids[:8]] = 20.0

    expected, expected_tpf = _reference_low_confidence_update(
        logits,
        input_ids,
        input_start=input_start,
        mask_id=mask_id,
        threshold=threshold,
        num_to_transfer=4,
    )
    actual = input_ids.clone()

    actual_tpf = triton_decode.low_confidence_update_triton(
        logits,
        actual,
        input_start=input_start,
        mask_id=mask_id,
        threshold=threshold,
        num_to_transfer=4,
    )

    assert actual_tpf == expected_tpf
    assert torch.equal(actual, expected)


def test_argmax_wrapper_falls_back_when_triton_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logits = _make_logits(7, 257, noncontiguous=False)
    expected_ids, expected_probs = _reference_argmax_confidence(logits)

    def fail_triton(*args, **kwargs):
        raise RuntimeError("injected Triton failure")

    monkeypatch.setattr(triton_decode, "argmax_confidence_triton", fail_triton)
    caplog.set_level(logging.ERROR)

    actual_ids, actual_probs = _argmax_confidence_from_logits(
        logits,
        use_triton=True,
    )

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_probs, expected_probs)
    assert "falling back" in caplog.text
