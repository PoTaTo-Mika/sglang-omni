# SPDX-License-Identifier: Apache-2.0
"""Contracts for the batched dots.tts acoustic tail.

The engine replaces the upstream per-request streaming solver with a batched,
KV-cached one, so the tests that matter are equivalence checks against the
reference full-recompute DiT path and against the single-row mask builder.
"""

from __future__ import annotations

import torch

from sglang_omni.models.dots_tts.components.backbone.dit import DiT
from sglang_omni.models.dots_tts.components.backbone.dit_inference import (
    DiTInferenceContext,
    EagerDiTRunner,
)
from sglang_omni.models.dots_tts.components.backbone.encoder import VAESemanticEncoder
from sglang_omni.models.dots_tts.components.backbone.inference_utils import (
    build_causal_update_mask,
)
from sglang_omni.models.dots_tts.components.model_config import (
    _DiTConfig,
    _EncoderConfig,
)
from sglang_omni.models.dots_tts.tail import (
    DotsTtsAcousticTail,
    DotsTtsTailSpec,
    batched_causal_update_mask,
)

FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


class _TailModelStub(torch.nn.Module):
    """The subset of the serving model ``DiTInferenceContext`` reads."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_patch_size = 1
        self.latent_patch_size = PATCH_SIZE
        self.latent_dim = LATENT_DIM
        self.fm_hidden_size = FM_HIDDEN
        self.velocity_field_predictor = DiT(
            in_dim=FM_HIDDEN,
            out_dim=LATENT_DIM,
            transformer_config=_DiTConfig(
                num_layers=2,
                num_heads=2,
                hidden_size=FM_HIDDEN,
                ffn_hidden_size=64,
                modulation=True,
                qk_norm=True,
                rotary_bias=True,
            ),
            mode="meanflow",
        )
        self.coordinate_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        self.latent_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        self.config = type("_Cfg", (), {"fm_sigma": 0.0})()


def _patch_encoder() -> VAESemanticEncoder:
    encoder_config = _EncoderConfig(
        num_layers=1,
        num_heads=2,
        hidden_size=FM_HIDDEN,
        ffn_hidden_size=64,
        causal=True,
    )
    config = type(
        "_EncCfg", (), {"patch_size": PATCH_SIZE, "PatchEncoder": encoder_config}
    )()
    return VAESemanticEncoder(in_dim=LATENT_DIM, out_dim=FM_HIDDEN, config=config)


def _meanflow_modulations(dit: torch.nn.Module, g_cond: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, NFE + 1, dtype=g_cond.dtype)
    times, durations = grid[:-1], grid[1:] - grid[:-1]
    condition = dit.time_embedder(times)
    if dit.duration_embedder is not None:
        condition = condition + dit.duration_embedder(durations)
    return dit.fused_adaln(condition + g_cond.reshape(1, -1))


def _build_tail(
    model: _TailModelStub, *, num_slots: int, patch_capacity: int
) -> DotsTtsAcousticTail:
    return DotsTtsAcousticTail(
        dit_context=DiTInferenceContext.from_core(model),
        patch_encoder=_patch_encoder(),
        spec=DotsTtsTailSpec(
            nfe=NFE,
            patch_capacity=patch_capacity,
            num_slots=num_slots,
            hidden_patch_size=1,
            latent_patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
            fm_hidden_size=FM_HIDDEN,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_batched_mask_matches_the_single_row_builder() -> None:
    valid = torch.tensor([0, 3, 7])
    batched = batched_causal_update_mask(
        capacity_tokens=8,
        valid_persistent=valid,
        prev_len=3,
        current_len=3,
    )

    assert batched.shape == (3, 1, 6, 14)
    for row, valid_persistent in enumerate(valid.tolist()):
        reference = build_causal_update_mask(
            capacity_tokens=8,
            valid_persistent_tokens=valid_persistent,
            prev_len=3,
            current_len=3,
            device=torch.device("cpu"),
        )
        assert torch.equal(batched[row : row + 1], reference)


def test_kv_cached_tail_matches_the_full_recompute_solver() -> None:
    torch.manual_seed(1234)
    model = _TailModelStub().eval()
    patch_capacity = 8
    tail = _build_tail(model, num_slots=1, patch_capacity=patch_capacity)
    context = DiTInferenceContext.from_core(model)
    eager = EagerDiTRunner(context=context)
    unit = tail.spec.unit_len

    prompt_patches = 3
    g_cond = torch.randn(1, FM_HIDDEN)
    all_mods = _meanflow_modulations(context.dit, g_cond)
    fm_rows = torch.randn(prompt_patches * unit, FM_HIDDEN)

    slot = tail.acquire_slot()
    tail.seed_fm_history(slot, fm_rows=fm_rows, all_mods=all_mods)

    reference_sequence = torch.zeros(1, tail.spec.fm_capacity, FM_HIDDEN)
    reference_sequence[0, : fm_rows.size(0)] = fm_rows
    reference_len = fm_rows.size(0)

    for step in range(3):
        hidden_row = torch.randn(1, FM_HIDDEN)
        reference_sequence[0, reference_len] = hidden_row[0]
        reference_len += 1

        torch.manual_seed(step)
        expected = eager.decode_next(
            sequence=reference_sequence,
            fm_seq_len=reference_len,
            g_cond=g_cond,
            nfe=NFE,
            meanflow=True,
        )

        torch.manual_seed(step)
        actual = tail.sample_patches(
            [slot],
            fm_hidden_rows=hidden_row,
            latent_proj=model.latent_proj,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        latent_rows = model.latent_proj(expected)[0]
        reference_sequence[0, reference_len : reference_len + PATCH_SIZE] = latent_rows
        reference_len += PATCH_SIZE
        assert tail.fm_seq_len(slot) == reference_len


def test_slots_stay_independent_when_batched_together() -> None:
    torch.manual_seed(4321)
    model = _TailModelStub().eval()
    solo = _build_tail(model, num_slots=1, patch_capacity=8)
    shared = _build_tail(model, num_slots=2, patch_capacity=8)
    unit = solo.spec.unit_len

    g_cond = torch.randn(1, FM_HIDDEN)
    all_mods = _meanflow_modulations(DiTInferenceContext.from_core(model).dit, g_cond)
    # A second slot with a longer history so the batch has to pad and mask.
    fm_rows = torch.randn(2 * unit, FM_HIDDEN)
    other_rows = torch.randn(4 * unit, FM_HIDDEN)

    solo_slot = solo.acquire_slot()
    solo.set_slot_seed(solo_slot, 99)
    solo.seed_fm_history(solo_slot, fm_rows=fm_rows, all_mods=all_mods)
    first = shared.acquire_slot()
    second = shared.acquire_slot()
    shared.set_slot_seed(first, 99)
    shared.seed_fm_history(first, fm_rows=fm_rows, all_mods=all_mods)
    shared.seed_fm_history(second, fm_rows=other_rows, all_mods=all_mods)

    for _step in range(2):
        hidden = torch.randn(2, FM_HIDDEN)
        expected = solo.sample_patches(
            [solo_slot],
            fm_hidden_rows=hidden[:1],
            latent_proj=model.latent_proj,
        )
        batched = shared.sample_patches(
            [first, second],
            fm_hidden_rows=hidden,
            latent_proj=model.latent_proj,
        )
        torch.testing.assert_close(batched[:1], expected, rtol=2e-4, atol=2e-4)


def test_slot_pool_is_bounded_and_reusable() -> None:
    model = _TailModelStub().eval()
    tail = _build_tail(model, num_slots=2, patch_capacity=8)

    first = tail.acquire_slot()
    second = tail.acquire_slot()
    assert {first, second} == {0, 1}
    try:
        tail.acquire_slot()
    except RuntimeError as error:
        assert "ran out of slots" in str(error)
    else:  # pragma: no cover - the pool must be bounded
        raise AssertionError("acquire_slot must fail once the pool is exhausted")

    tail.release_slot(first)
    assert tail.acquire_slot() == first
