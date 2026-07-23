from __future__ import annotations

import pytest

from benchmarks.audar_tts.run_quality_benchmark import _is_arabic_text, _select_targets
from benchmarks.audar_tts.summarize_quality import _quality_metrics


def _generation_result(samples: list[dict]) -> dict:
    return {
        "successful_samples": len(samples),
        "truncated_samples": 0,
        "samples": samples,
    }


def test_select_targets_keeps_fixed_arabic_subset() -> None:
    selected = _select_targets(
        [
            {"id": 1, "transcription": "short"},
            {
                "id": 2,
                "transcription": "مرحبا بكم في هذا الاختبار العربي الواضح",
            },
            {
                "id": 3,
                "transcription": "هذا النص العربي يحتوي على الرقم 123 هنا",
            },
            {
                "id": 4,
                "transcription": "هذا مثال عربي ثان لاختبار جودة الكلام",
            },
        ],
        samples=2,
        min_words=6,
        max_words=20,
    )

    assert [sample["dataset_id"] for sample in selected] == ["2", "4"]
    assert [sample["sample_id"] for sample in selected] == [
        "fleurs-ar-eg-0001",
        "fleurs-ar-eg-0003",
    ]
    assert _is_arabic_text(selected[0]["target_text"])


def test_arabic_quality_uses_target_and_asr_text_directly() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": "one",
                "target_text": "مرحبا بكم في هذا العالم الجميل",
                "is_success": True,
                "reached_max_new_tokens": False,
            },
            {
                "sample_id": "two",
                "target_text": "هذا اختبار عربي بسيط وواضح جدا",
                "is_success": True,
                "reached_max_new_tokens": False,
            },
        ]
    )
    result = _quality_metrics(
        {
            "config": {"asr_model": "test-asr"},
            "summary": {"wer_corpus": 1 / 12},
            "per_sample": [
                {
                    "id": "one",
                    "is_success": True,
                    "ref_norm": "مرحبا بكم في هذا العالم الجميل",
                    "hyp_norm": "مرحبا بكم في هذا العالم الجميل",
                },
                {
                    "id": "two",
                    "is_success": True,
                    "ref_norm": "هذا اختبار عربي بسيط وواضح جدا",
                    "hyp_norm": "هذا اختبار عربي بسيط جدا",
                },
            ],
        },
        generation,
    )

    assert result["sample_count"] == 2
    assert result["asr_model"] == "test-asr"
    assert result["arabic_wer"] == pytest.approx(1 / 12)
    assert 0 < result["arabic_cer"] < 1
    assert 0 < result["arabic_bleu"] < 100
    assert 0 < result["arabic_chrf_pp"] < 100


def test_arabic_quality_rejects_asr_reference_not_derived_from_target() -> None:
    generation = _generation_result(
        [
            {
                "sample_id": sample_id,
                "target_text": target_text,
                "is_success": True,
                "reached_max_new_tokens": False,
            }
            for sample_id, target_text in (
                ("one", "النص العربي الأول هنا"),
                ("two", "النص العربي الثاني هنا"),
            )
        ]
    )
    wer = {
        "config": {"asr_model": "test-asr"},
        "summary": {"wer_corpus": 0.0},
        "per_sample": [
            {
                "id": "one",
                "is_success": True,
                "ref_norm": "مرجع خاطئ",
                "hyp_norm": "مرجع خاطئ",
            },
            {
                "id": "two",
                "is_success": True,
                "ref_norm": "النص العربي الثاني هنا",
                "hyp_norm": "النص العربي الثاني هنا",
            },
        ],
    }

    with pytest.raises(ValueError, match="ASR reference does not match target text"):
        _quality_metrics(wer, generation)
