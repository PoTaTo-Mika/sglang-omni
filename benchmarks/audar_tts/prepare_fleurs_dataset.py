# SPDX-License-Identifier: Apache-2.0
"""Materialize the fixed FLEURS Arabic smoke-test subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

FLEURS_REPO = "google/fleurs"
FLEURS_CONFIG = "ar_eg"
FLEURS_SPLIT = "test"
FLEURS_REVISION = "ab93cf03f9d0cd083c853fad065a6377067408aa"
FLEURS_FILE = "ar_eg/test-00000-of-00001.parquet"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=20)
    return parser.parse_args()


def _is_arabic_text(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    arabic_letters = [
        char for char in letters if "ARABIC" in unicodedata.name(char, "")
    ]
    return len(arabic_letters) / len(letters) >= 0.9


def _select_targets(
    rows: Iterable[Mapping[str, Any]],
    *,
    samples: int,
    min_words: int,
    max_words: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        text = " ".join(str(row["transcription"]).split())
        word_count = len(text.split())
        if not min_words <= word_count <= max_words:
            continue
        if any(char.isdigit() for char in text):
            continue
        if not _is_arabic_text(text) or text in seen:
            continue
        seen.add(text)
        selected.append(
            {
                "sample_id": f"fleurs-ar-eg-{row_index:04d}",
                "target_text": text,
                "word_count": word_count,
                "source_row_index": row_index,
                "source_id": str(row.get("id", row_index)),
            }
        )
        if len(selected) == samples:
            break
    if len(selected) != samples:
        raise ValueError(f"selected {len(selected)} of {samples} requested texts")
    return selected


def main() -> None:
    args = _parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.min_words <= 0 or args.max_words < args.min_words:
        raise ValueError("invalid word-count range")

    from datasets import Dataset, load_dataset

    parquet_path = hf_hub_download(
        FLEURS_REPO,
        FLEURS_FILE,
        revision=FLEURS_REVISION,
        repo_type="dataset",
    )
    source = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    ).select_columns(["id", "transcription"])
    selected = _select_targets(
        source,
        samples=args.samples,
        min_words=args.min_words,
        max_words=args.max_words,
    )

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "test-00000-of-00001.parquet"
    Dataset.from_list(selected).to_parquet(output_path)
    manifest = {
        "schema_version": 1,
        "source": {
            "repo": FLEURS_REPO,
            "config": FLEURS_CONFIG,
            "split": FLEURS_SPLIT,
            "revision": FLEURS_REVISION,
            "file": FLEURS_FILE,
            "license": "cc-by-4.0",
        },
        "selection": {
            "samples": args.samples,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "minimum_arabic_letter_fraction": 0.9,
            "exclude_digits": True,
            "text_column": "transcription",
        },
        "artifact": {
            "path": output_path.relative_to(args.output_dir).as_posix(),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
