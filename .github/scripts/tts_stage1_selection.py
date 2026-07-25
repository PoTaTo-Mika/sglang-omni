# SPDX-License-Identifier: Apache-2.0
"""Resolve one immutable TTS model selection before H100 jobs are scheduled."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

from tts_stage1_artifacts import SCHEMA_VERSION, write_json_atomic

MODELS = ("higgs", "moss")
MODEL_LABELS = {"higgs": "run-higgs", "moss": "run-moss"}
MODEL_CONFIG_CLASSES = {
    "higgs": "HiggsTtsPipelineConfig",
    "moss": "MossTTSLocalPipelineConfig",
}
MODEL_CONFIG_PATHS = {
    "higgs": "examples/mps_dp/configs/higgs_h100_dp3.yaml",
    "moss": "examples/mps_dp/configs/moss_local_h100_dp2.yaml",
}
SUPPORTED_TOPOLOGIES = frozenset({"multi_gpu", "mps_shared"})


class SelectionResult(NamedTuple):
    schema_version: int
    selected_model: str
    selection_digest: str
    selection_reason: str
    tts_stage1_topology: str
    resolved_mps_config: str
    resolved_config_class: str
    run_id: str
    run_attempt: str

    def as_dict(self) -> dict[str, Any]:
        return self._asdict()


def _load_weight_share_registry(repository_root: Path) -> dict[str, str]:
    registry_path = repository_root / "examples" / "mps_dp" / "config.py"
    tree = ast.parse(registry_path.read_text(encoding="utf-8"), registry_path)
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "WEIGHT_SHARE_VALIDATED_CONFIGS":
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
            break
    raise ValueError(f"Cannot find WEIGHT_SHARE_VALIDATED_CONFIGS in {registry_path}")


def _config_class(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "config_cls":
            return value.strip().strip("'\"")
    return None


def _resolve_config(
    *,
    repository_root: Path,
    model: str,
) -> tuple[str, str]:
    config_class = MODEL_CONFIG_CLASSES[model]
    registry = _load_weight_share_registry(repository_root)
    if config_class not in registry:
        raise ValueError(
            f"{config_class} is not in the merged #1124 "
            "WEIGHT_SHARE_VALIDATED_CONFIGS registry"
        )

    relative_path = MODEL_CONFIG_PATHS[model]
    path = repository_root / relative_path
    if not path.is_file():
        raise ValueError(f"Validated MPS config is missing: {relative_path}")
    actual_config_class = _config_class(path)
    if actual_config_class != config_class:
        raise ValueError(
            f"Validated MPS config {relative_path} declares "
            f"{actual_config_class!r}, expected {config_class!r}"
        )
    return relative_path, config_class


def select_tts_stage1(
    *,
    run_id: str,
    run_attempt: str,
    labels: list[str],
    override: str,
    repository_root: str | Path,
    topology: str = "multi_gpu",
) -> SelectionResult:
    """Resolve the model once from override, label, or stable run-id digest."""

    if not run_id:
        raise ValueError("github.run_id is required")
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"Unsupported TTS Stage 1 topology: {topology!r}")

    label_set = set(labels)
    if set(MODEL_LABELS.values()) <= label_set:
        raise ValueError(
            "Conflicting TTS model labels: run-higgs and run-moss cannot both be set"
        )

    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    normalized_override = override.strip().lower()
    if normalized_override:
        if normalized_override not in MODELS:
            raise ValueError(
                f"Unsupported tts_ci_model={override!r}. Expected one of: "
                f"{' '.join(MODELS)}"
            )
        selected_model = normalized_override
        reason = "workflow_dispatch_override"
    else:
        selected_model = ""
        reason = ""
        for model, label in MODEL_LABELS.items():
            if label in label_set:
                selected_model = model
                reason = f"label:{label}"
                break
        if not selected_model:
            selected_model = MODELS[int(digest, 16) % len(MODELS)]
            reason = "stable_run_id_sha256"

    resolved_config, config_class = _resolve_config(
        repository_root=Path(repository_root).resolve(),
        model=selected_model,
    )
    return SelectionResult(
        schema_version=SCHEMA_VERSION,
        selected_model=selected_model,
        selection_digest=digest,
        selection_reason=reason,
        tts_stage1_topology=topology,
        resolved_mps_config=resolved_config,
        resolved_config_class=config_class,
        run_id=run_id,
        run_attempt=run_attempt,
    )


def write_preflight(path: str | Path, result: SelectionResult) -> None:
    write_json_atomic(path, result.as_dict())


def _write_github_outputs(path: str | Path, result: SelectionResult) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        for key in (
            "selected_model",
            "selection_digest",
            "selection_reason",
            "tts_stage1_topology",
            "resolved_mps_config",
            "resolved_config_class",
            "run_attempt",
        ):
            handle.write(f"{key}={getattr(result, key)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--override", default="")
    parser.add_argument("--topology", default="multi_gpu")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--github-output")
    args = parser.parse_args()

    labels = json.loads(args.labels_json)
    if not isinstance(labels, list) or not all(
        isinstance(item, str) for item in labels
    ):
        parser.error("--labels-json must be a JSON array of strings")
    try:
        result = select_tts_stage1(
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            labels=labels,
            override=args.override,
            topology=args.topology,
            repository_root=args.repository_root,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    write_preflight(args.output, result)
    if args.github_output:
        _write_github_outputs(args.github_output, result)
    print(json.dumps(result.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
