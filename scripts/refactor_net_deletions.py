"""Track refactor net line deletion while excluding test files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


TEST_DIRECTORY_NAMES = frozenset(
    {
        "test",
        "tests",
        "unit_test",
        "unit_tests",
        "integration_test",
        "integration_tests",
    }
)
TEST_FILE_NAMES = frozenset({"conftest.py"})
TEST_FILE_SUFFIXES = (
    "_test.py",
    "_tests.py",
    ".test.js",
    ".test.jsx",
    ".test.ts",
    ".test.tsx",
    ".spec.js",
    ".spec.jsx",
    ".spec.ts",
    ".spec.tsx",
)


@dataclass(frozen=True)
class FileStat:
    paths: tuple[str, ...]
    added: int | None
    deleted: int | None
    is_test: bool

    @property
    def display_path(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0]
        return f"{self.paths[0]} -> {self.paths[-1]}"


@dataclass
class Totals:
    files: int = 0
    binary_files: int = 0
    added: int = 0
    deleted: int = 0

    @property
    def net_deleted(self) -> int:
        return self.deleted - self.added

    def add(self, file_stat: FileStat) -> None:
        self.files += 1
        if file_stat.added is None or file_stat.deleted is None:
            self.binary_files += 1
            return
        self.added += file_stat.added
        self.deleted += file_stat.deleted

    def to_dict(self) -> dict[str, int]:
        return {
            "files": self.files,
            "binary_files": self.binary_files,
            "added": self.added,
            "deleted": self.deleted,
            "net_deleted": self.net_deleted,
        }


@dataclass(frozen=True)
class DiffSpec:
    label: str
    git_args: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    repo: str
    diff_label: str
    mode: str
    totals: dict[str, Totals]
    files: tuple[FileStat, ...]

    @property
    def target_met(self) -> bool:
        return self.totals["non_test"].net_deleted > 0


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = tuple(part for part in PurePosixPath(normalized).parts if part != ".")
    if not parts:
        return False

    if any(part.lower() in TEST_DIRECTORY_NAMES for part in parts[:-1]):
        return True

    name = parts[-1].lower()
    if name in TEST_FILE_NAMES:
        return True
    if name.startswith("test_"):
        return True
    return any(name.endswith(suffix) for suffix in TEST_FILE_SUFFIXES)


def parse_numstat_z(data: bytes) -> tuple[FileStat, ...]:
    tokens = data.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()

    results: list[FileStat] = []
    index = 0
    while index < len(tokens):
        header = _decode_git_path(tokens[index])
        fields = header.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"invalid --numstat -z record header: {header!r}")

        added = _parse_numstat_count(fields[0])
        deleted = _parse_numstat_count(fields[1])
        path_field = fields[2]

        if path_field:
            paths = (path_field,)
            index += 1
        else:
            if index + 2 >= len(tokens):
                raise ValueError("rename --numstat -z record is missing paths")
            paths = (
                _decode_git_path(tokens[index + 1]),
                _decode_git_path(tokens[index + 2]),
            )
            index += 3

        results.append(
            FileStat(
                paths=paths,
                added=added,
                deleted=deleted,
                is_test=any(is_test_path(path) for path in paths),
            )
        )

    return tuple(results)


def build_report(repo: Path, diff_spec: DiffSpec, mode: str) -> Report:
    raw = _run_git(
        repo,
        (
            "diff",
            "--numstat",
            "-z",
            "--find-renames",
            *diff_spec.git_args,
        ),
    )
    files = parse_numstat_z(raw)

    all_totals = Totals()
    test_totals = Totals()
    non_test_totals = Totals()
    for file_stat in files:
        all_totals.add(file_stat)
        if file_stat.is_test:
            test_totals.add(file_stat)
        else:
            non_test_totals.add(file_stat)

    return Report(
        repo=str(repo),
        diff_label=diff_spec.label,
        mode=mode,
        totals={
            "non_test": non_test_totals,
            "test": test_totals,
            "all": all_totals,
        },
        files=files,
    )


def format_text(report: Report, list_test_files: bool, list_non_test_files: bool) -> str:
    lines = [
        "Refactor net deletion tracking",
        f"Repo: {report.repo}",
        f"Diff: {report.diff_label}",
        f"Mode: {report.mode}",
        "",
        _format_table(report),
        "",
        f"Target: non-test net deleted > 0 ({_format_target(report)})",
        "",
        "Test files are excluded from the progress target.",
    ]

    if list_test_files:
        lines.extend(["", "Excluded test files:"])
        lines.extend(_format_file_list(report.files, is_test=True))

    if list_non_test_files:
        lines.extend(["", "Counted non-test files:"])
        lines.extend(_format_file_list(report.files, is_test=False))

    return "\n".join(lines)


def format_markdown(
    report: Report, list_test_files: bool, list_non_test_files: bool
) -> str:
    lines = [
        "### Refactor Net Deletion Tracking",
        "",
        f"- Diff: `{report.diff_label}`",
        f"- Mode: `{report.mode}`",
        f"- Target: non-test net deleted > 0 (**{_format_target(report)}**)",
        "",
        "| Category | Files | Binary | Added | Deleted | Net deleted |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, label in (("non_test", "Non-test"), ("test", "Test"), ("all", "All")):
        totals = report.totals[key]
        lines.append(
            "| "
            f"{label} | {totals.files} | {totals.binary_files} | "
            f"{totals.added} | {totals.deleted} | {totals.net_deleted} |"
        )

    if list_test_files:
        lines.extend(["", "<details><summary>Excluded test files</summary>", ""])
        lines.extend(f"- `{path}`" for path in _matching_paths(report.files, True))
        lines.extend(["", "</details>"])

    if list_non_test_files:
        lines.extend(["", "<details><summary>Counted non-test files</summary>", ""])
        lines.extend(f"- `{path}`" for path in _matching_paths(report.files, False))
        lines.extend(["", "</details>"])

    return "\n".join(lines)


def format_json(report: Report) -> str:
    payload = {
        "repo": report.repo,
        "diff": report.diff_label,
        "mode": report.mode,
        "target": {
            "name": "non_test_net_deleted_gt_zero",
            "met": report.target_met,
        },
        "totals": {key: value.to_dict() for key, value in report.totals.items()},
        "files": [
            {
                **asdict(file_stat),
                "display_path": file_stat.display_path,
            }
            for file_stat in report.files
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track refactor net line deletion while excluding test files from "
            "the progress target."
        )
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository path. Defaults to the current directory.",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref for branch comparisons. Defaults to origin/main.",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head ref for branch comparisons. Defaults to HEAD.",
    )
    parser.add_argument(
        "--range",
        dest="diff_range",
        help="Explicit git diff range, such as origin/main...HEAD.",
    )
    parser.add_argument(
        "--mode",
        choices=("merge-base", "direct", "worktree", "staged"),
        default="merge-base",
        help=(
            "Diff mode: merge-base uses base...head, direct uses base head, "
            "worktree compares the working tree with merge-base(base, head), "
            "and staged compares the index with merge-base(base, head)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--list-test-files",
        action="store_true",
        help="List changed test files excluded from the progress target.",
    )
    parser.add_argument(
        "--list-non-test-files",
        action="store_true",
        help="List changed non-test files counted toward the progress target.",
    )
    parser.add_argument(
        "--fail-on-nonpositive",
        action="store_true",
        help="Exit with status 1 when non-test net deleted is not positive.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = Path(args.repo).resolve()

    diff_spec = resolve_diff_spec(repo, args)
    report = build_report(repo, diff_spec, args.mode)

    if args.format == "json":
        output = format_json(report)
    elif args.format == "markdown":
        output = format_markdown(report, args.list_test_files, args.list_non_test_files)
    else:
        output = format_text(report, args.list_test_files, args.list_non_test_files)

    print(output)
    if args.fail_on_nonpositive and not report.target_met:
        return 1
    return 0


def resolve_diff_spec(repo: Path, args: argparse.Namespace) -> DiffSpec:
    if args.diff_range:
        return DiffSpec(label=args.diff_range, git_args=(args.diff_range,))

    if args.mode == "merge-base":
        label = f"{args.base}...{args.head}"
        return DiffSpec(label=label, git_args=(label,))

    if args.mode == "direct":
        label = f"{args.base}..{args.head}"
        return DiffSpec(label=label, git_args=(args.base, args.head))

    merge_base = _run_git(repo, ("merge-base", args.base, args.head)).decode().strip()
    if args.mode == "worktree":
        return DiffSpec(
            label=f"working tree vs merge-base({args.base}, {args.head}) {merge_base}",
            git_args=(merge_base,),
        )
    if args.mode == "staged":
        return DiffSpec(
            label=f"staged vs merge-base({args.base}, {args.head}) {merge_base}",
            git_args=("--cached", merge_base),
        )

    raise ValueError(f"unsupported mode: {args.mode}")


def _run_git(repo: Path, args: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return completed.stdout


def _decode_git_path(value: bytes) -> str:
    return value.decode("utf-8", "surrogateescape")


def _parse_numstat_count(value: str) -> int | None:
    if value == "-":
        return None
    return int(value)


def _format_table(report: Report) -> str:
    rows = [("Category", "Files", "Binary", "Added", "Deleted", "Net deleted")]
    for key, label in (("non_test", "non-test"), ("test", "test"), ("all", "all")):
        totals = report.totals[key]
        rows.append(
            (
                label,
                str(totals.files),
                str(totals.binary_files),
                str(totals.added),
                str(totals.deleted),
                str(totals.net_deleted),
            )
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )


def _format_target(report: Report) -> str:
    return "met" if report.target_met else "not met"


def _matching_paths(files: Sequence[FileStat], is_test: bool) -> tuple[str, ...]:
    return tuple(file_stat.display_path for file_stat in files if file_stat.is_test is is_test)


def _format_file_list(files: Sequence[FileStat], is_test: bool) -> list[str]:
    paths = _matching_paths(files, is_test)
    if not paths:
        return ["  (none)"]
    return [f"  {path}" for path in paths]


if __name__ == "__main__":
    raise SystemExit(main())
