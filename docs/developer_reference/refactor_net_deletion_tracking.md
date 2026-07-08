# Refactor Net Deletion Tracking

Use `scripts/refactor_net_deletions.py` to track whether a refactor is reducing
non-test code over time. The TTS refactor roadmap lives in
[#985](https://github.com/sgl-project/sglang-omni/issues/985); this script is
the line-count companion for PRs and issue updates.

The progress target is:

```text
non-test net deleted = deleted non-test lines - added non-test lines
```

The target is met when `non-test net deleted > 0`. Test changes are still
reported, but they are excluded from the progress target so that added coverage
does not hide whether the refactor is actually shrinking implementation code.

## Test File Detection

A changed file is treated as a test file when either of these rules match:

- Any parent directory is named `test`, `tests`, `unit_test`, `unit_tests`,
  `integration_test`, or `integration_tests`.
- The basename is `conftest.py`, starts with `test_`, or ends with a common test
  suffix such as `_test.py`, `_tests.py`, `.test.ts`, or `.spec.tsx`.

Because the whole path under `tests/` is excluded, fixtures, test data, helper
modules, and CI-only test utilities do not count toward the non-test deletion
target.

## Common Commands

For a PR branch, compare against the merge base with `origin/main`:

```bash
python scripts/refactor_net_deletions.py \
  --base origin/main \
  --head HEAD \
  --format markdown \
  --list-test-files
```

For local tracked work before committing, include the working tree:

```bash
python scripts/refactor_net_deletions.py \
  --base origin/main \
  --head HEAD \
  --mode worktree \
  --list-test-files \
  --list-non-test-files
```

For an issue or PR comment, use Markdown output. For automation, use JSON:

```bash
python scripts/refactor_net_deletions.py --format json
```

If a future CI job should enforce the target, add `--fail-on-nonpositive`.
Leave that flag off for normal tracking because some intermediate refactor PRs
may add shared infrastructure before later PRs delete model-local code.
