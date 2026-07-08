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
python3 scripts/refactor_net_deletions.py \
  --base origin/main \
  --head HEAD \
  --format markdown \
  --list-test-files
```

For local tracked work before committing, include the working tree:

```bash
python3 scripts/refactor_net_deletions.py \
  --base origin/main \
  --head HEAD \
  --mode worktree \
  --list-test-files \
  --list-non-test-files
```

For an issue or PR comment, use Markdown output. For automation, use JSON:

```bash
python3 scripts/refactor_net_deletions.py --format json
```

If a future CI job should enforce the target, add `--fail-on-nonpositive`.
Leave that flag off for normal tracking because some intermediate refactor PRs
may add shared infrastructure before later PRs delete model-local code.

## HTML Dashboard

The same script can write a static dashboard. Serve the output directory with a
plain local HTTP server, then expose that localhost port with
[Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
or [ngrok](https://ngrok.com/docs/getting-started).

For the whole TTS refactor, choose one stable baseline and keep using it. The
parent of the first landed TTS refactor PR is a reasonable baseline:

```bash
git rev-parse 4e4c98a^1
```

Generate the dashboard once:

```bash
python3 scripts/refactor_net_deletions.py \
  --base 4e4c98a^1 \
  --head origin/main \
  --format html \
  --output /data/jaxan/tts-refactor-dashboard/index.html \
  --title "TTS Refactor Progress" \
  --refresh-seconds 300 \
  --path sglang_omni/models/fishaudio_s2_pro \
  --path sglang_omni/models/higgs_tts \
  --path sglang_omni/models/moss_tts \
  --path sglang_omni/models/moss_tts_local \
  --path sglang_omni/models/qwen3_tts \
  --path sglang_omni/models/voxtral_tts \
  --path sglang_omni/pipeline \
  --path sglang_omni/scheduling \
  --path sglang_omni/serve \
  --path tests/unit_test/fishaudio_s2_pro \
  --path tests/unit_test/higgs_tts \
  --path tests/unit_test/moss_tts \
  --path tests/unit_test/moss_tts_local \
  --path tests/unit_test/qwen3_tts \
  --path tests/unit_test/voxtral_tts \
  --path tests/test_model/test_tts_ci.py \
  --path tests/test_model/tts_ci_config.py \
  --list-test-files \
  --list-non-test-files
```

For an H100 host, keep the checkout and dashboard under persistent storage such
as `/data/jaxan`. Run the refresher and server in separate `tmux` panes:

```bash
while true; do
  git -C /data/jaxan/sglang-omni fetch origin main
  python3 /data/jaxan/sglang-omni/scripts/refactor_net_deletions.py \
    --repo /data/jaxan/sglang-omni \
    --base 4e4c98a^1 \
    --head origin/main \
    --format html \
    --output /data/jaxan/tts-refactor-dashboard/index.html \
    --title "TTS Refactor Progress" \
    --refresh-seconds 300 \
    --path sglang_omni/models/fishaudio_s2_pro \
    --path sglang_omni/models/higgs_tts \
    --path sglang_omni/models/moss_tts \
    --path sglang_omni/models/moss_tts_local \
    --path sglang_omni/models/qwen3_tts \
    --path sglang_omni/models/voxtral_tts \
    --path sglang_omni/pipeline \
    --path sglang_omni/scheduling \
    --path sglang_omni/serve \
    --path tests/unit_test/fishaudio_s2_pro \
    --path tests/unit_test/higgs_tts \
    --path tests/unit_test/moss_tts \
    --path tests/unit_test/moss_tts_local \
    --path tests/unit_test/qwen3_tts \
    --path tests/unit_test/voxtral_tts \
    --path tests/test_model/test_tts_ci.py \
    --path tests/test_model/tts_ci_config.py \
    --list-test-files \
    --list-non-test-files
  sleep 300
done
```

```bash
python3 -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory /data/jaxan/tts-refactor-dashboard
```

Expose it temporarily with Cloudflare:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

Or with ngrok:

```bash
ngrok http 127.0.0.1:8765
```

Only expose the generated dashboard directory. Do not serve the full checkout or
any directory containing credentials, caches, checkpoints, or private artifacts.
