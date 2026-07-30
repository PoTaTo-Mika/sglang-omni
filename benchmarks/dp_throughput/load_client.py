# SPDX-License-Identifier: Apache-2.0
"""Continuous closed-loop seed-tts load generator for throughput / GPU-idle profiling.

Keeps exactly ``--conc`` requests in flight at all times: each of ``conc`` worker
coroutines re-issues the next request the instant its previous one completes, for
``--secs``. No inter-batch drain, so the server stays saturated for the whole
measurement window (this is what makes the device-level SM% window clean).
Reuses the in-repo ``make_tts_send_fn`` + ``load_seedtts_samples`` so requests
match ``benchmark_tts_seedtts.py`` and hit ``/v1/audio/speech``.

Run one instance per replica (or several against a router) as a numactl-pinned
subprocess so N clients are N independent GILs (rules out client-GIL as the serial
bottleneck). Point it at a server port (``--port``) or any base URL (``--base-url``,
e.g. a router).

Reads throughput over the steady window [warmup_secs, secs] and a per-5s-bucket
coefficient of variation so you can confirm the load was steady.
"""
import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

# Make the repo importable without installing (this file lives in benchmarks/dp_throughput).
_REPO = os.environ.get("REPO") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import aiohttp  # noqa: E402

from benchmarks.dataset.prepare import DATASETS  # noqa: E402
from benchmarks.dataset.seedtts import load_seedtts_samples  # noqa: E402
from benchmarks.tasks.tts import _build_tts_payload, make_tts_send_fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=0)
ap.add_argument(
    "--base-url", default=None, help="override http://host:port (e.g. a router)"
)
ap.add_argument("--model", required=True)
ap.add_argument("--conc", type=int, required=True)
ap.add_argument("--secs", type=int, default=90)
ap.add_argument("--warmup-secs", type=int, default=15)
ap.add_argument("--voice-clone", action="store_true")
ap.add_argument(
    "--ref-format",
    choices=["flat", "references"],
    default="references",
    help="reference payload shape; Higgs requires 'references'",
)
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--lang", default="en")
ap.add_argument("--meta", default=None, help="dataset id; default DATASETS['seedtts']")
ap.add_argument(
    "--save-latencies",
    action="store_true",
    help="include sorted steady-window request latencies in the output JSON",
)
ap.add_argument("--out", required=True)
args = ap.parse_args()

meta = args.meta or os.environ.get("SEEDTTS_META") or DATASETS["seedtts"]
samples = load_seedtts_samples(meta, None, split=args.lang)
if not samples:
    raise SystemExit("no seed-tts samples loaded")
N = len(samples)

base_url = args.base_url or f"http://127.0.0.1:{args.port}"
api_url = f"{base_url}/v1/audio/speech"
send_fn = make_tts_send_fn(
    args.model,
    api_url,
    no_ref_audio=not args.voice_clone,
    ref_format=args.ref_format,
    max_new_tokens=args.max_new_tokens,
    stream=False,
    save_audio_dir=None,
    temperature=0.7,
)

# Fail before generating load if the benchmark drifts back to a request shape
# that Higgs preprocessing does not consume.
probe_payload = _build_tts_payload(
    samples[0],
    args.model,
    no_ref_audio=not args.voice_clone,
    ref_format=args.ref_format,
    max_new_tokens=args.max_new_tokens,
    temperature=0.7,
)
if probe_payload.get("input") != samples[0].target_text or not probe_payload["input"]:
    raise SystemExit(
        "invalid TTS payload: non-empty target text is required in 'input'"
    )
if args.voice_clone:
    references = probe_payload.get("references")
    if args.ref_format != "references" or not references:
        raise SystemExit(
            "invalid Higgs voice-clone payload: --ref-format references is required"
        )
    reference = references[0]
    if (
        reference.get("audio_path") != samples[0].ref_audio
        or reference.get("text") != samples[0].ref_text
    ):
        raise SystemExit("invalid Higgs voice-clone payload: reference data was lost")
print(
    f"[client {base_url}] endpoint=/v1/audio/speech "
    f"payload_check=input:{len(probe_payload['input'])}chars "
    f"references:{len(probe_payload.get('references', []))}",
    flush=True,
)

comp_ts = []  # completion timestamps (rel to t0), success only
lat_s = []  # (completion_ts, latency_seconds), success only
prompt_tokens = []
completion_tokens = []
audio_durations = []
errors = [0]
error_examples = []
state = {"stop": None, "t0": None}


async def worker(session, k):
    i = k
    while True:
        if time.perf_counter() >= state["stop"]:
            return
        s = samples[i % N]
        i += args.conc
        t_req = time.perf_counter()
        r = await send_fn(session, s)
        if getattr(r, "is_success", False):
            t_done = time.perf_counter()
            comp_ts.append(t_done - state["t0"])
            lat_s.append((t_done - state["t0"], t_done - t_req))
            if getattr(r, "prompt_tokens", 0) > 0:
                prompt_tokens.append(r.prompt_tokens)
            if getattr(r, "completion_tokens", 0) > 0:
                completion_tokens.append(r.completion_tokens)
            if getattr(r, "audio_duration_s", 0) > 0:
                audio_durations.append(r.audio_duration_s)
        else:
            errors[0] += 1
            if len(error_examples) < 5:
                error_examples.append(getattr(r, "error", "unknown request error"))


async def main():
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        state["t0"] = time.perf_counter()
        state["stop"] = state["t0"] + args.secs
        await asyncio.gather(
            *[asyncio.create_task(worker(session, k)) for k in range(args.conc)]
        )


asyncio.run(main())

w0, w1 = args.warmup_secs, args.secs
win = [t for t in comp_ts if w0 <= t <= w1]
qps_window = len(win) / max(1e-9, (w1 - w0))
buckets = {}
for t in comp_ts:
    b = int(t // 5) * 5
    buckets[b] = buckets.get(b, 0) + 1
bucket_qps = {b: round(c / 5.0, 2) for b, c in sorted(buckets.items())}
bvals = [v for b, v in bucket_qps.items() if w0 <= b < w1]

lat_win = sorted(l for t, l in lat_s if w0 <= t <= w1)


def _pct(p):
    return (
        round(lat_win[min(len(lat_win) - 1, int(p * len(lat_win)))], 3)
        if lat_win
        else None
    )


agg = {
    "base_url": base_url,
    "api_url": api_url,
    "payload_check": {
        "input_chars": len(probe_payload["input"]),
        "references": len(probe_payload.get("references", [])),
    },
    "conc": args.conc,
    "secs": args.secs,
    "warmup_secs": args.warmup_secs,
    "n_completions_total": len(comp_ts),
    "n_completions_window": len(win),
    "errors": errors[0],
    "error_examples": error_examples,
    "qps_window": round(qps_window, 4),
    "qps_overall": round(len(comp_ts) / max(1e-9, args.secs), 4),
    "bucket_qps_cv": (
        round(st.pstdev(bvals) / (st.mean(bvals) + 1e-9), 3) if bvals else None
    ),
    "lat_mean_s": round(st.mean(lat_win), 3) if lat_win else None,
    "lat_p50_s": _pct(0.50),
    "lat_p95_s": _pct(0.95),
    "prompt_tokens_mean": (round(st.mean(prompt_tokens), 2) if prompt_tokens else None),
    "completion_tokens_mean": (
        round(st.mean(completion_tokens), 2) if completion_tokens else None
    ),
    "audio_duration_mean_s": (
        round(st.mean(audio_durations), 3) if audio_durations else None
    ),
    "bucket_qps": bucket_qps,
}
if args.save_latencies:
    agg["latency_window_s"] = [round(value, 6) for value in lat_win]
json.dump(agg, open(args.out, "w"), indent=2)
print(
    f"[client {base_url}] qps_window={qps_window:.3f} n_win={len(win)} "
    f"errors={errors[0]} cv={agg['bucket_qps_cv']}",
    flush=True,
)
if errors[0]:
    raise SystemExit(f"{errors[0]} request(s) failed; benchmark is invalid")
