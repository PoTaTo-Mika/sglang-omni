# SPDX-License-Identifier: Apache-2.0
"""Small fixed-seed nano-vLLM reference used by VoxCPM GPU parity jobs."""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np


async def _run(args: argparse.Namespace) -> None:
    from nanovllm_voxcpm import VoxCPM

    server = VoxCPM.from_pretrained(
        args.model,
        inference_timesteps=args.inference_timesteps,
        max_num_batched_tokens=32768,
        max_num_seqs=2,
        max_model_len=32768,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        devices=[0],
    )
    await server.wait_for_ready()
    try:
        chunks = []
        async for chunk in server.generate(
            target_text=args.text,
            max_generate_length=args.max_new_tokens,
            temperature=args.temperature,
            cfg_value=args.cfg_value,
            seed=args.seed,
        ):
            chunks.append(chunk)
        waveform = np.concatenate(chunks).astype(np.float32, copy=False)
        print(
            json.dumps(
                {
                    "shape": list(waveform.shape),
                    "dtype": str(waveform.dtype),
                    "mean": float(waveform.mean()),
                    "std": float(waveform.std()),
                    "max_abs": float(np.abs(waveform).max()),
                },
                sort_keys=True,
            )
        )
    finally:
        await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--inference-timesteps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
