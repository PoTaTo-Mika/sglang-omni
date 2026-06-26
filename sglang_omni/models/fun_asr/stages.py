# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Fun-ASR-Nano inference.

Mirrors :mod:`sglang_omni.models.qwen3_asr.stages` but adapts the
feature-extractor / token-count path to Fun-ASR-Nano:

* The feature extractor is ``FunAsrNanoFeatureExtractor`` (80-mel + LFR), not
  Whisper. It is a custom class (not a transformers built-in), so it is
  registered with ``AutoFeatureExtractor`` in
  :mod:`configuration_fun_asr` (imported below for its registration side
  effects: it also registers the ``fun_asr_nano`` ``AutoConfig``).
* The 30 s audio-token ceiling is ``fun_asr_low_frame_rate_length(nb_max_frames)``
  (``nb_max_frames`` is already the post-LFR frame count for a 30 s clip; the
  adaptor's three stride-2 reductions remain) — 63 tokens for the default
  frontend, vs Qwen3-ASR's ``nb_max_frames // 2``.
* ``model_arch_override="FunAsrNanoForConditionalGeneration"`` selects our
  :class:`sglang_model.FunAsrNanoForConditionalGeneration` via the sglang-omni
  model registry (see ``sglang_model_runner._register_omni_model``).
"""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoFeatureExtractor, AutoTokenizer

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.fun_asr.configuration_fun_asr import (  # noqa: F401 — registers fun_asr_nano AutoConfig + FunAsrNanoFeatureExtractor
    FunAsrNanoConfig,
)
from sglang_omni.models.fun_asr.request_builders import (
    make_fun_asr_scheduler_adapters,
)
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version


def create_sglang_fun_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    # bf16 (not fp16): the Fun-ASR adaptor emits audio embeddings with large
    # magnitude (std ~29, max ~650) — the reference (demo_vllm.py) also defaults
    # to bf16. fp16's ~65504 range overflows to NaN inside the Qwen3 attention/MLP
    # on these embeddings, producing degenerate output (token 0 / "!" repetition).
    # bf16 shares fp32's exponent range and is stable. The checkpoint config is
    # dtype=bfloat16.
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    mm_attention_backend: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
):

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path, trust_remote_code=True
    )

    # nb_max_frames is the post-LFR frame count for a 30 s clip
    # (FunAsrNanoFeatureExtractor applies LFR internally). Only the adaptor's
    # three stride-2 reductions remain, so the audio-token ceiling is
    # fun_asr_low_frame_rate_length(nb_max_frames) (=63 for the default frontend).
    encoder_token_count = int(
        fun_asr_low_frame_rate_length(feature_extractor.nb_max_frames)
    )

    # Fun-ASR's ChatML prompt is ~23 tokens bare (system+user+assistant headers
    # + "语音转写：") and grows with the target-language suffix (语音转写成英文：,
    # +itn 不进行文本规整) and the hotwords context prefix (74+ tokens for 3
    # hotwords). Unlike Qwen3-ASR (whose 30 s audio ceiling is ~1500 tokens, so
    # a +8 prompt buffer is negligible), Fun-ASR's 30 s ceiling is only 63, so
    # the prompt overhead is a significant fraction of the context budget — a
    # +8 buffer rejects even a 24 s clip (74 input + 256 max_new > 326 kv). 128
    # comfortably covers the bare/en/itn prompt and up to ~5 hotwords; very long
    # hotword lists need a server_args_overrides context_length bump.
    prompt_overhead_tokens = 128

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": enable_torch_compile,
        "torch_compile_max_bs": max_running_requests,
        "cuda_graph_max_bs": max_running_requests,
        "mem_fraction_static": mem_fraction_static,
        "max_running_requests": max_running_requests,
        "max_prefill_tokens": 4096,
        "chunked_prefill_size": 4096,
        "sampling_backend": "pytorch",
        "dtype": dtype,
    }
    if mm_attention_backend is not None:
        overrides["mm_attention_backend"] = mm_attention_backend
    else:
        sm_version = get_visible_gpu_sm_version(gpu_id)
        if sm_version is not None and sm_version >= 100:
            overrides["mm_attention_backend"] = "triton_attn"
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        model_path,
        context_length=encoder_token_count + int(max_new_tokens) + prompt_overhead_tokens,
        **overrides,
    )

    # Temporarily disable CUDA graphs during infrastructure init (weights load +
    # memory-pool sizing), then re-enable and capture graphs on the built model.
    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="FunAsrNanoForConditionalGeneration",
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_new_tokens=max_new_tokens,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_fun_asr_executor(*args, **kwargs):
    return create_sglang_fun_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_fun_asr_executor", "create_fun_asr_executor"]
