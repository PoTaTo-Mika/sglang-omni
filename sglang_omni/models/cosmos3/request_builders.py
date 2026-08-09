# SPDX-License-Identifier: Apache-2.0
"""SGLang request and result adapters for Cosmos3 text generation."""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import GenerationConfig, PreTrainedTokenizerBase

from sglang_omni.models.cosmos3.payload_types import (
    Cosmos3PipelineState,
    PromptInputs,
    TextOutput,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"

_TRANSPORT_SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
}


def _value_or_default(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _request_value(params: dict[str, Any], name: str, default: Any) -> Any:
    value = params.get(name)
    return default if value is None else value


def build_cosmos3_sampling_kwargs(
    params: dict[str, Any],
    *,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Merge request sampling overrides over Cosmos3-Nano defaults."""

    do_sample = bool(
        _value_or_default(
            generation_config.do_sample if generation_config is not None else None,
            True,
        )
    )
    default_temperature = _value_or_default(
        generation_config.temperature if generation_config is not None else None,
        0.7,
    )
    if not do_sample:
        default_temperature = 0.0
    sampling_seed = params.get("seed")
    if sampling_seed is None:
        sampling_seed = params.get("sampling_seed")
    return {
        "max_new_tokens": _request_value(params, "max_new_tokens", 2048),
        "temperature": _request_value(params, "temperature", default_temperature),
        "top_p": _request_value(
            params,
            "top_p",
            _value_or_default(
                generation_config.top_p if generation_config is not None else None,
                0.8,
            ),
        ),
        "top_k": _request_value(
            params,
            "top_k",
            _value_or_default(
                generation_config.top_k if generation_config is not None else None,
                20,
            ),
        ),
        "min_p": _request_value(params, "min_p", 0.0),
        "repetition_penalty": _request_value(
            params,
            "repetition_penalty",
            _value_or_default(
                (
                    generation_config.repetition_penalty
                    if generation_config is not None
                    else None
                ),
                1.0,
            ),
        ),
        "stop": params.get("stop") or [],
        "stop_token_ids": params.get("stop_token_ids") or [],
        "sampling_seed": sampling_seed,
    }


def _effective_request_params(
    params: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Remove transport placeholders that must not shadow Nano defaults.

    OpenAI-compatible request parsing materializes generic sampling defaults.
    The explicit-field marker distinguishes those placeholders from values the
    caller actually supplied. Sparse internal requests predate that marker, so
    non-placeholder values remain valid overrides for backward compatibility.
    """

    effective = dict(params)
    raw_explicit_fields = metadata.get(EXPLICIT_GENERATION_PARAMS_KEY)
    has_explicit_marker = raw_explicit_fields is not None
    explicit_fields = set(raw_explicit_fields or ())

    for field, transport_default in _TRANSPORT_SAMPLING_DEFAULTS.items():
        if field in explicit_fields:
            continue
        value = effective.get(field)
        if has_explicit_marker or value is None or value == transport_default:
            effective.pop(field, None)
    return effective


def _eos_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    generation_config: GenerationConfig | None,
) -> set[int] | None:
    values = _value_or_default(
        generation_config.eos_token_id if generation_config is not None else None,
        tokenizer.eos_token_id,
    )
    if values is None:
        return None
    if isinstance(values, int):
        values = [values]
    result = {int(value) for value in values if value is not None and int(value) >= 0}
    return result or None


def build_sglang_text_request(
    state: Cosmos3PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    generation_config: GenerationConfig | None = None,
    request_id: str | None = None,
) -> SGLangARRequestData:
    """Build one SGLang request from the canonical preprocessing state."""

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prompt = cast(PromptInputs, state.prompt)
    input_ids = prompt["input_ids"]

    sampling_kwargs = build_cosmos3_sampling_kwargs(
        params,
        generation_config=generation_config,
    )
    sampling_params = SamplingParams(**sampling_kwargs)
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=request_id or "cosmos3-request",
        origin_input_text=prompt["prompt_text"],
        origin_input_ids=input_ids.tolist(),
        sampling_params=sampling_params,
        eos_token_ids=_eos_token_ids(tokenizer, generation_config),
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer

    return SGLangARRequestData(
        input_ids=input_ids,
        attention_mask=prompt["attention_mask"],
        max_new_tokens=int(sampling_kwargs["max_new_tokens"]),
        temperature=float(sampling_kwargs["temperature"]),
        top_p=float(sampling_kwargs["top_p"]),
        top_k=int(sampling_kwargs["top_k"]),
        repetition_penalty=float(sampling_kwargs["repetition_penalty"]),
        output_ids=req.output_ids,
        req=req,
    )


def apply_text_result(
    state: Cosmos3PipelineState,
    *,
    stage_name: str,
    result: SGLangARRequestData,
) -> TextOutput:
    text_out: TextOutput = {
        "output_ids": list(result.output_ids),
        "is_final": True,
    }
    finish_reason = result.finish_reason
    if finish_reason is not None:
        text_out["finish_reason"] = finish_reason
    output_token_logprobs = result.output_token_logprobs
    if output_token_logprobs:
        text_out["output_token_logprobs"] = output_token_logprobs
    weight_version = result.weight_version
    if weight_version is not None:
        text_out["weight_version"] = weight_version
    state.text_out = text_out
    state.engine_outputs[stage_name] = text_out
    return text_out


def make_text_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    generation_config: GenerationConfig | None = None,
    stage_name: str = THINKER_STAGE,
):
    """Build StagePayload-to-SGLang adapters for the text thinker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = Cosmos3PipelineState.from_dict(payload.data)
        params = _effective_request_params(
            payload.request.params,
            payload.request.metadata,
        )
        req_data = build_sglang_text_request(
            state,
            params=params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            generation_config=generation_config,
            request_id=payload.request_id,
        )
        req_data.stage_payload = payload
        req_data.return_logprob = bool(payload.request.params.get("return_logprob"))
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = cast(StagePayload, data.stage_payload)
        state = Cosmos3PipelineState.from_dict(payload.data)
        apply_text_result(state, stage_name=stage_name, result=data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


def make_text_stream_output_builder(*, decode_stage: str = DECODE_STAGE):
    """Forward generated token ids to the stream-aware detokenizer."""

    def stream_output_builder(
        request_id: str,
        req_data: SGLangARRequestData,
        req_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        req = req_data.req
        if req.inflight_middle_chunks > 0:
            return []
        if req_output.data is None:
            return []
        payload = cast(StagePayload, req_data.stage_payload)
        if not bool(payload.request.params.get("stream", False)):
            return []
        token_id = int(req_output.data)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=torch.tensor([token_id], dtype=torch.long),
                target=decode_stage,
                metadata={"token_id": token_id},
            )
        ]

    return stream_output_builder


__all__ = [
    "DECODE_STAGE",
    "THINKER_STAGE",
    "apply_text_result",
    "build_cosmos3_sampling_kwargs",
    "build_sglang_text_request",
    "make_text_scheduler_adapters",
    "make_text_stream_output_builder",
]
