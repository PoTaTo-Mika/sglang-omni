# SPDX-License-Identifier: Apache-2.0
"""Request/result builders for LLaDA2-Uni pipeline stages."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.llada2_uni.components.preprocessor import (
    DUMMY_IMAGE_TOKEN_ID,
    IMAGE_TOKEN_OFFSET,
)
from sglang_omni.models.llada2_uni.config import (
    DEFAULT_THINKER_MAX_NEW_TOKENS,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.interleaved import (
    CFGBranchPlan,
    InterleavedGenerationConfig,
    build_cfg_plan,
    parse_image_header,
)
from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniPipelineState,
    ThinkerOutput,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangDLLMRequestData

logger = logging.getLogger(__name__)


def _align_cfg_branch_group(
    *,
    tokenizer: Any,
    branches: dict[str, list[int]],
    existing_left_pad_lens: dict[str, int | None],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Left-pad every CFG branch to the longest physical prompt."""
    target_length = max(len(input_ids) for input_ids in branches.values())
    requires_padding = any(
        len(input_ids) < target_length for input_ids in branches.values()
    )
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if requires_padding and mask_token_id is None:
        raise ValueError("LLaDA2 tokenizer has no mask_token_id for CFG padding")

    aligned: dict[str, list[int]] = {}
    left_pad_lens: dict[str, int] = {}
    for branch_name, input_ids in branches.items():
        existing_left_pad_len = int(existing_left_pad_lens.get(branch_name) or 0)
        if existing_left_pad_len < 0 or existing_left_pad_len > len(input_ids):
            raise ValueError(
                f"CFG {branch_name} has invalid left-pad length "
                f"{existing_left_pad_len} for {len(input_ids)} input tokens"
            )
        added_left_pad_len = target_length - len(input_ids)
        padding = (
            [int(mask_token_id)] * added_left_pad_len if added_left_pad_len else []
        )
        aligned[branch_name] = padding + list(input_ids)
        left_pad_lens[branch_name] = existing_left_pad_len + added_left_pad_len
    return aligned, left_pad_lens


def _attach_cfg_branches(
    req: Any,
    *,
    tokenizer: Any,
    conditional_input_ids: list[int],
    uncond_input_ids: list[int],
    cfg_scale: float,
    cfg_rescale: float,
    uncond_left_pad_len: int | None = None,
    uncond_img_input_ids: list[int] | None = None,
    uncond_img_left_pad_len: int | None = None,
    cfg_image_scale: float | None = None,
) -> None:
    """Align and attach the CFG branches consumed by ``DllmScheduler``."""
    branches = {
        "conditional": list(conditional_input_ids),
        "unconditional": list(uncond_input_ids),
    }
    existing_left_pad_lens = {
        "conditional": getattr(req, "_dllm_left_pad_len", 0),
        "unconditional": uncond_left_pad_len,
    }
    if uncond_img_input_ids is not None:
        branches["image-unconditional"] = list(uncond_img_input_ids)
        existing_left_pad_lens["image-unconditional"] = uncond_img_left_pad_len

    aligned, left_pad_lens = _align_cfg_branch_group(
        tokenizer=tokenizer,
        branches=branches,
        existing_left_pad_lens=existing_left_pad_lens,
    )
    req.origin_input_ids = aligned["conditional"]
    req._dllm_left_pad_len = left_pad_lens["conditional"]
    req._uncond_input_ids = aligned["unconditional"]
    req._uncond_left_pad_len = left_pad_lens["unconditional"]
    req._cfg_scale = float(cfg_scale)
    req._cfg_rescale = float(cfg_rescale)

    if uncond_img_input_ids is None:
        return
    req._uncond_img_input_ids = aligned["image-unconditional"]
    req._uncond_img_left_pad_len = left_pad_lens["image-unconditional"]
    req._cfg_image_scale = float(0.0 if cfg_image_scale is None else cfg_image_scale)


def build_encoder_request(
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
) -> dict[str, Any]:
    """Build encoder request dict from pipeline state."""
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return {"_skip": True, "_result": {}}
    if inputs.get("_skip"):
        return {"_skip": True, "_result": inputs.get("_result", {})}
    return dict(inputs)


def apply_encoder_result(
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    """Apply encoder result to pipeline state."""
    state.encoder_outs[stage_name] = result


def merge_image_tokens_for_thinker(state: LLaDA2UniPipelineState) -> None:
    """Merge VQ token IDs from image encoder output into prompt input_ids.

    Replaces DUMMY_IMAGE_TOKEN_ID placeholders with actual VQ token IDs
    offset by image_token_offset.
    """
    image_out = state.encoder_outs.get(IMAGE_STAGE)
    if not image_out:
        return

    image_token_ids_list = image_out.get("image_token_ids")
    if not image_token_ids_list:
        return

    prompt = state.prompt
    if not isinstance(prompt, dict) or "input_ids" not in prompt:
        return

    input_ids = prompt["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.flatten().tolist()

    all_vq_tokens = []
    for token_ids in image_token_ids_list:
        all_vq_tokens.extend(tid + IMAGE_TOKEN_OFFSET for tid in token_ids)

    if not all_vq_tokens:
        return

    new_ids = []
    vq_idx = 0
    for tid in input_ids:
        if tid == DUMMY_IMAGE_TOKEN_ID:
            if vq_idx >= len(all_vq_tokens):
                raise ValueError(
                    f"More placeholders than VQ tokens ({len(all_vq_tokens)})"
                )
            new_ids.append(all_vq_tokens[vq_idx])
            vq_idx += 1
        else:
            new_ids.append(tid)

    if vq_idx != len(all_vq_tokens):
        raise ValueError(
            f"VQ token count mismatch: {len(all_vq_tokens)} VQ tokens "
            f"but only {vq_idx} placeholders"
        )

    prompt["input_ids"] = torch.tensor([new_ids], dtype=torch.long)

    # Also merge into uncond_input_ids for edit/T2I CFG — the unconditional
    # prompt also contains DUMMY_IMAGE_TOKEN_ID placeholders that must be
    # replaced with real VQ codes from the encoder output.
    uncond_ids = state.stream_state.get("uncond_input_ids")
    if uncond_ids is not None and any(
        tid == DUMMY_IMAGE_TOKEN_ID for tid in uncond_ids
    ):
        state.stream_state["uncond_input_ids"] = _replace_dummy_tokens(
            uncond_ids, all_vq_tokens
        )


def _replace_dummy_tokens(input_ids: list[int], vq_tokens: list[int]) -> list[int]:
    """Replace DUMMY_IMAGE_TOKEN_ID entries with actual VQ token IDs."""
    new_ids = []
    vq_idx = 0
    for tid in input_ids:
        if tid == DUMMY_IMAGE_TOKEN_ID and vq_idx < len(vq_tokens):
            new_ids.append(vq_tokens[vq_idx])
            vq_idx += 1
        else:
            new_ids.append(tid)
    if vq_idx < len(vq_tokens):
        new_ids.extend(vq_tokens[vq_idx:])
    return new_ids


def build_dllm_thinker_request(
    state: LLaDA2UniPipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    dllm_config: Any,
    request_id: str | None = None,
) -> SGLangDLLMRequestData:
    """Build SGLangDLLMRequestData for the LLaDA2-Uni thinker."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    input_ids_list = input_ids.to(dtype=torch.long).flatten().tolist()

    # Override max_new_tokens for model-controlled image and interleaved phases.
    max_new_tokens = params.get("max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS)
    ss = state.stream_state if isinstance(state.stream_state, dict) else {}
    is_thinking_phase1 = ss.get("thinking_mode") and ss.get("thinking_phase") == 1
    interleaved_config = None
    interleaved_phase = None
    if state.task_kind == "interleaved":
        interleaved_config = InterleavedGenerationConfig.from_metadata(
            state.request_metadata
        )
        interleaved_phase = ss.get("interleaved_phase", "text")

    if is_thinking_phase1:
        max_new_tokens = 2048
    elif state.task_kind in ("t2i", "edit"):
        image_info = ss.get("image_info", [])
        if image_info:
            gh = int(image_info[0].get("grid_h", 0))
            gw = int(image_info[0].get("grid_w", 0))
            if gh > 0 and gw > 0:
                max_new_tokens = gh * gw
    elif interleaved_phase == "text":
        max_seq_len = int(ss.get("interleaved_max_seq_len", 8192))
        available = max_seq_len - len(input_ids_list)
        if available <= 0:
            raise ValueError("interleaved thinker exhausted its context before EOS")
        max_new_tokens = min(interleaved_config.text_max_new_tokens, available)
    elif interleaved_phase == "image":
        current_frame = ss.get("interleaved_current_frame", {})
        remaining = int(current_frame.get("remaining_image_tokens", 0))
        if remaining <= 0:
            raise ValueError("interleaved image phase has no remaining tokens")
        # The reference image path reserves room for EOI and terminates only
        # after the model emits it.
        max_new_tokens = remaining + 1

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=params.get("temperature", 0.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", -1),
        min_p=params.get("min_p", 0.0),
        repetition_penalty=params.get("repetition_penalty", 1.0),
        stop=params.get("stop") or [],
        stop_token_ids=params.get("stop_token_ids") or [],
        sampling_seed=params.get("seed"),
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = {eos_token_id} if eos_token_id is not None else set()
    if is_thinking_phase1:
        boi_id = getattr(tokenizer, "boi_token_id", None)
        if boi_id is None:
            boi_id = tokenizer.convert_tokens_to_ids("<boi>")
        if boi_id is not None:
            eos_token_ids = (eos_token_ids or set()) | {boi_id}
    elif interleaved_phase == "text":
        boi_id = tokenizer.convert_tokens_to_ids("<boi>")
        eos_token_ids = (eos_token_ids or set()) | {int(boi_id)}
    elif interleaved_phase == "image":
        eoi_id = tokenizer.convert_tokens_to_ids("<|/image|>")
        eos_token_ids = {int(eoi_id)}

    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
        dllm_config=dllm_config,
    )
    req.tokenizer = tokenizer

    req.omni_model_inputs = None
    req._omni_consumed = None

    # Attach CFG metadata if present (edit and t2i with cfg_scale > 1.0)
    uncond_ids = (
        state.stream_state.get("uncond_input_ids")
        if isinstance(state.stream_state, dict)
        else None
    )
    if uncond_ids is not None and not is_thinking_phase1:
        ig_meta = state.request_metadata.get("image_generation", {})
        uncond_img_ids = ss.get("uncond_img_input_ids")
        _attach_cfg_branches(
            req,
            tokenizer=tokenizer,
            conditional_input_ids=input_ids_list,
            uncond_input_ids=list(uncond_ids),
            uncond_left_pad_len=int(ss.get("uncond_left_pad_len", 0)),
            cfg_scale=ss.get(
                "cfg_scale",
                ig_meta.get("cfg_text_scale", ig_meta.get("cfg_scale", 1.0)),
            ),
            cfg_rescale=ss.get(
                "cfg_rescale",
                ig_meta.get("cfg_rescale", 0.7),
            ),
            uncond_img_input_ids=(
                list(uncond_img_ids) if uncond_img_ids is not None else None
            ),
            uncond_img_left_pad_len=(
                int(ss.get("uncond_img_left_pad_len", 0))
                if uncond_img_ids is not None
                else None
            ),
            cfg_image_scale=ss.get(
                "cfg_image_scale",
                ig_meta.get("cfg_image_scale", 0.0),
            ),
        )

    if interleaved_phase == "image":
        raw_plan = ss.get("interleaved_cfg_plan")
        if not isinstance(raw_plan, dict):
            raise TypeError("interleaved image phase is missing its CFG plan")
        branches = raw_plan.get("branches", {})
        mode = raw_plan.get("mode")
        if mode == "simple":
            _attach_cfg_branches(
                req,
                tokenizer=tokenizer,
                conditional_input_ids=input_ids_list,
                uncond_input_ids=list(branches["uncond"]),
                cfg_scale=float(raw_plan["cfg_scale"]),
                cfg_rescale=float(raw_plan["cfg_rescale"]),
            )
        elif mode == "editing":
            _attach_cfg_branches(
                req,
                tokenizer=tokenizer,
                conditional_input_ids=input_ids_list,
                uncond_input_ids=list(branches["no_text"]),
                cfg_scale=float(raw_plan["cfg_text_scale"]),
                cfg_rescale=float(raw_plan["cfg_rescale"]),
                uncond_img_input_ids=list(branches["no_image"]),
                cfg_image_scale=float(raw_plan["cfg_image_scale"]),
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"unsupported interleaved CFG mode: {mode!r}")

    # Attach task metadata for algorithm-level decisions.
    req._task_kind = state.task_kind
    if is_thinking_phase1:
        req._is_thinking_phase1 = True
    if interleaved_phase is not None:
        req._interleaved_phase = interleaved_phase
    if state.task_kind == "interleaved":
        if interleaved_phase == "image":
            dllm_steps = ss.get(
                "interleaved_image_dllm_steps",
                ss.get("interleaved_dllm_steps"),
            )
        else:
            # Match i2t: text uses the scheduler's default block schedule.
            dllm_steps = None
    else:
        dllm_steps = ss.get("dllm_steps")
    if dllm_steps is not None:
        req._dllm_steps = int(dllm_steps)

    data = SGLangDLLMRequestData(
        output_ids=req.output_ids,
        req=req,
    )
    return data


def apply_dllm_thinker_result(
    state: LLaDA2UniPipelineState,
    *,
    stage_name: str,
    output_ids: list[int],
    finish_reason: str | None = None,
) -> ThinkerOutput:
    """Apply DLLM thinker result to pipeline state."""
    thinker_out: ThinkerOutput = {
        "output_ids": output_ids,
        "is_final": True,
    }
    if finish_reason is not None:
        thinker_out["finish_reason"] = finish_reason

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def _thinking_phase1_to_phase2(
    state: LLaDA2UniPipelineState,
    tokenizer: Any,
) -> None:
    """Transition from thinking Phase 1 (text) to Phase 2 (image VQ tokens)."""
    ss = state.stream_state
    output_ids = state.thinker_out["output_ids"]

    boi_id = getattr(tokenizer, "boi_token_id", None)
    if boi_id is None:
        boi_id = tokenizer.convert_tokens_to_ids("<boi>")

    boi_pos = None
    for i, tid in enumerate(output_ids):
        if tid == boi_id:
            boi_pos = i
            break

    if boi_pos is None:
        raise RuntimeError(
            "Thinking Phase 1 did not produce <boi> "
            f"after generating {len(output_ids)} token(s)"
        )

    thinking_text = tokenizer.decode(output_ids[:boi_pos], skip_special_tokens=True)
    ss["thinking_text"] = thinking_text

    # Rebuild prompt for Phase 2: original prompt + Phase 1 output through <boi>.
    original_ids = state.prompt["input_ids"].flatten().tolist()
    phase2_ids = original_ids + output_ids[: boi_pos + 1]

    image_info = ss.get("image_info", [{}])
    grid_h = image_info[0].get("grid_h", 32) if image_info else 32
    grid_w = image_info[0].get("grid_w", 32) if image_info else 32

    state.prompt = {"input_ids": torch.tensor([phase2_ids], dtype=torch.long)}

    # Build CFG unconditional for Phase 2 if cfg_scale > 1
    cfg_scale = ss.get("cfg_scale", 1.0)
    if cfg_scale > 1.0:
        from sglang_omni.models.llada2_uni.components.preprocessor import (
            SYSTEM_PROMPT_T2I_THINKING,
            UNCOND_TEXT,
        )

        sys_encoded = tokenizer.encode(
            f"<role>SYSTEM</role>{SYSTEM_PROMPT_T2I_THINKING}<role>HUMAN</role>"
            f"{UNCOND_TEXT}<role>ASSISTANT</role>",
            add_special_tokens=False,
        )
        # Append full image header: <soi><reserved_h><reserved_w><boi>
        soi_id = tokenizer.convert_tokens_to_ids("<|image|>")
        boi_tok = tokenizer.convert_tokens_to_ids("<boi>")
        h_token_ids = tokenizer.encode(
            f"<|reserved_token_{grid_h}|>", add_special_tokens=False
        )
        w_token_ids = tokenizer.encode(
            f"<|reserved_token_{grid_w}|>", add_special_tokens=False
        )
        img_header = [soi_id] + h_token_ids + w_token_ids + [boi_tok]
        ss["uncond_input_ids"] = sys_encoded + img_header

    ss["thinking_phase"] = 2
    ss["thinking_needs_reentry"] = True
    state.thinker_out = None
    state.engine_outputs.pop(THINKER_STAGE, None)


def _serialize_cfg_plan(plan: CFGBranchPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "branches": {name: list(ids) for name, ids in plan.branches.items()},
        "cfg_scale": plan.cfg_scale,
        "cfg_text_scale": plan.cfg_text_scale,
        "cfg_image_scale": plan.cfg_image_scale,
        "cfg_rescale": plan.cfg_rescale,
    }


def _mark_interleaved_done(
    state: LLaDA2UniPipelineState,
    *,
    finish_reason: str,
) -> None:
    stream_state = state.stream_state
    stream_state["interleaved_phase"] = "done"
    stream_state["interleaved_done"] = True
    stream_state["interleaved_finish_reason"] = finish_reason
    stream_state.pop("interleaved_needs_reentry", None)
    prompt = state.prompt or {}
    input_ids = prompt.get("input_ids")
    final_length = int(input_ids.numel()) if isinstance(input_ids, torch.Tensor) else 0
    prompt_length = int(stream_state.get("interleaved_prompt_length", 0))
    completion_tokens = max(final_length - prompt_length, 0)
    stream_state["interleaved_usage"] = {
        "prompt_tokens": prompt_length,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_length + completion_tokens,
    }


def _reject_image_tokens_in_text(
    token_ids: list[int],
    tokenizer: Any,
    *,
    frame_index: int,
) -> None:
    offenders = [
        (index, token_id)
        for index, token_id in enumerate(token_ids)
        if token_id >= IMAGE_TOKEN_OFFSET
    ]
    if not offenders:
        return

    details = ", ".join(
        f"offset={index} id={token_id} token={tokenizer.convert_ids_to_tokens(token_id)!r}"
        for index, token_id in offenders[:8]
    )
    logger.error(
        "Interleaved text phase emitted image token(s): frame=%d %s",
        frame_index,
        details,
    )
    raise ValueError(
        f"interleaved text phase emitted image token(s) for frame {frame_index}: "
        f"{details}"
    )


def _interleaved_text_to_image_or_done(
    state: LLaDA2UniPipelineState,
    tokenizer: Any,
    *,
    finish_reason: str | None = None,
) -> None:
    """Consume a text phase ending in BOI or EOS and prepare the next phase."""

    thinker_out = state.thinker_out or {}
    output_ids = [int(token_id) for token_id in thinker_out.get("output_ids", [])]
    if not output_ids:
        raise ValueError("interleaved text phase produced no tokens")
    prompt = state.prompt or {}
    prompt_tensor = prompt.get("input_ids")
    if not isinstance(prompt_tensor, torch.Tensor):
        raise TypeError("interleaved thinker prompt must be a tensor")
    prompt_ids = prompt_tensor.flatten().tolist()
    boi_id = int(tokenizer.convert_tokens_to_ids("<boi>"))
    stream_state = state.stream_state
    full_ids = prompt_ids + output_ids
    if output_ids[-1] != boi_id:
        state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
        segment_start = int(
            stream_state.get("interleaved_segment_start", len(prompt_ids))
        )
        trailing_ids = full_ids[segment_start:]
        _reject_image_tokens_in_text(
            trailing_ids,
            tokenizer,
            frame_index=int(stream_state.get("interleaved_frame_index", 0)) + 1,
        )
        stream_state["interleaved_trailing_text"] = tokenizer.decode(
            trailing_ids, skip_special_tokens=True
        )
        _mark_interleaved_done(state, finish_reason=finish_reason or "stop")
        return

    header = parse_image_header(output_ids, tokenizer)
    config = InterleavedGenerationConfig.from_metadata(state.request_metadata)
    if (
        header.image_token_count <= 0
        or header.image_token_count > config.max_image_tokens
    ):
        raise ValueError(
            f"interleaved image grid {header.grid_h}x{header.grid_w} requires "
            f"{header.image_token_count} tokens; limit is {config.max_image_tokens}"
        )

    max_seq_len = int(state.stream_state.get("interleaved_max_seq_len", 8192))
    required_length = len(full_ids) + header.image_token_count + 1
    if required_length > max_seq_len:
        raise ValueError(
            "interleaved frame would exceed thinker context: "
            f"required={required_length}, max={max_seq_len}"
        )
    frame_index = int(state.stream_state.get("interleaved_frame_index", 0))
    plan = build_cfg_plan(
        full_ids=full_ids,
        header=header,
        frame_index=frame_index,
        tokenizer=tokenizer,
        config=config,
    )
    segment_start = int(stream_state.get("interleaved_segment_start", len(prompt_ids)))
    header_start = len(full_ids) - len(header.token_ids)
    current_text_ids = full_ids[segment_start:header_start]
    _reject_image_tokens_in_text(
        current_text_ids,
        tokenizer,
        frame_index=frame_index + 1,
    )
    current_text = tokenizer.decode(current_text_ids, skip_special_tokens=True)
    state.stream_state["interleaved_current_frame"] = {
        "index": frame_index + 1,
        "text": current_text,
        "text_ids": current_text_ids,
        "grid_h": header.grid_h,
        "grid_w": header.grid_w,
        "image_token_count": header.image_token_count,
        "remaining_image_tokens": header.image_token_count,
        "vq_tokens": [],
        "cfg_mode": plan.mode,
    }
    state.stream_state["interleaved_cfg_plan"] = _serialize_cfg_plan(plan)
    state.stream_state["interleaved_phase"] = "image"
    state.stream_state["image_info"] = [
        {"grid_h": header.grid_h, "grid_w": header.grid_w}
    ]
    state.stream_state["interleaved_needs_reentry"] = True
    state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
    state.thinker_out = None
    state.engine_outputs.pop(THINKER_STAGE, None)


def _interleaved_image_to_text_or_done(
    state: LLaDA2UniPipelineState,
    tokenizer: Any,
) -> None:
    """Validate the VQ result and emit the completed frame."""

    stream_state = state.stream_state
    thinker_out = state.thinker_out or {}
    output_ids = [int(token_id) for token_id in thinker_out.get("output_ids", [])]
    current_frame = stream_state.get("interleaved_current_frame", {})
    remaining = int(current_frame.get("remaining_image_tokens", 0))
    eoi_id = int(tokenizer.convert_tokens_to_ids("<|/image|>"))

    if not output_ids:
        raise ValueError("interleaved image phase produced no tokens")

    eoi_positions = [
        index for index, token_id in enumerate(output_ids) if token_id == eoi_id
    ]
    if len(eoi_positions) > 1:
        raise ValueError("interleaved image phase produced multiple EOI tokens")
    has_eoi = bool(eoi_positions)
    if has_eoi and eoi_positions[0] != len(output_ids) - 1:
        raise ValueError("interleaved image phase produced tokens after EOI")

    image_output_ids = output_ids[: eoi_positions[0]] if has_eoi else output_ids
    if len(image_output_ids) > remaining:
        raise ValueError(
            f"interleaved frame {current_frame.get('index')} produced "
            f"{len(image_output_ids)} VQ tokens with {remaining} remaining"
        )
    if any(token_id < IMAGE_TOKEN_OFFSET for token_id in image_output_ids):
        raise ValueError(
            "interleaved image phase produced a non-image token before EOI"
        )

    prompt = state.prompt or {}
    prompt_tensor = prompt.get("input_ids")
    if not isinstance(prompt_tensor, torch.Tensor):
        raise TypeError("interleaved thinker prompt must be a tensor")
    prompt_ids = prompt_tensor.flatten().tolist()
    accumulated = list(current_frame.get("vq_tokens", []))
    accumulated.extend(image_output_ids)
    remaining -= len(image_output_ids)
    current_frame["vq_tokens"] = accumulated
    current_frame["remaining_image_tokens"] = remaining

    raw_plan = stream_state.get("interleaved_cfg_plan", {})
    branches = raw_plan.get("branches", {}) if isinstance(raw_plan, dict) else {}
    for branch_ids in branches.values():
        branch_ids.extend(image_output_ids)

    state.prompt = {
        "input_ids": torch.tensor([prompt_ids + image_output_ids], dtype=torch.long)
    }
    if not has_eoi:
        if remaining <= 0:
            raise ValueError("interleaved image phase completed VQ tokens without EOI")
        state.thinker_out = None
        state.engine_outputs.pop(THINKER_STAGE, None)
        stream_state["interleaved_needs_reentry"] = True
        return

    if remaining != 0:
        raise ValueError(
            f"interleaved image phase emitted EOI with {remaining} VQ tokens remaining"
        )

    expected = int(current_frame.get("image_token_count", 0))
    if len(accumulated) != expected:
        raise ValueError(
            f"interleaved frame {current_frame.get('index')} produced "
            f"{len(accumulated)} VQ tokens; expected {expected}"
        )

    full_ids = prompt_ids + image_output_ids + [eoi_id]
    state.prompt = {"input_ids": torch.tensor([full_ids], dtype=torch.long)}
    final_thinker_out = dict(thinker_out)
    final_thinker_out["output_ids"] = accumulated
    state.thinker_out = final_thinker_out
    state.engine_outputs[THINKER_STAGE] = final_thinker_out

    frame_index = int(current_frame["index"])
    stream_state["interleaved_frame_index"] = frame_index
    segments = stream_state.setdefault("interleaved_segments", [])
    segments.append(
        {
            "frame_index": frame_index,
            "text": current_frame.get("text", ""),
            "grid_h": current_frame.get("grid_h"),
            "grid_w": current_frame.get("grid_w"),
            "cfg_mode": current_frame.get("cfg_mode"),
        }
    )
    stream_state["interleaved_phase"] = "text"
    stream_state["interleaved_segment_start"] = len(full_ids)
    stream_state["interleaved_emit_frame"] = True
    stream_state.pop("interleaved_cfg_plan", None)
    stream_state.pop("interleaved_current_frame", None)

    max_frames = int(stream_state.get("interleaved_max_frames", 10))
    if frame_index >= max_frames:
        _mark_interleaved_done(state, finish_reason="max_frames")
    else:
        stream_state["interleaved_needs_reentry"] = True


def _advance_interleaved_state(
    state: LLaDA2UniPipelineState,
    tokenizer: Any,
    *,
    completed_phase: str,
    finish_reason: str | None = None,
) -> None:
    if completed_phase == "text":
        _interleaved_text_to_image_or_done(
            state, tokenizer, finish_reason=finish_reason
        )
    elif completed_phase == "image":
        _interleaved_image_to_text_or_done(state, tokenizer)
    else:
        raise ValueError(f"unsupported interleaved phase: {completed_phase!r}")


def make_dllm_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    dllm_config: Any,
    stage_name: str = THINKER_STAGE,
):
    """Build StagePayload <-> scheduler adapters for the dLLM thinker."""

    def request_builder(payload: StagePayload) -> SGLangDLLMRequestData:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        # Route flags describe the previous result and must not leak into the
        # next self-loop pass.
        if isinstance(state.stream_state, dict):
            state.stream_state.pop("thinking_needs_reentry", None)
            state.stream_state.pop("interleaved_needs_reentry", None)
            state.stream_state.pop("interleaved_emit_frame", None)
            # Update payload data with cleared flag
            payload = StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=state.to_dict(),
                timing=payload.timing,
            )
        data = build_dllm_thinker_request(
            state,
            params=payload.request.params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            dllm_config=dllm_config,
            request_id=payload.request_id,
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: SGLangDLLMRequestData) -> StagePayload:
        payload = data.stage_payload
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        completed_interleaved_phase = (
            state.stream_state.get("interleaved_phase")
            if state.task_kind == "interleaved"
            else None
        )
        apply_dllm_thinker_result(
            state,
            stage_name=stage_name,
            output_ids=data.output_ids,
            finish_reason=data.finish_reason,
        )

        # Thinking mode Phase 1 → Phase 2 transition
        ss = state.stream_state if isinstance(state.stream_state, dict) else {}
        if ss.get("thinking_mode") and ss.get("thinking_phase") == 1:
            _thinking_phase1_to_phase2(state, tokenizer)
        elif completed_interleaved_phase is not None:
            _advance_interleaved_state(
                state,
                tokenizer,
                completed_phase=str(completed_interleaved_phase),
                finish_reason=data.finish_reason,
            )

        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
            timing=payload.timing,
        )

    return request_builder, result_adapter
