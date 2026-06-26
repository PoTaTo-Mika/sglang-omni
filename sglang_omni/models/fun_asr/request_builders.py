# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Fun-ASR-Nano.

Fun-ASR-Nano is a Qwen3-0.6B causal LM that ingests audio as multimodal
embeddings, structurally analogous to Qwen3-ASR but with three differences
that this adapter accounts for:

* **Audio placeholder** is ``<|object_ref_start|>`` (id 151646, the
  ``audio_token_index`` in ``config.json``), expanded to N copies — one per
  adaptor audio token. The HF chat template
  (``checkpoints/.../chat_template.jinja``) emits the placeholder bare, with
  **no** ``<|audio_start|>``/``<|audio_end|>`` wrappers (the original funasr
  ``<|startofspeech|>``/``<|endofspeech|>`` markers are retired in the HF
  build).
* **Feature extractor** is the SenseVoice 80-mel WavFrontend + LFR
  (``FunAsrNanoFeatureExtractor``), not Whisper. LFR is applied inside the
  extractor, so ``feature_attention_mask.sum()`` is already the post-LFR frame
  count and only the adaptor's three stride-2 reductions remain — hence
  :func:`fun_asr_low_frame_rate_length` (not the full mel→token chain).
* **Prompt** is the Fun-ASR ChatML instruction
  ``语音转写成{language}：`` (``model.py:get_prompt``) placed *before* the
  audio placeholders, with **no forced assistant prefix** — the model
  generates the transcript directly after ``<|im_start|>assistant\n``
  (unlike Qwen3-ASR's ``language <Lang><asr_text>`` forced prefix).
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .tool_funcs.audio_lengths import fun_asr_low_frame_rate_length

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

# Audio placeholder token. config.json "audio_token_index": 151646 maps to
# <|object_ref_start|> in the Qwen3 tokenizer (see tokenizer_config.json
# added_tokens_decoder). The HF chat_template.jinja emits this token bare for
# each audio content item; the processor expands one placeholder into N copies
# (N = adaptor audio-token count). Here we build the N copies directly.
_AUDIO_PAD = "<|object_ref_start|>"


@dataclass
class FunASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str | None = None
    engine_start_s: float = 0.0


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def load_audio(source: Any) -> np.ndarray:
    """Load audio as a 1-D float32 numpy waveform at 16 kHz mono."""
    import torchaudio

    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)

    if isinstance(source, bytes):
        audio, sample_rate = torchaudio.load(io.BytesIO(source))
    elif isinstance(source, str):
        audio, sample_rate = torchaudio.load(source)
    else:
        raise ValueError(f"Unsupported Fun-ASR audio input: {type(source).__name__}")

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != _SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, _SAMPLE_RATE)
    return audio.cpu().numpy()


def _audio_fingerprint(audio: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()


def _audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)


def _decode_token_ids(
    tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _resolve_language(lang_raw: str | None) -> str | None:
    """Map a request ``language`` param to a Fun-ASR target-language name.

    Fun-ASR's prompt is ``语音转写：`` (transcribe as-is, multilingual) by
    default and ``语音转写成{language}：`` to force a target language
    (e.g. ``语音转写成英文``). ``None`` ⇒ as-is. Common short codes are mapped
    to the Chinese language names the model was trained on; anything else is
    passed through verbatim so callers can request e.g. ``日文``.
    """
    if lang_raw is None:
        return None
    lang = lang_raw.strip().lower()
    if lang in ("", "auto", "null", "none"):
        return None
    # Chinese / as-is: the bare ``语音转写：`` form is what the training data
    # uses for Chinese audio, so we don't append a target language.
    if lang in ("zh", "cn", "chinese", "中文"):
        return None
    if lang in ("en", "english", "英文"):
        return "英文"
    # Passthrough: allow callers to supply the exact target name.
    return lang_raw.strip()


def _build_prompt_text(language: str | None, itn: bool, hotwords: list[str]) -> str:
    """Reproduce ``FunAsrNano.get_prompt`` (Fun-ASR/model.py:552-563)."""
    prompt = ""
    if hotwords:
        joined = ", ".join(hotwords)
        prompt += (
            "请结合上下文信息，更加准确地完成语音转写任务。"
            "如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
        )
        prompt += f"热词列表：[{joined}]\n"
    if language is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{language}"
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："


def make_fun_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    feature_extractor: Any = None,
) -> tuple[
    Callable[[StagePayload], FunASRRequestData], Callable[[FunASRRequestData], StagePayload]
]:
    """Build (request_builder, result_adapter) for Fun-ASR-Nano.

    ``feature_extractor`` is the ``FunAsrNanoFeatureExtractor`` (80-mel + LFR)
    loaded by ``stages.create_sglang_fun_asr_executor``. It must be provided.
    """
    if feature_extractor is None:
        raise ValueError("Fun-ASR processor is missing a feature_extractor")

    audio_pad_token_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_PAD))
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)

    def _build_prompt_ids(num_audio_tokens: int, prompt_text: str) -> list[int]:
        # ChatML per chat_template.jinja: system + user(text then N×audio
        # placeholder) + assistant header. No <|audio_start|>/<|audio_end|>
        # wrappers — the HF template emits <|object_ref_start|> bare.
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt_text}{_AUDIO_PAD * num_audio_tokens}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return tokenizer(prompt, add_special_tokens=False).input_ids

    def request_builder(payload: StagePayload) -> FunASRRequestData:
        params = payload.request.params or {}
        audio = load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        fingerprint = _audio_fingerprint(audio)

        # FunAsrNanoFeatureExtractor applies LFR internally, so the returned
        # attention_mask sums to the post-LFR frame count (T_lfr). Only the
        # adaptor's three stride-2 reductions remain → fun_asr_low_frame_rate_length.
        # No 30s windowing: Fun-ASR's encoder is variable-length (unlike
        # Whisper), so padding="longest" pads to this clip's true length only.
        extracted = feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        features = extracted["input_features"]  # [1, 560, T_lfr]
        feature_attention_mask = extracted.get("attention_mask")
        if feature_attention_mask is None:
            feature_attention_mask = torch.ones(
                (features.shape[0], features.shape[-1]), dtype=torch.long
            )
        num_lfr_frames = int(feature_attention_mask.sum().item())
        num_audio_tokens = int(fun_asr_low_frame_rate_length(num_lfr_frames))
        logger.debug(
            f"[fun-asr] lfr_frames={num_lfr_frames} "
            f"num_audio_tokens={num_audio_tokens} feat_shape={tuple(features.shape)}"
        )

        lang_raw = params.get("language")
        language = _resolve_language(lang_raw)
        itn = bool(params.get("itn", True))
        hotwords = list(params.get("hotwords") or [])
        prompt_text = _build_prompt_text(language, itn, hotwords)
        input_ids = _build_prompt_ids(num_audio_tokens, prompt_text)

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=_audio_fingerprint_int(fingerprint),
            feature=features,
            model_specific_data={
                "feature_attention_mask": feature_attention_mask,
            },
        )
        # general_mm_embed_routine locates audio positions by matching each
        # item's pad_value against input_ids. The omni scheduler does not run
        # pad_input_ids for us, so compute the pad_value, replace the N
        # <|object_ref_start|> placeholders with it, and record the placeholder
        # span as item.offsets. SGLang treats offsets as inclusive.
        audio_item.set_pad_value()
        if audio_pad_token_id not in input_ids:
            raise RuntimeError(
                f"Fun-ASR prompt missing audio placeholder {_AUDIO_PAD!r} "
                f"(id {audio_pad_token_id}); prompt_text={prompt_text!r}"
            )
        audio_start = input_ids.index(audio_pad_token_id)
        input_ids = [
            audio_item.pad_value if tok == audio_pad_token_id else tok
            for tok in input_ids
        ]
        audio_item.offsets = [(audio_start, audio_start + num_audio_tokens - 1)]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=num_audio_tokens,
        )
        mm_inputs.audio_token_id = audio_pad_token_id
        # Fun-ASR's Qwen3 LLM uses plain 1-D positions (rope_type "default",
        # not mrope). sglang's prefill indexes mm_input.mrope_positions during
        # prefill and does not compute a default, so supply degenerate [3, seq]
        # positions broadcasting the text position — same as Qwen3-ASR's ASR
        # degenerate MRoPE.
        seq_len = len(input_ids)
        positions = torch.arange(seq_len, dtype=torch.long)
        mm_inputs.mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        mm_inputs.mrope_position_delta = torch.tensor([0], dtype=torch.long)

        # Fun-ASR's reference inference (Fun-ASR/model.py) calls llm.generate
        # with no sampling args ⇒ greedy. Default to greedy; allow override.
        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        logger.debug(
            f"[fun-asr] sampling temp={temperature} "
            f"max_new_tokens={request_max_new_tokens} params={dict(params)}"
        )
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return FunASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=lang_raw,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: FunASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        # Fun-ASR generates the transcript directly after <|im_start|>assistant\n
        # — no forced prefix marker to strip (unlike Qwen3-ASR's <asr_text>).
        # skip_special_tokens=True drops the trailing <|im_end|>.
        text = _decode_token_ids(tokenizer, output_ids, skip_special_tokens=True)
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        logger.debug(
            f"[fun-asr] n_out={len(output_ids)} ids={output_ids[:40]} text={text!r}"
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "FunASRRequestData",
    "load_audio",
    "make_fun_asr_scheduler_adapters",
]
