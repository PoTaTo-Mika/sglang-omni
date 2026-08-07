from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel


def test_weight_loader_routes_every_checkpoint_namespace() -> None:
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: list[tuple[str, torch.Tensor]] = []

        def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
            self.received = list(weights)

    class _Flow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for root in (
                "patch_encoder",
                "hidden_proj",
                "latent_proj",
                "coordinate_proj",
                "xvec_proj",
                "velocity_field_predictor",
                "eos_proj",
            ):
                setattr(self, root, nn.Linear(1, 1, bias=False))

    model = DotsTTSSGLangModel.__new__(DotsTTSSGLangModel)
    nn.Module.__init__(model)
    model.qwen2 = _Backbone()
    model.flow = _Flow()
    flow_weights = [
        (name, torch.full_like(parameter, float(index + 1)))
        for index, (name, parameter) in enumerate(model.flow.named_parameters())
    ]
    codec_weights = [
        (f"{root}.unused", torch.ones(1))
        for root in (
            "audio_encoder",
            "dec_mi_layer",
            "decoder",
            "enc_mi_layer",
            "model",
            "post_proj",
            "pre_proj",
            "resample",
        )
    ]

    loaded = model.load_weights(
        [
            ("llm.model.embed_tokens.weight", torch.tensor([11.0])),
            *flow_weights,
            *codec_weights,
        ]
    )

    assert [name for name, _tensor in model.qwen2.received] == [
        "model.embed_tokens.weight"
    ]
    assert loaded == {
        *(f"flow.{name}" for name, _tensor in flow_weights),
    }
    for index, (_name, parameter) in enumerate(model.flow.named_parameters()):
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, float(index + 1)),
        )

    name, parameter = next(iter(model.flow.named_parameters()))
    partial_weight = torch.full_like(parameter, 99.0)
    assert model.load_weights([(name, partial_weight)]) == {f"flow.{name}"}
    torch.testing.assert_close(parameter, partial_weight)

    with pytest.raises(AssertionError, match="Unexpected dots.tts checkpoint weight"):
        model.load_weights([("new_acoustic_block.weight", torch.ones(1, 1))])


def test_graph_feedback_buffer_routes_decode_input_embeds() -> None:
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.weight = nn.Parameter(torch.zeros(1))
            self.seen_embeds: list[torch.Tensor] = []

        def forward(self, *, input_ids, positions, forward_batch, input_embeds, **_):
            self.seen_embeds.append(input_embeds)
            return input_embeds

    model = DotsTTSSGLangModel.__new__(DotsTTSSGLangModel)
    nn.Module.__init__(model)
    model.qwen2 = _Backbone()

    assert model.graph_feedback_buffer is None
    model.enable_graph_feedback(3)
    buffer = model.graph_feedback_buffer
    assert buffer is not None and buffer.shape == (3, 4)

    buffer[:2].copy_(torch.full((2, 4), 8.0))
    decode_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        input_embeds=None,
    )
    out = model.forward(torch.tensor([1, 2]), torch.tensor([0, 0]), decode_batch)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, torch.full((2, 4), 8.0))
    assert model.qwen2.seen_embeds[0].data_ptr() == buffer.data_ptr()

    prefill_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        input_embeds=torch.full((1, 4), 2.0),
    )
    out = model.forward(torch.tensor([3]), torch.tensor([0]), prefill_batch)
    torch.testing.assert_close(out, torch.full((1, 4), 2.0))
