from __future__ import annotations

import asyncio
from types import SimpleNamespace

from benchmarks.benchmarker.data import RequestResult
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_results_config,
    _config_from_args,
)
from benchmarks.metrics.performance import _request_result_to_dict
from benchmarks.tasks.tts import make_tts_send_fn


def _config_from_cli(*args: str) -> TtsSeedttsBenchmarkConfig:
    parser = _build_arg_parser()
    return _config_from_args(parser.parse_args(list(args)))


def test_seedtts_benchmark_batch_args_default_to_64() -> None:
    config = _config_from_cli()

    assert config.max_running_requests == 64
    assert config.cuda_graph_max_bs == 64

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 64
    assert results_config["cuda_graph_max_bs"] == 64


def test_seedtts_benchmark_batch_args_are_independent() -> None:
    config = _config_from_cli(
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "128",
    )

    assert config.max_running_requests == 32
    assert config.cuda_graph_max_bs == 128

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 32
    assert results_config["cuda_graph_max_bs"] == 128


def test_speed_result_preserves_server_request_id() -> None:
    result = RequestResult(
        request_id="sample-1",
        server_request_id="speech-runtime-1",
        is_success=True,
    )

    assert _request_result_to_dict(result)["server_request_id"] == "speech-runtime-1"


class _ResponseWithRequestId:
    status = 500
    headers = {"X-Request-Id": "speech-runtime-2"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def text(self) -> str:
        return "expected test error"


class _SessionWithRequestId:
    def post(self, url, *, json):
        return _ResponseWithRequestId()


def test_tts_sender_captures_server_request_id() -> None:
    send = make_tts_send_fn(
        "model",
        "http://localhost/v1/audio/speech",
        no_ref_audio=True,
    )
    sample = SimpleNamespace(sample_id="sample-2", target_text="hello")

    result = asyncio.run(send(_SessionWithRequestId(), sample))

    assert result.request_id == "sample-2"
    assert result.server_request_id == "speech-runtime-2"


def test_request_result_positional_constructor_remains_compatible() -> None:
    result = RequestResult("sample-positional", "original text", True)

    assert result.request_id == "sample-positional"
    assert result.text == "original text"
    assert result.is_success is True
    assert result.server_request_id == ""
