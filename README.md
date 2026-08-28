<div align="center">

<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="SGLang-Omni logo" width="400"></img>

### Serving runtime for speech, audio, and multimodal models

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
</p>

<p>
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="#quick-start"><b>Quick start</b></a> |
<a href="./docs/supported_models.md"><b>Supported models</b></a> |
<a href="https://slack.sglang.io"><b>Slack</b></a>
</p>

</div>

--------------------------------------------------------------------------------

## About

SGLang-Omni is a serving runtime for speech, audio, and multimodal models built
on [SGLang](https://github.com/sgl-project/sglang). It coordinates multi-stage
inference pipelines and exposes OpenAI-compatible APIs.

Stages can use separate schedulers and can be placed across processes and
accelerators. See the [pipeline lifecycle](./docs/developer_reference/pipeline.md),
[communication design](./docs/developer_reference/communication.md), and
[stage placement](./docs/user_guide/deployment/stage_placement.md) for details.

## Supported workloads

- Multimodal and omni chat with text and optional audio output.
- Text-to-speech and music generation.
- Speech transcription, translation, and diarization.

Availability, streaming behavior, and accelerator coverage vary by model. See
the [supported-model and accelerator matrices](./docs/supported_models.md).

## Quick start

This path uses the NVIDIA CUDA development image. The `dev` tag moves with
`main`; use the [installation guide](./docs/get_started/installation.md) for
digest pinning and other installation methods.

```bash
docker pull hongccc/sglang-omni:dev

docker run -it \
  --shm-size 32g \
  --gpus all \
  --ipc host \
  --network host \
  --privileged \
  hongccc/sglang-omni:dev \
  /bin/zsh
```

Inside the container, install SGLang-Omni and start a Higgs Audio v3 server:

```bash
pip install uv
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.3"

sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --port 8000
```

Send a speech request from another shell:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "Hello from SGLang-Omni."
  }' \
  --output output.wav
```

For Intel GPUs, use the [XPU installation guide](./docs/get_started/installation_xpu.md).

## Documentation

[Installation](./docs/get_started/installation.md) ·
[Supported models](./docs/supported_models.md) ·
[Speech API](./docs/user_guide/serving/speech_api.md) ·
[Transcription API](./docs/user_guide/serving/transcription_api.md) ·
[Cookbook](./docs/cookbook/) ·
[Deployment](./docs/user_guide/deployment/stage_placement.md) ·
[Benchmarks](./docs/benchmarks/methodology.md) ·
[Developer guide](./docs/developer_reference/main.md)

## Community

Join the [SGLang Slack](https://slack.sglang.io), read the
[project blog](https://lmsys.org/blog/), or open a
[GitHub issue](https://github.com/sgl-project/sglang-omni/issues).

Organizations interested in SGLang-Omni can contact Chenyang Zhao at
[zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org).
