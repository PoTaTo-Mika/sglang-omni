from __future__ import annotations

import asyncio
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

from benchmarks.dataset import dailyomni, prepare
from benchmarks.dataset.videomme import VideoAMMESample
from benchmarks.tasks.video_understanding import make_video_send_fn


def _write_dailyomni_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / "dailyomni"
    dataset_dir.mkdir()
    (dataset_dir / "qa.json").write_text(json.dumps(rows), encoding="utf-8")
    (dataset_dir / "Videos").mkdir()
    for video_id in {
        row["video_id"] for row in rows if not row["video_id"].startswith(".")
    }:
        video_dir = dataset_dir / "Videos" / video_id
        video_dir.mkdir(parents=True)
        (video_dir / f"{video_id}_video.mp4").write_bytes(b"video")
        (video_dir / f"{video_id}_audio.wav").write_bytes(b"audio")
    return dataset_dir


def _row(**overrides) -> dict:
    row = {
        "Question": "What happened at the same time?",
        "Choice": ["A. First", "B. Second", "C. Third", "D. Fourth"],
        "Answer": "B",
        "video_id": "video-id-01",
        "Type": "AV Event Alignment",
        "content_parent_category": "Lifestyle",
        "content_fine_category": "Daily Routines",
        "video_category": "People & Blogs",
        "video_duration": "30s",
    }
    row.update(overrides)
    return row


def test_load_dailyomni_samples_maps_official_schema(tmp_path: Path) -> None:
    dataset_dir = _write_dailyomni_fixture(
        tmp_path,
        [_row(), _row(Question="Which event came first?", Answer="A")],
    )

    samples = dailyomni.load_dailyomni_samples(repo_id=str(dataset_dir))

    assert [sample.sample_id for sample in samples] == [
        "dailyomni:0",
        "dailyomni:1",
    ]
    assert samples[0].video_id == "video-id-01"
    assert samples[0].options == ["First", "Second", "Third", "Fourth"]
    assert samples[0].answer == "B"
    assert samples[0].duration == "30s"
    assert samples[0].domain == "People & Blogs"
    assert samples[0].task_type == "AV Event Alignment"
    assert samples[0].sub_category == "Daily Routines"
    assert samples[0].audio_path.endswith("video-id-01_audio.wav")
    assert "given video and audio together" in samples[0].prompt
    assert "B. Second" in samples[0].prompt


def test_load_dailyomni_samples_uses_input_mode_and_max_samples(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_dailyomni_fixture(
        tmp_path,
        [_row(), _row(Question="Another question")],
    )

    samples = dailyomni.load_dailyomni_samples(
        repo_id=str(dataset_dir),
        input_mode="audio",
        max_samples=1,
    )

    assert len(samples) == 1
    assert "based on the given audio" in samples[0].prompt


def test_visual_mode_does_not_require_audio_file(tmp_path: Path) -> None:
    dataset_dir = _write_dailyomni_fixture(tmp_path, [_row()])
    (dataset_dir / "Videos" / "video-id-01" / "video-id-01_audio.wav").unlink()

    samples = dailyomni.load_dailyomni_samples(
        repo_id=str(dataset_dir),
        input_mode="visual",
    )

    assert len(samples) == 1
    assert "based on the given video" in samples[0].prompt


def test_load_dailyomni_samples_rejects_escaping_video_id(tmp_path: Path) -> None:
    dataset_dir = _write_dailyomni_fixture(tmp_path, [_row(video_id="../escape")])

    with pytest.raises(ValueError, match="escapes media root"):
        dailyomni.load_dailyomni_samples(repo_id=str(dataset_dir))


def test_dailyomni_archive_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "Videos.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe path"):
        dailyomni._safe_extract(archive_path, tmp_path / "extracted")

    assert not (tmp_path / "escape.txt").exists()


def test_prepare_dailyomni_downloads_metadata_and_media(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_hf_hub_download(repo_id: str, filename: str, repo_type: str) -> str:
        calls.append((repo_id, filename, repo_type))
        return filename

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(get_dataset_config_names=None, load_dataset=None),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    prepare.download_dataset("liarliar/Daily-Omni", quiet=True)

    assert calls == [
        ("liarliar/Daily-Omni", "qa.json", "dataset"),
        ("liarliar/Daily-Omni", "Videos.tar", "dataset"),
    ]


class _FakeResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "B"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }


class _FakeSession:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def post(self, _url: str, *, json: dict) -> _FakeResponse:
        self.payload = json
        return _FakeResponse()


@pytest.mark.parametrize(
    ("enable_video", "enable_audio", "expected_media_keys"),
    [
        (True, True, {"videos", "audios"}),
        (True, False, {"videos"}),
        (False, True, {"audios"}),
    ],
)
def test_video_send_fn_supports_dailyomni_input_modes(
    enable_video: bool,
    enable_audio: bool,
    expected_media_keys: set[str],
) -> None:
    sample = VideoAMMESample(
        sample_id="dailyomni:0",
        video_path="/tmp/video.mp4",
        audio_path="/tmp/audio.wav",
        question="Question",
        options=["First", "Second", "Third", "Fourth"],
        answer="B",
        prompt="Prompt",
        all_choices=["A", "B", "C", "D"],
        index2ans={"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
    )
    session = _FakeSession()
    send_fn = make_video_send_fn(
        "test-multimodal-model",
        "http://localhost/v1/chat/completions",
        enable_video_input=enable_video,
        enable_audio_input=enable_audio,
    )

    result = asyncio.run(send_fn(session, sample))

    assert result.is_success
    assert session.payload is not None
    assert {key for key in ("videos", "audios") if key in session.payload} == (
        expected_media_keys
    )
