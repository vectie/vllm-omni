from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_omni.benchmarks.data_modules.video_mme_dataset import VideoMMEDataset

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "youtube-id.mp4").write_bytes(b"video")
    annotations = [
        {
            "video_id": "001",
            "videoID": "youtube-id",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "History",
            "questions": [
                {
                    "question_id": "001-1",
                    "task_type": "Counting Problem",
                    "question": "How many objects are visible?",
                    "options": ["A. One", "B. Two", "C. Three", "D. Four"],
                    "answer": "B",
                }
            ],
        }
    ]
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    return annotations_path, video_dir


def test_local_official_json_builds_video_request(tmp_path: Path) -> None:
    annotations, video_dir = _write_fixture(tmp_path)
    dataset = VideoMMEDataset(
        annotations_json=str(annotations),
        video_dir=str(video_dir),
        disable_shuffle=True,
    )

    requests = dataset.sample(_Tokenizer(), num_requests=1, no_oversample=True)

    assert len(requests) == 1
    request = requests[0]
    assert request.video_mme_gold_answer == "B"
    assert request.video_mme_duration == "short"
    assert "Respond with only the letter" in request.prompt
    content = request.omni_chat_messages[0]["content"]
    assert content[0]["video_url"]["url"].startswith("file://")


def test_duration_filter_does_not_admit_other_buckets(tmp_path: Path) -> None:
    annotations, video_dir = _write_fixture(tmp_path)
    dataset = VideoMMEDataset(
        annotations_json=str(annotations),
        video_dir=str(video_dir),
        duration="long",
        disable_shuffle=True,
    )
    assert dataset.sample(_Tokenizer(), num_requests=1, no_oversample=True) == []


def test_video_directory_is_required(tmp_path: Path) -> None:
    annotations, _ = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="video-mme-video-dir"):
        VideoMMEDataset(annotations_json=str(annotations), video_dir="")
