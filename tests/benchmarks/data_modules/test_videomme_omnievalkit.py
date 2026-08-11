from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules.videomme_dataset import (
    VIDEOMME_DEFAULT_MAX_FRAMES,
    VIDEOMME_USER_PROMPT_TEMPLATE,
    VideoMMEDataset,
    VideoMMESampleRequest,
)
from vllm_omni.benchmarks.data_modules.videomme_eval import compute_videomme_accuracy_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_omnievalkit_prompt_and_option_format_are_pinned() -> None:
    dataset = VideoMMEDataset.__new__(VideoMMEDataset)
    dataset.use_subtitle = False

    fields = {
        "question": "What happens next?",
        "options": ["A. It stops", "B. It moves", "C. It falls", "D. It opens"],
        "video_id": "video",
    }
    prompt = dataset._build_user_prompt(fields)

    assert prompt == VIDEOMME_USER_PROMPT_TEMPLATE.format(
        question="What happens next?",
        options="A. It stops\nB. It moves\nC. It falls\nD. It opens",
    )
    assert "Only respond with the letter." in prompt


def test_minicpm_frames_pack_is_image_only_and_pins_processor_kwargs(monkeypatch) -> None:
    dataset = VideoMMEDataset.__new__(VideoMMEDataset)
    dataset.pack_mode = "minicpm-frames"
    frame = {"type": "image_url", "image_url": {"url": "file:///frame.jpg"}}
    monkeypatch.setattr(dataset, "_get_minicpm_frame_parts", lambda *_args, **kwargs: [frame])

    parts, extra = dataset._compose_multimodal("video")

    assert parts == [frame]
    assert extra == {
        "mm_processor_kwargs": {
            "use_audio_in_video": False,
            "max_slice_nums": 1,
            "use_image_id": False,
        }
    }


def test_omnievalkit_timestamp_sampling_caps_long_video_at_96_frames() -> None:
    timestamps = VideoMMEDataset._sample_timestamps(120.0, VIDEOMME_DEFAULT_MAX_FRAMES)

    assert len(timestamps) == 96
    assert timestamps == sorted(timestamps)
    assert timestamps[0] > 0.0
    assert timestamps[-1] < 120.0


def _request(request_id: str, gold: str, duration: str) -> VideoMMESampleRequest:
    return VideoMMESampleRequest(
        prompt="question",
        prompt_len=1,
        expected_output_len=1,
        multi_modal_data=None,
        request_id=request_id,
        videomme_gold_answer=gold,
        videomme_video_id="video",
        videomme_question_id=request_id,
        videomme_duration=duration,
        videomme_domain="Knowledge",
        videomme_sub_category="History",
        videomme_task_type="Reasoning",
    )


def test_videomme_accuracy_is_fail_closed_and_keeps_official_breakdowns() -> None:
    requests = [_request("q1", "B", "short"), _request("q2", "C", "long")]
    outputs = [
        SimpleNamespace(success=True, generated_text="B", error=""),
        SimpleNamespace(success=True, generated_text="No option", error=""),
    ]

    metrics = compute_videomme_accuracy_metrics(requests, outputs, include_per_item=True)

    assert metrics is not None
    assert metrics["videomme_accuracy"] == pytest.approx(0.5)
    assert metrics["videomme_parse_failed"] == 1
    assert metrics["videomme_per_duration_accuracy"] == {
        "short": 1.0,
        "medium": None,
        "long": 0.0,
    }
    assert len(metrics["videomme_eval_items"]) == 2
