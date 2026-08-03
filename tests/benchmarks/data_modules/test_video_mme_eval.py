from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules.video_mme_dataset import VideoMMESampleRequest
from vllm_omni.benchmarks.data_modules.video_mme_eval import compute_video_mme_accuracy_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _request(request_id: str, gold: str, duration: str = "short") -> VideoMMESampleRequest:
    return VideoMMESampleRequest(
        prompt="question",
        prompt_len=1,
        expected_output_len=1,
        multi_modal_data=None,
        request_id=request_id,
        video_mme_gold_answer=gold,
        video_mme_video_id="video",
        video_mme_question_id=request_id,
        video_mme_duration=duration,
        video_mme_domain="Knowledge",
        video_mme_sub_category="History",
        video_mme_task_type="Counting",
    )


def test_accuracy_tracks_failures_and_official_breakdowns() -> None:
    requests = [_request("q1", "B"), _request("q2", "C"), _request("q3", "A", "long")]
    outputs = [
        SimpleNamespace(success=True, generated_text="B", error=""),
        SimpleNamespace(success=True, generated_text="D.", error=""),
        SimpleNamespace(success=False, generated_text="", error="timeout"),
    ]

    metrics = compute_video_mme_accuracy_metrics(requests, outputs, include_per_item=True)

    assert metrics is not None
    assert metrics["video_mme_accuracy"] == pytest.approx(0.5)
    assert metrics["video_mme_accuracy_incl_http_fail"] == pytest.approx(1 / 3)
    assert metrics["video_mme_request_failed"] == 1
    assert metrics["video_mme_per_duration_accuracy"] == {"short": 0.5, "long": None}
    assert len(metrics["video_mme_eval_items"]) == 3


def test_non_video_mme_requests_are_ignored() -> None:
    assert compute_video_mme_accuracy_metrics([SimpleNamespace()], [SimpleNamespace()]) is None
