from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniSampleRequest
from vllm_omni.benchmarks.data_modules.daily_omni_eval import compute_daily_omni_accuracy_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _request(gold: str) -> DailyOmniSampleRequest:
    return DailyOmniSampleRequest(
        prompt="question",
        prompt_len=1,
        expected_output_len=1,
        multi_modal_data=None,
        request_id="q1",
        daily_omni_gold_answer=gold,
        daily_omni_video_id="video",
    )


@pytest.mark.parametrize("gold", ["B", "B. Full selected option text"])
def test_accuracy_accepts_official_and_full_option_gold(gold: str) -> None:
    output = SimpleNamespace(success=True, generated_text="B", error="")

    metrics = compute_daily_omni_accuracy_metrics([_request(gold)], [output], include_per_item=True)

    assert metrics is not None
    assert metrics["daily_omni_accuracy"] == 1.0
    assert metrics["daily_omni_correct"] == 1
    assert metrics["daily_omni_eval_items"][0]["gold_normalized"] == "B"
    assert metrics["daily_omni_eval_items"][0]["correct"] is True
