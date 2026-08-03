from __future__ import annotations

import pytest

from vllm_omni.benchmarks.quality_gate import build_report, compare_quality_results

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_two_percentage_point_boundary_passes() -> None:
    comparisons = compare_quality_results(
        {
            "daily_omni_accuracy_incl_http_fail": 0.80,
            "daily_omni_evaluated": 100,
        },
        {
            "daily_omni_accuracy_incl_http_fail": 0.78,
            "daily_omni_evaluated": 100,
        },
    )
    assert build_report(comparisons)["passed"] is True
    assert comparisons[0].regression_pp == pytest.approx(2.0)


def test_accuracy_drop_over_budget_fails() -> None:
    comparisons = compare_quality_results(
        {"video_mme_accuracy_incl_http_fail": 0.70, "video_mme_evaluated": 2700},
        {"video_mme_accuracy_incl_http_fail": 0.679, "video_mme_evaluated": 2700},
    )
    assert build_report(comparisons)["passed"] is False
    assert comparisons[0].regression_pp == pytest.approx(2.1)


def test_wer_increase_uses_lower_is_better_direction() -> None:
    comparisons = compare_quality_results(
        {"seed_tts_content_error_mean": 0.20, "seed_tts_content_evaluated": 100},
        {"seed_tts_content_error_mean": 0.215, "seed_tts_content_evaluated": 100},
    )
    assert comparisons[0].passed is True
    assert comparisons[0].regression_pp == pytest.approx(1.5)


def test_evaluated_count_mismatch_fails_even_when_score_improves() -> None:
    comparisons = compare_quality_results(
        {"seed_tts_sim_mean": 0.70, "seed_tts_content_evaluated": 100},
        {"seed_tts_sim_mean": 0.80, "seed_tts_content_evaluated": 99},
    )
    assert comparisons[0].passed is False
    assert "count mismatch" in comparisons[0].reason


def test_required_missing_metric_fails_closed() -> None:
    comparisons = compare_quality_results(
        {"daily_omni_accuracy_incl_http_fail": 0.8, "daily_omni_evaluated": 10},
        {"daily_omni_evaluated": 10},
        required_metrics={"daily_omni_accuracy_incl_http_fail"},
    )
    assert comparisons[0].passed is False
    assert comparisons[0].reason == "metric missing or non-finite"


def test_no_known_metrics_is_rejected() -> None:
    with pytest.raises(ValueError, match="No recognized quality metrics"):
        compare_quality_results({"foo": 1}, {"foo": 1})
