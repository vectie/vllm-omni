from __future__ import annotations

import pytest

from vllm_omni.benchmarks.quality_gate import (
    _parse_required_evaluated_counts,
    build_report,
    compare_quality_results,
)

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
        {
            "seed_tts_content_error_mean": 0.20,
            "seed_tts_content_evaluated": 100,
            "seed_tts_content_protocol": "seed-tts-in-tree",
        },
        {
            "seed_tts_content_error_mean": 0.215,
            "seed_tts_content_evaluated": 100,
            "seed_tts_content_protocol": "seed-tts-in-tree",
        },
    )
    assert comparisons[0].passed is True
    assert comparisons[0].regression_pp == pytest.approx(1.5)


def test_evaluated_count_mismatch_fails_even_when_score_improves() -> None:
    comparisons = compare_quality_results(
        {"seed_tts_sim_mean": 0.70, "seed_tts_sim_evaluated": 100},
        {"seed_tts_sim_mean": 0.80, "seed_tts_sim_evaluated": 99},
    )
    assert comparisons[0].passed is False
    assert "count mismatch" in comparisons[0].reason


@pytest.mark.parametrize("count", [None, 0, -1, 1.5, "not-a-count", True])
def test_missing_or_invalid_evaluated_count_fails_closed(count) -> None:
    comparisons = compare_quality_results(
        {"seed_tts_sim_mean": 0.70, "seed_tts_sim_evaluated": count},
        {"seed_tts_sim_mean": 0.80, "seed_tts_sim_evaluated": count},
    )
    assert comparisons[0].passed is False
    assert "missing or invalid" in comparisons[0].reason


def test_required_missing_metric_fails_closed() -> None:
    comparisons = compare_quality_results(
        {"daily_omni_accuracy_incl_http_fail": 0.8, "daily_omni_evaluated": 10},
        {"daily_omni_evaluated": 10},
        required_metrics={"daily_omni_accuracy_incl_http_fail"},
    )
    assert comparisons[0].passed is False
    assert comparisons[0].reason == "metric missing or non-finite"


def test_seed_tts_protocol_mismatch_fails_closed() -> None:
    comparisons = compare_quality_results(
        {
            "seed_tts_sim_mean": 0.70,
            "seed_tts_sim_evaluated": 100,
            "seed_tts_sim_protocol": "proxy-a",
        },
        {
            "seed_tts_sim_mean": 0.71,
            "seed_tts_sim_evaluated": 100,
            "seed_tts_sim_protocol": "proxy-b",
        },
    )
    assert comparisons[0].passed is False
    assert "protocol" in comparisons[0].reason


def test_official_seed_tts_protocol_can_be_required() -> None:
    common = {
        "seed_tts_sim_mean": 0.70,
        "seed_tts_sim_evaluated": 100,
        "seed_tts_sim_protocol": "seed-tts-eval-official-wavlm-large-sv",
    }
    comparisons = compare_quality_results(common, common, require_seed_tts_official=True)
    assert comparisons[0].passed is True


def test_required_suite_size_rejects_matching_partial_runs() -> None:
    common = {"video_mme_accuracy_incl_http_fail": 0.70, "video_mme_evaluated": 27}
    comparisons = compare_quality_results(
        common,
        common,
        required_evaluated_counts={"video_mme_evaluated": 2700},
    )
    assert comparisons[0].passed is False
    assert "required suite size" in comparisons[0].reason


def test_required_suite_size_also_requires_its_metric() -> None:
    comparisons = compare_quality_results(
        {
            "daily_omni_accuracy_incl_http_fail": 0.70,
            "daily_omni_evaluated": 1197,
            "video_mme_evaluated": 2700,
        },
        {
            "daily_omni_accuracy_incl_http_fail": 0.70,
            "daily_omni_evaluated": 1197,
            "video_mme_evaluated": 2700,
        },
        required_evaluated_counts={"video_mme_evaluated": 2700},
    )
    assert any(item.metric == "video_mme_accuracy_incl_http_fail" and not item.passed for item in comparisons)


def test_parse_required_evaluated_counts() -> None:
    assert _parse_required_evaluated_counts(["daily_omni_evaluated=1197", "video_mme_evaluated=2700"]) == {
        "daily_omni_evaluated": 1197,
        "video_mme_evaluated": 2700,
    }


@pytest.mark.parametrize("value", ["missing", "field=", "field=0", "field=bad"])
def test_invalid_required_evaluated_count_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_required_evaluated_counts([value])


def test_no_known_metrics_is_rejected() -> None:
    with pytest.raises(ValueError, match="No recognized quality metrics"):
        compare_quality_results({"foo": 1}, {"foo": 1})
