from __future__ import annotations

import pytest

from vllm_omni.benchmarks.performance_gate import (
    _reject_duplicate_paths,
    build_report,
    compare_performance_results,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _runs(metric: str, values: list[float], *, completed: int = 100, failed: int = 0):
    return [{"completed": completed, "failed": failed, metric: value} for value in values]


def test_target_uses_median_of_three_repeated_runs() -> None:
    comparisons = compare_performance_results(
        _runs("mean_audio_ttfp_ms", [1000, 1100, 900]),
        _runs("mean_audio_ttfp_ms", [800, 850, 900]),
        target_metrics={"mean_audio_ttfp_ms"},
        min_improvement_percent=10,
    )
    assert comparisons[0].passed is True
    assert comparisons[0].baseline_median == 1000
    assert comparisons[0].candidate_median == 850
    assert comparisons[0].change_percent == pytest.approx(-15)
    assert build_report(comparisons)["passed"] is True


def test_target_regression_fails() -> None:
    comparisons = compare_performance_results(
        _runs("mean_audio_rtf", [0.8, 0.9, 1.0]),
        _runs("mean_audio_rtf", [0.9, 1.0, 1.1]),
        target_metrics={"mean_audio_rtf"},
    )
    assert comparisons[0].passed is False


def test_guard_allows_only_bounded_regression() -> None:
    baseline = [
        {"completed": 100, "failed": 0, "target": 10.0, "guard": 100.0},
        {"completed": 100, "failed": 0, "target": 11.0, "guard": 100.0},
        {"completed": 100, "failed": 0, "target": 12.0, "guard": 100.0},
    ]
    candidate = [
        {"completed": 100, "failed": 0, "target": 9.0, "guard": 103.0},
        {"completed": 100, "failed": 0, "target": 10.0, "guard": 103.0},
        {"completed": 100, "failed": 0, "target": 11.0, "guard": 103.0},
    ]
    comparisons = compare_performance_results(
        baseline,
        candidate,
        target_metrics={"target"},
        guard_metrics={"guard"},
        max_guard_regression_percent=2,
    )
    assert comparisons[0].role == "target"
    assert comparisons[0].passed is True
    assert comparisons[1].role == "guard"
    assert comparisons[1].passed is False


def test_higher_is_better_target_supports_throughput() -> None:
    comparisons = compare_performance_results(
        _runs("request_throughput", [10, 11, 12]),
        _runs("request_throughput", [12, 13, 14]),
        target_metrics={"request_throughput"},
        higher_is_better_metrics={"request_throughput"},
        min_improvement_percent=10,
    )
    assert comparisons[0].direction == "higher"
    assert comparisons[0].passed is True


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        (_runs("metric", [1, 1]), _runs("metric", [1, 1]), "at least 3"),
        (_runs("metric", [1, 1, 1]), _runs("metric", [1, 1, 1, 1]), "run counts differ"),
        (_runs("metric", [1, 1, 1], failed=1), _runs("metric", [1, 1, 1]), "failed requests"),
        (
            _runs("metric", [1, 1, 1], completed=100),
            _runs("metric", [1, 1, 1], completed=99),
            "completed counts differ",
        ),
    ],
)
def test_invalid_run_sets_fail_closed(baseline, candidate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_performance_results(baseline, candidate, target_metrics={"metric"})


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), "bad", True])
def test_invalid_metric_value_fails_closed(value) -> None:
    baseline = _runs("metric", [1, 1, 1])
    candidate = _runs("metric", [1, value, 1])
    comparisons = compare_performance_results(baseline, candidate, target_metrics={"metric"})
    assert comparisons[0].passed is False
    assert comparisons[0].change_percent is None


def test_duplicate_result_paths_are_rejected(tmp_path) -> None:
    result = tmp_path / "run.json"
    with pytest.raises(ValueError, match="must be distinct"):
        _reject_duplicate_paths("Baseline", [result, result])
