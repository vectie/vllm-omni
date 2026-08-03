"""Fail-closed promotion gate for repeated serving benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class PerformanceComparison:
    metric: str
    role: Literal["target", "guard"]
    direction: Literal["higher", "lower"]
    baseline_median: float | None
    candidate_median: float | None
    change_percent: float | None
    threshold_percent: float
    passed: bool
    reason: str


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _validate_run_set(
    name: str,
    runs: list[dict[str, Any]],
    *,
    min_runs: int,
) -> int:
    if len(runs) < min_runs:
        raise ValueError(f"{name} has {len(runs)} runs; at least {min_runs} are required")
    completed_values: list[int] = []
    for index, run in enumerate(runs, start=1):
        completed = _positive_int(run.get("completed"))
        failed = _nonnegative_int(run.get("failed"))
        if completed is None:
            raise ValueError(f"{name} run {index} has missing or invalid completed count")
        if failed is None:
            raise ValueError(f"{name} run {index} has missing or invalid failed count")
        if failed:
            raise ValueError(f"{name} run {index} has {failed} failed requests")
        completed_values.append(completed)
    if len(set(completed_values)) != 1:
        raise ValueError(f"{name} completed counts differ across runs: {completed_values}")
    return completed_values[0]


def compare_performance_results(
    baseline_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    *,
    target_metrics: set[str],
    guard_metrics: set[str] | None = None,
    higher_is_better_metrics: set[str] | None = None,
    min_runs: int = 3,
    min_improvement_percent: float = 0.0,
    max_guard_regression_percent: float = 2.0,
) -> list[PerformanceComparison]:
    """Compare medians across repeated runs.

    Target metrics must improve by at least ``min_improvement_percent``.
    Guard metrics may regress by at most ``max_guard_regression_percent``.
    Every run must complete the same positive number of requests with zero
    failures, and every selected metric must be finite and positive.
    """

    guard_metrics = set(guard_metrics or ())
    target_metrics = set(target_metrics)
    higher_is_better_metrics = set(higher_is_better_metrics or ())
    if not target_metrics:
        raise ValueError("At least one target metric is required")
    overlap = target_metrics & guard_metrics
    if overlap:
        raise ValueError(f"Metrics cannot be both targets and guards: {sorted(overlap)}")
    unknown_directions = higher_is_better_metrics - target_metrics - guard_metrics
    if unknown_directions:
        raise ValueError(f"Direction specified for unselected metrics: {sorted(unknown_directions)}")
    if min_runs <= 0:
        raise ValueError("min_runs must be positive")
    if min_improvement_percent < 0 or max_guard_regression_percent < 0:
        raise ValueError("Performance thresholds must be non-negative")
    if len(baseline_runs) != len(candidate_runs):
        raise ValueError(f"Baseline and candidate run counts differ: {len(baseline_runs)} != {len(candidate_runs)}")

    baseline_completed = _validate_run_set("baseline", baseline_runs, min_runs=min_runs)
    candidate_completed = _validate_run_set("candidate", candidate_runs, min_runs=min_runs)
    if baseline_completed != candidate_completed:
        raise ValueError(
            f"Baseline and candidate completed counts differ: {baseline_completed} != {candidate_completed}"
        )

    comparisons: list[PerformanceComparison] = []
    for role, metrics, threshold in (
        ("target", target_metrics, min_improvement_percent),
        ("guard", guard_metrics, max_guard_regression_percent),
    ):
        for metric in sorted(metrics):
            direction: Literal["higher", "lower"] = "higher" if metric in higher_is_better_metrics else "lower"
            baseline_values = [_positive_finite(run.get(metric)) for run in baseline_runs]
            candidate_values = [_positive_finite(run.get(metric)) for run in candidate_runs]
            if any(value is None for value in baseline_values + candidate_values):
                comparisons.append(
                    PerformanceComparison(
                        metric=metric,
                        role=role,
                        direction=direction,
                        baseline_median=None,
                        candidate_median=None,
                        change_percent=None,
                        threshold_percent=threshold,
                        passed=False,
                        reason="metric missing, non-finite, or non-positive in one or more runs",
                    )
                )
                continue

            baseline_median = statistics.median(value for value in baseline_values if value is not None)
            candidate_median = statistics.median(value for value in candidate_values if value is not None)
            change_percent = (candidate_median - baseline_median) / baseline_median * 100.0
            improvement_percent = change_percent if direction == "higher" else -change_percent
            if role == "target":
                passed = improvement_percent >= threshold
                reason = (
                    f"median improvement {improvement_percent:.3f}% "
                    f"{'meets' if passed else 'does not meet'} required {threshold:.3f}%"
                )
            else:
                regression_percent = -improvement_percent
                passed = regression_percent <= threshold
                reason = (
                    f"median regression {regression_percent:.3f}% "
                    f"{'is within' if passed else 'exceeds'} allowed regression {threshold:.3f}%"
                )
            comparisons.append(
                PerformanceComparison(
                    metric=metric,
                    role=role,
                    direction=direction,
                    baseline_median=baseline_median,
                    candidate_median=candidate_median,
                    change_percent=change_percent,
                    threshold_percent=threshold,
                    passed=passed,
                    reason=reason,
                )
            )
    return comparisons


def build_report(comparisons: list[PerformanceComparison]) -> dict[str, Any]:
    return {
        "passed": bool(comparisons) and all(item.passed for item in comparisons),
        "comparisons": [asdict(item) for item in comparisons],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", type=Path, required=True, help="Baseline result JSON")
    parser.add_argument("--candidate", action="append", type=Path, required=True, help="Candidate result JSON")
    parser.add_argument(
        "--target-metric",
        action="append",
        required=True,
        help="Lower-is-better metric that must improve; may be repeated.",
    )
    parser.add_argument(
        "--guard-metric",
        action="append",
        default=[],
        help="Lower-is-better metric with bounded regression; may be repeated.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="append",
        default=[],
        metavar="METRIC",
        help="Mark a selected metric as higher-is-better; may be repeated.",
    )
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--min-improvement-percent", type=float, default=0.0)
    parser.add_argument("--max-guard-regression-percent", type=float, default=2.0)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser


def _reject_duplicate_paths(name: str, paths: list[Path]) -> set[Path]:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} result paths must be distinct")
    return set(resolved)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline_paths = _reject_duplicate_paths("Baseline", args.baseline)
    candidate_paths = _reject_duplicate_paths("Candidate", args.candidate)
    overlap = baseline_paths & candidate_paths
    if overlap:
        raise ValueError(f"Baseline and candidate paths overlap: {sorted(map(str, overlap))}")
    comparisons = compare_performance_results(
        [_load_json_object(path) for path in args.baseline],
        [_load_json_object(path) for path in args.candidate],
        target_metrics=set(args.target_metric),
        guard_metrics=set(args.guard_metric),
        higher_is_better_metrics=set(args.higher_is_better),
        min_runs=args.min_runs,
        min_improvement_percent=args.min_improvement_percent,
        max_guard_regression_percent=args.max_guard_regression_percent,
    )
    report = build_report(comparisons)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
