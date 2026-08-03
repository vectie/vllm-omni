"""Compare benchmark result JSON files against a quality-regression budget."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class QualityMetricSpec:
    key: str
    direction: Literal["higher", "lower"]
    evaluated_key: str | None = None


QUALITY_METRICS: tuple[QualityMetricSpec, ...] = (
    QualityMetricSpec("daily_omni_accuracy_incl_http_fail", "higher", "daily_omni_evaluated"),
    QualityMetricSpec("video_mme_accuracy_incl_http_fail", "higher", "video_mme_evaluated"),
    QualityMetricSpec("seed_tts_sim_mean", "higher", "seed_tts_content_evaluated"),
    QualityMetricSpec("seed_tts_content_error_mean", "lower", "seed_tts_content_evaluated"),
)


@dataclass(frozen=True)
class QualityComparison:
    metric: str
    baseline: float | None
    candidate: float | None
    regression_pp: float | None
    allowed_regression_pp: float
    passed: bool
    reason: str = ""


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def compare_quality_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_regression_pp: float = 2.0,
    required_metrics: set[str] | None = None,
) -> list[QualityComparison]:
    """Compare all recognized common quality metrics.

    Accuracy-like values are stored as fractions, so a delta of ``0.02`` is a
    two-percentage-point regression. WER is lower-is-better; SIM and MCQ
    accuracy are higher-is-better. Matching evaluated counts are required to
    prevent a candidate from passing by silently dropping difficult requests.
    """
    if max_regression_pp < 0:
        raise ValueError("max_regression_pp must be non-negative")
    required_metrics = required_metrics or set()
    known = {spec.key for spec in QUALITY_METRICS}
    unknown = required_metrics - known
    if unknown:
        raise ValueError(f"Unknown required quality metrics: {sorted(unknown)}")

    comparisons: list[QualityComparison] = []
    for spec in QUALITY_METRICS:
        present = spec.key in baseline or spec.key in candidate or spec.key in required_metrics
        if not present:
            continue
        base_value = _finite_float(baseline.get(spec.key))
        candidate_value = _finite_float(candidate.get(spec.key))
        if base_value is None or candidate_value is None:
            comparisons.append(
                QualityComparison(
                    metric=spec.key,
                    baseline=base_value,
                    candidate=candidate_value,
                    regression_pp=None,
                    allowed_regression_pp=max_regression_pp,
                    passed=False,
                    reason="metric missing or non-finite",
                )
            )
            continue

        if spec.evaluated_key:
            base_count = baseline.get(spec.evaluated_key)
            candidate_count = candidate.get(spec.evaluated_key)
            if base_count != candidate_count:
                comparisons.append(
                    QualityComparison(
                        metric=spec.key,
                        baseline=base_value,
                        candidate=candidate_value,
                        regression_pp=None,
                        allowed_regression_pp=max_regression_pp,
                        passed=False,
                        reason=(
                            f"evaluated count mismatch: {spec.evaluated_key} "
                            f"baseline={base_count!r} candidate={candidate_count!r}"
                        ),
                    )
                )
                continue

        regression = base_value - candidate_value if spec.direction == "higher" else candidate_value - base_value
        regression_pp = regression * 100.0
        comparisons.append(
            QualityComparison(
                metric=spec.key,
                baseline=base_value,
                candidate=candidate_value,
                regression_pp=regression_pp,
                allowed_regression_pp=max_regression_pp,
                passed=regression_pp <= max_regression_pp + 1e-12,
            )
        )

    if not comparisons:
        raise ValueError("No recognized quality metrics found in the result files")
    missing_required = required_metrics - {comparison.metric for comparison in comparisons}
    if missing_required:
        raise ValueError(f"Required quality metrics were not compared: {sorted(missing_required)}")
    return comparisons


def build_report(comparisons: list[QualityComparison]) -> dict[str, Any]:
    return {
        "passed": all(comparison.passed for comparison in comparisons),
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline vllm bench serve result JSON")
    parser.add_argument("candidate", type=Path, help="Optimized candidate result JSON")
    parser.add_argument("--max-regression-pp", type=float, default=2.0)
    parser.add_argument(
        "--require-metric",
        action="append",
        default=[],
        choices=[spec.key for spec in QUALITY_METRICS],
        help="Require this metric to exist in both files; may be repeated.",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    comparisons = compare_quality_results(
        _load_json(args.baseline),
        _load_json(args.candidate),
        max_regression_pp=args.max_regression_pp,
        required_metrics=set(args.require_metric),
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
