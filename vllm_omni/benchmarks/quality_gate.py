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
    protocol_key: str | None = None


QUALITY_METRICS: tuple[QualityMetricSpec, ...] = (
    QualityMetricSpec("daily_omni_accuracy_incl_http_fail", "higher", "daily_omni_evaluated"),
    QualityMetricSpec("video_mme_accuracy_incl_http_fail", "higher", "video_mme_evaluated"),
    QualityMetricSpec("seed_tts_sim_mean", "higher", "seed_tts_sim_evaluated", "seed_tts_sim_protocol"),
    QualityMetricSpec(
        "seed_tts_content_error_mean",
        "lower",
        "seed_tts_content_evaluated",
        "seed_tts_content_protocol",
    ),
)

_OFFICIAL_SEED_TTS_PROTOCOLS = {
    "seed_tts_sim_mean": "seed-tts-eval-official-wavlm-large-sv",
    "seed_tts_content_error_mean": "seed-tts-eval-official",
}


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


def _positive_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if result <= 0 or result != value:
        return None
    return result


def compare_quality_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_regression_pp: float = 2.0,
    required_metrics: set[str] | None = None,
    require_seed_tts_official: bool = False,
    required_evaluated_counts: dict[str, int] | None = None,
) -> list[QualityComparison]:
    """Compare all recognized common quality metrics.

    Accuracy-like values are stored as fractions, so a delta of ``0.02`` is a
    two-percentage-point regression. WER is lower-is-better; SIM and MCQ
    accuracy are higher-is-better. Matching evaluated counts are required to
    prevent a candidate from passing by silently dropping difficult requests.
    """
    if max_regression_pp < 0:
        raise ValueError("max_regression_pp must be non-negative")
    required_metrics = set(required_metrics or ())
    required_evaluated_counts = required_evaluated_counts or {}
    known = {spec.key for spec in QUALITY_METRICS}
    unknown = required_metrics - known
    if unknown:
        raise ValueError(f"Unknown required quality metrics: {sorted(unknown)}")
    known_count_keys = {spec.evaluated_key for spec in QUALITY_METRICS if spec.evaluated_key}
    unknown_count_keys = set(required_evaluated_counts) - known_count_keys
    if unknown_count_keys:
        raise ValueError(f"Unknown evaluated-count fields: {sorted(unknown_count_keys)}")
    if any(_positive_count(value) is None for value in required_evaluated_counts.values()):
        raise ValueError("Required evaluated counts must be positive integers")
    required_metrics.update(spec.key for spec in QUALITY_METRICS if spec.evaluated_key in required_evaluated_counts)

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
            raw_base_count = baseline.get(spec.evaluated_key)
            raw_candidate_count = candidate.get(spec.evaluated_key)
            base_count = _positive_count(raw_base_count)
            candidate_count = _positive_count(raw_candidate_count)
            if base_count is None or candidate_count is None:
                comparisons.append(
                    QualityComparison(
                        metric=spec.key,
                        baseline=base_value,
                        candidate=candidate_value,
                        regression_pp=None,
                        allowed_regression_pp=max_regression_pp,
                        passed=False,
                        reason=(
                            f"evaluated count missing or invalid: {spec.evaluated_key} "
                            f"baseline={raw_base_count!r} candidate={raw_candidate_count!r}"
                        ),
                    )
                )
                continue
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
                            f"baseline={raw_base_count!r} candidate={raw_candidate_count!r}"
                        ),
                    )
                )
                continue
            expected_count = required_evaluated_counts.get(spec.evaluated_key)
            if expected_count is not None and base_count != expected_count:
                comparisons.append(
                    QualityComparison(
                        metric=spec.key,
                        baseline=base_value,
                        candidate=candidate_value,
                        regression_pp=None,
                        allowed_regression_pp=max_regression_pp,
                        passed=False,
                        reason=(
                            f"evaluated count does not match required suite size: {spec.evaluated_key} "
                            f"actual={base_count} required={expected_count}"
                        ),
                    )
                )
                continue

        if spec.protocol_key:
            base_protocol = baseline.get(spec.protocol_key)
            candidate_protocol = candidate.get(spec.protocol_key)
            expected_protocol = _OFFICIAL_SEED_TTS_PROTOCOLS.get(spec.key) if require_seed_tts_official else None
            if (
                not isinstance(base_protocol, str)
                or not base_protocol
                or base_protocol != candidate_protocol
                or (expected_protocol is not None and base_protocol != expected_protocol)
            ):
                comparisons.append(
                    QualityComparison(
                        metric=spec.key,
                        baseline=base_value,
                        candidate=candidate_value,
                        regression_pp=None,
                        allowed_regression_pp=max_regression_pp,
                        passed=False,
                        reason=(
                            f"evaluation protocol missing or mismatched: {spec.protocol_key} "
                            f"baseline={base_protocol!r} candidate={candidate_protocol!r}"
                            + (f" required={expected_protocol!r}" if expected_protocol else "")
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
    parser.add_argument(
        "--require-seed-tts-official",
        action="store_true",
        help="Require official Seed-TTS WER and WavLM-large-SV protocols for requested Seed-TTS metrics.",
    )
    parser.add_argument(
        "--require-evaluated-count",
        action="append",
        default=[],
        metavar="FIELD=COUNT",
        help="Require an exact positive evaluated count in both files; may be repeated.",
    )
    return parser


def _parse_required_evaluated_counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        field, separator, raw_count = value.partition("=")
        if not separator or not field or not raw_count:
            raise ValueError(f"Invalid --require-evaluated-count value: {value!r}; expected FIELD=COUNT")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"Invalid evaluated count in {value!r}") from exc
        if count <= 0:
            raise ValueError(f"Evaluated count must be positive in {value!r}")
        if field in result and result[field] != count:
            raise ValueError(f"Conflicting required evaluated counts for {field!r}")
        result[field] = count
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    comparisons = compare_quality_results(
        _load_json(args.baseline),
        _load_json(args.candidate),
        max_regression_pp=args.max_regression_pp,
        required_metrics=set(args.require_metric),
        require_seed_tts_official=args.require_seed_tts_official,
        required_evaluated_counts=_parse_required_evaluated_counts(args.require_evaluated_count),
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
