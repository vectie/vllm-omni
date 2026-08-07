# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed exact-output gate for batch-invariance qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_FIELDS = ("generated_texts",)


def _load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _nonempty_errors(result: dict[str, Any]) -> list[dict[str, Any]]:
    errors = result.get("errors")
    if not isinstance(errors, list):
        return [{"kind": "invalid_errors", "actual_type": type(errors).__name__}]
    return [{"kind": "request_error", "index": index, "error": error} for index, error in enumerate(errors) if error]


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> dict[str, Any]:
    """Compare exact per-request outputs and fail closed on incomplete data."""
    mismatches: list[dict[str, Any]] = []
    mismatches.extend(_nonempty_errors(baseline))
    mismatches.extend(_nonempty_errors(candidate))

    baseline_completed = baseline.get("completed")
    candidate_completed = candidate.get("completed")
    if not isinstance(baseline_completed, int) or not isinstance(candidate_completed, int):
        mismatches.append(
            {
                "kind": "invalid_completed",
                "baseline_type": type(baseline_completed).__name__,
                "candidate_type": type(candidate_completed).__name__,
            }
        )
    elif baseline_completed != candidate_completed:
        mismatches.append(
            {
                "kind": "completed",
                "baseline": baseline_completed,
                "candidate": candidate_completed,
            }
        )

    for field in fields:
        baseline_values = baseline.get(field)
        candidate_values = candidate.get(field)
        if not isinstance(baseline_values, list) or not isinstance(candidate_values, list):
            mismatches.append(
                {
                    "kind": "missing_or_invalid_field",
                    "field": field,
                    "baseline_type": type(baseline_values).__name__,
                    "candidate_type": type(candidate_values).__name__,
                }
            )
            continue
        if len(baseline_values) != len(candidate_values):
            mismatches.append(
                {
                    "kind": "length",
                    "field": field,
                    "baseline": len(baseline_values),
                    "candidate": len(candidate_values),
                }
            )
        if isinstance(baseline_completed, int) and len(baseline_values) != baseline_completed:
            mismatches.append(
                {
                    "kind": "completed_field_length",
                    "field": field,
                    "side": "baseline",
                    "completed": baseline_completed,
                    "length": len(baseline_values),
                }
            )
        if isinstance(candidate_completed, int) and len(candidate_values) != candidate_completed:
            mismatches.append(
                {
                    "kind": "completed_field_length",
                    "field": field,
                    "side": "candidate",
                    "completed": candidate_completed,
                    "length": len(candidate_values),
                }
            )
        for index, (expected, actual) in enumerate(zip(baseline_values, candidate_values, strict=False)):
            if expected != actual:
                mismatches.append(
                    {
                        "kind": "value",
                        "field": field,
                        "index": index,
                        "baseline": expected,
                        "candidate": actual,
                    }
                )

    return {"passed": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches}


def build_report(
    baseline_path: Path,
    candidate_paths: list[Path],
    *,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> dict[str, Any]:
    baseline = _load_result(baseline_path)
    candidates = []
    for candidate_path in candidate_paths:
        comparison = compare_results(
            baseline,
            _load_result(candidate_path),
            fields=fields,
        )
        candidates.append({"path": str(candidate_path), **comparison})
    return {
        "passed": all(candidate["passed"] for candidate in candidates),
        "baseline": str(baseline_path),
        "fields": list(fields),
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument(
        "--field",
        action="append",
        dest="fields",
        help="Exact list-valued result field to compare (repeatable)",
    )
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)

    report = build_report(
        args.baseline,
        args.candidates,
        fields=tuple(args.fields or DEFAULT_FIELDS),
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report_json is not None:
        args.report_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
