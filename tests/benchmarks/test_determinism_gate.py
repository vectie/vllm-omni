# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.benchmarks.determinism_gate import compare_results


def test_determinism_gate_accepts_exact_outputs():
    baseline = {
        "completed": 2,
        "errors": ["", ""],
        "generated_texts": ["a", "b"],
        "audio_content_sha256s": ["one", "two"],
    }
    candidate = dict(baseline)

    report = compare_results(
        baseline,
        candidate,
        fields=("generated_texts", "audio_content_sha256s"),
    )

    assert report["passed"] is True
    assert report["mismatch_count"] == 0


def test_determinism_gate_reports_request_and_value_mismatches():
    baseline = {"completed": 2, "errors": ["", ""], "generated_texts": ["a", "b"]}
    candidate = {"completed": 1, "errors": ["", "timeout"], "generated_texts": ["a", "c"]}

    report = compare_results(baseline, candidate)

    assert report["passed"] is False
    assert {item["kind"] for item in report["mismatches"]} == {
        "completed",
        "completed_field_length",
        "request_error",
        "value",
    }


def test_determinism_gate_fails_closed_on_missing_field():
    report = compare_results(
        {"completed": 1, "errors": [""]},
        {"completed": 1, "errors": [""]},
    )

    assert report["passed"] is False
    assert report["mismatches"][0]["kind"] == "missing_or_invalid_field"


def test_determinism_gate_fails_closed_on_missing_run_metadata():
    report = compare_results(
        {"generated_texts": []},
        {"generated_texts": []},
    )

    assert report["passed"] is False
    assert {item["kind"] for item in report["mismatches"]} == {
        "invalid_completed",
        "invalid_errors",
    }
