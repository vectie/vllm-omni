"""Merge official Seed-TTS WER/SIM reports into a serving benchmark JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

_WER_RE = re.compile(r"^WER:\s*([0-9]+(?:\.[0-9]+)?)%\s*$")
_SIM_RE = re.compile(r"^ASV:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _parse_official_report(path: Path, pattern: re.Pattern[str], *, percent: bool) -> tuple[float, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metric: float | None = None
    evaluated = 0
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            metric = float(match.group(1)) / (100.0 if percent else 1.0)
        elif "\t" in line and not line.lower().startswith("utt\t"):
            evaluated += 1
    if metric is None or not math.isfinite(metric):
        raise ValueError(f"Official Seed-TTS metric summary not found in {path}")
    if evaluated <= 0:
        raise ValueError(f"Official Seed-TTS report contains no evaluated rows: {path}")
    return metric, evaluated


def merge_official_seed_tts_results(
    benchmark: dict[str, Any],
    *,
    wer_report: Path | None = None,
    sim_report: Path | None = None,
) -> dict[str, Any]:
    if wer_report is None and sim_report is None:
        raise ValueError("At least one official Seed-TTS report is required")
    result = dict(benchmark)
    result["seed_tts_eval_protocol"] = "seed-tts-eval-official"
    reports = dict(result.get("seed_tts_official_reports") or {})
    if wer_report is not None:
        wer, evaluated = _parse_official_report(wer_report, _WER_RE, percent=True)
        result.update(
            {
                "seed_tts_content_error_mean": wer,
                "seed_tts_content_evaluated": evaluated,
                "seed_tts_content_metric": "wer",
                "seed_tts_content_protocol": "seed-tts-eval-official",
            }
        )
        reports["wer"] = str(wer_report)
    if sim_report is not None:
        similarity, evaluated = _parse_official_report(sim_report, _SIM_RE, percent=False)
        result.update(
            {
                "seed_tts_sim_mean": similarity,
                "seed_tts_sim_evaluated": evaluated,
                "seed_tts_sim_protocol": "seed-tts-eval-official-wavlm-large-sv",
            }
        )
        reports["sim"] = str(sim_report)
    result["seed_tts_official_reports"] = reports
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="Saved vllm bench serve result JSON")
    parser.add_argument("--wer-report", type=Path, default=None, help="Official cal_wer.sh wav_res_ref_text.wer")
    parser.add_argument("--sim-report", type=Path, default=None, help="Official cal_sim.sh wav_res_ref_text.wer")
    parser.add_argument("--output", type=Path, required=True, help="Merged result JSON (input is never overwritten)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = merge_official_seed_tts_results(
        _load_json_object(args.benchmark),
        wer_report=args.wer_report,
        sim_report=args.sim_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
