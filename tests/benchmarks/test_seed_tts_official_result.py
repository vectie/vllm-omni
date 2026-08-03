from __future__ import annotations

import pytest

from vllm_omni.benchmarks.seed_tts_official_result import merge_official_seed_tts_results

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_merge_official_seed_tts_reports(tmp_path) -> None:
    wer_report = tmp_path / "wer.txt"
    wer_report.write_text(
        "utt\twav_res\tres_wer\nitem1\t0.01\ttext\nitem2\t0.03\ttext\nWER: 2.0%\n",
        encoding="utf-8",
    )
    sim_report = tmp_path / "sim.txt"
    sim_report.write_text("a|b\t0.7\nc|d\t0.8\nASV: 0.75\nASV-var: 0.01\n", encoding="utf-8")

    result = merge_official_seed_tts_results(
        {"completed": 2},
        wer_report=wer_report,
        sim_report=sim_report,
    )

    assert result["seed_tts_content_error_mean"] == pytest.approx(0.02)
    assert result["seed_tts_content_evaluated"] == 2
    assert result["seed_tts_sim_mean"] == pytest.approx(0.75)
    assert result["seed_tts_sim_evaluated"] == 2
    assert result["seed_tts_content_protocol"] == "seed-tts-eval-official"
    assert result["seed_tts_sim_protocol"] == "seed-tts-eval-official-wavlm-large-sv"


@pytest.mark.parametrize(
    "content",
    [
        "item1\t0.1\n",
        "WER: 1.0%\n",
        "",
    ],
)
def test_invalid_official_report_is_rejected(tmp_path, content: str) -> None:
    report = tmp_path / "invalid.txt"
    report.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Official Seed-TTS"):
        merge_official_seed_tts_results({}, wer_report=report)
