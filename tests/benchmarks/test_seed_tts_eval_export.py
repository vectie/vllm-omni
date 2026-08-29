from __future__ import annotations

import sys
import wave
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules import seed_tts_eval
from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSSampleRequest
from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
    _save_seed_tts_official_audio,
    compute_seed_tts_wer_metrics,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_official_export_uses_exact_utterance_id(tmp_path) -> None:
    pcm = b"\x01\x00\x02\x00"

    result = _save_seed_tts_official_audio(
        pcm,
        output_dir=tmp_path,
        utterance_id="seed_en_0001.wav",
    )

    assert result == str(tmp_path / "seed_en_0001.wav")
    with wave.open(result, "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(2) == pcm


def test_official_export_only_skips_wer_dependencies(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEED_TTS_OFFICIAL_EXPORT_DIR", str(tmp_path))
    monkeypatch.delenv("SEED_TTS_WER_EVAL", raising=False)
    monkeypatch.setattr(
        seed_tts_eval,
        "_missing_deps_message",
        lambda _lang: pytest.fail("export-only mode must not initialize ASR"),
    )
    request = SeedTTSSampleRequest(
        prompt="hello",
        prompt_len=1,
        seed_tts_utterance_id="seed_en_0002",
        seed_tts_locale="en",
    )
    output = SimpleNamespace(success=True, tts_output_pcm_bytes=b"\x01\x00\x02\x00")

    result = compute_seed_tts_wer_metrics([request], [output])

    assert result is not None
    assert result["seed_tts_eval_protocol"] == "seed-tts-official-export-only"
    assert result["seed_tts_official_exported"] == 1
    assert result["seed_tts_content_evaluated"] == 0
    assert (tmp_path / "seed_en_0002.wav").is_file()


@pytest.mark.parametrize("utterance_id", ["", ".", "..", "../escape", "nested/item", "nested\\item"])
def test_official_export_rejects_unsafe_utterance_id(tmp_path, utterance_id: str) -> None:
    with pytest.raises(ValueError, match="Unsafe Seed-TTS utterance id"):
        _save_seed_tts_official_audio(
            b"\x00\x00",
            output_dir=tmp_path,
            utterance_id=utterance_id,
        )


def test_zh_asr_initialization_disables_network_update_checks(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    model = object()

    def fake_auto_model(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "funasr",
        SimpleNamespace(AutoModel=fake_auto_model),
    )
    monkeypatch.setenv("SEED_TTS_EVAL_DEVICE", "cpu")
    monkeypatch.delenv("SEED_TTS_PARAFORMER_MODEL", raising=False)
    monkeypatch.setattr(seed_tts_eval, "_zh_paraformer", None)
    monkeypatch.setattr(seed_tts_eval, "_device", None)

    seed_tts_eval._ensure_zh_asr()

    assert seed_tts_eval._zh_paraformer is model
    assert calls == [
        {
            "model": seed_tts_eval.PARAFORMER_MODEL_ID,
            "device": "cpu",
            "disable_update": True,
        }
    ]


def test_zh_asr_accepts_offline_paraformer_path(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []
    model = object()

    def fake_auto_model(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "funasr",
        SimpleNamespace(AutoModel=fake_auto_model),
    )
    monkeypatch.setenv("SEED_TTS_EVAL_DEVICE", "cpu")
    monkeypatch.setenv("SEED_TTS_PARAFORMER_MODEL", str(tmp_path))
    monkeypatch.setattr(seed_tts_eval, "_zh_paraformer", None)
    monkeypatch.setattr(seed_tts_eval, "_device", None)

    seed_tts_eval._ensure_zh_asr()

    assert seed_tts_eval._zh_paraformer is model
    assert calls == [
        {
            "model": str(tmp_path),
            "device": "cpu",
            "disable_update": True,
        }
    ]
