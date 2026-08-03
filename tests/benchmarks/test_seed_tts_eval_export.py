from __future__ import annotations

import wave

import pytest

from vllm_omni.benchmarks.data_modules.seed_tts_eval import _save_seed_tts_official_audio

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


@pytest.mark.parametrize("utterance_id", ["", ".", "..", "../escape", "nested/item", "nested\\item"])
def test_official_export_rejects_unsafe_utterance_id(tmp_path, utterance_id: str) -> None:
    with pytest.raises(ValueError, match="Unsafe Seed-TTS utterance id"):
        _save_seed_tts_official_audio(
            b"\x00\x00",
            output_dir=tmp_path,
            utterance_id=utterance_id,
        )
