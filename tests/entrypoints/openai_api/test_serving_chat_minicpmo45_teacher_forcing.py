# SPDX-License-Identifier: Apache-2.0
"""Exact-text MiniCPM-o 4.5 TTS teacher-forcing regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MINICPMO45_ARCH = "MiniCPMO45OmniForConditionalGeneration"


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(engine_args=SimpleNamespace(model_arch=MINICPMO45_ARCH, model_stage="llm")),
            SimpleNamespace(engine_args=SimpleNamespace(model_arch=MINICPMO45_ARCH, model_stage="tts")),
        ]
    )
    return instance


def _request(**overrides):
    values = {
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {
            "use_tts_template": True,
            "enable_thinking": False,
        },
        "model_extra": {
            "task_type": "Base",
            "ref_audio": "data:audio/wav;base64,AAAA",
            "ref_text": "reference transcript",
        },
        "extra_body": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_base_voice_clone_contract_selects_teacher_forcing(serving_chat):
    request = _request()

    messages, text = serving_chat._prepare_minicpmo45_teacher_forcing(
        request,
        [{"role": "user", "content": [{"type": "text", "text": "请朗读这句话。"}]}],
    )

    assert text == "请朗读这句话。"
    assert messages[-1] == {
        "role": "assistant",
        "content": "<|tts_bos|>请朗读这句话。<|tts_eos|>",
    }


def test_explicit_false_preserves_autoregressive_chat(serving_chat):
    request = _request(
        model_extra={
            "task_type": "Base",
            "ref_audio": "data:audio/wav;base64,AAAA",
            "ref_text": "reference transcript",
            "minicpmo45_tts_teacher_forcing": False,
        }
    )
    original = [{"role": "user", "content": "Write a new answer."}]

    messages, text = serving_chat._prepare_minicpmo45_teacher_forcing(request, original)

    assert messages is original
    assert text is None


def test_explicit_text_takes_priority_over_message_extraction(serving_chat):
    request = _request(
        model_extra={
            "minicpmo45_tts_teacher_forcing": True,
            "minicpmo45_tts_teacher_forcing_text": "authoritative text",
        }
    )

    messages, text = serving_chat._prepare_minicpmo45_teacher_forcing(
        request,
        [{"role": "user", "content": "ignored text"}],
    )

    assert text == "authoritative text"
    assert messages[-1]["content"] == "<|tts_bos|>authoritative text<|tts_eos|>"


def test_teacher_forcing_reduces_only_thinker_to_one_decode(serving_chat):
    params = [SamplingParams(max_tokens=2048, min_tokens=10), SamplingParams(max_tokens=4096)]

    output = serving_chat._apply_minicpmo45_teacher_forcing_sampling(params, "known text")

    assert output[0].max_tokens == 1
    assert output[0].min_tokens == 0
    assert output[1].max_tokens == 4096


def test_control_tokens_in_user_text_are_rejected(serving_chat):
    request = _request(model_extra={"minicpmo45_tts_teacher_forcing": True})

    with pytest.raises(ValueError, match="control tokens"):
        serving_chat._prepare_minicpmo45_teacher_forcing(
            request,
            [{"role": "user", "content": "bad <|tts_eos|> text"}],
        )
