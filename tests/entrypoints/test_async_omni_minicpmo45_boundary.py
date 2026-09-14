# SPDX-License-Identifier: Apache-2.0
"""Engine-boundary regressions for MiniCPM-o 4.5 audio handoff."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _params():
    return [
        SamplingParams(output_kind=RequestOutputKind.DELTA),
        SamplingParams(output_kind=RequestOutputKind.FINAL_ONLY),
        SimpleNamespace(),
    ]


def _stages(arch: str = "MiniCPMO45OmniForConditionalGeneration"):
    return [
        SimpleNamespace(engine_args=SimpleNamespace(model_arch=arch, model_stage="llm")),
        {"engine_args": {"model_arch": arch, "model_stage": "tts"}},
        SimpleNamespace(engine_args=SimpleNamespace(model_arch="MiniCPMO45Code2Wav", model_stage="code2wav")),
    ]


def test_audio_request_enforces_complete_thinker_at_engine_boundary():
    params = _params()

    result = AsyncOmni._enforce_minicpmo45_audio_stage_boundary(
        params,
        _stages(),
        ["text", "audio"],
    )

    assert result[0].output_kind is RequestOutputKind.FINAL_ONLY
    assert result[1].output_kind is RequestOutputKind.DELTA


@pytest.mark.parametrize("modalities", [None, [], ["text"]])
def test_non_audio_request_keeps_existing_output_kinds(modalities):
    params = _params()

    AsyncOmni._enforce_minicpmo45_audio_stage_boundary(params, _stages(), modalities)

    assert params[0].output_kind is RequestOutputKind.DELTA
    assert params[1].output_kind is RequestOutputKind.FINAL_ONLY


def test_other_pipeline_is_untouched():
    params = _params()

    AsyncOmni._enforce_minicpmo45_audio_stage_boundary(
        params,
        _stages("SomeOtherOmniForConditionalGeneration"),
        ["audio"],
    )

    assert params[0].output_kind is RequestOutputKind.DELTA
    assert params[1].output_kind is RequestOutputKind.FINAL_ONLY
