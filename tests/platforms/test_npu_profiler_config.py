# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Metrics:
    AiCoreNone = object()
    PipeUtilization = object()


def test_resolve_aic_metrics_default(monkeypatch):
    from vllm_omni.platforms.npu.profiler import _resolve_aic_metrics

    monkeypatch.delenv("VLLM_OMNI_NPU_PROFILER_AIC_METRICS", raising=False)
    torch_npu = SimpleNamespace(profiler=SimpleNamespace(AiCMetrics=_Metrics))
    assert _resolve_aic_metrics(torch_npu) is _Metrics.AiCoreNone


def test_resolve_aic_metrics_case_insensitive(monkeypatch):
    from vllm_omni.platforms.npu.profiler import _resolve_aic_metrics

    monkeypatch.setenv("VLLM_OMNI_NPU_PROFILER_AIC_METRICS", "pipeutilization")
    torch_npu = SimpleNamespace(profiler=SimpleNamespace(AiCMetrics=_Metrics))
    assert _resolve_aic_metrics(torch_npu) is _Metrics.PipeUtilization


def test_resolve_aic_metrics_rejects_unknown(monkeypatch):
    from vllm_omni.platforms.npu.profiler import _resolve_aic_metrics

    monkeypatch.setenv("VLLM_OMNI_NPU_PROFILER_AIC_METRICS", "not-a-counter")
    torch_npu = SimpleNamespace(profiler=SimpleNamespace(AiCMetrics=_Metrics))
    with pytest.raises(ValueError, match="not-a-counter"):
        _resolve_aic_metrics(torch_npu)
