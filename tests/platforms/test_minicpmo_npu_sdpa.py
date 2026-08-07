# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_npu_sdpa_auto_uses_sticky_math_fallback(monkeypatch):
    from vllm_omni.platforms.npu.models import cosyvoice2_dit_attn as adapter

    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_SDPA_BACKEND", raising=False)
    monkeypatch.setattr(adapter, "_FUSED_SDPA_AVAILABLE", None)
    expected = torch.ones(1, 1, 2, 4)
    calls = 0

    def fake_sdpa(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ACL 161001")
        return expected

    monkeypatch.setattr(adapter.F, "scaled_dot_product_attention", fake_sdpa)
    q = torch.zeros(1, 1, 2, 4)
    assert adapter._npu_sdpa(q, q, q, attn_mask=None) is expected
    assert adapter._FUSED_SDPA_AVAILABLE is False
    assert adapter._npu_sdpa(q, q, q, attn_mask=None) is expected
    assert calls == 3


def test_npu_sdpa_fused_policy_propagates_failure(monkeypatch):
    from vllm_omni.platforms.npu.models import cosyvoice2_dit_attn as adapter

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_SDPA_BACKEND", "fused")
    monkeypatch.setattr(
        adapter.F, "scaled_dot_product_attention", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    q = torch.zeros(1, 1, 2, 4)
    with pytest.raises(RuntimeError, match="bad"):
        adapter._npu_sdpa(q, q, q, attn_mask=None)


def test_npu_math_context_propagates_model_failure():
    from vllm_omni.platforms.npu.models.cosyvoice2_dit_attn import (
        npu_math_sdpa_context,
    )

    with pytest.raises(RuntimeError, match="model failed"):
        with npu_math_sdpa_context():
            raise RuntimeError("model failed")


def test_token2wav_context_propagates_model_failure():
    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    with pytest.raises(RuntimeError, match="model failed"):
        with npu_token2wav_sdpa_context():
            raise RuntimeError("model failed")
