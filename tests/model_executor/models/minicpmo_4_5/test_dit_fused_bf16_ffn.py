# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
    _dit_flat_capture_conv_mlp_partition,
    _dit_fused_conv_bf16_ffn_residual,
    _npu_bf16_ffn,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bf16_ffn_flattens_bsh_and_preserves_residual_shape(monkeypatch):
    seen: dict[str, object] = {}

    def fake_npu_ffn(
        x: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        activation: str,
        *,
        bias1: torch.Tensor,
        bias2: torch.Tensor,
        inner_precise: int,
    ):
        seen.update(
            x_shape=tuple(x.shape),
            weight1_shape=tuple(weight1.shape),
            weight2_shape=tuple(weight2.shape),
            activation=activation,
            bias1_dtype=bias1.dtype,
            bias2_dtype=bias2.dtype,
            inner_precise=inner_precise,
        )
        return torch.zeros((x.shape[0], weight2.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(torch.ops.npu, "npu_ffn", fake_npu_ffn)

    output = _npu_bf16_ffn(
        torch.randn(2, 50, 16, dtype=torch.bfloat16),
        torch.zeros(16, 64, dtype=torch.bfloat16),
        torch.zeros(64, dtype=torch.float32),
        torch.zeros(64, 16, dtype=torch.bfloat16),
        torch.zeros(16, dtype=torch.float32),
    )

    assert output.shape == (2, 50, 16)
    assert seen == {
        "x_shape": (100, 16),
        "weight1_shape": (16, 64),
        "weight2_shape": (64, 16),
        "activation": "gelu",
        "bias1_dtype": torch.float32,
        "bias2_dtype": torch.float32,
        "inner_precise": 0,
    }


def test_bf16_ffn_preparation_materializes_cube_orientation_once():
    module = nn.Module()
    module.fc1 = nn.Linear(16, 64, bias=True, dtype=torch.bfloat16)
    module.fc2 = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    estimator = SimpleNamespace(blocks=[SimpleNamespace(mlp=module)])

    BatchedToken2Wav._prepare_npu_dit_fused_bf16_ffn_weights(estimator)

    assert module._minicpmo_bf16_ffn_fc1_weight_kn.shape == (16, 64)
    assert module._minicpmo_bf16_ffn_fc2_weight_kn.shape == (64, 16)
    assert module._minicpmo_bf16_ffn_fc1_weight_kn.is_contiguous()
    assert module._minicpmo_bf16_ffn_fc2_weight_kn.is_contiguous()
    assert module._minicpmo_bf16_ffn_fc1_bias_fp32.dtype == torch.float32
    assert module._minicpmo_bf16_ffn_fc2_bias_fp32.dtype == torch.float32


def test_flat_capture_selects_native_bf16_ffn_partition():
    selected = _dit_flat_capture_conv_mlp_partition(
        fused_conv_pack=True,
        cache_major=False,
        post_attention=False,
        fused_bf16_ffn=True,
    )

    assert selected is _dit_fused_conv_bf16_ffn_residual
