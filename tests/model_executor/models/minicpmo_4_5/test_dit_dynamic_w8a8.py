# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    _npu_dynamic_w8a8_linear,
    _quantize_dynamic_w8a8_weight,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dynamic_w8a8_weight_is_persistent_cube_orientation():
    weight = torch.linspace(-1.0, 1.0, 7 * 13).reshape(7, 13)

    quantized, scale = _quantize_dynamic_w8a8_weight(weight)

    assert quantized.shape == (13, 7)
    assert quantized.dtype == torch.int8
    assert quantized.is_contiguous()
    assert scale.shape == (7,)
    assert scale.dtype == torch.float32


def test_dynamic_w8a8_weight_roundtrip_is_per_output_channel():
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(32, 64, generator=generator)

    quantized, scale = _quantize_dynamic_w8a8_weight(weight)
    restored = quantized.transpose(0, 1).float() * scale[:, None]
    channel_error = (restored - weight).abs().amax(dim=1)

    assert torch.all(channel_error <= scale * 0.501)


def test_dynamic_w8a8_zero_weight_has_finite_scale_and_exact_roundtrip():
    weight = torch.zeros(4, 8)

    quantized, scale = _quantize_dynamic_w8a8_weight(weight)

    assert torch.count_nonzero(quantized) == 0
    assert torch.isfinite(scale).all()
    assert (quantized.transpose(0, 1).float() * scale[:, None]).equal(weight)


def test_dynamic_w8a8_linear_flattens_tokens_for_a2_scale_abi(monkeypatch):
    seen: dict[str, tuple[int, ...]] = {}

    def fake_dynamic_quant(x: torch.Tensor):
        seen["quant_input"] = tuple(x.shape)
        return torch.zeros_like(x, dtype=torch.int8), torch.ones(x.shape[0])

    def fake_quant_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        pertoken_scale: torch.Tensor,
        bias: torch.Tensor,
        output_dtype: torch.dtype,
    ):
        del scale, bias
        seen["matmul_input"] = tuple(x.shape)
        seen["pertoken_scale"] = tuple(pertoken_scale.shape)
        return torch.zeros((x.shape[0], weight.shape[-1]), dtype=output_dtype)

    monkeypatch.setattr(torch.ops.npu, "npu_dynamic_quant", fake_dynamic_quant)
    monkeypatch.setattr(torch.ops.npu, "npu_quant_matmul", fake_quant_matmul)

    output = _npu_dynamic_w8a8_linear(
        torch.randn(2, 50, 16, dtype=torch.bfloat16),
        torch.zeros(16, 32, dtype=torch.int8),
        torch.ones(32),
        torch.zeros(32, dtype=torch.bfloat16),
    )

    assert output.shape == (2, 50, 32)
    assert seen == {
        "quant_input": (100, 16),
        "matmul_input": (100, 16),
        "pertoken_scale": (100,),
    }

