# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
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

