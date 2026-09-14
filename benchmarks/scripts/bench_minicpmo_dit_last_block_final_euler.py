# SPDX-License-Identifier: Apache-2.0
"""Benchmark the 910C last-DiT-block-to-CFM-Euler GE replay."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    _dit_final_cfg_euler_from_modulation,
    _dit_fused_conv_mlp_final_euler_residual,
    _dit_fused_conv_mlp_residual,
    _ensure_torchair_broadcast_alias,
)


def _measure_us(function: Callable[[], object], iterations: int) -> float:
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1_000


def _weights(checkpoint: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "decoder.estimator.blocks.15"
    conv1 = state[f"{prefix}.conv.block.1.weight"]
    conv2 = state[f"{prefix}.conv.block.6.weight"]
    values = (
        conv1.permute(0, 2, 1).reshape(512, 1536).contiguous(),
        state[f"{prefix}.conv.block.1.bias"],
        state[f"{prefix}.conv.block.3.weight"],
        state[f"{prefix}.conv.block.3.bias"],
        conv2.permute(0, 2, 1).reshape(512, 1536).contiguous(),
        state[f"{prefix}.conv.block.6.bias"],
        state[f"{prefix}.mlp.fc1.weight"],
        state[f"{prefix}.mlp.fc1.bias"],
        state[f"{prefix}.mlp.fc2.weight"],
        state[f"{prefix}.mlp.fc2.bias"],
        state["decoder.estimator.final_layer.linear.weight"],
        state["decoder.estimator.final_layer.linear.bias"],
    )
    return tuple(value.to(device=device) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    weights = _weights(args.checkpoint, device)
    base_weights = weights[:10]
    final_weight, final_bias = weights[10:]
    dtype = weights[0].dtype
    hidden = torch.full((2, 50, 512), 0.125, device=device, dtype=dtype)
    conv_input = torch.full_like(hidden, -0.25)
    cnn_cache = torch.full((2, 1024, 2), 0.0625, device=device, dtype=dtype)
    modulation = torch.full((2, 1, 512), 0.03125, device=device, dtype=dtype)
    final_modulation = torch.full((2, 1, 1024), 0.015625, device=device, dtype=dtype)
    state = torch.full((1, 80, 50), -0.125, device=device, dtype=dtype)
    delta = torch.tensor(0.125, device=device, dtype=dtype)
    cfg_rate = 0.7
    base_args = (
        hidden,
        conv_input,
        cnn_cache,
        modulation,
        modulation,
        modulation,
        modulation,
        *base_weights,
    )

    from torch_npu.dynamo import torchair
    from vllm_ascend.compilation.minicpmo_causal_conv import (
        register_minicpmo_causal_conv_pack_converter,
    )
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    register_minicpmo_causal_conv_pack_converter()
    _ensure_torchair_broadcast_alias()
    base_graph = torch.compile(
        _dit_fused_conv_mlp_residual,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    fused_graph = torch.compile(
        _dit_fused_conv_mlp_final_euler_residual,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )

    def control() -> tuple[torch.Tensor, torch.Tensor]:
        output, cache = base_graph(*base_args)
        return (
            _dit_final_cfg_euler_from_modulation(
                output,
                final_modulation,
                final_weight,
                final_bias,
                state,
                delta,
                cfg_rate,
            ),
            cache,
        )

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_graph(
            *base_args,
            final_modulation,
            final_weight,
            final_bias,
            state,
            delta,
            cfg_rate,
        )

    with torch.inference_mode():
        expected_x, expected_cache = control()
        actual_x, actual_cache = candidate()
        for _ in range(args.warmups):
            control()
            candidate()
        control_trials: list[float] = []
        candidate_trials: list[float] = []
        for trial in range(args.trials):
            first, second = (candidate, control) if trial % 2 else (control, candidate)
            first_result = _measure_us(first, args.iterations)
            second_result = _measure_us(second, args.iterations)
            if trial % 2:
                candidate_trials.append(first_result)
                control_trials.append(second_result)
            else:
                control_trials.append(first_result)
                candidate_trials.append(second_result)

    control_median = statistics.median(control_trials)
    candidate_median = statistics.median(candidate_trials)
    difference = (actual_x - expected_x).abs()
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": str(hidden.dtype),
                "control_median_us": control_median,
                "candidate_median_us": candidate_median,
                "speedup": control_median / candidate_median,
                "control_trials_us": control_trials,
                "candidate_trials_us": candidate_trials,
                "x_max_abs_error": float(difference.max().item()),
                "x_mean_abs_error": float(difference.mean().item()),
                "cache_max_abs_error": float(
                    (actual_cache - expected_cache).abs().max().item()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
