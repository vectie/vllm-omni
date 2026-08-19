# SPDX-License-Identifier: Apache-2.0
"""Screen one wide Cube projection for all MiniCPM-o DiT AdaLN blocks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    _ensure_torchair_broadcast_alias,
)

_BLOCK_COUNT = 16
_HIDDEN_SIZE = 512
_MODULATION_SIZE = 9 * _HIDDEN_SIZE


def _one_modulation(
    time_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.silu(time_embedding), weight, bias)


def _wide_modulations(
    time_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output = F.linear(F.silu(time_embedding), weight, bias)
    return output.reshape(
        time_embedding.shape[0],
        time_embedding.shape[1],
        _BLOCK_COUNT,
        _MODULATION_SIZE,
    )


def _measure_us(function: Callable[[], object], iterations: int) -> float:
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1_000


def _load_weights(
    checkpoint: Path,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = tuple(
        state[f"decoder.estimator.blocks.{index}.adaLN_modulation.1.weight"].to(device)
        for index in range(_BLOCK_COUNT)
    )
    biases = tuple(
        state[f"decoder.estimator.blocks.{index}.adaLN_modulation.1.bias"].to(device)
        for index in range(_BLOCK_COUNT)
    )
    return weights, biases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    weights, biases = _load_weights(args.checkpoint, device)
    packed_weight = torch.cat(weights, dim=0).contiguous()
    packed_bias = torch.cat(biases, dim=0).contiguous()
    time_embedding = weights[0].new_full((2, 1, _HIDDEN_SIZE), 0.125)

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    one_graph = torch.compile(
        _one_modulation,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    wide_graph = torch.compile(
        _wide_modulations,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )

    def control() -> None:
        for weight, bias in zip(weights, biases, strict=True):
            one_graph(time_embedding, weight, bias)

    def candidate() -> torch.Tensor:
        return wide_graph(time_embedding, packed_weight, packed_bias)

    with torch.inference_mode():
        expected = torch.stack(
            [
                one_graph(time_embedding, weight, bias)
                for weight, bias in zip(weights, biases, strict=True)
            ],
            dim=2,
        )
        actual = candidate()
        for _ in range(args.warmups):
            control()
            candidate()
        torch.npu.synchronize()

        control_trials: list[float] = []
        candidate_trials: list[float] = []
        for trial in range(args.trials):
            if trial % 2:
                candidate_trials.append(_measure_us(candidate, args.iterations))
                control_trials.append(_measure_us(control, args.iterations))
            else:
                control_trials.append(_measure_us(control, args.iterations))
                candidate_trials.append(_measure_us(candidate, args.iterations))

    control_median = statistics.median(control_trials)
    candidate_median = statistics.median(candidate_trials)
    difference = (actual - expected).abs()
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": str(time_embedding.dtype),
                "blocks": _BLOCK_COUNT,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "trials": args.trials,
                "control_median_us": control_median,
                "candidate_median_us": candidate_median,
                "speedup": control_median / candidate_median,
                "control_trials_us": control_trials,
                "candidate_trials_us": candidate_trials,
                "max_abs_error": float(difference.max().item()),
                "mean_abs_error": float(difference.mean().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
