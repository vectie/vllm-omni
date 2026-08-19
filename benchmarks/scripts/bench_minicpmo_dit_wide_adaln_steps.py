# SPDX-License-Identifier: Apache-2.0
"""Screen one all-step, all-block MiniCPM-o DiT AdaLN projection."""

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


def _wide_step(
    time_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output = F.linear(F.silu(time_embedding), weight, bias)
    return output.reshape(2, 1, _BLOCK_COUNT, _MODULATION_SIZE)


def _wide_all_steps(
    time_embeddings: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output = F.linear(F.silu(time_embeddings), weight, bias)
    return output.reshape(
        time_embeddings.shape[0],
        2,
        1,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = tuple(
        state[f"decoder.estimator.blocks.{index}.adaLN_modulation.1.weight"].to(device)
        for index in range(_BLOCK_COUNT)
    )
    biases = tuple(
        state[f"decoder.estimator.blocks.{index}.adaLN_modulation.1.bias"].to(device)
        for index in range(_BLOCK_COUNT)
    )
    packed_weight = torch.cat(weights, dim=0).contiguous()
    packed_bias = torch.cat(biases, dim=0).contiguous()
    time_embeddings = torch.linspace(
        -0.25,
        0.25,
        args.steps * 2 * _HIDDEN_SIZE,
        device=device,
        dtype=weights[0].dtype,
    ).reshape(args.steps, 2, 1, _HIDDEN_SIZE)

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    step_graph = torch.compile(
        _wide_step,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    all_steps_graph = torch.compile(
        _wide_all_steps,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )

    def control() -> tuple[torch.Tensor, ...]:
        return tuple(
            step_graph(time_embeddings[step], packed_weight, packed_bias)
            for step in range(args.steps)
        )

    def candidate() -> torch.Tensor:
        return all_steps_graph(time_embeddings, packed_weight, packed_bias)

    with torch.inference_mode():
        expected = torch.stack(control())
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
                "dtype": str(time_embeddings.dtype),
                "steps": args.steps,
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
