# SPDX-License-Identifier: Apache-2.0
"""Screen a CFM-invariant factorization of MiniCPM-o's DiT input projection."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F


def _control_step(
    state: torch.Tensor,
    invariant: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    estimator_input = torch.cat((state, invariant), dim=1)
    return F.linear(estimator_input.transpose(1, 2), weight, bias)


def _invariant_projection(
    invariant: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return F.linear(invariant.transpose(1, 2), weight, bias)


def _state_projection(
    state: torch.Tensor,
    weight: torch.Tensor,
    invariant_projection: torch.Tensor,
) -> torch.Tensor:
    return F.linear(state.transpose(1, 2), weight) + invariant_projection


def _invariant_projection_btc(
    invariant: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    batch, width, channels = invariant.shape
    return F.linear(invariant.reshape(batch * width, channels), weight, bias).reshape(
        batch,
        width,
        -1,
    )


def _state_projection_btc(
    state: torch.Tensor,
    weight: torch.Tensor,
    invariant_projection: torch.Tensor,
) -> torch.Tensor:
    batch, width, channels = state.shape
    projected = F.linear(state.reshape(batch * width, channels), weight).reshape(
        batch,
        width,
        -1,
    )
    return projected + invariant_projection


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
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weight = state_dict["decoder.estimator.in_proj.weight"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    bias = state_dict["decoder.estimator.in_proj.bias"].to(
        device=device,
        dtype=torch.bfloat16,
    )
    state_channels = 80
    if tuple(weight.shape) != (512, 320):
        raise ValueError(f"expected MiniCPM-o in_proj [512,320], got {tuple(weight.shape)}")
    state_weight = weight[:, :state_channels].contiguous()
    invariant_weight = weight[:, state_channels:].contiguous()
    states = torch.linspace(
        -0.25,
        0.25,
        args.steps * 2 * state_channels * args.width,
        device=device,
        dtype=torch.bfloat16,
    ).reshape(args.steps, 2, state_channels, args.width)
    invariant = torch.linspace(
        0.125,
        -0.125,
        2 * 240 * args.width,
        device=device,
        dtype=torch.bfloat16,
    ).reshape(2, 240, args.width)
    states_btc = tuple(
        states[step].transpose(1, 2).contiguous()
        for step in range(args.steps)
    )
    invariant_btc = invariant.transpose(1, 2).contiguous()

    def control() -> tuple[torch.Tensor, ...]:
        return tuple(
            _control_step(states[step], invariant, weight, bias)
            for step in range(args.steps)
        )

    def factorized() -> tuple[torch.Tensor, ...]:
        fixed = _invariant_projection(invariant, invariant_weight, bias)
        return tuple(
            _state_projection(states[step], state_weight, fixed)
            for step in range(args.steps)
        )

    from torch_npu.dynamo import torchair

    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    invariant_graph = torch.compile(
        _invariant_projection_btc,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    state_graph = torch.compile(
        _state_projection_btc,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )

    def factorized_graph() -> tuple[torch.Tensor, ...]:
        fixed = invariant_graph(invariant_btc, invariant_weight, bias)
        return tuple(
            state_graph(states_btc[step], state_weight, fixed)
            for step in range(args.steps)
        )

    calls = {
        "control": control,
        "factorized_eager": factorized,
        "factorized_graph": factorized_graph,
    }
    with torch.inference_mode():
        expected = control()
        outputs = {name: call() for name, call in calls.items()}
        for _ in range(args.warmups):
            for call in calls.values():
                call()
        torch.npu.synchronize()
        trials = {name: [] for name in calls}
        orders = (
            tuple(calls),
            tuple(reversed(calls)),
            ("factorized_eager", "control", "factorized_graph"),
        )
        for trial in range(args.trials):
            for name in orders[trial % len(orders)]:
                trials[name].append(_measure_us(calls[name], args.iterations))

    medians = {name: statistics.median(values) for name, values in trials.items()}
    errors = {}
    for name, output in outputs.items():
        difference = torch.stack(output) - torch.stack(expected)
        errors[name] = {
            "max_abs": float(difference.abs().max().item()),
            "mean_abs": float(difference.abs().float().mean().item()),
        }
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": str(weight.dtype),
                "steps": args.steps,
                "width": args.width,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "trials": args.trials,
                "median_us": medians,
                "speedup_vs_control": {
                    name: medians["control"] / value
                    for name, value in medians.items()
                },
                "trials_us": trials,
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
