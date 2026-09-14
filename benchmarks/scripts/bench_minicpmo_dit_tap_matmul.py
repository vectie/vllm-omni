# SPDX-License-Identifier: Apache-2.0
"""Screen GE-visible no-pack formulations for MiniCPM-o DiT causal Conv."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F


def _native_pack_linear(
    hidden: torch.Tensor,
    cache: torch.Tensor,
    flat_weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed, new_cache = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        hidden,
        cache,
    )
    return F.linear(packed, flat_weight, bias).reshape(2, 50, 512), new_cache


def _tap_inputs(
    hidden: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    extended = torch.cat((cache.transpose(1, 2), hidden), dim=1)
    return (
        extended[:, :50, :],
        extended[:, 1:51, :],
        extended[:, 2:52, :],
        extended[:, -2:, :].transpose(1, 2),
    )


def _three_tap_linear(
    hidden: torch.Tensor,
    cache: torch.Tensor,
    weight0: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tap0, tap1, tap2, new_cache = _tap_inputs(hidden, cache)
    output = F.linear(tap0.reshape(100, 512), weight0, bias)
    output = output + F.linear(tap1.reshape(100, 512), weight1)
    output = output + F.linear(tap2.reshape(100, 512), weight2)
    return output.reshape(2, 50, 512), new_cache


def _batched_tap_matmul(
    hidden: torch.Tensor,
    cache: torch.Tensor,
    tap_weights: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tap0, tap1, tap2, new_cache = _tap_inputs(hidden, cache)
    taps = torch.stack(
        (
            tap0.reshape(100, 512),
            tap1.reshape(100, 512),
            tap2.reshape(100, 512),
        ),
        dim=0,
    )
    output = torch.bmm(taps, tap_weights).sum(dim=0)
    return (output + bias).reshape(2, 50, 512), new_cache


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
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    prefix = f"decoder.estimator.blocks.{args.block}.conv.block.1"
    weight = state[f"{prefix}.weight"].to(device)
    bias = state[f"{prefix}.bias"].to(device)
    flat_weight = weight.permute(0, 2, 1).reshape(512, 1536).contiguous()
    tap_weights = tuple(weight[:, :, tap].contiguous() for tap in range(3))
    batched_weights = weight.permute(2, 1, 0).contiguous()
    hidden = weight.new_full((2, 50, 512), 0.125)
    cache = weight.new_full((2, 512, 2), -0.0625)

    from torch_npu.dynamo import torchair
    from vllm_ascend.compilation.minicpmo_causal_conv import (
        register_minicpmo_causal_conv_pack_converter,
    )
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    register_minicpmo_causal_conv_pack_converter()
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    graphs = {
        "control": torch.compile(
            _native_pack_linear,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        ),
        "three_linear": torch.compile(
            _three_tap_linear,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        ),
        "batched_matmul": torch.compile(
            _batched_tap_matmul,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        ),
    }
    calls = {
        "control": lambda: graphs["control"](hidden, cache, flat_weight, bias),
        "three_linear": lambda: graphs["three_linear"](
            hidden,
            cache,
            *tap_weights,
            bias,
        ),
        "batched_matmul": lambda: graphs["batched_matmul"](
            hidden,
            cache,
            batched_weights,
            bias,
        ),
    }

    with torch.inference_mode():
        expected_output, expected_cache = calls["control"]()
        outputs = {name: call() for name, call in calls.items()}
        for _ in range(args.warmups):
            for call in calls.values():
                call()
        torch.npu.synchronize()

        trials = {name: [] for name in calls}
        orders = (
            tuple(calls),
            tuple(reversed(calls)),
            ("three_linear", "control", "batched_matmul"),
        )
        for trial in range(args.trials):
            for name in orders[trial % len(orders)]:
                trials[name].append(_measure_us(calls[name], args.iterations))

    medians = {name: statistics.median(values) for name, values in trials.items()}
    print(
        json.dumps(
            {
                "device": args.device,
                "block": args.block,
                "dtype": str(hidden.dtype),
                "warmups": args.warmups,
                "iterations": args.iterations,
                "trials": args.trials,
                "median_us": medians,
                "speedup_vs_control": {
                    name: medians["control"] / median
                    for name, median in medians.items()
                },
                "trials_us": trials,
                "max_abs_error": {
                    name: float((output - expected_output).abs().max().item())
                    for name, (output, _) in outputs.items()
                },
                "cache_max_abs_error": {
                    name: float((new_cache - expected_cache).abs().max().item())
                    for name, (_, new_cache) in outputs.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
