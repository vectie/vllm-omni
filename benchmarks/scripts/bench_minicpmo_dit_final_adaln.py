# SPDX-License-Identifier: Apache-2.0
"""Screen all-step AdaLN packing through MiniCPM-o's DiT final layer."""

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
_BLOCK_MODULATION = 9 * _HIDDEN_SIZE
_FINAL_MODULATION = 2 * _HIDDEN_SIZE


def _all_adaln_steps(
    time_embeddings: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.silu(time_embeddings), packed_weight, packed_bias)


def _final_control(
    hidden: torch.Tensor,
    time_embedding: torch.Tensor,
    modulation_weight: torch.Tensor,
    modulation_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    modulation = F.linear(
        F.silu(time_embedding),
        modulation_weight,
        modulation_bias,
    )
    return _final_from_modulation(
        hidden,
        modulation,
        output_weight,
        output_bias,
    )


def _final_from_modulation(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    shift, scale = modulation.chunk(2, dim=-1)
    hidden = F.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6)
    hidden = hidden * (1 + scale) + shift
    return F.linear(hidden, output_weight, output_bias)


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
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    block_weights = tuple(
        state[f"decoder.estimator.blocks.{index}.adaLN_modulation.1.weight"].to(device)
        for index in range(_BLOCK_COUNT)
    )
    block_biases = tuple(
        state[f"decoder.estimator.blocks.{index}.adaLN_modulation.1.bias"].to(device)
        for index in range(_BLOCK_COUNT)
    )
    final_weight = state["decoder.estimator.final_layer.adaLN_modulation.1.weight"].to(device)
    final_bias = state["decoder.estimator.final_layer.adaLN_modulation.1.bias"].to(device)
    output_weight = state["decoder.estimator.final_layer.linear.weight"].to(device)
    output_bias = state["decoder.estimator.final_layer.linear.bias"].to(device)
    packed_block_weight = torch.cat(block_weights, dim=0).contiguous()
    packed_block_bias = torch.cat(block_biases, dim=0).contiguous()
    packed_all_weight = torch.cat((*block_weights, final_weight), dim=0).contiguous()
    packed_all_bias = torch.cat((*block_biases, final_bias), dim=0).contiguous()
    time_embeddings = torch.linspace(
        -0.25,
        0.25,
        args.steps * 2 * _HIDDEN_SIZE,
        device=device,
        dtype=final_weight.dtype,
    ).reshape(args.steps, 2, 1, _HIDDEN_SIZE)
    hidden = torch.linspace(
        -0.125,
        0.125,
        args.steps * 2 * args.width * _HIDDEN_SIZE,
        device=device,
        dtype=final_weight.dtype,
    ).reshape(args.steps, 2, args.width, _HIDDEN_SIZE)

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    all_adaln_graph = torch.compile(
        _all_adaln_steps,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    final_graph = torch.compile(
        _final_from_modulation,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )

    def control() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        blocks = all_adaln_graph(
            time_embeddings,
            packed_block_weight,
            packed_block_bias,
        )
        finals = tuple(
            _final_control(
                hidden[step],
                time_embeddings[step],
                final_weight,
                final_bias,
                output_weight,
                output_bias,
            )
            for step in range(args.steps)
        )
        return blocks, finals

    def candidate(*, use_final_graph: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        all_modulations = all_adaln_graph(
            time_embeddings,
            packed_all_weight,
            packed_all_bias,
        )
        blocks = all_modulations[..., : _BLOCK_COUNT * _BLOCK_MODULATION]
        final_modulations = all_modulations[
            ..., _BLOCK_COUNT * _BLOCK_MODULATION :
        ]
        final_function = final_graph if use_final_graph else _final_from_modulation
        finals = tuple(
            final_function(
                hidden[step],
                final_modulations[step],
                output_weight,
                output_bias,
            )
            for step in range(args.steps)
        )
        return blocks, finals

    with torch.inference_mode():
        expected_blocks, expected_finals = control()
        actual_blocks, actual_finals = candidate(use_final_graph=False)
        graph_blocks, graph_finals = candidate(use_final_graph=True)
        for _ in range(args.warmups):
            control()
            candidate(use_final_graph=False)
            candidate(use_final_graph=True)
        calls = {
            "control": control,
            "candidate_eager_final": lambda: candidate(use_final_graph=False),
            "candidate_graph_final": lambda: candidate(use_final_graph=True),
        }
        trials = {name: [] for name in calls}
        orders = (
            tuple(calls),
            tuple(reversed(calls)),
            ("candidate_eager_final", "control", "candidate_graph_final"),
        )
        for trial in range(args.trials):
            for name in orders[trial % len(orders)]:
                trials[name].append(_measure_us(calls[name], args.iterations))

    medians = {name: statistics.median(values) for name, values in trials.items()}
    final_difference = torch.stack(actual_finals) - torch.stack(expected_finals)
    graph_final_difference = torch.stack(graph_finals) - torch.stack(expected_finals)
    block_difference = actual_blocks - expected_blocks
    graph_block_difference = graph_blocks - expected_blocks
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": str(final_weight.dtype),
                "steps": args.steps,
                "width": args.width,
                "median_us": medians,
                "speedup_vs_control": {
                    name: medians["control"] / value
                    for name, value in medians.items()
                },
                "trials_us": trials,
                "block_max_abs_error": float(block_difference.abs().max().item()),
                "final_max_abs_error": float(final_difference.abs().max().item()),
                "final_mean_abs_error": float(final_difference.abs().mean().item()),
                "graph_block_max_abs_error": float(
                    graph_block_difference.abs().max().item()
                ),
                "graph_final_max_abs_error": float(
                    graph_final_difference.abs().max().item()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
