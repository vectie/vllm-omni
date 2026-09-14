# SPDX-License-Identifier: Apache-2.0
"""Screen frozen DiT Conv+MLP graph weights on an Ascend NPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
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


def _block_weights(
    checkpoint: Path,
    *,
    block: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = f"decoder.estimator.blocks.{block}"
    conv1 = state[f"{prefix}.conv.block.1.weight"]
    conv2 = state[f"{prefix}.conv.block.6.weight"]
    weights = (
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
    )
    return tuple(weight.to(device) for weight in weights)


def _mark_frozen(weights: tuple[torch.Tensor, ...]) -> None:
    for weight in weights:
        torch._dynamo.mark_static_address(weight, guard=True)


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
    dynamic_weights = _block_weights(
        args.checkpoint,
        block=args.block,
        device=device,
    )
    frozen_weights = tuple(weight.detach().clone() for weight in dynamic_weights)

    hidden = dynamic_weights[0].new_full((2, 50, 512), 0.125)
    conv_input = dynamic_weights[0].new_full((2, 50, 512), -0.25)
    cnn_cache = dynamic_weights[0].new_full((2, 1024, 2), 0.0625)
    modulation = dynamic_weights[0].new_full((2, 1, 512), 0.03125)
    graph_args = (
        hidden,
        conv_input,
        cnn_cache,
        modulation,
        modulation,
        modulation,
        modulation,
    )

    from torch_npu.dynamo import torchair
    from vllm_ascend.compilation.minicpmo_causal_conv import (
        register_minicpmo_causal_conv_pack_converter,
    )
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    register_minicpmo_causal_conv_pack_converter()
    _ensure_torchair_broadcast_alias()

    dynamic_graph = torch.compile(
        _dit_fused_conv_mlp_residual,
        backend=torchair.get_npu_backend(
            compiler_config=torchair.CompilerConfig()
        ),
        fullgraph=True,
        dynamic=False,
    )
    _mark_frozen(frozen_weights)
    frozen_config = torchair.CompilerConfig()
    frozen_config.experimental_config.frozen_parameter.value = True
    frozen_graph = torch.compile(
        _dit_fused_conv_mlp_residual,
        backend=torchair.get_npu_backend(compiler_config=frozen_config),
        fullgraph=True,
        dynamic=False,
    )

    def control() -> tuple[torch.Tensor, torch.Tensor]:
        return dynamic_graph(*graph_args, *dynamic_weights)

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return frozen_graph(*graph_args, *frozen_weights)

    with torch.inference_mode():
        expected_hidden, expected_cache = control()
        actual_hidden, actual_cache = candidate()
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
    print(
        json.dumps(
            {
                "device": args.device,
                "block": args.block,
                "dtype": str(hidden.dtype),
                "warmups": args.warmups,
                "iterations": args.iterations,
                "trials": args.trials,
                "control_median_us": control_median,
                "candidate_median_us": candidate_median,
                "speedup": control_median / candidate_median,
                "control_trials_us": control_trials,
                "candidate_trials_us": candidate_trials,
                "hidden_max_abs_error": float(
                    (actual_hidden - expected_hidden).abs().max().item()
                ),
                "cache_max_abs_error": float(
                    (actual_cache - expected_cache).abs().max().item()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
