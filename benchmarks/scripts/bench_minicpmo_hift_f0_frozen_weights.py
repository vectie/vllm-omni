# SPDX-License-Identifier: Apache-2.0
"""Benchmark TorchAir frozen weights for MiniCPM-o's HiFT F0 graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from flashcosyvoice.modules.hifigan import HiFTGenerator

from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
    _ensure_torchair_broadcast_alias,
    _hift_f0_features,
    _mark_frozen_graph_weights,
    materialize_hift_weight_norm_for_npu,
)


def _load_hift(checkpoint: Path, device: torch.device) -> HiFTGenerator:
    hift = HiFTGenerator().eval()
    state = {
        name.removeprefix("generator."): value
        for name, value in torch.load(checkpoint, map_location="cpu", weights_only=True).items()
    }
    hift.load_state_dict(state, strict=True)
    return hift.to(device)


def _measure(function, args: tuple[torch.Tensor, ...], *, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        function(*args)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / iterations


def _control_features(value: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
    return _hift_f0_features(value, *weights)


def _frozen_features(value: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
    return _hift_f0_features(value, *weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", type=int, default=58)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    hift = _load_hift(args.checkpoint, device)
    materialize_hift_weight_norm_for_npu(hift.f0_predictor)
    convolutions = [
        layer
        for layer in hift.f0_predictor.condnet
        if isinstance(layer, torch.nn.Conv1d)
    ]
    frozen_weights = tuple(
        tensor for layer in convolutions for tensor in (layer.weight, layer.bias)
    )
    control_weights = tuple(weight.detach().clone() for weight in frozen_weights)
    value = frozen_weights[0].new_full(
        (1, int(convolutions[0].in_channels), args.width),
        0.125,
    )

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    control_config = torchair.CompilerConfig()
    control_graph = torch.compile(
        _control_features,
        backend=torchair.get_npu_backend(compiler_config=control_config),
        fullgraph=True,
        dynamic=False,
    )
    _mark_frozen_graph_weights(frozen_weights)
    frozen_config = torchair.CompilerConfig()
    frozen_config.experimental_config.frozen_parameter.value = True
    frozen_graph = torch.compile(
        _frozen_features,
        backend=torchair.get_npu_backend(compiler_config=frozen_config),
        fullgraph=True,
        dynamic=False,
    )

    control_args = (value, *control_weights)
    frozen_args = (value, *frozen_weights)
    with torch.inference_mode():
        control = control_graph(*control_args)
        frozen = frozen_graph(*frozen_args)
        expected = hift.f0_predictor.condnet(value)
        control_us = _measure(
            control_graph,
            control_args,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        frozen_us = _measure(
            frozen_graph,
            frozen_args,
            warmups=args.warmups,
            iterations=args.iterations,
        )

    print(
        json.dumps(
            {
                "device": args.device,
                "shape": list(value.shape),
                "warmups": args.warmups,
                "iterations": args.iterations,
                "control_us": control_us,
                "frozen_us": frozen_us,
                "speedup": control_us / frozen_us,
                "control_max_abs_error": float((control - expected).abs().max().item()),
                "frozen_max_abs_error": float((frozen - expected).abs().max().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
