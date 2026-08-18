# SPDX-License-Identifier: Apache-2.0
"""Benchmark the opt-in MiniCPM-o HiFT residual-block graph on Ascend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from flashcosyvoice.modules.hifigan import HiFTGenerator

from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
    _hift_resblock_stage_shape,
    materialize_hift_weight_norm_for_npu,
    prepare_hift_resblock_graph_for_npu,
)


def _measure(function, value: torch.Tensor, *, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        function(value)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function(value)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / iterations


def _load_hift(checkpoint: Path, device: torch.device) -> HiFTGenerator:
    hift = HiFTGenerator().eval()
    state = {
        name.removeprefix("generator."): value
        for name, value in torch.load(checkpoint, map_location="cpu", weights_only=True).items()
    }
    hift.load_state_dict(state, strict=True)
    return hift.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--mel-width", type=int, default=58)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    hift = _load_hift(args.checkpoint, device)
    materialize_hift_weight_norm_for_npu(hift)

    shape = _hift_resblock_stage_shape(hift, mel_width=args.mel_width, stage=args.stage)
    value = torch.linspace(-0.5, 0.5, steps=shape[1] * shape[2], device=device).reshape(shape)
    first = args.stage * hift.num_kernels
    last = first + hift.num_kernels
    blocks = list(hift.resblocks[first:last])

    eager_us = [
        _measure(block.forward, value, warmups=args.warmups, iterations=args.iterations)
        for block in blocks
    ]
    compiled = prepare_hift_resblock_graph_for_npu(
        hift,
        stage=args.stage,
        mel_width=args.mel_width,
    )

    graph_us: list[float] = []
    max_abs_error: list[float] = []
    for block in blocks:
        expected = block._step_audio2_original_forward(value)
        actual = block(value)
        max_abs_error.append(float((actual - expected).abs().max().item()))
        graph_us.append(
            _measure(block, value, warmups=args.warmups, iterations=args.iterations)
        )

    eager_total = sum(eager_us)
    graph_total = sum(graph_us)
    print(
        json.dumps(
            {
                "device": args.device,
                "stage": args.stage,
                "mel_width": args.mel_width,
                "shape": shape,
                "compiled_blocks": compiled,
                "eager_us_per_block": eager_us,
                "graph_us_per_block": graph_us,
                "eager_us_total": eager_total,
                "graph_us_total": graph_total,
                "speedup": eager_total / graph_total,
                "max_abs_error_per_block": max_abs_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
