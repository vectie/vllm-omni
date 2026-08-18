# SPDX-License-Identifier: Apache-2.0
"""Benchmark MiniCPM-o HiFT source-noise scratch reuse on Ascend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from flashcosyvoice.modules.hifigan import HiFTGenerator

from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
    prepare_hift_source_noise_scratch_for_npu,
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


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--mel-width", type=int, default=58)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    hift = _load_hift(args.checkpoint, device)
    source = hift.m_source
    original_forward = source.forward
    waveform_width = args.mel_width * int(source.l_sin_gen.upsample_scale)
    value = torch.linspace(
        80,
        440,
        steps=waveform_width,
        device=device,
        dtype=next(hift.parameters()).dtype,
    ).reshape(1, waveform_width, 1)

    eager_us = _measure(
        original_forward,
        value,
        warmups=args.warmups,
        iterations=args.iterations,
    )
    prepare_hift_source_noise_scratch_for_npu(hift)
    scratch_us = _measure(
        source.forward,
        value,
        warmups=args.warmups,
        iterations=args.iterations,
    )

    _seed(args.seed)
    expected = original_forward(value)
    expected_next = torch.randn(32, device=device)
    torch.npu.synchronize()
    _seed(args.seed)
    actual = source(value)
    actual_noise = actual[1].clone()
    actual_next = torch.randn(32, device=device)
    torch.npu.synchronize()
    scratch_pointer = actual[1].data_ptr()
    reused_pointer = source(value)[1].data_ptr()

    max_abs_error = [
        float((actual[0] - expected[0]).abs().max().item()),
        float((actual_noise - expected[1]).abs().max().item()),
        float((actual[2] - expected[2]).abs().max().item()),
    ]
    next_rng_max_abs_error = float((actual_next - expected_next).abs().max().item())
    print(
        json.dumps(
            {
                "device": args.device,
                "mel_width": args.mel_width,
                "shape": list(value.shape),
                "eager_us": eager_us,
                "scratch_us": scratch_us,
                "speedup": eager_us / scratch_us,
                "max_abs_error_per_output": max_abs_error,
                "next_rng_max_abs_error": next_rng_max_abs_error,
                "scratch_pointer_reused": scratch_pointer == reused_pointer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
