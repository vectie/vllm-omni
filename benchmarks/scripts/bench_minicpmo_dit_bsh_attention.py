# SPDX-License-Identifier: Apache-2.0
"""Compare MiniCPM-o cached attention in BNSD and sequence-major BSH."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

_BATCH = 2
_HEADS = 8
_HEAD_DIM = 64
_HIDDEN = _HEADS * _HEAD_DIM


def _measure_us(function: Callable[[], object], iterations: int) -> float:
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1_000


def _fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_fusion_attention(
        query=query,
        key=key,
        value=value,
        head_num=_HEADS,
        input_layout=layout,
        scale=_HEAD_DIM**-0.5,
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        sparse_mode=0,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--cache-length", type=int, default=402)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    q_bsh = torch.randn(
        _BATCH,
        args.width,
        _HIDDEN,
        device=device,
        dtype=dtype,
    )
    k_bsh = torch.randn_like(q_bsh)
    v_bsh = torch.randn_like(q_bsh)
    cached_k_bsh = torch.randn(
        _BATCH,
        args.cache_length,
        _HIDDEN,
        device=device,
        dtype=dtype,
    )
    cached_v_bsh = torch.randn_like(cached_k_bsh)
    projection_weight = torch.randn(
        _HIDDEN,
        _HIDDEN,
        device=device,
        dtype=dtype,
    )
    projection_bias = torch.randn(
        _HIDDEN,
        device=device,
        dtype=dtype,
    )

    def to_bnsd(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            _BATCH, value.shape[1], _HEADS, _HEAD_DIM
        ).transpose(1, 2)

    q_bnsd = to_bnsd(q_bsh)
    k_bnsd = to_bnsd(k_bsh)
    v_bnsd = to_bnsd(v_bsh)
    cached_k_bnsd = to_bnsd(cached_k_bsh)
    cached_v_bnsd = to_bnsd(cached_v_bsh)

    def control() -> torch.Tensor:
        full_k = torch.cat((k_bnsd, cached_k_bnsd), dim=2)
        full_v = torch.cat((v_bnsd, cached_v_bnsd), dim=2)
        hidden = _fusion_attention(q_bnsd, full_k, full_v, "BNSD")
        hidden = hidden.transpose(1, 2).reshape(_BATCH, args.width, _HIDDEN)
        return F.linear(hidden, projection_weight, projection_bias)

    def candidate() -> torch.Tensor:
        full_k = torch.cat((k_bsh, cached_k_bsh), dim=1)
        full_v = torch.cat((v_bsh, cached_v_bsh), dim=1)
        hidden = _fusion_attention(q_bsh, full_k, full_v, "BSH")
        return F.linear(hidden, projection_weight, projection_bias)

    with torch.inference_mode():
        expected = control()
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
    difference = (actual.float() - expected.float()).abs()
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": str(dtype),
                "width": args.width,
                "cache_length": args.cache_length,
                "warmups": args.warmups,
                "iterations": args.iterations,
                "trials": args.trials,
                "control_bnsd_median_us": control_median,
                "candidate_bsh_median_us": candidate_median,
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
