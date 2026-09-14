# SPDX-License-Identifier: Apache-2.0
"""Screen a fused MiniCPM-o final-layer, CFG, and Euler epilogue."""

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

_HIDDEN_SIZE = 512
_OUTPUT_SIZE = 80


def _cfm_epilogue(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    state: torch.Tensor,
    delta: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    cfg_rate: float,
) -> torch.Tensor:
    shift, scale = modulation.chunk(2, dim=-1)
    normalized = F.layer_norm(hidden, (_HIDDEN_SIZE,), eps=1e-6)
    estimate = F.linear(
        normalized * (1 + scale) + shift,
        output_weight,
        output_bias,
    ).transpose(1, 2)
    conditional, unconditional = estimate.chunk(2, dim=0)
    velocity = (1.0 + cfg_rate) * conditional - cfg_rate * unconditional
    return state + delta * velocity


def _cfm_guidance_before_output(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    state: torch.Tensor,
    delta: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    cfg_rate: float,
) -> torch.Tensor:
    shift, scale = modulation.chunk(2, dim=-1)
    normalized = F.layer_norm(hidden, (_HIDDEN_SIZE,), eps=1e-6)
    modulated = normalized * (1 + scale) + shift
    conditional, unconditional = modulated.chunk(2, dim=0)
    guided_hidden = (
        (1.0 + cfg_rate) * conditional - cfg_rate * unconditional
    )
    velocity = F.linear(
        guided_hidden,
        output_weight,
        output_bias,
    ).transpose(1, 2)
    return state + delta * velocity


def _final_modulate(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
) -> torch.Tensor:
    shift, scale = modulation.chunk(2, dim=-1)
    normalized = F.layer_norm(hidden, (_HIDDEN_SIZE,), eps=1e-6)
    return normalized * (1 + scale) + shift


def _final_modulate_addcmul(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
) -> torch.Tensor:
    shift, scale = modulation.chunk(2, dim=-1)
    normalized = F.layer_norm(hidden, (_HIDDEN_SIZE,), eps=1e-6)
    return torch.addcmul(normalized + shift, normalized, scale)


def _output_cfg_update(
    modulated: torch.Tensor,
    state: torch.Tensor,
    delta: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    cfg_rate: float,
) -> torch.Tensor:
    estimate = F.linear(modulated, output_weight, output_bias).transpose(1, 2)
    conditional, unconditional = estimate.chunk(2, dim=0)
    velocity = (1.0 + cfg_rate) * conditional - cfg_rate * unconditional
    return state + delta * velocity


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
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    final_weight = state_dict[
        "decoder.estimator.final_layer.adaLN_modulation.1.weight"
    ].to(device)
    final_bias = state_dict[
        "decoder.estimator.final_layer.adaLN_modulation.1.bias"
    ].to(device)
    output_weight = state_dict["decoder.estimator.final_layer.linear.weight"].to(
        device
    )
    output_bias = state_dict["decoder.estimator.final_layer.linear.bias"].to(device)
    dtype = output_weight.dtype

    hidden_steps = torch.linspace(
        -0.125,
        0.125,
        args.steps * 2 * args.width * _HIDDEN_SIZE,
        device=device,
        dtype=dtype,
    ).reshape(args.steps, 2, args.width, _HIDDEN_SIZE)
    time_embeddings = torch.linspace(
        -0.25,
        0.25,
        args.steps * 2 * _HIDDEN_SIZE,
        device=device,
        dtype=dtype,
    ).reshape(args.steps, 2, 1, _HIDDEN_SIZE)
    modulations = F.linear(F.silu(time_embeddings), final_weight, final_bias)
    modulated_steps = torch.stack(
        [
            _final_modulate(hidden_steps[step], modulations[step])
            for step in range(args.steps)
        ]
    )
    initial_state = torch.linspace(
        -0.1,
        0.1,
        args.width * _OUTPUT_SIZE,
        device=device,
        dtype=dtype,
    ).reshape(1, _OUTPUT_SIZE, args.width)
    timeline = 1 - torch.cos(
        torch.linspace(0, 1, args.steps + 1, device=device, dtype=torch.float32)
        * 0.5
        * torch.pi
    )
    deltas = (timeline[1:] - timeline[:-1]).to(dtype=dtype)

    def eager() -> torch.Tensor:
        state = initial_state
        for step in range(args.steps):
            state = _cfm_epilogue(
                hidden_steps[step],
                modulations[step],
                state,
                deltas[step],
                output_weight,
                output_bias,
                args.cfg_rate,
            )
        return state

    def guided_before_output() -> torch.Tensor:
        state = initial_state
        for step in range(args.steps):
            state = _cfm_guidance_before_output(
                hidden_steps[step],
                modulations[step],
                state,
                deltas[step],
                output_weight,
                output_bias,
                args.cfg_rate,
            )
        return state

    def modulate_only() -> tuple[torch.Tensor, ...]:
        return tuple(
            _final_modulate(hidden_steps[step], modulations[step])
            for step in range(args.steps)
        )

    def output_cfg_only() -> torch.Tensor:
        state = initial_state
        for step in range(args.steps):
            state = _output_cfg_update(
                modulated_steps[step],
                state,
                deltas[step],
                output_weight,
                output_bias,
                args.cfg_rate,
            )
        return state

    def addcmul_epilogue() -> torch.Tensor:
        state = initial_state
        for step in range(args.steps):
            modulated = _final_modulate_addcmul(
                hidden_steps[step],
                modulations[step],
            )
            state = _output_cfg_update(
                modulated,
                state,
                deltas[step],
                output_weight,
                output_bias,
                args.cfg_rate,
            )
        return state

    def modulate_addcmul_only() -> tuple[torch.Tensor, ...]:
        return tuple(
            _final_modulate_addcmul(hidden_steps[step], modulations[step])
            for step in range(args.steps)
        )

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    epilogue_graph = torch.compile(
        _cfm_epilogue,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    guided_before_output_graph = torch.compile(
        _cfm_guidance_before_output,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )

    def graph() -> torch.Tensor:
        state = initial_state
        for step in range(args.steps):
            state = epilogue_graph(
                hidden_steps[step],
                modulations[step],
                state,
                deltas[step],
                output_weight,
                output_bias,
                args.cfg_rate,
            )
        return state

    def guided_graph() -> torch.Tensor:
        state = initial_state
        for step in range(args.steps):
            state = guided_before_output_graph(
                hidden_steps[step],
                modulations[step],
                state,
                deltas[step],
                output_weight,
                output_bias,
                args.cfg_rate,
            )
        return state

    calls = {
        "eager": eager,
        "graph": graph,
        "guided_before_output": guided_before_output,
        "guided_graph": guided_graph,
        "modulate_only": modulate_only,
        "modulate_addcmul_only": modulate_addcmul_only,
        "output_cfg_only": output_cfg_only,
        "addcmul_epilogue": addcmul_epilogue,
    }
    with torch.inference_mode():
        outputs = {name: function() for name, function in calls.items()}
        for _ in range(args.warmups):
            for function in calls.values():
                function()
        trials = {name: [] for name in calls}
        orders = (
            tuple(calls),
            tuple(reversed(calls)),
            (
                "modulate_only",
                "addcmul_epilogue",
                "guided_before_output",
                "eager",
                "modulate_addcmul_only",
                "output_cfg_only",
                "guided_graph",
                "graph",
            ),
        )
        for trial in range(args.trials):
            order = orders[trial % len(orders)]
            for name in order:
                trials[name].append(_measure_us(calls[name], args.iterations))

    medians = {name: statistics.median(values) for name, values in trials.items()}
    comparable_outputs = {
        name: output
        for name, output in outputs.items()
        if isinstance(output, torch.Tensor) and output.shape == outputs["eager"].shape
    }
    differences = {
        name: output - outputs["eager"]
        for name, output in comparable_outputs.items()
    }
    print(
        json.dumps(
            {
                "device": args.device,
                "dtype": str(dtype),
                "steps": args.steps,
                "width": args.width,
                "cfg_rate": args.cfg_rate,
                "median_us": medians,
                "speedup_vs_eager": {
                    name: medians["eager"] / value
                    for name, value in medians.items()
                },
                "trials_us": trials,
                "errors": {
                    name: {
                        "max_abs": float(difference.abs().max().item()),
                        "mean_abs": float(difference.abs().mean().item()),
                    }
                    for name, difference in differences.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
