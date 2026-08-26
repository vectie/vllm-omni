# SPDX-License-Identifier: Apache-2.0
"""Screen one GE-visible BSH MiniCPM-o DiT block against the split path."""

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
    _dit_attention_preamble_bsh_from_modulation,
    _dit_conv_mlp_residual,
    _dit_fused_conv_mlp_residual,
    _dit_fused_full_block_bsh_from_modulation,
    _dit_full_block_bsh_standard_conv_from_modulation,
    _ensure_torchair_broadcast_alias,
)


def _measure_us(function: Callable[[], object], iterations: int) -> float:
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1_000


def _causal_pack_reference(
    value: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, channels = value.shape
    history = torch.cat((cache, value.transpose(1, 2)), dim=2)
    packed = torch.stack(
        [
            history[:, :, offset : offset + 3]
            .transpose(1, 2)
            .reshape(batch, channels * 3)
            for offset in range(frames)
        ],
        dim=1,
    ).reshape(batch * frames, channels * 3)
    return packed, value[:, -2:, :].transpose(1, 2).contiguous()


def _weights(
    checkpoint: Path,
    *,
    block: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = f"decoder.estimator.blocks.{block}"
    conv1 = state[f"{prefix}.conv.block.1.weight"]
    conv2 = state[f"{prefix}.conv.block.6.weight"]
    values = (
        state[f"{prefix}.adaLN_modulation.1.weight"],
        state[f"{prefix}.adaLN_modulation.1.bias"],
        state[f"{prefix}.attn.to_q.weight"],
        state[f"{prefix}.attn.to_q.bias"],
        state[f"{prefix}.attn.to_k.weight"],
        state[f"{prefix}.attn.to_k.bias"],
        state[f"{prefix}.attn.to_v.weight"],
        state[f"{prefix}.attn.to_v.bias"],
        state[f"{prefix}.attn.q_norm.weight"],
        state[f"{prefix}.attn.q_norm.bias"],
        state[f"{prefix}.attn.k_norm.weight"],
        state[f"{prefix}.attn.k_norm.bias"],
        state[f"{prefix}.attn.proj.weight"],
        state[f"{prefix}.attn.proj.bias"],
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
    return tuple(value.to(device=device, dtype=dtype) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--cache-length", type=int, default=402)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    import torch_npu
    from torch_npu.dynamo import torchair
    from vllm_ascend.compilation.minicpmo_causal_conv import (
        register_minicpmo_causal_conv_pack_converter,
    )
    from vllm_ascend.compilation.minicpmo_fusion_attention import (
        register_minicpmo_fusion_attention_v3_converter,
    )
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.bfloat16
    loaded_weights = _weights(
        args.checkpoint,
        block=args.block,
        device=device,
        dtype=dtype,
    )
    adaln_weight, adaln_bias = loaded_weights[:2]
    weights = loaded_weights[2:]
    standard_weights = (
        *weights[:12],
        weights[12].reshape(512, 3, 512).permute(0, 2, 1).contiguous(),
        *weights[13:16],
        weights[16].reshape(512, 3, 512).permute(0, 2, 1).contiguous(),
        *weights[17:],
    )
    torch.manual_seed(0)
    hidden = torch.randn(2, args.width, 512, device=device, dtype=dtype)
    time_embedding = torch.randn(2, 1, 512, device=device, dtype=dtype)
    modulation = F.linear(F.silu(time_embedding), adaln_weight, adaln_bias)
    att_cache = torch.randn(
        2, 2, args.cache_length, 512, device=device, dtype=dtype
    )
    cnn_cache = torch.randn(2, 1024, 2, device=device, dtype=dtype)

    enable_custom_op()
    register_minicpmo_causal_conv_pack_converter()
    register_minicpmo_fusion_attention_v3_converter()
    _ensure_torchair_broadcast_alias()
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    preamble_graph = torch.compile(
        _dit_attention_preamble_bsh_from_modulation,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    conv_graph = torch.compile(
        _dit_fused_conv_mlp_residual,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    candidate_graph = torch.compile(
        _dit_fused_full_block_bsh_from_modulation,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    standard_candidate_graph = torch.compile(
        _dit_full_block_bsh_standard_conv_from_modulation,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )

    def fused_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=8,
            input_layout="BSH",
            scale=64**-0.5,
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            sparse_mode=0,
        )[0]

    def control() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        supplied, q, k, v = preamble_graph(hidden, modulation, *weights[:10])
        full_k = torch.cat((k, att_cache[0]), dim=1)
        full_v = torch.cat((v, att_cache[1]), dim=1)
        new_att = torch.stack((full_k, full_v), dim=0)
        attention = F.linear(fused_attention(q, full_k, full_v), *weights[10:12])
        mods = supplied.chunk(9, dim=-1)
        updated = hidden + mods[2] * attention
        conv_input = F.layer_norm(updated, (512,), eps=1e-6)
        conv_input = conv_input * (1 + mods[7]) + mods[6]
        updated, new_cnn = conv_graph(
            updated,
            conv_input,
            cnn_cache,
            mods[8],
            mods[3],
            mods[4],
            mods[5],
            *weights[12:],
        )
        return updated, new_cnn, new_att

    def candidate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return candidate_graph(hidden, modulation, att_cache, cnn_cache, *weights)

    def standard_candidate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return standard_candidate_graph(
            hidden,
            modulation,
            att_cache,
            cnn_cache,
            *standard_weights,
        )

    def canonical() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        supplied, q, k, v = _dit_attention_preamble_bsh_from_modulation(
            hidden, modulation, *weights[:10]
        )
        full_k = torch.cat((k, att_cache[0]), dim=1)
        full_v = torch.cat((v, att_cache[1]), dim=1)
        new_att = torch.stack((full_k, full_v), dim=0)
        attention = F.linear(fused_attention(q, full_k, full_v), *weights[10:12])
        mods = supplied.chunk(9, dim=-1)
        updated = hidden + mods[2] * attention
        conv_input = F.layer_norm(updated, (512,), eps=1e-6)
        conv_input = conv_input * (1 + mods[7]) + mods[6]
        conv1_weight = (
            weights[12].reshape(512, 3, 512).permute(0, 2, 1).contiguous()
        )
        conv2_weight = (
            weights[16].reshape(512, 3, 512).permute(0, 2, 1).contiguous()
        )
        updated, new_cnn = _dit_conv_mlp_residual(
            updated,
            conv_input,
            cnn_cache,
            mods[8],
            mods[3],
            mods[4],
            mods[5],
            conv1_weight,
            weights[13],
            weights[14],
            weights[15],
            conv2_weight,
            weights[17],
            *weights[18:],
        )
        return updated, new_cnn, new_att

    with torch.inference_mode():
        graph_preamble = tuple(
            result.clone()
            for result in preamble_graph(hidden, modulation, *weights[:10])
        )
        eager_preamble = tuple(
            result.clone()
            for result in _dit_attention_preamble_bsh_from_modulation(
                hidden, modulation, *weights[:10]
            )
        )
        graph_full_k = torch.cat((graph_preamble[2], att_cache[0]), dim=1)
        graph_full_v = torch.cat((graph_preamble[3], att_cache[1]), dim=1)
        eager_full_k = torch.cat((eager_preamble[2], att_cache[0]), dim=1)
        eager_full_v = torch.cat((eager_preamble[3], att_cache[1]), dim=1)
        graph_attention = fused_attention(
            graph_preamble[1], graph_full_k, graph_full_v
        ).clone()
        eager_attention = fused_attention(
            eager_preamble[1], eager_full_k, eager_full_v
        ).clone()
        mods = eager_preamble[0].chunk(9, dim=-1)
        attention_projection = F.linear(
            eager_attention, *weights[10:12]
        )
        post_attention = hidden + mods[2] * attention_projection
        pack_input = F.layer_norm(post_attention, (512,), eps=1e-6)
        pack_input = pack_input * (1 + mods[7]) + mods[6]
        cache1 = cnn_cache[:, :512]
        native_packed, native_cache = (
            torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
                pack_input, cache1
            )
        )
        reference_packed, reference_cache = _causal_pack_reference(
            pack_input, cache1
        )
        native_packed = native_packed.clone()
        native_cache = native_cache.clone()
        reference_packed = reference_packed.clone()
        reference_cache = reference_cache.clone()
        reference = tuple(result.clone() for result in canonical())
        expected = tuple(result.clone() for result in control())
        eager = tuple(
            result.clone()
            for result in _dit_fused_full_block_bsh_from_modulation(
                hidden,
                modulation,
                att_cache,
                cnn_cache,
                *weights,
            )
        )
        actual = tuple(result.clone() for result in candidate())
        standard_actual = tuple(
            result.clone() for result in standard_candidate()
        )
        for _ in range(args.warmups):
            control()
            candidate()
            standard_candidate()
        torch.npu.synchronize()
        control_trials: list[float] = []
        candidate_trials: list[float] = []
        standard_candidate_trials: list[float] = []
        for trial in range(args.trials):
            if trial % 2:
                standard_candidate_trials.append(
                    _measure_us(standard_candidate, args.iterations)
                )
                candidate_trials.append(_measure_us(candidate, args.iterations))
                control_trials.append(_measure_us(control, args.iterations))
            else:
                control_trials.append(_measure_us(control, args.iterations))
                candidate_trials.append(_measure_us(candidate, args.iterations))
                standard_candidate_trials.append(
                    _measure_us(standard_candidate, args.iterations)
                )

    control_median = statistics.median(control_trials)
    candidate_median = statistics.median(candidate_trials)
    standard_candidate_median = statistics.median(standard_candidate_trials)
    errors = [
        (result.float() - reference.float()).abs()
        for result, reference in zip(actual, expected, strict=True)
    ]
    canonical_errors = [
        (result.float() - reference_value.float()).abs()
        for result, reference_value in zip(actual, reference, strict=True)
    ]
    standard_errors = [
        (result.float() - reference_value.float()).abs()
        for result, reference_value in zip(
            standard_actual, reference, strict=True
        )
    ]
    standard_split_errors = [
        (result.float() - split_value.float()).abs()
        for result, split_value in zip(
            standard_actual, expected, strict=True
        )
    ]
    split_canonical_errors = [
        (result.float() - reference_value.float()).abs()
        for result, reference_value in zip(expected, reference, strict=True)
    ]
    eager_errors = [
        (result.float() - reference.float()).abs()
        for result, reference in zip(eager, expected, strict=True)
    ]
    preamble_errors = [
        (result.float() - reference.float()).abs()
        for result, reference in zip(
            eager_preamble, graph_preamble, strict=True
        )
    ]
    attention_error = (
        eager_attention.float() - graph_attention.float()
    ).abs()
    print(
        json.dumps(
            {
                "device": args.device,
                "block": args.block,
                "width": args.width,
                "cache_length": args.cache_length,
                "control_split_median_us": control_median,
                "candidate_full_block_median_us": candidate_median,
                "standard_full_block_median_us": standard_candidate_median,
                "speedup": control_median / candidate_median,
                "standard_speedup": control_median / standard_candidate_median,
                "control_trials_us": control_trials,
                "candidate_trials_us": candidate_trials,
                "standard_candidate_trials_us": standard_candidate_trials,
                "hidden_max_abs_error": float(errors[0].max().item()),
                "cnn_max_abs_error": float(errors[1].max().item()),
                "cnn_first_cache_max_abs_error": float(
                    errors[1][:, :512].max().item()
                ),
                "cnn_second_cache_max_abs_error": float(
                    errors[1][:, 512:].max().item()
                ),
                "attention_cache_max_abs_error": float(errors[2].max().item()),
                "canonical_hidden_max_abs_error": float(
                    canonical_errors[0].max().item()
                ),
                "canonical_cnn_max_abs_error": float(
                    canonical_errors[1].max().item()
                ),
                "standard_hidden_max_abs_error": float(
                    standard_errors[0].max().item()
                ),
                "standard_hidden_mean_abs_error": float(
                    standard_errors[0].mean().item()
                ),
                "canonical_hidden_max_abs_value": float(
                    reference[0].float().abs().max().item()
                ),
                "standard_cnn_max_abs_error": float(
                    standard_errors[1].max().item()
                ),
                "standard_attention_cache_max_abs_error": float(
                    standard_errors[2].max().item()
                ),
                "standard_vs_split_hidden_max_abs_error": float(
                    standard_split_errors[0].max().item()
                ),
                "standard_vs_split_cnn_max_abs_error": float(
                    standard_split_errors[1].max().item()
                ),
                "split_vs_canonical_hidden_max_abs_error": float(
                    split_canonical_errors[0].max().item()
                ),
                "split_vs_canonical_cnn_max_abs_error": float(
                    split_canonical_errors[1].max().item()
                ),
                "eager_hidden_max_abs_error": float(
                    eager_errors[0].max().item()
                ),
                "eager_cnn_max_abs_error": float(eager_errors[1].max().item()),
                "eager_cnn_first_cache_max_abs_error": float(
                    eager_errors[1][:, :512].max().item()
                ),
                "eager_cnn_second_cache_max_abs_error": float(
                    eager_errors[1][:, 512:].max().item()
                ),
                "eager_attention_cache_max_abs_error": float(
                    eager_errors[2].max().item()
                ),
                "preamble_modulation_max_abs_error": float(
                    preamble_errors[0].max().item()
                ),
                "preamble_q_max_abs_error": float(
                    preamble_errors[1].max().item()
                ),
                "preamble_k_max_abs_error": float(
                    preamble_errors[2].max().item()
                ),
                "preamble_v_max_abs_error": float(
                    preamble_errors[3].max().item()
                ),
                "attention_max_abs_error": float(attention_error.max().item()),
                "attention_mean_abs_error": float(attention_error.mean().item()),
                "direct_pack_max_abs_error": float(
                    (native_packed.float() - reference_packed.float())
                    .abs()
                    .max()
                    .item()
                ),
                "direct_pack_cache_max_abs_error": float(
                    (native_cache.float() - reference_cache.float())
                    .abs()
                    .max()
                    .item()
                ),
                "pack_input_npu_format": int(
                    torch_npu.get_npu_format(pack_input)
                ),
                "cnn_cache_npu_format": int(
                    torch_npu.get_npu_format(cnn_cache)
                ),
                "native_packed_npu_format": int(
                    torch_npu.get_npu_format(native_packed)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
