# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

import importlib
import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_SILENCE_TOKEN = 4218
_NPU_DIT_MLP_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH"
_NPU_DIT_MLP_GRAPH_WIDTH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH"
_NPU_DIT_GRAPH_BUCKETS_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_GRAPH_BUCKETS"
_NPU_DIT_PREAMBLE_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_PREAMBLE_GRAPH"
_NPU_DIT_WIDE_ADALN_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_ADALN"
_NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT = 1.0e-6
_NPU_DIT_CONV_MLP_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_CONV_MLP_GRAPH"
_NPU_DIT_PROMPT_CONV_MLP_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_PROMPT_CONV_MLP_GRAPH"
_NPU_DIT_FULL_BLOCK_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_GRAPH"
_NPU_DIT_FULL_STACK_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_STACK_GRAPH"
_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS"
_NPU_DIT_FUSED_CONV_PACK_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK"
_NPU_DIT_CACHE_MAJOR_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR"
_NPU_DIT_POST_ATTN_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_POST_ATTN_GRAPH"
_NPU_DIT_QKV_PACK_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_QKV_PACK"
_NPU_DIT_ATTN_CACHE_OUT_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_ATTN_CACHE_OUT"
_NPU_CFM_STACKED_CACHE_OUT_ENV = "VLLM_OMNI_MINICPMO45_NPU_CFM_STACKED_CACHE_OUT"
_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH"
)
_NPU_DIT_FUSED_CONV_BLOCK_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_BLOCK"
_NPU_DIT_FUSED_CONV_LINEAR_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_LINEAR"
logger = logging.getLogger(__name__)


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def _npu_dit_mlp_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_MLP_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_MLP_GRAPH_ENV}={raw!r}")


def _npu_dit_preamble_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_PREAMBLE_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_PREAMBLE_GRAPH_ENV}={raw!r}")


def _npu_dit_wide_adaln_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_WIDE_ADALN_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_WIDE_ADALN_ENV}={raw!r}")


def _npu_dit_conv_mlp_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_CONV_MLP_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_CONV_MLP_GRAPH_ENV}={raw!r}")


def _npu_dit_prompt_conv_mlp_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_PROMPT_CONV_MLP_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_PROMPT_CONV_MLP_GRAPH_ENV}={raw!r}")


def _npu_dit_full_block_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FULL_BLOCK_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_FULL_BLOCK_GRAPH_ENV}={raw!r}")


def _npu_dit_full_stack_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FULL_STACK_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_FULL_STACK_GRAPH_ENV}={raw!r}")


def _npu_dit_fused_conv_pack_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FUSED_CONV_PACK_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_FUSED_CONV_PACK_ENV}={raw!r}")


def _npu_dit_cache_major_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_CACHE_MAJOR_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_CACHE_MAJOR_ENV}={raw!r}")


def _npu_dit_post_attn_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_POST_ATTN_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_POST_ATTN_GRAPH_ENV}={raw!r}")


def _npu_dit_qkv_pack_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_QKV_PACK_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_QKV_PACK_ENV}={raw!r}")


def _npu_dit_attn_cache_out_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_ATTN_CACHE_OUT_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_ATTN_CACHE_OUT_ENV}={raw!r}")


def _npu_cfm_stacked_cache_out_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_CFM_STACKED_CACHE_OUT_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_CFM_STACKED_CACHE_OUT_ENV}={raw!r}")


def _npu_single_request_cache_passthrough_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH_ENV}={raw!r}")


def _npu_dit_fused_conv_block_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FUSED_CONV_BLOCK_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_FUSED_CONV_BLOCK_ENV}={raw!r}")


def _npu_dit_fused_conv_linear_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FUSED_CONV_LINEAR_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_FUSED_CONV_LINEAR_ENV}={raw!r}")


def _npu_dit_mlp_graph_width(config_value: Any = None) -> int:
    env_value = os.environ.get(_NPU_DIT_MLP_GRAPH_WIDTH_ENV)
    raw = env_value if env_value not in (None, "") else (50 if config_value is None else config_value)
    try:
        width = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {_NPU_DIT_MLP_GRAPH_WIDTH_ENV}={raw!r}") from exc
    if width <= 0:
        raise ValueError(f"{_NPU_DIT_MLP_GRAPH_WIDTH_ENV} must be positive, got {width}")
    return width


def _npu_dit_graph_buckets(config_value: Any = None) -> tuple[int, ...]:
    """Return additional static DiT widths to compile beside the stream width."""
    env_value = os.environ.get(_NPU_DIT_GRAPH_BUCKETS_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw in (None, ""):
        return ()
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    widths: list[int] = []
    for value in values:
        try:
            width = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {_NPU_DIT_GRAPH_BUCKETS_ENV}={raw!r}") from exc
        if width <= 0:
            raise ValueError(f"{_NPU_DIT_GRAPH_BUCKETS_ENV} widths must be positive, got {width}")
        if width not in widths:
            widths.append(width)
    return tuple(widths)


def _npu_dit_full_block_cache_buckets(config_value: Any = None) -> tuple[int, ...]:
    """Return fixed attention-cache lengths eligible for full-block replay."""
    env_value = os.environ.get(_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw in (None, ""):
        return ()
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    lengths: list[int] = []
    for value in values:
        try:
            length = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS_ENV}={raw!r}") from exc
        if length <= 0:
            raise ValueError(
                f"{_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS_ENV} lengths must be positive, got {length}"
            )
        if length not in lengths:
            lengths.append(length)
    return tuple(lengths)


def _dit_attention_cache_length(att_cache: torch.Tensor | None) -> int:
    """Return sequence length from [layers, batch, heads, sequence, width]."""
    return int(att_cache.shape[-2]) if att_cache is not None else 0


def _ensure_torchair_broadcast_alias() -> None:
    """Repair a TorchAir registration-order incompatibility in vLLM workers.

    vLLM's distributed initialization can register ``npu_define::broadcast``
    before TorchAir imports its experimental converters. In that order,
    TorchAir skips defining its module-local ``op_broadcast`` name and a later
    converter import fails even though the operator itself is available.
    Populate the missing alias without changing the installed torch-npu tree.
    """
    module = importlib.import_module(
        "torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast"
    )
    if not hasattr(module, "op_broadcast"):
        broadcast = torch.ops.npu_define.broadcast
        module.op_broadcast = getattr(broadcast, "default", broadcast)


def _dit_mlp_residual(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> torch.Tensor:
    """Fixed-shape DiT MLP/residual partition used by the NPU graph path."""
    hidden = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    hidden = hidden * (1 + scale) + shift
    hidden = F.linear(hidden, fc1_weight, fc1_bias)
    hidden = F.gelu(hidden, approximate="tanh")
    hidden = F.linear(hidden, fc2_weight, fc2_bias)
    return x + gate * hidden


def _dit_wide_adaln(
    time_embedding: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor,
) -> torch.Tensor:
    """Project all 16 DiT block modulations through one wide Cube GEMM."""
    modulation = F.linear(F.silu(time_embedding), packed_weight, packed_bias)
    return modulation.reshape(2, 1, 16, 9 * 512)


def _dit_wide_adaln_steps(
    time_embeddings: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor,
) -> torch.Tensor:
    """Project every fixed CFM timestep and DiT block in one Cube GEMM."""
    modulation = F.linear(F.silu(time_embeddings), packed_weight, packed_bias)
    return modulation.reshape(time_embeddings.shape[0], 2, 1, 16, 9 * 512)


def _dit_attention_from_modulation(
    x: torch.Tensor,
    modulation: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build projected Q/K/V from a supplied block modulation."""
    shift_msa = modulation[:, :, :512]
    scale_msa = modulation[:, :, 512:1024]
    hidden = F.layer_norm(x, (512,), eps=1e-6)
    hidden = hidden * (1 + scale_msa) + shift_msa
    width = x.shape[1]
    q = F.linear(hidden, q_weight, q_bias).reshape(2, width, 8, 64).transpose(1, 2)
    k = F.linear(hidden, k_weight, k_bias).reshape(2, width, 8, 64).transpose(1, 2)
    v = F.linear(hidden, v_weight, v_bias).reshape(2, width, 8, 64).transpose(1, 2)
    q = F.layer_norm(q, (64,), q_norm_weight, q_norm_bias, 1e-5)
    k = F.layer_norm(k, (64,), k_norm_weight, k_norm_bias, 1e-5)
    return modulation, q, k, v


def _dit_attention_preamble(
    x: torch.Tensor,
    time_embedding: torch.Tensor,
    adaln_weight: torch.Tensor,
    adaln_bias: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the fixed MiniCPM-o DiT block front into one NPU graph.

    The production profile guards the model-specific 512-wide, 8x64-head
    layout before entering this function. Returning the complete modulation
    tensor lets the eager convolution and the separately compiled MLP consume
    the same values without repeating the AdaLN projection.
    """
    modulation = F.linear(F.silu(time_embedding), adaln_weight, adaln_bias)
    return _dit_attention_from_modulation(
        x,
        modulation,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
    )


def _dit_attention_preamble_from_modulation(
    x: torch.Tensor,
    modulation: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention preamble consuming one row from the wide AdaLN bank."""
    return _dit_attention_from_modulation(
        x,
        modulation,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
    )


def _dit_attention_preamble_qkv_pack(
    x: torch.Tensor,
    time_embedding: torch.Tensor,
    adaln_weight: torch.Tensor,
    adaln_bias: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention preamble using one native BSH-to-BNSD QKV layout node."""
    modulation = F.linear(F.silu(time_embedding), adaln_weight, adaln_bias)
    shift_msa = modulation[:, :, :512]
    scale_msa = modulation[:, :, 512:1024]
    hidden = F.layer_norm(x, (512,), eps=1e-6)
    hidden = hidden * (1 + scale_msa) + shift_msa
    q = F.linear(hidden, q_weight, q_bias)
    k = F.linear(hidden, k_weight, k_bias)
    v = F.linear(hidden, v_weight, v_bias)
    q, k, v = torch.ops._C_ascend.npu_minicpmo_qkv_pack(q, k, v)
    q = F.layer_norm(q, (64,), q_norm_weight, q_norm_bias, 1e-5)
    k = F.layer_norm(k, (64,), k_norm_weight, k_norm_bias, 1e-5)
    return modulation, q, k, v


def _dit_conv_mlp_residual(
    hidden: torch.Tensor,
    conv_input: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_conv: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed 910C DiT Conv/cache + gated MLP megagraph partition."""
    cache1, cache2 = cnn_cache.split((512, 512), dim=1)
    first_input = torch.cat((cache1, conv_input.transpose(1, 2)), dim=2)
    new_cache1 = first_input[:, :, -2:]
    convolution = F.conv1d(first_input, conv1_weight, conv1_bias).transpose(1, 2)
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    second_input = torch.cat((cache2, convolution.transpose(1, 2)), dim=2)
    new_cache2 = second_input[:, :, -2:]
    convolution = F.conv1d(second_input, conv2_weight, conv2_bias).transpose(1, 2)
    hidden = hidden + gate_conv * convolution
    hidden = _dit_mlp_residual(
        hidden,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    return hidden, torch.cat((new_cache1, new_cache2), dim=1)


def _dit_fused_conv_mlp_residual(
    hidden: torch.Tensor,
    conv_input: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_conv: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    conv1_flat_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_flat_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """910C Conv/cache + MLP graph using the native causal-pack boundary."""
    cache1, cache2 = cnn_cache.split((512, 512), dim=1)
    packed, new_cache1 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(conv_input, cache1)
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(2, 50, 512)
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(convolution, cache2)
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(2, 50, 512)
    hidden = hidden + gate_conv * convolution
    hidden = _dit_mlp_residual(
        hidden,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    return hidden, torch.cat((new_cache1, new_cache2), dim=1)


def _dit_cache_major_conv_mlp_residual(
    hidden: torch.Tensor,
    conv_input: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_conv: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    conv1_flat_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_flat_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """910C Conv/cache graph retaining contiguous ``[batch, taps, channels]`` state."""
    cache1, cache2 = cnn_cache.split((512, 512), dim=2)
    packed, new_cache1 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(conv_input, cache1)
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(2, 50, 512)
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(convolution, cache2)
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(2, 50, 512)
    hidden = hidden + gate_conv * convolution
    hidden = _dit_mlp_residual(
        hidden,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    return hidden, torch.cat((new_cache1, new_cache2), dim=2)


def _dit_cache_major_post_attention_conv_mlp_residual(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_msa: torch.Tensor,
    shift_conv: torch.Tensor,
    scale_conv: torch.Tensor,
    gate_conv: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    conv1_flat_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_flat_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """910C post-attention residual, norm3, Conv/cache, and MLP graph."""
    hidden = hidden + gate_msa * attention
    conv_input = F.layer_norm(hidden, (512,), eps=1e-6)
    conv_input = conv_input * (1 + scale_conv) + shift_conv
    return _dit_cache_major_conv_mlp_residual(
        hidden,
        conv_input,
        cnn_cache,
        gate_conv,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        conv1_flat_weight,
        conv1_bias,
        conv_norm_weight,
        conv_norm_bias,
        conv2_flat_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )


def _dit_fused_conv_linear_mlp_residual(
    hidden: torch.Tensor,
    conv_input: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_conv: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    conv1_flat_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_flat_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """910C graph with two fused causal-pack + Cube projection stages."""
    cache1, cache2 = cnn_cache.split((512, 512), dim=1)
    convolution, new_cache1 = torch.ops._C_ascend.npu_minicpmo_causal_conv_linear(
        conv_input,
        cache1,
        conv1_flat_weight,
        conv1_bias,
    )
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    convolution, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_linear(
        convolution,
        cache2,
        conv2_flat_weight,
        conv2_bias,
    )
    hidden = hidden + gate_conv * convolution
    hidden = _dit_mlp_residual(
        hidden,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    return hidden, torch.cat((new_cache1, new_cache2), dim=1)


def _dit_fused_conv_block_mlp_residual(
    hidden: torch.Tensor,
    conv_input: torch.Tensor,
    cnn_cache: torch.Tensor,
    gate_conv: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    conv1_flat_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_flat_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile the aggressive Conv profile as a fully visible GE graph.

    The eager MIX kernel is valuable in isolation, but treating the whole
    block as one custom node prevents GE from optimizing its GEMMs,
    normalization, activation, residual, and the following MLP together.  Use
    the native causal-pack kernel only for the layout transform and let
    TorchAir lower the compute-heavy portion through its normal ATen path.
    """
    return _dit_fused_conv_mlp_residual(
        hidden,
        conv_input,
        cnn_cache,
        gate_conv,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        conv1_flat_weight,
        conv1_bias,
        conv_norm_weight,
        conv_norm_bias,
        conv2_flat_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )


def _dit_explicit_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Small-shape attention that remains decomposed inside the GE graph."""
    batch_size, num_heads, query_width, head_dim = query.shape
    key_width = key.shape[2]
    flat_query = query.reshape(batch_size * num_heads, query_width, head_dim)
    flat_key = key.reshape(batch_size * num_heads, key_width, head_dim)
    flat_value = value.reshape(batch_size * num_heads, key_width, head_dim)
    scores = torch.bmm(flat_query, flat_key.transpose(1, 2)) * 0.125
    attention = torch.bmm(torch.softmax(scores, dim=-1), flat_value)
    return attention.reshape(batch_size, num_heads, query_width, head_dim)


def _dit_fused_full_block(
    x: torch.Tensor,
    time_embedding: torch.Tensor,
    att_cache: torch.Tensor,
    cnn_cache: torch.Tensor,
    adaln_weight: torch.Tensor,
    adaln_bias: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    conv1_flat_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_norm_bias: torch.Tensor,
    conv2_flat_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    explicit_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GE-visible steady DiT block with cached attention, Conv, and MLP."""
    modulation, q, k, v = _dit_attention_preamble(
        x,
        time_embedding,
        adaln_weight,
        adaln_bias,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
    )
    k_cache, v_cache = att_cache.chunk(2, dim=3)
    k = torch.cat((k, k_cache), dim=2)
    v = torch.cat((v, v_cache), dim=2)
    new_att_cache = torch.cat((k, v), dim=3)
    if explicit_attention:
        attention = _dit_explicit_attention(q, k, v)
    else:
        attention = F.scaled_dot_product_attention(q, k, v)
    attention = attention.transpose(1, 2).reshape(x.shape[0], x.shape[1], 512)
    attention = F.linear(attention, proj_weight, proj_bias)
    modulations = modulation.chunk(9, dim=-1)
    hidden = x + modulations[2] * attention
    conv_input = F.layer_norm(hidden, (512,), eps=1e-6)
    conv_input = conv_input * (1 + modulations[7]) + modulations[6]
    hidden, new_cnn_cache = _dit_fused_conv_mlp_residual(
        hidden,
        conv_input,
        cnn_cache,
        modulations[8],
        modulations[3],
        modulations[4],
        modulations[5],
        conv1_flat_weight,
        conv1_bias,
        conv_norm_weight,
        conv_norm_bias,
        conv2_flat_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )
    return hidden, new_cnn_cache, new_att_cache


class _DiTFullStackGraph(nn.Module):
    """All DiT blocks behind one GE boundary for a fixed cache shape."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cnn_results: list[torch.Tensor] = []
        att_results: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            conv1 = block.conv.block[1]
            conv_norm = block.conv.block[3]
            conv2 = block.conv.block[6]
            hidden, new_cnn, new_att = _dit_fused_full_block(
                hidden,
                time_embedding,
                att_cache[block_idx],
                cnn_cache[block_idx],
                block.adaLN_modulation[1].weight,
                block.adaLN_modulation[1].bias,
                block.attn.to_q.weight,
                block.attn.to_q.bias,
                block.attn.to_k.weight,
                block.attn.to_k.bias,
                block.attn.to_v.weight,
                block.attn.to_v.bias,
                block.attn.q_norm.weight,
                block.attn.q_norm.bias,
                block.attn.k_norm.weight,
                block.attn.k_norm.bias,
                block.attn.proj.weight,
                block.attn.proj.bias,
                conv1._minicpmo_flat_weight,
                conv1.bias,
                conv_norm.weight,
                conv_norm.bias,
                conv2._minicpmo_flat_weight,
                conv2.bias,
                block.mlp.fc1.weight,
                block.mlp.fc1.bias,
                block.mlp.fc2.weight,
                block.mlp.fc2.bias,
                True,
            )
            cnn_results.append(new_cnn)
            att_results.append(new_att)
        return hidden, torch.stack(cnn_results), torch.stack(att_results)


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    projected_speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(
        self,
        token2wav: Any,
        *,
        npu_dit_mlp_graph: Any = None,
        npu_dit_mlp_graph_width: Any = None,
        npu_dit_graph_buckets: Any = None,
        npu_dit_preamble_graph: Any = None,
        npu_dit_wide_adaln: Any = None,
        npu_dit_conv_mlp_graph: Any = None,
        npu_dit_prompt_conv_mlp_graph: Any = None,
        npu_dit_full_block_graph: Any = None,
        npu_dit_full_stack_graph: Any = None,
        npu_dit_full_block_cache_buckets: Any = None,
        npu_dit_fused_conv_pack: Any = None,
        npu_dit_cache_major: Any = None,
        npu_dit_post_attn_graph: Any = None,
        npu_dit_qkv_pack: Any = None,
        npu_dit_attn_cache_out: Any = None,
        npu_cfm_stacked_cache_out: Any = None,
        npu_single_request_cache_passthrough: Any = None,
        npu_dit_fused_conv_block: Any = None,
        npu_dit_fused_conv_linear: Any = None,
    ):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        hift_parameter = next(self.hift.parameters(), None)
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "cfm_timeline_base",
            1 - torch.cos(torch.linspace(0, 1, self.n_timesteps + 1, dtype=torch.float32) * 0.5 * torch.pi),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        self._timeline_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
        self._cfm_delta_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
        self._timestep_embedding_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._cfg_workspace: dict[tuple[str, tuple[int, ...], torch.dtype, str, int | None], torch.Tensor] = {}
        self._npu_cfm_graphs: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._npu_cfm_graph_disabled = False
        self._npu_dit_mlp_graph_enabled = _npu_dit_mlp_graph_enabled(npu_dit_mlp_graph)
        self._npu_dit_mlp_graph_width = _npu_dit_mlp_graph_width(npu_dit_mlp_graph_width)
        extra_graph_widths = _npu_dit_graph_buckets(npu_dit_graph_buckets)
        self._npu_dit_graph_widths = tuple(
            dict.fromkeys((self._npu_dit_mlp_graph_width, *extra_graph_widths))
        )
        self._npu_dit_mlp_graph: Any | None = None
        self._npu_dit_mlp_graph_disabled = False
        self._npu_dit_mlp_graph_disabled_widths: set[int] = set()
        self._npu_dit_mlp_graph_used = False
        self._npu_dit_preamble_graph_enabled = _npu_dit_preamble_graph_enabled(npu_dit_preamble_graph)
        self._npu_dit_preamble_graph: Any | None = None
        self._npu_dit_qkv_preamble_graph: Any | None = None
        self._npu_dit_preamble_graph_disabled = False
        self._npu_dit_preamble_graph_disabled_widths: set[int] = set()
        self._npu_dit_preamble_graph_used = False
        self._npu_dit_wide_adaln_enabled = _npu_dit_wide_adaln_enabled(
            npu_dit_wide_adaln
        )
        self._npu_dit_wide_adaln_graph: Any | None = None
        self._npu_dit_wide_adaln_steps_graph: Any | None = None
        self._npu_dit_wide_adaln_used = False
        self._npu_dit_wide_adaln_steps_used = False
        self.register_buffer(
            "_npu_dit_wide_adaln_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_npu_dit_wide_adaln_bias",
            None,
            persistent=False,
        )
        self._npu_dit_conv_mlp_graph_enabled = _npu_dit_conv_mlp_graph_enabled(npu_dit_conv_mlp_graph)
        self._npu_dit_conv_mlp_graph: Any | None = None
        self._npu_dit_conv_mlp_graph_disabled = False
        self._npu_dit_conv_mlp_graph_used = False
        self._npu_dit_prompt_conv_mlp_graph_enabled = _npu_dit_prompt_conv_mlp_graph_enabled(
            npu_dit_prompt_conv_mlp_graph
        )
        self._npu_dit_prompt_conv_mlp_graph: Any | None = None
        self._npu_dit_prompt_conv_mlp_graph_disabled_widths: set[int] = set()
        self._npu_dit_prompt_conv_mlp_graph_used_widths: set[int] = set()
        self._npu_dit_full_block_graph_enabled = _npu_dit_full_block_graph_enabled(
            npu_dit_full_block_graph
        )
        self._npu_dit_full_block_cache_buckets = _npu_dit_full_block_cache_buckets(
            npu_dit_full_block_cache_buckets
        )
        self._npu_dit_full_block_graph: Any | None = None
        self._npu_dit_full_block_graph_disabled_lengths: set[int] = set()
        self._npu_dit_full_block_graph_used_lengths: set[int] = set()
        self._npu_dit_full_stack_graph_enabled = _npu_dit_full_stack_graph_enabled(
            npu_dit_full_stack_graph
        )
        self._npu_dit_full_stack_graph: Any | None = None
        self._npu_dit_full_stack_graph_disabled_lengths: set[int] = set()
        self._npu_dit_full_stack_graph_used_lengths: set[int] = set()
        self._npu_dit_fused_conv_pack_enabled = _npu_dit_fused_conv_pack_enabled(npu_dit_fused_conv_pack)
        self._npu_dit_fused_conv_pack_used = False
        self._npu_dit_cache_major_enabled = _npu_dit_cache_major_enabled(npu_dit_cache_major)
        self._npu_dit_cache_major_used = False
        self._npu_dit_post_attn_graph_enabled = _npu_dit_post_attn_graph_enabled(
            npu_dit_post_attn_graph
        )
        self._npu_dit_post_attn_graph_used = False
        self._npu_dit_qkv_pack_enabled = _npu_dit_qkv_pack_enabled(npu_dit_qkv_pack)
        self._npu_dit_qkv_pack_used = False
        self._npu_dit_attn_cache_out_enabled = _npu_dit_attn_cache_out_enabled(
            npu_dit_attn_cache_out
        )
        self._npu_dit_attn_cache_out_used = False
        self._npu_cfm_stacked_cache_out_enabled = _npu_cfm_stacked_cache_out_enabled(
            npu_cfm_stacked_cache_out
        )
        self._npu_cfm_stacked_cache_out_used = False
        self._npu_single_request_cache_passthrough_enabled = (
            _npu_single_request_cache_passthrough_enabled(
                npu_single_request_cache_passthrough
            )
        )
        self._npu_single_request_cache_passthrough_used = False
        self._npu_dit_fused_conv_block_enabled = _npu_dit_fused_conv_block_enabled(npu_dit_fused_conv_block)
        self._npu_dit_fused_conv_block_used = False
        self._npu_dit_fused_conv_linear_enabled = _npu_dit_fused_conv_linear_enabled(npu_dit_fused_conv_linear)
        self._npu_dit_fused_conv_linear_used = False
        if self._npu_dit_wide_adaln_enabled and not self._npu_dit_preamble_graph_enabled:
            self._npu_dit_wide_adaln_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN requires the DiT preamble graph; disabling it"
            )
        if self._npu_dit_wide_adaln_enabled and self._npu_dit_qkv_pack_enabled:
            self._npu_dit_qkv_pack_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN currently uses the ordinary QKV preamble; disabling native QKV pack"
            )
        if self._npu_dit_cache_major_enabled and (
            not self._npu_dit_fused_conv_pack_enabled or self._npu_dit_fused_conv_linear_enabled
        ):
            self._npu_dit_cache_major_enabled = False
            logger.warning(
                "MiniCPM-o NPU cache-major state requires causal-pack without fused Conv+Linear; disabling it"
            )
        if self._npu_dit_post_attn_graph_enabled and not self._npu_dit_cache_major_enabled:
            self._npu_dit_post_attn_graph_enabled = False
            logger.warning(
                "MiniCPM-o NPU post-attention graph requires cache-major Conv+MLP; disabling it"
            )
        self._warmup_npu_dit_mlp_graph()
        self._warmup_npu_dit_wide_adaln_graph()
        self._warmup_npu_dit_preamble_graph()
        self._warmup_npu_dit_conv_mlp_graph()
        self._warmup_npu_dit_prompt_conv_mlp_graphs()
        self._warmup_npu_dit_full_block_graphs()
        self._warmup_npu_dit_full_stack_graphs()

    def _warmup_npu_dit_mlp_graph(self) -> None:
        """Compile the fixed-width post-convolution partition during startup."""
        if not self._npu_dit_mlp_graph_enabled:
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        if not blocks:
            self._npu_dit_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT MLP graph disabled: estimator blocks are unavailable")
            return
        block = blocks[0]
        weight = getattr(getattr(block, "mlp", None), "fc1", None)
        weight = getattr(weight, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.device.type != "npu":
            self._npu_dit_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT MLP graph disabled: estimator is not on NPU")
            return
        hidden_size = int(weight.shape[1])
        shift = weight.new_zeros((2, 1, hidden_size))
        scale = weight.new_zeros((2, 1, hidden_size))
        gate = weight.new_zeros((2, 1, hidden_size))
        graph_fn = self._get_npu_dit_mlp_graph()
        if graph_fn is None:
            return
        for width in self._npu_dit_graph_widths:
            try:
                x = weight.new_zeros((2, width, hidden_size))
                with torch.inference_mode():
                    graph_fn(
                        x,
                        shift,
                        scale,
                        gate,
                        block.mlp.fc1.weight,
                        block.mlp.fc1.bias,
                        block.mlp.fc2.weight,
                        block.mlp.fc2.bias,
                    )
                torch.npu.synchronize()
                logger.info(
                    "Compiled MiniCPM-o NPU DiT MLP graph partition for CFG batch=2, width=%d, hidden=%d",
                    width,
                    hidden_size,
                )
            except Exception:
                self._npu_dit_mlp_graph_disabled_widths.add(width)
                logger.warning(
                    "MiniCPM-o NPU DiT MLP graph compilation failed at width=%d; using eager blocks for that width",
                    width,
                    exc_info=True,
                )
        if self._npu_dit_mlp_graph_width in self._npu_dit_mlp_graph_disabled_widths:
            self._npu_dit_mlp_graph_disabled = True

    def _get_npu_dit_mlp_graph(self):
        if self._npu_dit_mlp_graph_disabled:
            return None
        if self._npu_dit_mlp_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_mlp_graph = torch.compile(
                _dit_mlp_residual,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_mlp_graph

    def _warmup_npu_dit_wide_adaln_graph(self) -> None:
        """Pack and compile the current-step, all-block AdaLN projection."""
        if not self._npu_dit_wide_adaln_enabled:
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        if not blocks or len(blocks) != 16:
            self._npu_dit_wide_adaln_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN disabled: expected 16 estimator blocks"
            )
            return
        try:
            projections = [block.adaLN_modulation[1] for block in blocks]
        except (AttributeError, IndexError, TypeError):
            self._npu_dit_wide_adaln_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN disabled: block projections are unavailable"
            )
            return
        if not all(
            isinstance(projection, nn.Linear)
            and tuple(projection.weight.shape) == (9 * 512, 512)
            and projection.bias is not None
            and projection.weight.device.type == "npu"
            for projection in projections
        ):
            self._npu_dit_wide_adaln_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN disabled: block projections are incompatible"
            )
            return

        self._npu_dit_wide_adaln_weight = torch.cat(
            [projection.weight.detach() for projection in projections],
            dim=0,
        ).contiguous()
        self._npu_dit_wide_adaln_bias = torch.cat(
            [projection.bias.detach() for projection in projections],
            dim=0,
        ).contiguous()
        graph_fn = self._get_npu_dit_wide_adaln_graph()
        # Exercise both the packed weights and biases. An all-zero embedding
        # would reduce this startup parity gate to a bias-only comparison.
        time_embedding = projections[0].weight.new_full((2, 1, 512), 0.125)
        try:
            with torch.inference_mode():
                actual = graph_fn(
                    time_embedding,
                    self._npu_dit_wide_adaln_weight,
                    self._npu_dit_wide_adaln_bias,
                )
                expected = torch.stack(
                    [block.adaLN_modulation(time_embedding) for block in blocks],
                    dim=2,
                )
                difference = (actual - expected).abs()
                max_abs_drift = float(difference.max().item())
                if not torch.isfinite(actual).all() or (
                    max_abs_drift > _NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT
                ):
                    raise RuntimeError(
                        "MiniCPM-o wide AdaLN exceeded its startup drift bound: "
                        f"max_abs_drift={max_abs_drift:.9g}, "
                        f"limit={_NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT:.9g}"
                    )
                step_embeddings = time_embedding.unsqueeze(0).expand(
                    self.n_timesteps,
                    -1,
                    -1,
                    -1,
                ).contiguous()
                step_actual = self._get_npu_dit_wide_adaln_steps_graph()(
                    step_embeddings,
                    self._npu_dit_wide_adaln_weight,
                    self._npu_dit_wide_adaln_bias,
                )
                step_expected = actual.unsqueeze(0).expand_as(step_actual)
                step_difference = (step_actual - step_expected).abs()
                step_max_abs_drift = float(step_difference.max().item())
                if not torch.isfinite(step_actual).all() or (
                    step_max_abs_drift > _NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT
                ):
                    raise RuntimeError(
                        "MiniCPM-o all-step wide AdaLN exceeded its startup drift bound: "
                        f"max_abs_drift={step_max_abs_drift:.9g}, "
                        f"limit={_NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT:.9g}"
                    )
            torch.npu.synchronize()
            logger.info(
                "Compiled bounded-drift MiniCPM-o wide AdaLN graphs for %d steps x 16 blocks; "
                "single_max_abs_drift=%.9g, steps_max_abs_drift=%.9g",
                self.n_timesteps,
                max_abs_drift,
                step_max_abs_drift,
            )
        except Exception:
            self._npu_dit_wide_adaln_enabled = False
            self._npu_dit_wide_adaln_graph = None
            self._npu_dit_wide_adaln_steps_graph = None
            self._npu_dit_wide_adaln_weight = None
            self._npu_dit_wide_adaln_bias = None
            logger.warning(
                "MiniCPM-o wide AdaLN compilation/parity gate failed; using per-block projections",
                exc_info=True,
            )

    def _get_npu_dit_wide_adaln_graph(self):
        if self._npu_dit_wide_adaln_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_wide_adaln_graph = torch.compile(
                _dit_wide_adaln,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_wide_adaln_graph

    def _get_npu_dit_wide_adaln_steps_graph(self):
        if self._npu_dit_wide_adaln_steps_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_wide_adaln_steps_graph = torch.compile(
                _dit_wide_adaln_steps,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_wide_adaln_steps_graph

    @staticmethod
    def _dit_preamble_compatible(block: nn.Module, width: int) -> bool:
        return (
            width > 0
            and not block.training
            and int(block.attn.num_heads) == 8
            and int(block.attn.head_dim) == 64
            and int(block.attn.to_q.in_features) == 512
            and int(block.adaLN_modulation[1].out_features) == 9 * 512
            and block.norm1.weight is None
            and block.norm1.bias is None
            and float(block.norm1.eps) == 1e-6
            and float(block.attn.q_norm.eps) == 1e-5
            and float(block.attn.k_norm.eps) == 1e-5
        )

    def _warmup_npu_dit_preamble_graph(self) -> None:
        if not self._npu_dit_preamble_graph_enabled:
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        block = blocks[0] if blocks else None
        weight = getattr(getattr(getattr(block, "attn", None), "to_q", None), "weight", None)
        if (
            block is None
            or not isinstance(weight, torch.Tensor)
            or weight.device.type != "npu"
            or not self._dit_preamble_compatible(block, self._npu_dit_mlp_graph_width)
        ):
            self._npu_dit_preamble_graph_disabled = True
            logger.warning(
                "MiniCPM-o NPU DiT preamble graph disabled: estimator does not match the fixed 2x50x512, 8x64 layout"
            )
            return
        time_embedding = weight.new_zeros((2, 1, 512))
        wide_modulations = None
        if self._npu_dit_wide_adaln_enabled:
            wide_modulations = self._get_npu_dit_wide_adaln_graph()(
                time_embedding,
                self._npu_dit_wide_adaln_weight,
                self._npu_dit_wide_adaln_bias,
            )
        if self._npu_dit_qkv_pack_enabled:
            try:
                from vllm_ascend.compilation.minicpmo_causal_conv import (
                    register_minicpmo_qkv_pack_converter,
                )

                register_minicpmo_qkv_pack_converter()
            except (ImportError, AttributeError, RuntimeError):
                self._npu_dit_qkv_pack_enabled = False
                logger.warning(
                    "MiniCPM-o native QKV layout pack unavailable; using ordinary transposes",
                    exc_info=True,
                )
        for width in self._npu_dit_graph_widths:
            try:
                graph_fn = self._get_npu_dit_preamble_graph(width)
                if graph_fn is None:
                    continue
                x = weight.new_zeros((2, width, 512))
                with torch.inference_mode():
                    self._call_npu_dit_preamble_graph(
                        graph_fn,
                        block,
                        x,
                        time_embedding,
                        None if wide_modulations is None else wide_modulations[:, :, 0, :],
                    )
                torch.npu.synchronize()
                if self._npu_dit_qkv_pack_enabled and width == self._npu_dit_mlp_graph_width:
                    logger.info(
                        "Compiled MiniCPM-o NPU DiT native-QKV preamble graph for 2x%dx512, 8x64 heads",
                        width,
                    )
                else:
                    logger.info(
                        "Compiled MiniCPM-o NPU DiT attention preamble graph for 2x%dx512, 8x64 heads",
                        width,
                    )
            except Exception:
                if self._npu_dit_qkv_pack_enabled and width == self._npu_dit_mlp_graph_width:
                    self._npu_dit_qkv_pack_enabled = False
                    self._npu_dit_qkv_preamble_graph = None
                    logger.warning(
                        "MiniCPM-o native QKV preamble compilation failed; retrying ordinary transposes",
                        exc_info=True,
                    )
                    try:
                        graph_fn = self._get_npu_dit_preamble_graph(width)
                        with torch.inference_mode():
                            self._call_npu_dit_preamble_graph(
                                graph_fn,
                                block,
                                x,
                                time_embedding,
                            )
                        torch.npu.synchronize()
                        logger.info(
                            "Compiled MiniCPM-o NPU DiT attention preamble graph for 2x%dx512 after QKV fallback",
                            width,
                        )
                        continue
                    except Exception:
                        logger.warning(
                            "MiniCPM-o ordinary-transpose preamble retry failed at width=%d",
                            width,
                            exc_info=True,
                        )
                self._npu_dit_preamble_graph_disabled_widths.add(width)
                logger.warning(
                    "MiniCPM-o NPU DiT preamble graph compilation failed at width=%d; using eager attention for that width",
                    width,
                    exc_info=True,
                )
        if self._npu_dit_mlp_graph_width in self._npu_dit_preamble_graph_disabled_widths:
            self._npu_dit_preamble_graph_disabled = True

    def _get_npu_dit_preamble_graph(self, width: int | None = None):
        if self._npu_dit_preamble_graph_disabled:
            return None
        from torch_npu.dynamo import torchair

        _ensure_torchair_broadcast_alias()
        compiler_config = torchair.CompilerConfig()
        if self._npu_dit_qkv_pack_enabled and width == self._npu_dit_mlp_graph_width:
            if self._npu_dit_qkv_preamble_graph is None:
                self._npu_dit_qkv_preamble_graph = torch.compile(
                    _dit_attention_preamble_qkv_pack,
                    backend=torchair.get_npu_backend(compiler_config=compiler_config),
                    fullgraph=True,
                    dynamic=False,
                )
            return self._npu_dit_qkv_preamble_graph
        if self._npu_dit_preamble_graph is None:
            self._npu_dit_preamble_graph = torch.compile(
                (
                    _dit_attention_preamble_from_modulation
                    if self._npu_dit_wide_adaln_enabled
                    else _dit_attention_preamble
                ),
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_preamble_graph

    def _call_npu_dit_preamble_graph(
        self,
        graph_fn: Any,
        block: nn.Module,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_args = (
            block.attn.to_q.weight,
            block.attn.to_q.bias,
            block.attn.to_k.weight,
            block.attn.to_k.bias,
            block.attn.to_v.weight,
            block.attn.to_v.bias,
            block.attn.q_norm.weight,
            block.attn.q_norm.bias,
            block.attn.k_norm.weight,
            block.attn.k_norm.bias,
        )
        if modulation is not None:
            return graph_fn(hidden, modulation, *attention_args)
        return graph_fn(
            hidden,
            time_embedding,
            block.adaLN_modulation[1].weight,
            block.adaLN_modulation[1].bias,
            *attention_args,
        )

    @staticmethod
    def _dit_conv_mlp_layout_compatible(block: nn.Module) -> bool:
        conv1 = block.conv.block[1]
        conv2 = block.conv.block[6]
        return (
            not block.training
            and int(conv1.in_channels) == 512
            and int(conv1.out_channels) == 512
            and tuple(conv1.kernel_size) == (3,)
            and int(conv2.in_channels) == 512
            and int(conv2.out_channels) == 512
            and tuple(conv2.kernel_size) == (3,)
            and block.norm2.weight is None
            and block.norm2.bias is None
            and float(block.norm2.eps) == 1e-6
            and int(block.mlp.fc1.in_features) == 512
            and int(block.mlp.fc1.out_features) == 2048
            and int(block.mlp.fc2.out_features) == 512
        )

    @staticmethod
    def _dit_conv_mlp_compatible(block: nn.Module, width: int) -> bool:
        return width == 50 and BatchedToken2Wav._dit_conv_mlp_layout_compatible(block)

    @staticmethod
    def _dit_post_attention_compatible(block: nn.Module, width: int) -> bool:
        return (
            BatchedToken2Wav._dit_conv_mlp_compatible(block, width)
            and block.norm3.weight is None
            and block.norm3.bias is None
            and float(block.norm3.eps) == 1e-6
        )

    def _warmup_npu_dit_conv_mlp_graph(self) -> None:
        if not self._npu_dit_conv_mlp_graph_enabled:
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        block = blocks[0] if blocks else None
        weight = getattr(getattr(getattr(block, "mlp", None), "fc1", None), "weight", None)
        if (
            block is None
            or not isinstance(weight, torch.Tensor)
            or weight.device.type != "npu"
            or not self._dit_conv_mlp_compatible(block, self._npu_dit_mlp_graph_width)
            or (
                self._npu_dit_post_attn_graph_enabled
                and not self._dit_post_attention_compatible(
                    block,
                    self._npu_dit_mlp_graph_width,
                )
            )
        ):
            self._npu_dit_conv_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT Conv+MLP graph disabled: block layout is incompatible")
            return
        hidden = weight.new_zeros((2, 50, 512))
        modulation = weight.new_zeros((2, 1, 512))
        cnn_cache = weight.new_zeros(
            (2, 2, 1024) if self._npu_dit_cache_major_enabled else (2, 1024, 2)
        )
        conv1 = block.conv.block[1]
        conv_norm = block.conv.block[3]
        conv2 = block.conv.block[6]
        if self._npu_dit_fused_conv_linear_enabled:
            try:
                from vllm_ascend.compilation.minicpmo_causal_conv import (
                    register_minicpmo_causal_conv_linear_converter,
                )

                register_minicpmo_causal_conv_linear_converter()
            except (ImportError, AttributeError, RuntimeError):
                self._npu_dit_fused_conv_linear_enabled = False
                logger.warning(
                    "MiniCPM-o native causal Conv+Linear unavailable; falling back to the causal-pack graph",
                    exc_info=True,
                )
        if (
            self._npu_dit_fused_conv_block_enabled or self._npu_dit_fused_conv_pack_enabled
        ) and not self._npu_dit_fused_conv_linear_enabled:
            try:
                from vllm_ascend.compilation.minicpmo_causal_conv import (
                    register_minicpmo_causal_conv_pack_converter,
                )

                register_minicpmo_causal_conv_pack_converter()
            except (ImportError, AttributeError, RuntimeError):
                self._npu_dit_fused_conv_block_enabled = False
                self._npu_dit_fused_conv_pack_enabled = False
                logger.warning(
                    "MiniCPM-o native causal Conv pack unavailable; compiling the standard Conv+MLP graph",
                    exc_info=True,
                )
        if (
            self._npu_dit_fused_conv_linear_enabled
            or self._npu_dit_fused_conv_block_enabled
            or self._npu_dit_fused_conv_pack_enabled
        ):
            for estimator_block in blocks:
                for index in (1, 6):
                    convolution = estimator_block.conv.block[index]
                    convolution.register_buffer(
                        "_minicpmo_flat_weight",
                        convolution.weight.detach().permute(0, 2, 1).reshape(512, 1536).contiguous(),
                        persistent=False,
                    )
        try:
            graph_fn = self._get_npu_dit_conv_mlp_graph()
            if graph_fn is None:
                return
            with torch.inference_mode():
                if self._npu_dit_post_attn_graph_enabled:
                    graph_fn(
                        hidden,
                        hidden,
                        cnn_cache,
                        modulation,
                        modulation,
                        modulation,
                        modulation,
                        modulation,
                        modulation,
                        modulation,
                        self._dit_conv_graph_weight(conv1),
                        conv1.bias,
                        conv_norm.weight,
                        conv_norm.bias,
                        self._dit_conv_graph_weight(conv2),
                        conv2.bias,
                        block.mlp.fc1.weight,
                        block.mlp.fc1.bias,
                        block.mlp.fc2.weight,
                        block.mlp.fc2.bias,
                    )
                else:
                    graph_fn(
                        hidden,
                        hidden,
                        cnn_cache,
                        modulation,
                        modulation,
                        modulation,
                        modulation,
                        self._dit_conv_graph_weight(conv1),
                        conv1.bias,
                        conv_norm.weight,
                        conv_norm.bias,
                        self._dit_conv_graph_weight(conv2),
                        conv2.bias,
                        block.mlp.fc1.weight,
                        block.mlp.fc1.bias,
                        block.mlp.fc2.weight,
                        block.mlp.fc2.bias,
                    )
            torch.npu.synchronize()
            if self._npu_dit_post_attn_graph_enabled:
                logger.info(
                    "Compiled MiniCPM-o NPU DiT post-attention Conv+MLP megagraph for 2x50x512"
                )
            else:
                logger.info("Compiled MiniCPM-o NPU DiT Conv+MLP megagraph for 2x50x512")
        except Exception:
            self._npu_dit_conv_mlp_graph = None
            self._npu_dit_conv_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT Conv+MLP graph compilation failed; using split path", exc_info=True)

    def _get_npu_dit_conv_mlp_graph(self):
        if self._npu_dit_conv_mlp_graph_disabled:
            return None
        if self._npu_dit_conv_mlp_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            if self._npu_dit_fused_conv_linear_enabled:
                graph_partition = _dit_fused_conv_linear_mlp_residual
            elif self._npu_dit_post_attn_graph_enabled:
                graph_partition = _dit_cache_major_post_attention_conv_mlp_residual
            elif self._npu_dit_cache_major_enabled:
                graph_partition = _dit_cache_major_conv_mlp_residual
            elif self._npu_dit_fused_conv_block_enabled or self._npu_dit_fused_conv_pack_enabled:
                # Select the proven pack partition directly so the aggressive
                # and competition profiles share one canonical GE graph and
                # cache key.
                graph_partition = _dit_fused_conv_mlp_residual
            else:
                graph_partition = _dit_conv_mlp_residual
            self._npu_dit_conv_mlp_graph = torch.compile(
                graph_partition,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_conv_mlp_graph

    def _warmup_npu_dit_prompt_conv_mlp_graphs(self) -> None:
        """Compile regular Conv/cache + MLP graphs for non-stream buckets.

        The width-50 stream graph keeps its native causal-pack kernel. Wider
        setup/finalization buckets use the standard Conv1d partition so GE can
        optimize its layout transitions without depending on a fixed-width
        custom converter.
        """
        if not self._npu_dit_prompt_conv_mlp_graph_enabled:
            return
        widths = tuple(
            width for width in self._npu_dit_graph_widths if width != self._npu_dit_mlp_graph_width
        )
        if not widths:
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        block = blocks[0] if blocks else None
        weight = getattr(
            getattr(getattr(block, "mlp", None), "fc1", None),
            "weight",
            None,
        )
        if (
            block is None
            or not isinstance(weight, torch.Tensor)
            or weight.device.type != "npu"
            or not self._dit_conv_mlp_layout_compatible(block)
        ):
            self._npu_dit_prompt_conv_mlp_graph_disabled_widths.update(widths)
            logger.warning(
                "MiniCPM-o NPU prompt Conv+MLP graphs disabled: block layout is incompatible"
            )
            return
        graph_fn = self._get_npu_dit_prompt_conv_mlp_graph()
        if graph_fn is None:
            return
        modulation = weight.new_zeros((2, 1, 512))
        cnn_cache = weight.new_zeros((2, 1024, 2))
        conv1 = block.conv.block[1]
        conv_norm = block.conv.block[3]
        conv2 = block.conv.block[6]
        for width in widths:
            try:
                hidden = weight.new_zeros((2, width, 512))
                with torch.inference_mode():
                    graph_fn(
                        hidden,
                        hidden,
                        cnn_cache,
                        modulation,
                        modulation,
                        modulation,
                        modulation,
                        conv1.weight,
                        conv1.bias,
                        conv_norm.weight,
                        conv_norm.bias,
                        conv2.weight,
                        conv2.bias,
                        block.mlp.fc1.weight,
                        block.mlp.fc1.bias,
                        block.mlp.fc2.weight,
                        block.mlp.fc2.bias,
                    )
                torch.npu.synchronize()
                logger.info(
                    "Compiled MiniCPM-o NPU prompt Conv+MLP megagraph for 2x%dx512",
                    width,
                )
            except Exception:
                self._npu_dit_prompt_conv_mlp_graph_disabled_widths.add(width)
                logger.warning(
                    "MiniCPM-o NPU prompt Conv+MLP graph compilation failed at width=%d; using split path",
                    width,
                    exc_info=True,
                )

    def _get_npu_dit_prompt_conv_mlp_graph(self):
        if self._npu_dit_prompt_conv_mlp_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_prompt_conv_mlp_graph = torch.compile(
                _dit_conv_mlp_residual,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_prompt_conv_mlp_graph

    def _warmup_npu_dit_full_block_graphs(self) -> None:
        """Compile complete steady block graphs for bounded cache lengths."""
        if not self._npu_dit_full_block_graph_enabled:
            return
        lengths = self._npu_dit_full_block_cache_buckets
        if not lengths:
            logger.warning("MiniCPM-o NPU full-block graph disabled: no cache buckets configured")
            self._npu_dit_full_block_graph_enabled = False
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        block = blocks[0] if blocks else None
        weight = getattr(getattr(getattr(block, "attn", None), "to_q", None), "weight", None)
        if (
            block is None
            or not isinstance(weight, torch.Tensor)
            or weight.device.type != "npu"
            or not self._dit_full_block_compatible(block)
            or not self._npu_dit_fused_conv_pack_enabled
        ):
            self._npu_dit_full_block_graph_enabled = False
            logger.warning("MiniCPM-o NPU full-block graph disabled: block or causal-pack layout is incompatible")
            return
        graph_fn = self._get_npu_dit_full_block_graph()
        hidden = weight.new_zeros((2, 50, 512))
        time_embedding = weight.new_zeros((2, 1, 512))
        cnn_cache = weight.new_zeros((2, 1024, 2))
        for cache_length in lengths:
            try:
                att_cache = weight.new_zeros((2, 8, cache_length, 128))
                with torch.inference_mode():
                    self._call_npu_dit_full_block_graph(
                        graph_fn,
                        block,
                        hidden,
                        time_embedding,
                        att_cache,
                        cnn_cache,
                    )
                torch.npu.synchronize()
                logger.info(
                    "Compiled MiniCPM-o NPU full DiT block graph for width=50, attention cache=%d",
                    cache_length,
                )
            except Exception:
                self._npu_dit_full_block_graph_disabled_lengths.add(cache_length)
                logger.warning(
                    "MiniCPM-o NPU full-block graph compilation failed at cache=%d; using split path",
                    cache_length,
                    exc_info=True,
                )

    @staticmethod
    def _dit_full_block_compatible(block: nn.Module) -> bool:
        return (
            BatchedToken2Wav._dit_preamble_compatible(block, 50)
            and BatchedToken2Wav._dit_conv_mlp_compatible(block, 50)
            and block.norm3.weight is None
            and block.norm3.bias is None
            and float(block.norm3.eps) == 1e-6
            and int(block.attn.proj.in_features) == 512
            and int(block.attn.proj.out_features) == 512
        )

    def _get_npu_dit_full_block_graph(self):
        if self._npu_dit_full_block_graph is None:
            from torch_npu.dynamo import torchair
            from vllm_ascend.compilation.minicpmo_fusion_attention import (
                register_minicpmo_fusion_attention_v3_converter,
            )

            _ensure_torchair_broadcast_alias()
            register_minicpmo_fusion_attention_v3_converter()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_full_block_graph = torch.compile(
                _dit_fused_full_block,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_full_block_graph

    def _call_npu_dit_full_block_graph(
        self,
        graph_fn: Any,
        block: nn.Module,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv1 = block.conv.block[1]
        conv_norm = block.conv.block[3]
        conv2 = block.conv.block[6]
        return graph_fn(
            hidden,
            time_embedding,
            att_cache,
            cnn_cache,
            block.adaLN_modulation[1].weight,
            block.adaLN_modulation[1].bias,
            block.attn.to_q.weight,
            block.attn.to_q.bias,
            block.attn.to_k.weight,
            block.attn.to_k.bias,
            block.attn.to_v.weight,
            block.attn.to_v.bias,
            block.attn.q_norm.weight,
            block.attn.q_norm.bias,
            block.attn.k_norm.weight,
            block.attn.k_norm.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
            self._dit_conv_graph_weight(conv1),
            conv1.bias,
            conv_norm.weight,
            conv_norm.bias,
            self._dit_conv_graph_weight(conv2),
            conv2.bias,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
            block.mlp.fc2.weight,
            block.mlp.fc2.bias,
        )

    def _warmup_npu_dit_full_stack_graphs(self) -> None:
        """Compile the whole 16-block stack for selected steady cache shapes."""
        if not self._npu_dit_full_stack_graph_enabled:
            return
        lengths = self._npu_dit_full_block_cache_buckets
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        weight = getattr(getattr(estimator, "in_proj", None), "weight", None)
        if (
            not lengths
            or not isinstance(blocks, nn.ModuleList)
            or not blocks
            or not isinstance(weight, torch.Tensor)
            or weight.device.type != "npu"
            or not self._npu_dit_fused_conv_pack_enabled
            or not all(self._dit_full_block_compatible(block) for block in blocks)
        ):
            self._npu_dit_full_stack_graph_enabled = False
            logger.warning("MiniCPM-o NPU full-stack graph disabled: stack or cache buckets are incompatible")
            return
        graph_fn = self._get_npu_dit_full_stack_graph(blocks)
        depth = len(blocks)
        hidden = weight.new_zeros((2, 50, 512))
        time_embedding = weight.new_zeros((2, 1, 512))
        cnn_cache = weight.new_zeros((depth, 2, 1024, 2))
        for cache_length in lengths:
            try:
                att_cache = weight.new_zeros((depth, 2, 8, cache_length, 128))
                with torch.inference_mode():
                    graph_fn(hidden, time_embedding, att_cache, cnn_cache)
                torch.npu.synchronize()
                logger.info(
                    "Compiled MiniCPM-o NPU full DiT stack graph: blocks=%d, width=50, attention cache=%d",
                    depth,
                    cache_length,
                )
            except Exception:
                self._npu_dit_full_stack_graph_disabled_lengths.add(cache_length)
                logger.warning(
                    "MiniCPM-o NPU full-stack graph compilation failed at cache=%d; using split path",
                    cache_length,
                    exc_info=True,
                )

    def _get_npu_dit_full_stack_graph(self, blocks: nn.ModuleList):
        if self._npu_dit_full_stack_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            module = _DiTFullStackGraph(blocks).eval()
            graph = torch.compile(
                module,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
            # The wrapper references the existing estimator parameters. Keep it
            # out of this module's registry so state_dict ownership is unchanged.
            object.__setattr__(self, "_npu_dit_full_stack_graph", graph)
        return self._npu_dit_full_stack_graph

    @staticmethod
    def _call_npu_dit_full_stack_graph(
        graph_fn: Any,
        estimator: nn.Module,
        estimator_input: torch.Tensor,
        time_embedding: torch.Tensor,
        old_cnn: torch.Tensor,
        old_att: torch.Tensor,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
    ) -> torch.Tensor:
        hidden = estimator.in_proj(estimator_input.transpose(1, 2))
        hidden, new_cnn, new_att = graph_fn(hidden, time_embedding, old_att, old_cnn)
        cnn_out.copy_(new_cnn)
        att_out[:, :, :, : new_att.shape[3], :].copy_(new_att)
        hidden = estimator.final_layer(hidden, time_embedding)
        return hidden.transpose(1, 2)

    def _dit_conv_graph_weight(self, convolution: nn.Conv1d) -> torch.Tensor:
        if not (
            self._npu_dit_fused_conv_linear_enabled
            or self._npu_dit_fused_conv_block_enabled
            or self._npu_dit_fused_conv_pack_enabled
        ):
            return convolution.weight
        flat_weight = getattr(convolution, "_minicpmo_flat_weight", None)
        if not isinstance(flat_weight, torch.Tensor):
            raise RuntimeError("MiniCPM-o native causal Conv weight was not prepacked")
        return flat_weight

    def _timeline_for(self, value: torch.Tensor) -> torch.Tensor:
        key = (value.device.type, value.device.index, value.dtype)
        timeline = self._timeline_cache.get(key)
        if timeline is None:
            timeline = self.cfm_timeline_base.to(device=value.device, dtype=value.dtype)
            self._timeline_cache[key] = timeline
        return timeline

    def _cfm_deltas_for(self, timeline: torch.Tensor) -> torch.Tensor:
        """Cache the invariant Euler step widths without changing rounding.

        CosyVoice derives every next width from the accumulated time rather
        than taking a direct timeline difference. Reproduce that recurrence
        once per device/dtype, then reuse the resulting scalars for every
        streamed chunk. This removes two tiny eager accelerator operations
        from each non-final CFM step while preserving the original update
        order and values.
        """
        key = (timeline.device.type, timeline.device.index, timeline.dtype)
        cached = self._cfm_delta_cache.get(key)
        if cached is not None:
            return cached
        time = timeline[0]
        dt = timeline[1] - timeline[0]
        deltas: list[torch.Tensor] = []
        for step in range(self.n_timesteps):
            deltas.append(dt)
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time
        cached = torch.stack(deltas).detach()
        self._cfm_delta_cache[key] = cached
        return cached

    def _cfg_pair(self, name: str, value: torch.Tensor, *, zero_unconditional: bool) -> torch.Tensor:
        key = (name, tuple(value.shape), value.dtype, value.device.type, value.device.index)
        pair = self._cfg_workspace.get(key)
        expected_shape = (int(value.shape[0]) * 2, *value.shape[1:])
        if pair is None or tuple(pair.shape) != expected_shape:
            pair = torch.empty(expected_shape, device=value.device, dtype=value.dtype)
            self._cfg_workspace[key] = pair
        batch_size = int(value.shape[0])
        pair[:batch_size].copy_(value)
        if zero_unconditional:
            pair[batch_size:].zero_()
        else:
            pair[batch_size:].copy_(value)
        return pair

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            # The reference speaker and projection weights are immutable for
            # the lifetime of this prompt-cache entry. Project once instead
            # of launching normalize + affine operators for every audio chunk.
            with self._autocast(values[2].device):
                projected_speaker = self.flow.spk_embed_affine_layer(
                    F.normalize(values[2], dim=1)
                ).detach()
            cached = PromptFeatures(
                speech_tokens=values[0],
                speaker_embedding=values[2],
                projected_speaker_embedding=projected_speaker,
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        self._prompt_features.pop((prompt_cache_id, prompt_wav), None)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.projected_speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    @staticmethod
    def _estimator_buffer_shapes(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
        *,
        cache_major: bool = False,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn_shape = (
            (depth, batch_size, cnn_width, cnn_channels)
            if cache_major
            else (depth, batch_size, cnn_channels, cnn_width)
        )
        att_shape = (depth, batch_size, heads, old_att_len + chunk_size, att_width)
        return cnn_shape, att_shape

    @classmethod
    def _estimator_buffers(
        cls,
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
        *,
        cache_major: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_shape, att_shape = cls._estimator_buffer_shapes(
            estimator,
            x,
            old_att,
            cache_major=cache_major,
        )
        return x.new_empty(cnn_shape), x.new_empty(att_shape)

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time_embedding: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        cnn_out: torch.Tensor | None = None,
        att_out: torch.Tensor | None = None,
        wide_modulations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cache_major = self._is_cache_major_cnn(cnn_cache)
        if cnn_out is None or att_out is None:
            if cnn_out is not None or att_out is not None:
                raise ValueError("cnn_out and att_out must be provided together")
            cnn_out, att_out = self._estimator_buffers(
                estimator,
                estimator_input,
                att_cache,
                cache_major=cache_major,
            )
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        graph_width = int(estimator_input.shape[2])
        use_mlp_graph = (
            self._npu_dit_mlp_graph_enabled
            and not self._npu_dit_mlp_graph_disabled
            and estimator_input.device.type == "npu"
            and int(estimator_input.shape[0]) == 2
            and graph_width in self._npu_dit_graph_widths
            and graph_width not in self._npu_dit_mlp_graph_disabled_widths
        )
        if use_mlp_graph:
            try:
                graph_fn = self._get_npu_dit_mlp_graph()
                if graph_fn is not None:
                    full_stack_cache_length = _dit_attention_cache_length(att_cache)
                    if (
                        graph_width == self._npu_dit_mlp_graph_width
                        and not cache_major
                        and isinstance(cnn_cache, torch.Tensor)
                        and isinstance(att_cache, torch.Tensor)
                        and self._npu_dit_full_stack_graph_enabled
                        and full_stack_cache_length in self._npu_dit_full_block_cache_buckets
                        and full_stack_cache_length
                        not in self._npu_dit_full_stack_graph_disabled_lengths
                    ):
                        stack_graph_fn = self._get_npu_dit_full_stack_graph(estimator.blocks)
                        result = self._call_npu_dit_full_stack_graph(
                            stack_graph_fn,
                            estimator,
                            estimator_input,
                            time_embedding,
                            cnn_cache,
                            att_cache,
                            cnn_out,
                            att_out,
                        )
                        if (
                            full_stack_cache_length
                            not in self._npu_dit_full_stack_graph_used_lengths
                        ):
                            logger.info(
                                "MiniCPM-o NPU full DiT stack graph replay active at attention cache=%d",
                                full_stack_cache_length,
                            )
                            self._npu_dit_full_stack_graph_used_lengths.add(
                                full_stack_cache_length
                            )
                        return result, cnn_out, att_out
                    preamble_graph_fn = None
                    if (
                        self._npu_dit_preamble_graph_enabled
                        and not self._npu_dit_preamble_graph_disabled
                        and graph_width not in self._npu_dit_preamble_graph_disabled_widths
                    ):
                        preamble_graph_fn = self._get_npu_dit_preamble_graph(graph_width)
                    conv_mlp_graph_fn = None
                    conv_mlp_standard_weights = False
                    full_block_graph_fn = None
                    full_block_cache_length = _dit_attention_cache_length(att_cache)
                    if full_block_graph_fn is None and (
                        graph_width == self._npu_dit_mlp_graph_width
                        and not cache_major
                        and cnn_cache is not None
                        and att_cache is not None
                        and self._npu_dit_full_block_graph_enabled
                        and full_block_cache_length in self._npu_dit_full_block_cache_buckets
                        and full_block_cache_length not in self._npu_dit_full_block_graph_disabled_lengths
                    ):
                        full_block_graph_fn = self._get_npu_dit_full_block_graph()
                    if full_block_graph_fn is None and (
                        graph_width == self._npu_dit_mlp_graph_width
                        and cnn_cache is not None
                        and att_cache is not None
                        and self._npu_dit_conv_mlp_graph_enabled
                        and not self._npu_dit_conv_mlp_graph_disabled
                    ):
                        conv_mlp_graph_fn = self._get_npu_dit_conv_mlp_graph()
                    elif full_block_graph_fn is None and (
                        graph_width != self._npu_dit_mlp_graph_width
                        and self._npu_dit_prompt_conv_mlp_graph_enabled
                        and graph_width not in self._npu_dit_prompt_conv_mlp_graph_disabled_widths
                    ):
                        conv_mlp_graph_fn = self._get_npu_dit_prompt_conv_mlp_graph()
                        conv_mlp_standard_weights = conv_mlp_graph_fn is not None
                    result = self._estimator_blocks_forward_chunk_mlp_graph(
                        estimator,
                        estimator_input,
                        time_embedding,
                        old_cnn,
                        old_att,
                        cnn_out,
                        att_out,
                        graph_fn,
                        preamble_graph_fn,
                        conv_mlp_graph_fn,
                        conv_mlp_standard_weights,
                        full_block_graph_fn,
                        wide_modulations,
                    )
                    if not self._npu_dit_mlp_graph_used:
                        logger.info(
                            "MiniCPM-o NPU DiT MLP graph replay active for CFG batch=2, width=%d",
                            graph_width,
                        )
                        self._npu_dit_mlp_graph_used = True
                    if preamble_graph_fn is not None:
                        if (
                            self._npu_dit_qkv_pack_enabled
                            and graph_width == self._npu_dit_mlp_graph_width
                            and not self._npu_dit_qkv_pack_used
                        ):
                            logger.info(
                                "MiniCPM-o NPU DiT native-QKV attention preamble graph replay active"
                            )
                            self._npu_dit_qkv_pack_used = True
                        elif not self._npu_dit_preamble_graph_used:
                            logger.info("MiniCPM-o NPU DiT attention preamble graph replay active")
                        self._npu_dit_preamble_graph_used = True
                    if conv_mlp_graph_fn is not None and conv_mlp_standard_weights:
                        if graph_width not in self._npu_dit_prompt_conv_mlp_graph_used_widths:
                            logger.info(
                                "MiniCPM-o NPU prompt Conv+MLP megagraph replay active at width=%d",
                                graph_width,
                            )
                            self._npu_dit_prompt_conv_mlp_graph_used_widths.add(graph_width)
                    elif conv_mlp_graph_fn is not None and not self._npu_dit_conv_mlp_graph_used:
                        if self._npu_dit_fused_conv_linear_enabled:
                            logger.info("MiniCPM-o NPU DiT fused Conv+Linear + MLP graph replay active")
                            self._npu_dit_fused_conv_linear_used = True
                        elif self._npu_dit_fused_conv_block_enabled:
                            logger.info("MiniCPM-o NPU DiT GE-visible Conv-block + MLP graph replay active")
                            self._npu_dit_fused_conv_block_used = True
                        elif self._npu_dit_post_attn_graph_enabled:
                            logger.info(
                                "MiniCPM-o NPU post-attention cache-major Conv+MLP megagraph replay active"
                            )
                            self._npu_dit_post_attn_graph_used = True
                            self._npu_dit_cache_major_used = True
                        else:
                            if self._npu_dit_cache_major_enabled:
                                logger.info("MiniCPM-o NPU cache-major Conv+MLP megagraph replay active")
                                self._npu_dit_cache_major_used = True
                            else:
                                logger.info("MiniCPM-o NPU DiT Conv+MLP megagraph replay active")
                        self._npu_dit_conv_mlp_graph_used = True
                    if (
                        full_block_graph_fn is not None
                        and full_block_cache_length not in self._npu_dit_full_block_graph_used_lengths
                    ):
                        logger.info(
                            "MiniCPM-o NPU full DiT block graph replay active at attention cache=%d",
                            full_block_cache_length,
                        )
                        self._npu_dit_full_block_graph_used_lengths.add(full_block_cache_length)
                    return result, cnn_out, att_out
            except Exception:
                self._npu_dit_mlp_graph_disabled_widths.add(graph_width)
                logger.warning(
                    "MiniCPM-o NPU DiT graph execution failed at width=%d; using eager blocks for that width",
                    graph_width,
                    exc_info=True,
                )
        if cache_major:
            old_cnn = old_cnn.transpose(-2, -1).contiguous()
            cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
        return result, cnn_out, att_out

    @staticmethod
    def _is_cache_major_cnn(cache: torch.Tensor | None) -> bool:
        return (
            isinstance(cache, torch.Tensor)
            and cache.ndim >= 2
            and int(cache.shape[-2]) == 2
            and int(cache.shape[-1]) == 1024
        )

    def _estimator_blocks_forward_chunk_mlp_graph(
        self,
        estimator: nn.Module,
        estimator_input: torch.Tensor,
        time_embedding: torch.Tensor,
        old_cnn: Any,
        old_att: Any,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
        graph_fn: Any,
        preamble_graph_fn: Any | None = None,
        conv_mlp_graph_fn: Any | None = None,
        conv_mlp_standard_weights: bool = False,
        full_block_graph_fn: Any | None = None,
        precomputed_wide_modulations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replay enabled shape-bucketed partitions with exact eager fallbacks."""
        hidden = estimator.in_proj(estimator_input.transpose(1, 2))
        wide_modulations = precomputed_wide_modulations
        if (
            wide_modulations is None
            and self._npu_dit_wide_adaln_enabled
            and preamble_graph_fn is not None
            and full_block_graph_fn is None
        ):
            wide_modulations = self._get_npu_dit_wide_adaln_graph()(
                time_embedding,
                self._npu_dit_wide_adaln_weight,
                self._npu_dit_wide_adaln_bias,
            )
            if not self._npu_dit_wide_adaln_used:
                logger.info(
                    "MiniCPM-o wide AdaLN graph replay active for 16 block projections"
                )
                self._npu_dit_wide_adaln_used = True
        for block_idx, block in enumerate(estimator.blocks):
            if full_block_graph_fn is not None:
                if not self._dit_full_block_compatible(block):
                    raise RuntimeError("MiniCPM-o NPU full-block graph encountered an incompatible block")
                hidden, new_cnn, new_att = self._call_npu_dit_full_block_graph(
                    full_block_graph_fn,
                    block,
                    hidden,
                    time_embedding,
                    old_att[block_idx],
                    old_cnn[block_idx],
                )
                cnn_out[block_idx].copy_(new_cnn)
                att_out[block_idx, :, :, : new_att.shape[2], :].copy_(new_att)
                continue
            if block.training or block.norm2.weight is not None or block.norm2.bias is not None:
                raise RuntimeError("MiniCPM-o NPU DiT MLP graph requires eval-mode affine-free norm2")
            if float(block.norm2.eps) != 1e-6:
                raise RuntimeError(f"MiniCPM-o NPU DiT MLP graph requires norm2 eps=1e-6, got {block.norm2.eps}")
            if preamble_graph_fn is None:
                modulation = block.adaLN_modulation(time_embedding)
                q = k = v = None
            else:
                if not BatchedToken2Wav._dit_preamble_compatible(block, int(hidden.shape[1])):
                    raise RuntimeError("MiniCPM-o NPU DiT preamble graph encountered an incompatible block")
                modulation, q, k, v = self._call_npu_dit_preamble_graph(
                    preamble_graph_fn,
                    block,
                    hidden,
                    time_embedding,
                    (
                        None
                        if wide_modulations is None
                        else wide_modulations[:, :, block_idx, :]
                    ),
                )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                shift_conv,
                scale_conv,
                gate_conv,
            ) = modulation.chunk(9, dim=-1)

            if q is None or k is None or v is None:
                attention_cache_written = False
                attention, new_att = block.attn.forward_chunk(
                    block.norm1(hidden) * (1 + scale_msa) + shift_msa,
                    old_att[block_idx],
                    None,
                )
            else:
                output_cache = (
                    att_out[block_idx]
                    if self._npu_dit_attn_cache_out_enabled
                    else None
                )
                attention, new_att = BatchedToken2Wav._attention_from_projected_qkv(
                    block.attn,
                    q,
                    k,
                    v,
                    old_att[block_idx],
                    output_cache=output_cache,
                )
                attention_cache_written = output_cache is not None
                if attention_cache_written and not self._npu_dit_attn_cache_out_used:
                    logger.info(
                        "MiniCPM-o NPU DiT attention cache direct-output active"
                    )
                    self._npu_dit_attn_cache_out_used = True
            post_attention_graph = (
                self._npu_dit_post_attn_graph_enabled
                and conv_mlp_graph_fn is not None
                and not conv_mlp_standard_weights
            )
            if not post_attention_graph:
                hidden = hidden + gate_msa * attention
                conv_input = block.norm3(hidden) * (1 + scale_conv) + shift_conv
            if conv_mlp_graph_fn is None:
                convolution, new_cnn = block.conv.forward_chunk(
                    conv_input,
                    old_cnn[block_idx],
                )
                hidden = hidden + gate_conv * convolution
                hidden = graph_fn(
                    hidden,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    block.mlp.fc1.weight,
                    block.mlp.fc1.bias,
                    block.mlp.fc2.weight,
                    block.mlp.fc2.bias,
                )
            else:
                if not BatchedToken2Wav._dit_conv_mlp_layout_compatible(block):
                    raise RuntimeError("MiniCPM-o NPU DiT Conv+MLP graph encountered an incompatible block")
                if post_attention_graph and not BatchedToken2Wav._dit_post_attention_compatible(
                    block,
                    int(hidden.shape[1]),
                ):
                    raise RuntimeError("MiniCPM-o NPU post-attention graph encountered an incompatible block")
                conv1 = block.conv.block[1]
                conv_norm = block.conv.block[3]
                conv2 = block.conv.block[6]
                block_cnn_cache = old_cnn[block_idx]
                if block_cnn_cache is None:
                    block_cnn_cache = hidden.new_zeros(
                        (hidden.shape[0], 2, 1024)
                        if self._npu_dit_cache_major_enabled and int(hidden.shape[1]) == 50
                        else (hidden.shape[0], 1024, 2)
                    )
                conv1_weight = (
                    conv1.weight if conv_mlp_standard_weights else self._dit_conv_graph_weight(conv1)
                )
                conv2_weight = (
                    conv2.weight if conv_mlp_standard_weights else self._dit_conv_graph_weight(conv2)
                )
                if post_attention_graph:
                    hidden, new_cnn = conv_mlp_graph_fn(
                        hidden,
                        attention,
                        block_cnn_cache,
                        gate_msa,
                        shift_conv,
                        scale_conv,
                        gate_conv,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                        conv1_weight,
                        conv1.bias,
                        conv_norm.weight,
                        conv_norm.bias,
                        conv2_weight,
                        conv2.bias,
                        block.mlp.fc1.weight,
                        block.mlp.fc1.bias,
                        block.mlp.fc2.weight,
                        block.mlp.fc2.bias,
                    )
                else:
                    hidden, new_cnn = conv_mlp_graph_fn(
                        hidden,
                        conv_input,
                        block_cnn_cache,
                        gate_conv,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                        conv1_weight,
                        conv1.bias,
                        conv_norm.weight,
                        conv_norm.bias,
                        conv2_weight,
                        conv2.bias,
                        block.mlp.fc1.weight,
                        block.mlp.fc1.bias,
                        block.mlp.fc2.weight,
                        block.mlp.fc2.bias,
                    )
            cnn_out[block_idx].copy_(new_cnn)
            if not attention_cache_written:
                att_out[block_idx, :, :, : new_att.shape[2], :].copy_(new_att)

        hidden = estimator.final_layer(hidden, time_embedding)
        return hidden.transpose(1, 2)

    @staticmethod
    def _attention_from_projected_qkv(
        attention_module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_cache: torch.Tensor | None,
        *,
        output_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_cache is not None:
            old_width = int(att_cache.shape[2]) if att_cache is not None else 0
            expected_shape = (
                *k.shape[:2],
                int(k.shape[2]) + old_width,
                int(k.shape[3]) * 2,
            )
            if tuple(output_cache.shape) != expected_shape:
                raise ValueError(
                    "DiT attention output cache shape mismatch: "
                    f"expected {expected_shape}, got {tuple(output_cache.shape)}"
                )
            full_k, full_v = output_cache.chunk(2, dim=3)
            if att_cache is None:
                full_k.copy_(k)
                full_v.copy_(v)
            else:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                torch.cat((k, k_cache), dim=2, out=full_k)
                torch.cat((v, v_cache), dim=2, out=full_v)
            k = full_k
            v = full_v
            new_att_cache = output_cache
        elif att_cache is not None:
            k_cache, v_cache = att_cache.chunk(2, dim=3)
            k = torch.cat((k, k_cache), dim=2)
            v = torch.cat((v, v_cache), dim=2)
            new_att_cache = torch.cat((k, v), dim=3)
        else:
            new_att_cache = torch.cat((k, v), dim=3)
        hidden = F.scaled_dot_product_attention(q, k, v)
        batch_size, _, width, _ = hidden.shape
        hidden = hidden.transpose(1, 2).reshape(batch_size, width, -1)
        hidden = attention_module.proj(hidden)
        hidden = attention_module.proj_drop(hidden)
        return hidden, new_att_cache

    def _estimator_time_embeddings(
        self,
        estimator: nn.Module,
        timeline: torch.Tensor,
        cfg_batch_size: int,
    ) -> torch.Tensor:
        """Cache graph-safe CFM timestep embeddings for inference.

        CosyVoice's ``TimestepEmbedder`` constructs its sinusoidal frequencies
        on CPU and calls ``.to(t)`` for every diffusion step. Besides repeating
        the same host-to-device copy and MLP work for every streamed chunk,
        that transfer is illegal while an Ascend NPU graph is being captured.
        Token2wav weights and the CFM timeline are immutable during serving, so
        compute the step embeddings once and reuse them by shape.

        Lightweight test/dummy estimators that do not expose CosyVoice's
        embedder attributes keep the generic eager behavior.
        """
        embedder = estimator.t_embedder
        frequency_size = getattr(embedder, "frequency_embedding_size", None)
        scale = getattr(embedder, "scale", None)
        mlp = getattr(embedder, "mlp", None)
        if not isinstance(frequency_size, int) or scale is None or not isinstance(mlp, nn.Module):
            return torch.stack(
                [embedder(timeline[step].expand(cfg_batch_size)).unsqueeze(1) for step in range(self.n_timesteps)]
            )

        key = (
            id(embedder),
            cfg_batch_size,
            timeline.device.type,
            timeline.device.index,
            timeline.dtype,
            self.n_timesteps,
        )
        cached = self._timestep_embedding_cache.get(key)
        if cached is not None:
            return cached

        half = frequency_size // 2
        # Match CosyVoice's CPU/default-dtype frequency construction exactly,
        # but perform its one transfer before any graph capture begins.
        frequencies = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half) / half).to(timeline)
        embeddings: list[torch.Tensor] = []
        time = timeline[0].expand(cfg_batch_size)
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            arguments = (time * float(scale))[:, None] * frequencies[None]
            sinusoidal = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
            if frequency_size % 2:
                sinusoidal = torch.cat((sinusoidal, torch.zeros_like(sinusoidal[:, :1])), dim=-1)
            embeddings.append(mlp(sinusoidal).unsqueeze(1))
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
        cached = torch.stack(embeddings).detach()
        self._timestep_embedding_cache[key] = cached
        return cached

    def _decode_cfm_eager(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        retain_cache_major = self._npu_dit_cache_major_enabled and mu.device.type == "npu"
        use_cache_major = (
            retain_cache_major
            and int(mu.shape[2]) == self._npu_dit_mlp_graph_width
            and self._npu_dit_conv_mlp_graph_enabled
            and not self._npu_dit_conv_mlp_graph_disabled
        )
        working_cnn_cache = cnn_cache
        if working_cnn_cache is not None:
            input_cache_major = self._is_cache_major_cnn(working_cnn_cache)
            if use_cache_major != input_cache_major:
                working_cnn_cache = working_cnn_cache.transpose(-2, -1).contiguous()
        timeline = self._timeline_for(mu)
        mu_cfg = self._cfg_pair("mu", mu, zero_unconditional=True)
        speakers_cfg = self._cfg_pair("speakers", speakers, zero_unconditional=True)
        cond_cfg = self._cfg_pair("cond", cond, zero_unconditional=True)
        time_embeddings = self._estimator_time_embeddings(estimator, timeline, batch_size * 2)
        wide_modulation_steps: torch.Tensor | None = None
        if self._npu_dit_wide_adaln_enabled:
            try:
                wide_modulation_steps = self._get_npu_dit_wide_adaln_steps_graph()(
                    time_embeddings,
                    self._npu_dit_wide_adaln_weight,
                    self._npu_dit_wide_adaln_bias,
                )
                if not self._npu_dit_wide_adaln_steps_used:
                    logger.info(
                        "MiniCPM-o all-step wide AdaLN replay active for %d CFM steps x 16 blocks",
                        self.n_timesteps,
                    )
                    self._npu_dit_wide_adaln_steps_used = True
            except Exception:
                self._npu_dit_wide_adaln_enabled = False
                self._npu_dit_wide_adaln_steps_graph = None
                wide_modulation_steps = None
                logger.warning(
                    "MiniCPM-o all-step wide AdaLN replay failed; using per-block projections",
                    exc_info=True,
                )
        deltas = self._cfm_deltas_for(timeline)
        direct_cache_output = self._npu_cfm_stacked_cache_out_enabled and mu.device.type == "npu"
        stacked_cnn_out: torch.Tensor | None = None
        stacked_att_out: torch.Tensor | None = None
        if direct_cache_output:
            first_old_cnn = working_cnn_cache[0] if working_cnn_cache is not None else None
            first_old_att = att_cache[0] if att_cache is not None else None
            cnn_shape, att_shape = self._estimator_buffer_shapes(
                estimator,
                mu_cfg,
                first_old_att,
                cache_major=self._is_cache_major_cnn(first_old_cnn),
            )
            stacked_cnn_out = mu.new_empty((self.n_timesteps, *cnn_shape))
            stacked_att_out = mu.new_empty((self.n_timesteps, *att_shape))
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        for step in range(self.n_timesteps):
            old_cnn = working_cnn_cache[step] if working_cnn_cache is not None else None
            old_att = att_cache[step] if att_cache is not None else None
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=self._cfg_pair("x", x, zero_unconditional=False),
                mu=mu_cfg,
                time_embedding=time_embeddings[step],
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=old_cnn,
                att_cache=old_att,
                cnn_out=stacked_cnn_out[step] if stacked_cnn_out is not None else None,
                att_out=stacked_att_out[step] if stacked_att_out is not None else None,
                wide_modulations=(
                    wide_modulation_steps[step]
                    if wide_modulation_steps is not None
                    else None
                ),
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + deltas[step] * velocity
            if stacked_cnn_out is None:
                next_cnn.append(step_cnn)
                next_att.append(step_att)
        stacked_cnn = stacked_cnn_out if stacked_cnn_out is not None else torch.stack(next_cnn)
        stacked_att = stacked_att_out if stacked_att_out is not None else torch.stack(next_att)
        if direct_cache_output and not self._npu_cfm_stacked_cache_out_used:
            logger.info("MiniCPM-o NPU direct stacked CFM cache output active")
            self._npu_cfm_stacked_cache_out_used = True
        # A compile or replay failure disables the graph inside
        # ``_estimator_step``. Return to the canonical layout in that case so
        # subsequent chunks do not pay two compatibility transposes.
        retain_cache_major = use_cache_major and not self._npu_dit_conv_mlp_graph_disabled
        if retain_cache_major != self._is_cache_major_cnn(stacked_cnn):
            stacked_cnn = stacked_cnn.transpose(-2, -1).contiguous()
        return x, stacked_cnn, stacked_att

    @staticmethod
    def _optional_tensor_signature(value: torch.Tensor | None) -> Any:
        if value is None:
            return None
        return tuple(value.shape), value.dtype, value.device.type, value.device.index

    @staticmethod
    def _npu_cfm_graph_cache_limit() -> int:
        raw = os.environ.get("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE", "4")
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "Invalid VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE=%r; using 4",
                raw,
            )
            return 4

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enabled = os.environ.get("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH", "0").strip().lower()
        if enabled not in {"1", "true", "yes", "on"} or mu.device.type != "npu" or self._npu_cfm_graph_disabled:
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )

        inputs = (mu, speakers, cond, cnn_cache, att_cache)
        key = tuple(self._optional_tensor_signature(value) for value in inputs)
        entry = self._npu_cfm_graphs.get(key)
        if entry is not None:
            self._npu_cfm_graphs.move_to_end(key)
            for static, current in zip(entry["inputs"], inputs, strict=True):
                if static is not None and current is not None:
                    static.copy_(current)
            entry["graph"].replay()
            return tuple(output.clone() for output in entry["outputs"])

        static_inputs = tuple(value.clone() if value is not None else None for value in inputs)
        static_mu, static_speakers, static_cond, static_cnn, static_att = static_inputs
        try:
            with torch.inference_mode():
                self._decode_cfm_eager(
                    static_mu,
                    static_speakers,
                    static_cond,
                    cnn_cache=static_cnn,
                    att_cache=static_att,
                )
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.inference_mode(), torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
                outputs = self._decode_cfm_eager(
                    static_mu,
                    static_speakers,
                    static_cond,
                    cnn_cache=static_cnn,
                    att_cache=static_att,
                )
            graph.replay()
        except Exception:
            self._npu_cfm_graph_disabled = True
            self._npu_cfm_graphs.clear()
            logger.warning("MiniCPM-o NPU CFM graph capture failed; using eager Code2Wav", exc_info=True)
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )

        self._npu_cfm_graphs[key] = {
            "graph": graph,
            "inputs": static_inputs,
            "outputs": outputs,
        }
        max_graphs = self._npu_cfm_graph_cache_limit()
        while len(self._npu_cfm_graphs) > max_graphs:
            self._npu_cfm_graphs.popitem(last=False)
        return tuple(output.clone() for output in outputs)

    def _split_flow_cache(
        self,
        cache: dict[str, torch.Tensor],
        batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        if self._npu_single_request_cache_passthrough_enabled and batch_size == 1:
            if not self._npu_single_request_cache_passthrough_used:
                logger.info("MiniCPM-o NPU single-request cache passthrough active")
                self._npu_single_request_cache_passthrough_used = True
            return [{name: value.detach() for name, value in cache.items()}]
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": torch.cat(
                        (
                            cache["estimator_att_cache"][:, :, row : row + 1],
                            cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                }
            )
        return result

    def _stack_flow_cache(self, states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        if self._npu_single_request_cache_passthrough_enabled and len(states) == 1:
            return states[0].flow_cache
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tokens, projected_speakers, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        flow_cache = self._stack_flow_cache(states)
        projected_speakers = features.projected_speaker_embedding.expand(batch_size, -1)
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )

        prompt_len = int(features.mels.shape[1])
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        speech, source = self.hift(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states
