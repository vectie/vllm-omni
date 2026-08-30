# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

import importlib
import logging
import os
import time
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
_NPU_DIT_WIDE_FINAL_ADALN_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_FINAL_ADALN"
)
_NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT = 1.0e-6
_NPU_DIT_FINAL_ADDCMUL_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FINAL_ADDCMUL"
_NPU_DIT_FINAL_ADDCMUL_MAX_ABS_DRIFT = 1.0e-6
_NPU_DIT_FUSED_FINAL_ADALN_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_FINAL_ADALN"
)
_NPU_DIT_FUSED_FINAL_ADALN_MAX_ABS_DRIFT = 2.0e-3
_NPU_DIT_FUSED_FINAL_ADALN_MEAN_ABS_DRIFT = 5.0e-4
_NPU_DIT_CONV_MLP_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_CONV_MLP_GRAPH"
_NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH"
)
_NPU_DIT_LAST_BLOCK_FINAL_EULER_MAX_ABS_DRIFT = 5.0e-3
_NPU_DIT_LAST_BLOCK_FINAL_EULER_MEAN_ABS_DRIFT = 5.0e-4
_NPU_DIT_LAST_BLOCK_FINAL_EULER_MIN_SAVING_US = 200.0
_NPU_DIT_LAST_BLOCK_FINAL_EULER_MIN_SPEEDUP = 1.10
_NPU_DIT_LAST_BLOCK_FINAL_EULER_PERF_TRIALS = 5
_NPU_DIT_LAST_BLOCK_FINAL_EULER_PERF_ITERATIONS = 20
_NPU_DIT_PROMPT_CONV_MLP_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_PROMPT_CONV_MLP_GRAPH"
_NPU_DIT_FULL_BLOCK_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_GRAPH"
_NPU_DIT_FULL_STACK_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_STACK_GRAPH"
_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS"
_NPU_DIT_FUSED_CONV_PACK_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK"
_NPU_DIT_CACHE_MAJOR_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR"
_NPU_DIT_POST_ATTN_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_POST_ATTN_GRAPH"
_NPU_DIT_QKV_PACK_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_QKV_PACK"
_NPU_DIT_FUSED_QKV_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_QKV"
_NPU_DIT_ATTN_CACHE_OUT_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_ATTN_CACHE_OUT"
_NPU_CFM_STACKED_CACHE_OUT_ENV = "VLLM_OMNI_MINICPMO45_NPU_CFM_STACKED_CACHE_OUT"
_NPU_CFM_FIXED_KV_SLABS_ENV = "VLLM_OMNI_MINICPMO45_NPU_CFM_FIXED_KV_SLABS"
_NPU_CFM_PLANAR_KV_SLABS_ENV = "VLLM_OMNI_MINICPMO45_NPU_CFM_PLANAR_KV_SLABS"
_NPU_CFM_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"
_NPU_CFM_CACHE_FILL_GRAPH_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH"
)
_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS"
)
_NPU_CFM_GRAPH_ATTN_MAX_ABS_DRIFT = 3.125e-2
_NPU_CFM_GRAPH_ATTN_MEAN_ABS_DRIFT = 3.0e-3
_NPU_INITIAL_CFM_TIMESTEPS_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"
)
_NPU_PROMPT_CFM_TIMESTEPS_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS"
)
_NPU_PROMPT_CACHE_MAX_FRAMES_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CACHE_MAX_FRAMES"
)
_NPU_STAGE2_TIMING_ENV = "VLLM_OMNI_MINICPMO45_NPU_STAGE2_TIMING"
_NPU_DIT_BSH_ATTENTION_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_BSH_ATTENTION"
_NPU_DIT_BSH_ATTENTION_MAX_ABS_DRIFT = 2.0e-2
_NPU_DIT_BSH_ATTENTION_MEAN_ABS_DRIFT = 2.0e-3
_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH"
)
_NPU_DIT_FUSED_CONV_BLOCK_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_BLOCK"
_NPU_DIT_FUSED_CONV_LINEAR_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_LINEAR"
_NPU_DIT_COMPUTE_DTYPE_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_COMPUTE_DTYPE"
_NPU_CFM_INTEGRATION_DTYPE_ENV = "VLLM_OMNI_MINICPMO45_NPU_CFM_INTEGRATION_DTYPE"
_NPU_DIT_DYNAMIC_W8A8_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_DYNAMIC_W8A8"
_NPU_DIT_FUSED_BF16_FFN_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_BF16_FFN"
_NPU_MATMUL_HF32_ENV = "VLLM_OMNI_MINICPMO45_NPU_MATMUL_HF32"
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


def _npu_dit_compute_dtype(config_value: Any = None) -> torch.dtype:
    """Resolve the opt-in precision used by the NPU CFM estimator.

    Token2Wav is intentionally loaded in FP32 because its encoder and HiFT
    contain FP32-only modules. The DiT estimator is a separate submodule, so
    it can use BF16 without changing prompt extraction, the flow encoder, or
    HiFT. A separate policy controls whether the CFM integration state stays
    FP32 or follows the estimator dtype.
    """
    env_value = os.environ.get(_NPU_DIT_COMPUTE_DTYPE_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return torch.float32
    normalized = str(raw).strip().lower().replace("torch.", "")
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(
        f"Invalid {_NPU_DIT_COMPUTE_DTYPE_ENV}={raw!r}; "
        "expected fp32 or bf16"
    )


def _npu_cfm_integration_dtype(config_value: Any = None) -> torch.dtype:
    """Resolve the opt-in CFM state/integrator dtype.

    FP32 remains the default quality boundary. BF16 is an experimental NPU
    mode that avoids two casts per estimator evaluation by keeping noise,
    CFG, and Euler recurrence in the estimator dtype. The completed mel is
    converted back to the flow/HiFT dtype exactly once.
    """
    env_value = os.environ.get(_NPU_CFM_INTEGRATION_DTYPE_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return torch.float32
    normalized = str(raw).strip().lower().replace("torch.", "")
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(
        f"Invalid {_NPU_CFM_INTEGRATION_DTYPE_ENV}={raw!r}; "
        "expected fp32 or bf16"
    )


def _npu_dit_dynamic_w8a8_enabled(config_value: Any = None) -> bool:
    """Resolve selective per-channel-weight/per-token-activation DiT W8A8."""
    env_value = os.environ.get(_NPU_DIT_DYNAMIC_W8A8_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_DYNAMIC_W8A8_ENV}={raw!r}")


def _npu_dit_fused_bf16_ffn_enabled(config_value: Any = None) -> bool:
    """Resolve the A2 dense-BF16 full-MLP fusion candidate."""
    env_value = os.environ.get(_NPU_DIT_FUSED_BF16_FFN_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_FUSED_BF16_FFN_ENV}={raw!r}")


def _npu_matmul_hf32_enabled(config_value: Any = None) -> bool:
    """Resolve the opt-in Stage-2 FP32 MatMul-to-HF32 Cube policy."""
    env_value = os.environ.get(_NPU_MATMUL_HF32_ENV)
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
    raise ValueError(f"Invalid {_NPU_MATMUL_HF32_ENV}={raw!r}")


def _npu_cfm_fixed_kv_slabs_enabled(config_value: Any = None) -> bool:
    """Resolve the opt-in, request-owned fixed estimator KV workspace."""
    env_value = os.environ.get(_NPU_CFM_FIXED_KV_SLABS_ENV)
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
    raise ValueError(f"Invalid {_NPU_CFM_FIXED_KV_SLABS_ENV}={raw!r}")


def _npu_cfm_graph_enabled() -> bool:
    """Whether fixed-shape steady CFM NPUGraph capture is enabled."""
    return os.environ.get(_NPU_CFM_GRAPH_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _npu_initial_cfm_timesteps(full_timesteps: int) -> int:
    """Resolve the optional first-audio-packet CFM solver width.

    The default preserves the full quality path. Experimental profiles may
    spend fewer solver evaluations on the short first packet while retaining
    the full solver for every subsequent chunk.
    """
    raw = os.environ.get(_NPU_INITIAL_CFM_TIMESTEPS_ENV)
    if raw in (None, ""):
        return full_timesteps
    try:
        timesteps = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_NPU_INITIAL_CFM_TIMESTEPS_ENV}={raw!r}; expected an "
            f"integer in [1, {full_timesteps}]"
        ) from exc
    if not 1 <= timesteps <= full_timesteps:
        raise ValueError(
            f"Invalid {_NPU_INITIAL_CFM_TIMESTEPS_ENV}={raw!r}; expected an "
            f"integer in [1, {full_timesteps}]"
        )
    return timesteps


def _npu_prompt_cfm_timesteps(full_timesteps: int) -> int:
    """Resolve the optional reference-prompt cache solver width."""
    raw = os.environ.get(_NPU_PROMPT_CFM_TIMESTEPS_ENV)
    if raw in (None, ""):
        return full_timesteps
    try:
        timesteps = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_NPU_PROMPT_CFM_TIMESTEPS_ENV}={raw!r}; expected an "
            f"integer in [1, {full_timesteps}]"
        ) from exc
    if not 1 <= timesteps <= full_timesteps:
        raise ValueError(
            f"Invalid {_NPU_PROMPT_CFM_TIMESTEPS_ENV}={raw!r}; expected an "
            f"integer in [1, {full_timesteps}]"
        )
    return timesteps


def _npu_prompt_cache_max_frames() -> int | None:
    """Limit only the DiT prompt-cache suffix; preserve full prompt encoding."""
    raw = os.environ.get(_NPU_PROMPT_CACHE_MAX_FRAMES_ENV)
    if raw in (None, ""):
        return None
    try:
        frames = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_NPU_PROMPT_CACHE_MAX_FRAMES_ENV}={raw!r}; expected a "
            "positive integer"
        ) from exc
    if frames <= 0:
        raise ValueError(
            f"Invalid {_NPU_PROMPT_CACHE_MAX_FRAMES_ENV}={raw!r}; expected a "
            "positive integer"
        )
    return frames


def _npu_stage2_timing_enabled() -> bool:
    """Enable synchronizing diagnostic timers outside submission profiles."""
    return os.environ.get(_NPU_STAGE2_TIMING_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _npu_cfm_cache_fill_graph_enabled() -> bool:
    """Whether fixed width-50 cache-fill CFM shapes get outer NPUGraphs."""
    return os.environ.get(
        _NPU_CFM_CACHE_FILL_GRAPH_ENV, "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _npu_cfm_cache_fill_graph_lengths() -> tuple[int, ...]:
    """Resolve the exact old-cache lengths admitted to cache-fill capture."""
    if not _npu_cfm_cache_fill_graph_enabled():
        return ()
    raw = os.environ.get(_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS_ENV, "302")
    try:
        lengths = tuple(
            dict.fromkeys(
                int(item.strip())
                for item in raw.split(",")
                if item.strip()
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS_ENV}={raw!r}"
        ) from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(
            f"Invalid {_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS_ENV}={raw!r}"
        )
    return lengths


def _npu_cfm_graph_phase(
    *,
    fixed_kv_slabs: bool,
    cache_fill_lengths: tuple[int, ...],
    steady_graph: bool,
    width: int,
    cache_length: int | None,
    has_cache_outputs: bool,
) -> str | None:
    """Classify only shapes whose addresses and cache semantics are stable."""
    if not fixed_kv_slabs:
        return "dynamic"
    if steady_graph:
        return "steady"
    if (
        cache_length in cache_fill_lengths
        and width == 50
        and has_cache_outputs
    ):
        return "cache-fill"
    return None


def _npu_cfm_planar_kv_slabs_enabled(config_value: Any = None) -> bool:
    """Resolve the opt-in contiguous K/V-plane cache representation."""
    env_value = os.environ.get(_NPU_CFM_PLANAR_KV_SLABS_ENV)
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
    raise ValueError(f"Invalid {_NPU_CFM_PLANAR_KV_SLABS_ENV}={raw!r}")


def _npu_dit_bsh_attention_enabled(config_value: Any = None) -> bool:
    """Resolve the opt-in BSH Q/K/V, cache, and fused-attention path."""
    env_value = os.environ.get(_NPU_DIT_BSH_ATTENTION_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_BSH_ATTENTION_ENV}={raw!r}")


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


def _npu_dit_wide_final_adaln_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_WIDE_FINAL_ADALN_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_WIDE_FINAL_ADALN_ENV}={raw!r}")


def _npu_dit_final_addcmul_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FINAL_ADDCMUL_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_FINAL_ADDCMUL_ENV}={raw!r}")


def _npu_dit_fused_final_adaln_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FUSED_FINAL_ADALN_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_FUSED_FINAL_ADALN_ENV}={raw!r}")


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


def _npu_dit_last_block_final_euler_graph_enabled(
    config_value: Any = None,
) -> bool:
    env_value = os.environ.get(_NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH_ENV)
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
    raise ValueError(
        f"Invalid {_NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH_ENV}={raw!r}"
    )


def _npu_dit_last_block_final_euler_perf_qualifies(
    control_us: float,
    candidate_us: float,
    *,
    min_saving_us: float = _NPU_DIT_LAST_BLOCK_FINAL_EULER_MIN_SAVING_US,
    min_speedup: float = _NPU_DIT_LAST_BLOCK_FINAL_EULER_MIN_SPEEDUP,
) -> bool:
    """Require enough device-time headroom for a wider graph to matter live.

    Small isolated wins repeatedly disappeared after graph scheduling, cache
    publication, and pipeline overlap were included. Keep the extension
    fail-closed unless it clears both an absolute and a relative device-time
    threshold on the loaded checkpoint.
    """
    if control_us <= 0.0 or candidate_us <= 0.0:
        return False
    return (
        control_us - candidate_us >= min_saving_us
        and control_us / candidate_us >= min_speedup
    )


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


def _npu_dit_fused_qkv_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_FUSED_QKV_ENV)
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
    raise ValueError(f"Invalid {_NPU_DIT_FUSED_QKV_ENV}={raw!r}")


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


def _quantize_dynamic_w8a8_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[out,in]`` weights per output channel for Ascend Cube.

    ``npu_quant_matmul`` consumes its weight as ``[in,out]``.  Keeping the
    transposed INT8 tensor persistent avoids a transpose/format conversion in
    every one of the 96 steady DiT block evaluations per audio chunk.
    """
    if weight.ndim != 2:
        raise ValueError(f"Dynamic W8A8 requires a matrix, got shape={tuple(weight.shape)}")
    source = weight.detach().to(dtype=torch.float32)
    scale = source.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
    quantized = torch.round(source / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return quantized.transpose(0, 1).contiguous(), scale.contiguous()


def _npu_dynamic_w8a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Graph-visible Ascend dynamic-W8A8 linear with FP32/BF16 output."""
    # The registered NPU op defaults to INT8. Calling the dispatcher directly
    # keeps the operation visible to TorchAir instead of tracing through the
    # Python convenience wrapper. ``npu_quant_matmul`` requires a one-
    # dimensional per-token scale on A2. Flatten every leading B/T dimension
    # before dynamic quantization; this preserves independent token scales and
    # lets the contiguous BSH producer feed Cube without a materializing
    # transpose. Restore the original leading dimensions after the matmul.
    output_shape = (*x.shape[:-1], weight.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    quantized_x, pertoken_scale = torch.ops.npu.npu_dynamic_quant(x_2d)
    output = torch.ops.npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=x.dtype,
    )
    return output.reshape(output_shape)


def _npu_bf16_ffn(
    x: torch.Tensor,
    fc1_weight_kn: torch.Tensor,
    fc1_bias_fp32: torch.Tensor,
    fc2_weight_kn: torch.Tensor,
    fc2_bias_fp32: torch.Tensor,
) -> torch.Tensor:
    """Graph-visible A2 dense FFN covering both projections and GELU."""
    output_shape = (*x.shape[:-1], fc2_weight_kn.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    output = torch.ops.npu.npu_ffn(
        x_2d,
        fc1_weight_kn,
        fc2_weight_kn,
        "gelu",
        bias1=fc1_bias_fp32,
        bias2=fc2_bias_fp32,
        inner_precise=0,
    )
    return output.reshape(output_shape)


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


def _dit_wide_adaln_steps_with_final(
    time_embeddings: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project every block and the final layer for all fixed CFM timesteps."""
    modulation = F.linear(F.silu(time_embeddings), packed_weight, packed_bias)
    block_width = 16 * 9 * 512
    block_modulation = modulation[..., :block_width].reshape(
        time_embeddings.shape[0],
        2,
        1,
        16,
        9 * 512,
    )
    return block_modulation, modulation[..., block_width:]


def _dit_final_from_modulation(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    norm: nn.Module,
    output: nn.Module,
) -> torch.Tensor:
    """Run the source final layer from a precomputed AdaLN modulation."""
    shift, scale = modulation.chunk(2, dim=-1)
    return output(norm(hidden) * (1 + scale) + shift)


def _dit_final_from_modulation_addcmul(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    norm: nn.Module,
    output: nn.Module,
) -> torch.Tensor:
    """Run final AdaLN with one fewer eager elementwise launch."""
    shift, scale = modulation.chunk(2, dim=-1)
    normalized = norm(hidden)
    return output(torch.addcmul(normalized + shift, normalized, scale))


def _dit_final_addcmul_drift_limit(dtype: torch.dtype) -> float:
    """Allow one storage ULP while keeping the FP32 gate unchanged.

    ``addcmul`` and the canonical multiply/add expression round at different
    points.  Requiring FP32's absolute tolerance from a BF16 graph therefore
    disables the path for the expected one-ULP difference even though the
    retained 32-row WER/SIM screen qualified this exact rewrite.  Do not scale
    the bound with tensor magnitude: one dtype epsilon is the complete extra
    numerical budget granted to the fused launch.
    """
    if dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        return max(
            _NPU_DIT_FINAL_ADDCMUL_MAX_ABS_DRIFT,
            float(torch.finfo(dtype).eps),
        )
    return _NPU_DIT_FINAL_ADDCMUL_MAX_ABS_DRIFT


def _dit_final_from_modulation_fused_npu(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    output: nn.Linear,
) -> torch.Tensor:
    """Fuse final LayerNorm, AdaLN modulation, and projection on Ascend 910C."""
    if output.bias is None:
        raise ValueError("MiniCPM-o fused final AdaLN requires an output bias")
    return torch.ops._C_ascend.npu_minicpmo_final_adaln(
        hidden,
        modulation,
        output.weight,
        output.bias,
    )


def _dit_final_cfg_euler_from_modulation(
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    x: torch.Tensor,
    delta: torch.Tensor,
    cfg_rate: float,
) -> torch.Tensor:
    """Finish one CFG-batch DiT step and update the Euler state."""
    shift, scale = modulation.chunk(2, dim=-1)
    normalized = F.layer_norm(hidden, (512,), eps=1e-6)
    projected = F.linear(
        torch.addcmul(normalized + shift, normalized, scale),
        output_weight,
        output_bias,
    ).transpose(1, 2)
    conditional = projected[:1]
    unconditional = projected[1:2]
    velocity = (1.0 + cfg_rate) * conditional - cfg_rate * unconditional
    return x + delta * velocity


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


def _dit_attention_from_modulation_bsh(
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
    """Build normalized Q/K/V without leaving the DiT's BSH layout.

    Q/K normalization still sees the exact 64-wide head dimension, but the
    head axis is never transposed across sequence. This representation feeds
    Ascend fused attention directly and keeps the fixed K/V slabs contiguous
    in the same sequence-major layout as the surrounding DiT blocks.
    """
    shift_msa = modulation[:, :, :512]
    scale_msa = modulation[:, :, 512:1024]
    hidden = F.layer_norm(x, (512,), eps=1e-6)
    hidden = hidden * (1 + scale_msa) + shift_msa
    batch, width, _ = hidden.shape
    q = F.linear(hidden, q_weight, q_bias).reshape(batch, width, 8, 64)
    k = F.linear(hidden, k_weight, k_bias).reshape(batch, width, 8, 64)
    q = F.layer_norm(q, (64,), q_norm_weight, q_norm_bias, 1e-5).reshape(
        batch, width, 512
    )
    k = F.layer_norm(k, (64,), k_norm_weight, k_norm_bias, 1e-5).reshape(
        batch, width, 512
    )
    v = F.linear(hidden, v_weight, v_bias)
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


def _dit_attention_preamble_bsh_from_modulation(
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
    """Attention preamble whose public outputs remain BSH."""
    return _dit_attention_from_modulation_bsh(
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


def _dit_attention_preamble_fused_qkv_from_modulation(
    x: torch.Tensor,
    modulation: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q/K/V with one graph-visible GEMM and retain BHSD outputs."""
    shift_msa = modulation[:, :, :512]
    scale_msa = modulation[:, :, 512:1024]
    hidden = F.layer_norm(x, (512,), eps=1e-6)
    hidden = hidden * (1 + scale_msa) + shift_msa
    width = x.shape[1]
    qkv = F.linear(hidden, qkv_weight, qkv_bias).reshape(2, width, 3, 8, 64)
    q = qkv[:, :, 0].transpose(1, 2)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    q = F.layer_norm(q, (64,), q_norm_weight, q_norm_bias, 1e-5)
    k = F.layer_norm(k, (64,), k_norm_weight, k_norm_bias, 1e-5)
    return modulation, q, k, v


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
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(conv_input.shape)
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(convolution, cache2)
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(conv_input.shape)
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


def _dit_fused_conv_bf16_ffn_residual(
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
    fc1_weight_kn: torch.Tensor,
    fc1_bias_fp32: torch.Tensor,
    fc2_weight_kn: torch.Tensor,
    fc2_bias_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-major causal Conv plus A2 native full-BF16 FFN."""
    cache1, cache2 = cnn_cache.split((512, 512), dim=1)
    packed, new_cache1 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        conv_input,
        cache1,
    )
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(
        conv_input.shape
    )
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        convolution,
        cache2,
    )
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(
        conv_input.shape
    )
    hidden = hidden + gate_conv * convolution

    mlp = F.layer_norm(hidden, (512,), eps=1e-6)
    mlp = mlp * (1 + scale_mlp) + shift_mlp
    mlp = _npu_bf16_ffn(
        mlp,
        fc1_weight_kn,
        fc1_bias_fp32,
        fc2_weight_kn,
        fc2_bias_fp32,
    )
    hidden = hidden + gate_mlp * mlp
    return hidden, torch.cat((new_cache1, new_cache2), dim=1)


def _dit_fused_conv_mlp_final_euler_residual(
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
    final_modulation: torch.Tensor,
    final_weight: torch.Tensor,
    final_bias: torch.Tensor,
    x: torch.Tensor,
    delta: torch.Tensor,
    cfg_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the last Conv+MLP replay with final projection, CFG, and Euler."""
    hidden, new_cache = _dit_fused_conv_mlp_residual(
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
    return (
        _dit_final_cfg_euler_from_modulation(
            hidden,
            final_modulation,
            final_weight,
            final_bias,
            x,
            delta,
            cfg_rate,
        ),
        new_cache,
    )


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
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(conv_input.shape)
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(convolution, cache2)
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(conv_input.shape)
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


def _dit_cache_major_conv_dynamic_w8a8_mlp_residual(
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
    fc1_scale: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_scale: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical cache-major Conv plus selective dynamic-W8A8 MLP graph.

    Convolution, normalization and residual arithmetic remain in the model's
    native precision.  Only the two Cube-dominant MLP projections use INT8,
    with per-token activation scales and persistent per-channel weight scales.
    """
    cache1, cache2 = cnn_cache.split((512, 512), dim=2)
    packed, new_cache1 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        conv_input,
        cache1,
    )
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(
        conv_input.shape
    )
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        convolution,
        cache2,
    )
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(
        conv_input.shape
    )
    hidden = hidden + gate_conv * convolution

    mlp = F.layer_norm(hidden, (512,), eps=1e-6)
    mlp = mlp * (1 + scale_mlp) + shift_mlp
    mlp = _npu_dynamic_w8a8_linear(mlp, fc1_weight, fc1_scale, fc1_bias)
    mlp = F.gelu(mlp, approximate="tanh")
    mlp = _npu_dynamic_w8a8_linear(mlp, fc2_weight, fc2_scale, fc2_bias)
    hidden = hidden + gate_mlp * mlp
    return hidden, torch.cat((new_cache1, new_cache2), dim=2)


def _dit_fused_conv_dynamic_w8a8_mlp_residual(
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
    fc1_scale: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_scale: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel-major causal Conv plus graph-capturable dynamic-W8A8 MLP.

    This variant keeps the proven outer-CFM graph cache ABI.  The quantization
    operators remain visible inside that raw graph instead of introducing a
    nested TorchAir executable or requiring the incompatible cache-major path.
    """
    cache1, cache2 = cnn_cache.split((512, 512), dim=1)
    packed, new_cache1 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        conv_input,
        cache1,
    )
    convolution = F.linear(packed, conv1_flat_weight, conv1_bias).reshape(
        conv_input.shape
    )
    convolution = F.layer_norm(
        convolution,
        (512,),
        conv_norm_weight,
        conv_norm_bias,
        1e-5,
    )
    convolution = F.mish(convolution)
    packed, new_cache2 = torch.ops._C_ascend.npu_minicpmo_causal_conv_pack(
        convolution,
        cache2,
    )
    convolution = F.linear(packed, conv2_flat_weight, conv2_bias).reshape(
        conv_input.shape
    )
    hidden = hidden + gate_conv * convolution

    mlp = F.layer_norm(hidden, (512,), eps=1e-6)
    mlp = mlp * (1 + scale_mlp) + shift_mlp
    mlp = _npu_dynamic_w8a8_linear(mlp, fc1_weight, fc1_scale, fc1_bias)
    mlp = F.gelu(mlp, approximate="tanh")
    mlp = _npu_dynamic_w8a8_linear(mlp, fc2_weight, fc2_scale, fc2_bias)
    hidden = hidden + gate_mlp * mlp
    return hidden, torch.cat((new_cache1, new_cache2), dim=1)


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


def _dit_flat_capture_conv_mlp_partition(
    *,
    fused_conv_pack: bool,
    cache_major: bool,
    post_attention: bool,
    dynamic_w8a8: bool = False,
    fused_bf16_ffn: bool = False,
):
    """Select the Conv/MLP partition embedded in the steady CFM graph.

    The outer NPUGraph must preserve the same cache layout as the separately
    compiled steady-width Conv/MLP graph. Falling back to the legacy
    channel-major partition here materializes two cache transposes in every
    DiT block and CFM step, even though the fixed slabs are already stored as
    ``[batch, taps, channels]``.
    """
    if not fused_conv_pack:
        return None
    if fused_bf16_ffn:
        return _dit_fused_conv_bf16_ffn_residual
    if dynamic_w8a8:
        return (
            _dit_cache_major_conv_dynamic_w8a8_mlp_residual
            if cache_major
            else _dit_fused_conv_dynamic_w8a8_mlp_residual
        )
    if cache_major and post_attention:
        return _dit_cache_major_post_attention_conv_mlp_residual
    if cache_major:
        return _dit_cache_major_conv_mlp_residual
    return _dit_fused_conv_mlp_residual


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


def _dit_fused_full_block_bsh_from_modulation(
    x: torch.Tensor,
    modulation: torch.Tensor,
    att_cache: torch.Tensor,
    cnn_cache: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One GE-visible BSH DiT block using the shared AdaLN slab.

    Unlike the older full-block diagnostic, this boundary does not recompute
    AdaLN and never converts the sequence-major K/V slabs to BHSD. It is small
    enough to screen one producer-consumer block without constructing a
    sixteen-block or six-step monolith.
    """
    modulation, q, k, v = _dit_attention_preamble_bsh_from_modulation(
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
    cache_k = att_cache.select(0, 0)
    cache_v = att_cache.select(0, 1)
    full_k = torch.cat((k, cache_k), dim=1)
    full_v = torch.cat((v, cache_v), dim=1)
    new_att_cache = torch.stack((full_k, full_v), dim=0)

    batch, query_width, hidden_size = q.shape
    num_heads = 8
    head_dim = 64
    if q.device.type == "npu":
        import torch_npu

        attention = torch_npu.npu_fusion_attention(
            query=q,
            key=full_k,
            value=full_v,
            head_num=num_heads,
            input_layout="BSH",
            scale=head_dim**-0.5,
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            sparse_mode=0,
        )[0]
    else:
        query = q.reshape(batch, query_width, num_heads, head_dim).transpose(1, 2)
        key = full_k.reshape(batch, full_k.shape[1], num_heads, head_dim).transpose(1, 2)
        value = full_v.reshape(batch, full_v.shape[1], num_heads, head_dim).transpose(1, 2)
        attention = F.scaled_dot_product_attention(query, key, value)
        attention = attention.transpose(1, 2).reshape(
            batch, query_width, hidden_size
        )
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


def _dit_full_block_bsh_standard_conv_from_modulation(
    x: torch.Tensor,
    modulation: torch.Tensor,
    att_cache: torch.Tensor,
    cnn_cache: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GE-visible BSH DiT block retaining canonical ``Conv1d`` math.

    This is the numerical control for the native causal-pack variant. Keeping
    the convolution visible to GE also lets the compiler propagate producer
    layouts without an opaque custom-op boundary.
    """
    modulation, q, k, v = _dit_attention_preamble_bsh_from_modulation(
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
    cache_k = att_cache.select(0, 0)
    cache_v = att_cache.select(0, 1)
    full_k = torch.cat((k, cache_k), dim=1)
    full_v = torch.cat((v, cache_v), dim=1)
    new_att_cache = torch.stack((full_k, full_v), dim=0)

    batch, query_width, hidden_size = q.shape
    num_heads = 8
    head_dim = 64
    if q.device.type == "npu":
        import torch_npu

        attention = torch_npu.npu_fusion_attention(
            query=q,
            key=full_k,
            value=full_v,
            head_num=num_heads,
            input_layout="BSH",
            scale=head_dim**-0.5,
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            sparse_mode=0,
        )[0]
    else:
        query = q.reshape(batch, query_width, num_heads, head_dim).transpose(1, 2)
        key = full_k.reshape(batch, full_k.shape[1], num_heads, head_dim).transpose(1, 2)
        value = full_v.reshape(batch, full_v.shape[1], num_heads, head_dim).transpose(1, 2)
        attention = F.scaled_dot_product_attention(query, key, value)
        attention = attention.transpose(1, 2).reshape(
            batch, query_width, hidden_size
        )
    attention = F.linear(attention, proj_weight, proj_bias)

    modulations = modulation.chunk(9, dim=-1)
    hidden = x + modulations[2] * attention
    conv_input = F.layer_norm(hidden, (512,), eps=1e-6)
    conv_input = conv_input * (1 + modulations[7]) + modulations[6]
    hidden, new_cnn_cache = _dit_conv_mlp_residual(
        hidden,
        conv_input,
        cnn_cache,
        modulations[8],
        modulations[3],
        modulations[4],
        modulations[5],
        conv1_weight,
        conv1_bias,
        conv_norm_weight,
        conv_norm_bias,
        conv2_weight,
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
    slab = state.estimator_kv_slabs
    slab_signature = (
        None
        if slab is None
        else (
            tensor_signature(slab.retained),
            tensor_signature(slab.append),
            tuple(tensor_signature(bank) for bank in slab.cnn_banks),
            slab.prompt_length,
            slab.logical_length,
            slab.active_cnn_bank,
            slab.planar,
            slab.bsh_attention,
            slab.cnn_cache_major,
        )
    )
    return flow, hift, slab_signature


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    projected_speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class FixedEstimatorKVSlabs:
    """Fixed-address storage for one request's six-step, 16-block KV state.

    ``retained`` is consumed by the next chunk. ``append`` is a separate
    full-output workspace, so compaction never performs overlapping copies.
    MiniCPM-o prepends each new chunk to the old cache; consequently the first
    ``prompt_length`` frames are not an immutable prompt after the first
    decode. Compaction deliberately preserves that exact ordering.
    """

    retained: torch.Tensor
    append: torch.Tensor
    cnn_banks: tuple[torch.Tensor, torch.Tensor]
    prompt_length: int
    logical_length: int
    active_cnn_bank: int
    planar: bool = False
    bsh_attention: bool = False
    cnn_cache_major: bool = False


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]
    estimator_kv_slabs: FixedEstimatorKVSlabs | None = None


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
        npu_dit_wide_final_adaln: Any = None,
        npu_dit_final_addcmul: Any = None,
        npu_dit_fused_final_adaln: Any = None,
        npu_dit_conv_mlp_graph: Any = None,
        npu_dit_last_block_final_euler_graph: Any = None,
        npu_dit_prompt_conv_mlp_graph: Any = None,
        npu_dit_full_block_graph: Any = None,
        npu_dit_full_stack_graph: Any = None,
        npu_dit_full_block_cache_buckets: Any = None,
        npu_dit_fused_conv_pack: Any = None,
        npu_dit_cache_major: Any = None,
        npu_dit_post_attn_graph: Any = None,
        npu_dit_qkv_pack: Any = None,
        npu_dit_fused_qkv: Any = None,
        npu_dit_attn_cache_out: Any = None,
        npu_cfm_stacked_cache_out: Any = None,
        npu_cfm_fixed_kv_slabs: Any = None,
        npu_cfm_planar_kv_slabs: Any = None,
        npu_dit_bsh_attention: Any = None,
        npu_single_request_cache_passthrough: Any = None,
        npu_dit_fused_conv_block: Any = None,
        npu_dit_fused_conv_linear: Any = None,
        npu_dit_compute_dtype: Any = None,
        npu_cfm_integration_dtype: Any = None,
        npu_dit_dynamic_w8a8: Any = None,
        npu_dit_fused_bf16_ffn: Any = None,
    ):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        hift_parameter = next(self.hift.parameters(), None)
        self._npu_matmul_hf32_enabled = _npu_matmul_hf32_enabled()
        if (
            self._npu_matmul_hf32_enabled
            and hift_parameter is not None
            and hift_parameter.device.type == "npu"
        ):
            npu_backend = getattr(torch, "npu", None)
            matmul_backend = getattr(npu_backend, "matmul", None)
            if matmul_backend is None or not hasattr(matmul_backend, "allow_hf32"):
                raise RuntimeError(
                    f"{_NPU_MATMUL_HF32_ENV}=1 requires torch.npu.matmul.allow_hf32"
                )
            matmul_backend.allow_hf32 = True
            logger.info(
                "MiniCPM-o Stage-2 HF32 MatMul active; FP32 tensor storage and "
                "non-MatMul precision boundaries are unchanged"
            )
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
        self.initial_cfm_timesteps = _npu_initial_cfm_timesteps(
            self.n_timesteps
        )
        self.prompt_cfm_timesteps = _npu_prompt_cfm_timesteps(
            self.n_timesteps
        )
        self._npu_prompt_cache_max_frames = _npu_prompt_cache_max_frames()
        self._npu_prompt_cache_limit_used = False
        self._npu_stage2_timing_enabled = _npu_stage2_timing_enabled()
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
        self._timeline_cache: dict[
            tuple[str, int | None, torch.dtype, int], torch.Tensor
        ] = {}
        self._cfm_delta_cache: dict[
            tuple[str, int | None, torch.dtype, int], torch.Tensor
        ] = {}
        self._timestep_embedding_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._cfg_workspace: dict[tuple[str, tuple[int, ...], torch.dtype, str, int | None], torch.Tensor] = {}
        self._npu_cfm_graphs: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._npu_cfm_graph_disabled = False
        self._npu_cfm_graph_capture_used = False
        self._npu_cfm_graph_replay_used = False
        self._npu_cfm_graph_capture_phases: set[str] = set()
        self._npu_cfm_graph_replay_phases: set[str] = set()
        self._npu_initial_cfm_used = False
        self._npu_prompt_cfm_used = False
        self._npu_cfm_cache_fill_graph_lengths = (
            _npu_cfm_cache_fill_graph_lengths()
        )
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
        self._npu_dit_fused_qkv_preamble_graph: Any | None = None
        self._npu_dit_preamble_graph_disabled = False
        self._npu_dit_preamble_graph_disabled_widths: set[int] = set()
        self._npu_dit_preamble_graph_used = False
        self._npu_dit_wide_adaln_enabled = _npu_dit_wide_adaln_enabled(
            npu_dit_wide_adaln
        )
        self._npu_dit_wide_final_adaln_enabled = (
            _npu_dit_wide_final_adaln_enabled(npu_dit_wide_final_adaln)
        )
        self._npu_dit_final_addcmul_enabled = _npu_dit_final_addcmul_enabled(
            npu_dit_final_addcmul
        )
        self._npu_dit_final_addcmul_used = False
        self._npu_dit_fused_final_adaln_enabled = (
            _npu_dit_fused_final_adaln_enabled(npu_dit_fused_final_adaln)
        )
        self._npu_dit_fused_final_adaln_used = False
        self._npu_dit_wide_adaln_graph: Any | None = None
        self._npu_dit_wide_adaln_steps_graph: Any | None = None
        self._npu_dit_wide_final_adaln_steps_graph: Any | None = None
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
        self.register_buffer(
            "_npu_dit_wide_final_adaln_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_npu_dit_wide_final_adaln_bias",
            None,
            persistent=False,
        )
        self._npu_dit_conv_mlp_graph_enabled = _npu_dit_conv_mlp_graph_enabled(npu_dit_conv_mlp_graph)
        self._npu_dit_conv_mlp_graph: Any | None = None
        self._npu_dit_conv_mlp_graph_disabled = False
        self._npu_dit_conv_mlp_graph_used = False
        self._npu_dit_last_block_final_euler_graph_enabled = (
            _npu_dit_last_block_final_euler_graph_enabled(
                npu_dit_last_block_final_euler_graph
            )
        )
        self._npu_dit_last_block_final_euler_graph: Any | None = None
        self._npu_dit_last_block_final_euler_graph_used = False
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
        self._npu_dit_fused_qkv_enabled = _npu_dit_fused_qkv_enabled(npu_dit_fused_qkv)
        self._npu_dit_fused_qkv_used = False
        self._npu_dit_attn_cache_out_enabled = _npu_dit_attn_cache_out_enabled(
            npu_dit_attn_cache_out
        )
        self._npu_dit_attn_cache_out_used = False
        self._npu_cfm_stacked_cache_out_enabled = _npu_cfm_stacked_cache_out_enabled(
            npu_cfm_stacked_cache_out
        )
        self._npu_cfm_stacked_cache_out_used = False
        self._npu_cfm_fixed_kv_slabs_enabled = _npu_cfm_fixed_kv_slabs_enabled(
            npu_cfm_fixed_kv_slabs
        )
        self._npu_cfm_fixed_kv_slabs_used = False
        self._npu_cfm_planar_kv_slabs_enabled = _npu_cfm_planar_kv_slabs_enabled(
            npu_cfm_planar_kv_slabs
        )
        if self._npu_cfm_planar_kv_slabs_enabled:
            self._npu_cfm_fixed_kv_slabs_enabled = True
        self._npu_cfm_planar_kv_slabs_used = False
        self._npu_dit_bsh_attention_enabled = _npu_dit_bsh_attention_enabled(
            npu_dit_bsh_attention
        )
        self._npu_dit_bsh_attention_used = False
        self._npu_dit_bsh_preamble_graph: Any | None = None
        if self._npu_dit_bsh_attention_enabled:
            self._npu_cfm_fixed_kv_slabs_enabled = True
            self._npu_cfm_planar_kv_slabs_enabled = True
        self._npu_cfm_fixed_kv_tail_fallback_used = False
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
        self._npu_dit_dynamic_w8a8_enabled = _npu_dit_dynamic_w8a8_enabled(
            npu_dit_dynamic_w8a8
        )
        self._npu_dit_dynamic_w8a8_used = False
        self._npu_dit_fused_bf16_ffn_enabled = _npu_dit_fused_bf16_ffn_enabled(
            npu_dit_fused_bf16_ffn
        )
        self._npu_dit_fused_bf16_ffn_used = False
        requested_dit_dtype = _npu_dit_compute_dtype(npu_dit_compute_dtype)
        requested_integration_dtype = _npu_cfm_integration_dtype(
            npu_cfm_integration_dtype
        )
        self._npu_dit_compute_dtype = torch.float32
        self._npu_cfm_integration_dtype = torch.float32
        self._npu_dit_mixed_precision_enabled = False
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        estimator_parameter = (
            next(estimator.parameters(), None)
            if isinstance(estimator, nn.Module)
            else None
        )
        if requested_dit_dtype != torch.float32:
            if (
                not isinstance(estimator, nn.Module)
                or not isinstance(estimator_parameter, torch.Tensor)
                or estimator_parameter.device.type != "npu"
            ):
                logger.warning(
                    "MiniCPM-o NPU DiT %s requested outside an NPU estimator; "
                    "retaining FP32",
                    requested_dit_dtype,
                )
            else:
                try:
                    estimator.to(dtype=requested_dit_dtype)
                    self._npu_dit_compute_dtype = requested_dit_dtype
                    self._npu_dit_mixed_precision_enabled = True
                    if requested_integration_dtype == requested_dit_dtype:
                        self._npu_cfm_integration_dtype = requested_integration_dtype
                    logger.info(
                        "MiniCPM-o NPU DiT mixed precision active: estimator=%s, "
                        "CFM integration=%s, HiFT=float32",
                        requested_dit_dtype,
                        self._npu_cfm_integration_dtype,
                    )
                except Exception:
                    # ``Module.to`` can have converted an early parameter
                    # before a later buffer fails. Restore the complete
                    # estimator so an opt-in precision failure cannot leave a
                    # partially converted serving path.
                    estimator.to(dtype=torch.float32)
                    logger.warning(
                        "MiniCPM-o NPU DiT precision conversion failed; retaining FP32",
                        exc_info=True,
                    )
        if (
            requested_integration_dtype != torch.float32
            and requested_integration_dtype != self._npu_dit_compute_dtype
        ):
            logger.warning(
                "MiniCPM-o NPU CFM integration dtype %s requires the same DiT "
                "compute dtype; retaining FP32 integration",
                requested_integration_dtype,
            )
        if (
            self._npu_dit_mixed_precision_enabled
            and self._npu_cfm_integration_dtype != self._npu_dit_compute_dtype
        ):
            # This diagnostic region was already rejected on the serving
            # gate, and its graph signature assumes that the CFM state and
            # DiT hidden state have the same dtype. Keep it out of the BF16
            # experiment instead of triggering a second compile or a hidden
            # state cast inside the six-step loop.
            self._npu_dit_last_block_final_euler_graph_enabled = False
        if self._npu_dit_wide_adaln_enabled and not self._npu_dit_preamble_graph_enabled:
            self._npu_dit_wide_adaln_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN requires the DiT preamble graph; disabling it"
            )
        if self._npu_dit_fused_qkv_enabled and not self._npu_dit_wide_adaln_enabled:
            self._npu_dit_fused_qkv_enabled = False
            logger.warning(
                "MiniCPM-o fused QKV requires wide AdaLN modulation; disabling it"
            )
        if self._npu_dit_fused_qkv_enabled:
            self._npu_dit_qkv_pack_enabled = False
        if self._npu_dit_wide_final_adaln_enabled and not self._npu_dit_wide_adaln_enabled:
            self._npu_dit_wide_final_adaln_enabled = False
            logger.warning(
                "MiniCPM-o wide final AdaLN requires all-step wide AdaLN; disabling it"
            )
        if self._npu_dit_final_addcmul_enabled and not self._npu_dit_wide_final_adaln_enabled:
            self._npu_dit_final_addcmul_enabled = False
            logger.warning(
                "MiniCPM-o final Addcmul requires wide final AdaLN; disabling it"
            )
        if self._npu_dit_fused_final_adaln_enabled and not self._npu_dit_wide_final_adaln_enabled:
            self._npu_dit_fused_final_adaln_enabled = False
            logger.warning(
                "MiniCPM-o fused final AdaLN requires wide final AdaLN; disabling it"
            )
        if self._npu_dit_wide_adaln_enabled and self._npu_dit_qkv_pack_enabled:
            self._npu_dit_qkv_pack_enabled = False
            logger.warning(
                "MiniCPM-o wide AdaLN currently uses the ordinary QKV preamble; disabling native QKV pack"
            )
        if self._npu_dit_bsh_attention_enabled and not (
            self._npu_dit_preamble_graph_enabled
            and self._npu_dit_wide_adaln_enabled
        ):
            self._npu_dit_bsh_attention_enabled = False
            logger.warning(
                "MiniCPM-o BSH attention requires the graph-visible preamble "
                "and wide AdaLN; disabling it"
            )
        if self._npu_dit_bsh_attention_enabled and (
            self._npu_dit_qkv_pack_enabled or self._npu_dit_fused_qkv_enabled
        ):
            self._npu_dit_qkv_pack_enabled = False
            self._npu_dit_fused_qkv_enabled = False
            logger.info(
                "MiniCPM-o BSH attention supersedes BHSD QKV packing candidates"
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
        if self._npu_dit_dynamic_w8a8_enabled and not (
            isinstance(estimator, nn.Module)
            and isinstance(estimator_parameter, torch.Tensor)
            and estimator_parameter.device.type == "npu"
            and self._npu_dit_conv_mlp_graph_enabled
            and self._npu_dit_fused_conv_pack_enabled
            and not self._npu_dit_post_attn_graph_enabled
            and not self._npu_dit_fused_conv_linear_enabled
        ):
            self._npu_dit_dynamic_w8a8_enabled = False
            logger.warning(
                "MiniCPM-o selective dynamic W8A8 requires the NPU "
                "causal-pack Conv+MLP graph; disabling it"
            )
        if self._npu_dit_fused_bf16_ffn_enabled and not (
            isinstance(estimator, nn.Module)
            and isinstance(estimator_parameter, torch.Tensor)
            and estimator_parameter.device.type == "npu"
            and estimator_parameter.dtype == torch.bfloat16
            and self._npu_dit_conv_mlp_graph_enabled
            and self._npu_dit_fused_conv_pack_enabled
            and not self._npu_dit_cache_major_enabled
            and not self._npu_dit_post_attn_graph_enabled
            and not self._npu_dit_fused_conv_linear_enabled
        ):
            self._npu_dit_fused_bf16_ffn_enabled = False
            logger.warning(
                "MiniCPM-o fused BF16 FFN requires the channel-major NPU "
                "causal-pack Conv+MLP graph and a BF16 DiT; disabling it"
            )
        if self._npu_dit_fused_bf16_ffn_enabled:
            if self._npu_dit_dynamic_w8a8_enabled:
                self._npu_dit_dynamic_w8a8_enabled = False
                logger.info(
                    "MiniCPM-o fused BF16 FFN supersedes selective dynamic W8A8"
                )
            self._npu_dit_last_block_final_euler_graph_enabled = False
            try:
                self._prepare_npu_dit_fused_bf16_ffn_weights(estimator)
            except Exception:
                self._npu_dit_fused_bf16_ffn_enabled = False
                logger.warning(
                    "MiniCPM-o fused BF16 FFN weight preparation failed; "
                    "retaining canonical BF16 MLPs",
                    exc_info=True,
                )
        if self._npu_dit_dynamic_w8a8_enabled:
            # The last-block fusion has a different argument contract. Keep
            # the quantized candidate confined to one canonical graph rather
            # than silently falling back to FP32 in the sixteenth block.
            self._npu_dit_last_block_final_euler_graph_enabled = False
            try:
                self._prepare_npu_dit_dynamic_w8a8_weights(estimator)
            except Exception:
                self._npu_dit_dynamic_w8a8_enabled = False
                logger.warning(
                    "MiniCPM-o selective dynamic W8A8 weight preparation failed; "
                    "retaining FP32 DiT MLP weights",
                    exc_info=True,
                )
        if self._npu_dit_last_block_final_euler_graph_enabled and not (
            self._npu_dit_conv_mlp_graph_enabled
            and self._npu_dit_fused_conv_pack_enabled
            and self._npu_dit_wide_final_adaln_enabled
            and self._npu_dit_final_addcmul_enabled
            and not self._npu_dit_cache_major_enabled
            and not self._npu_dit_post_attn_graph_enabled
            and not self._npu_dit_fused_conv_linear_enabled
            and not self._npu_dit_full_block_graph_enabled
            and not self._npu_dit_full_stack_graph_enabled
        ):
            self._npu_dit_last_block_final_euler_graph_enabled = False
            logger.warning(
                "MiniCPM-o last-block final Euler graph requires the accepted "
                "causal-pack Conv+MLP and final Addcmul profile; disabling it"
            )
        if (
            self._npu_dit_last_block_final_euler_graph_enabled
            and self._npu_dit_fused_final_adaln_enabled
        ):
            self._npu_dit_fused_final_adaln_enabled = False
            logger.info(
                "MiniCPM-o last-block final Euler graph supersedes the standalone fused final AdaLN"
            )
        self._warmup_npu_dit_mlp_graph()
        self._warmup_npu_dit_wide_adaln_graph()
        self._warmup_npu_dit_preamble_graph()
        self._warmup_npu_dit_conv_mlp_graph()
        self._warmup_npu_dit_last_block_final_euler_graph()
        self._warmup_npu_dit_prompt_conv_mlp_graphs()
        self._warmup_npu_dit_full_block_graphs()
        self._warmup_npu_dit_full_stack_graphs()

    @staticmethod
    def _prepare_npu_dit_dynamic_w8a8_weights(estimator: nn.Module) -> None:
        blocks = getattr(estimator, "blocks", None)
        if not blocks:
            raise ValueError("MiniCPM-o DiT estimator has no blocks")
        for block in blocks:
            mlp = block.mlp
            for name in ("fc1", "fc2"):
                linear = getattr(mlp, name)
                if not isinstance(linear, nn.Linear) or linear.bias is None:
                    raise ValueError(f"MiniCPM-o DiT {name} is not a biased Linear")
                quantized, scale = _quantize_dynamic_w8a8_weight(linear.weight)
                mlp.register_buffer(
                    f"_minicpmo_w8a8_{name}_weight",
                    quantized,
                    persistent=False,
                )
                mlp.register_buffer(
                    f"_minicpmo_w8a8_{name}_scale",
                    scale,
                    persistent=False,
                )

    @staticmethod
    def _prepare_npu_dit_fused_bf16_ffn_weights(estimator: nn.Module) -> None:
        blocks = getattr(estimator, "blocks", None)
        if not blocks:
            raise ValueError("MiniCPM-o DiT estimator has no blocks")
        for block in blocks:
            mlp = block.mlp
            for name in ("fc1", "fc2"):
                linear = getattr(mlp, name)
                if not isinstance(linear, nn.Linear) or linear.bias is None:
                    raise ValueError(f"MiniCPM-o DiT {name} is not a biased Linear")
                mlp.register_buffer(
                    f"_minicpmo_bf16_ffn_{name}_weight_kn",
                    linear.weight.detach().transpose(0, 1).contiguous(),
                    persistent=False,
                )
                mlp.register_buffer(
                    f"_minicpmo_bf16_ffn_{name}_bias_fp32",
                    linear.bias.detach().to(dtype=torch.float32),
                    persistent=False,
                )

    @staticmethod
    def _dit_fused_bf16_ffn_args(block: nn.Module) -> tuple[torch.Tensor, ...]:
        mlp = block.mlp
        return (
            mlp._minicpmo_bf16_ffn_fc1_weight_kn,
            mlp._minicpmo_bf16_ffn_fc1_bias_fp32,
            mlp._minicpmo_bf16_ffn_fc2_weight_kn,
            mlp._minicpmo_bf16_ffn_fc2_bias_fp32,
        )
    @staticmethod
    def _dit_dynamic_w8a8_mlp_args(block: nn.Module) -> tuple[torch.Tensor, ...]:
        mlp = block.mlp
        return (
            mlp._minicpmo_w8a8_fc1_weight,
            mlp._minicpmo_w8a8_fc1_scale,
            mlp.fc1.bias,
            mlp._minicpmo_w8a8_fc2_weight,
            mlp._minicpmo_w8a8_fc2_scale,
            mlp.fc2.bias,
        )

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
            self._warmup_npu_dit_wide_final_adaln_graph(
                estimator,
                step_embeddings,
                step_actual,
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

    def _warmup_npu_dit_wide_final_adaln_graph(
        self,
        estimator: nn.Module,
        step_embeddings: torch.Tensor,
        expected_blocks: torch.Tensor,
    ) -> None:
        if not self._npu_dit_wide_final_adaln_enabled:
            return
        try:
            final_layer = estimator.final_layer
            projection = final_layer.adaLN_modulation[1]
            if (
                not isinstance(projection, nn.Linear)
                or tuple(projection.weight.shape) != (2 * 512, 512)
                or projection.bias is None
                or projection.weight.device.type != "npu"
            ):
                raise TypeError("final AdaLN projection is incompatible")
            self._npu_dit_wide_final_adaln_weight = torch.cat(
                (self._npu_dit_wide_adaln_weight, projection.weight.detach()),
                dim=0,
            ).contiguous()
            self._npu_dit_wide_final_adaln_bias = torch.cat(
                (self._npu_dit_wide_adaln_bias, projection.bias.detach()),
                dim=0,
            ).contiguous()
            actual_blocks, actual_final = (
                self._get_npu_dit_wide_final_adaln_steps_graph()(
                    step_embeddings,
                    self._npu_dit_wide_final_adaln_weight,
                    self._npu_dit_wide_final_adaln_bias,
                )
            )
            expected_final = torch.stack(
                [
                    final_layer.adaLN_modulation(step_embeddings[step])
                    for step in range(self.n_timesteps)
                ]
            )
            block_drift = float((actual_blocks - expected_blocks).abs().max().item())
            final_drift = float((actual_final - expected_final).abs().max().item())
            if (
                not torch.isfinite(actual_blocks).all()
                or not torch.isfinite(actual_final).all()
                or block_drift > _NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT
                or final_drift > _NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT
            ):
                raise RuntimeError(
                    "wide final AdaLN exceeded its startup drift bound: "
                    f"block_max_abs_drift={block_drift:.9g}, "
                    f"final_max_abs_drift={final_drift:.9g}, "
                    f"limit={_NPU_DIT_WIDE_ADALN_MAX_ABS_DRIFT:.9g}"
                )
            torch.npu.synchronize()
            logger.info(
                "Compiled bounded-drift MiniCPM-o all-step AdaLN graph for "
                "16 blocks plus final layer; block_max_abs_drift=%.9g, "
                "final_max_abs_drift=%.9g",
                block_drift,
                final_drift,
            )
            self._warmup_npu_dit_final_addcmul(
                final_layer,
                actual_final[0],
            )
            self._warmup_npu_dit_fused_final_adaln(
                final_layer,
                actual_final[0],
            )
        except Exception:
            self._npu_dit_wide_final_adaln_enabled = False
            self._npu_dit_final_addcmul_enabled = False
            self._npu_dit_fused_final_adaln_enabled = False
            self._npu_dit_wide_final_adaln_steps_graph = None
            self._npu_dit_wide_final_adaln_weight = None
            self._npu_dit_wide_final_adaln_bias = None
            logger.warning(
                "MiniCPM-o wide final AdaLN compilation/parity gate failed; "
                "retaining block-only wide AdaLN",
                exc_info=True,
            )

    def _warmup_npu_dit_final_addcmul(
        self,
        final_layer: nn.Module,
        modulation: torch.Tensor,
    ) -> None:
        if not self._npu_dit_final_addcmul_enabled:
            return
        try:
            width = self._npu_dit_mlp_graph_width
            hidden = torch.linspace(
                -0.125,
                0.125,
                2 * width * 512,
                device=modulation.device,
                dtype=modulation.dtype,
            ).reshape(2, width, 512)
            expected = _dit_final_from_modulation(
                hidden,
                modulation,
                final_layer.norm_final,
                final_layer.linear,
            )
            actual = _dit_final_from_modulation_addcmul(
                hidden,
                modulation,
                final_layer.norm_final,
                final_layer.linear,
            )
            max_abs_drift = float((actual - expected).abs().max().item())
            drift_limit = _dit_final_addcmul_drift_limit(actual.dtype)
            if (
                not torch.isfinite(actual).all()
                or max_abs_drift > drift_limit
            ):
                raise RuntimeError(
                    "final Addcmul exceeded its startup drift bound: "
                    f"max_abs_drift={max_abs_drift:.9g}, "
                    f"limit={drift_limit:.9g}"
                )
            logger.info(
                "Validated dtype-bounded MiniCPM-o final Addcmul path; "
                "max_abs_drift=%.9g, limit=%.9g",
                max_abs_drift,
                drift_limit,
            )
        except Exception:
            self._npu_dit_final_addcmul_enabled = False
            logger.warning(
                "MiniCPM-o final Addcmul parity gate failed; retaining canonical AdaLN",
                exc_info=True,
            )

    def _warmup_npu_dit_fused_final_adaln(
        self,
        final_layer: nn.Module,
        modulation: torch.Tensor,
    ) -> None:
        if not self._npu_dit_fused_final_adaln_enabled:
            return
        try:
            norm = final_layer.norm_final
            if (
                tuple(norm.normalized_shape) != (512,)
                or norm.elementwise_affine
                or float(norm.eps) != 1.0e-6
            ):
                raise RuntimeError("fused final AdaLN requires affine-free LayerNorm(512, eps=1e-6)")
            hidden = torch.linspace(
                -0.125,
                0.125,
                2 * 50 * 512,
                device=modulation.device,
                dtype=modulation.dtype,
            ).reshape(2, 50, 512)
            expected = _dit_final_from_modulation_addcmul(
                hidden,
                modulation,
                norm,
                final_layer.linear,
            )
            actual = _dit_final_from_modulation_fused_npu(
                hidden,
                modulation,
                final_layer.linear,
            )
            error = (actual - expected).abs()
            max_abs_drift = float(error.max().item())
            mean_abs_drift = float(error.mean().item())
            if (
                not torch.isfinite(actual).all()
                or max_abs_drift > _NPU_DIT_FUSED_FINAL_ADALN_MAX_ABS_DRIFT
                or mean_abs_drift > _NPU_DIT_FUSED_FINAL_ADALN_MEAN_ABS_DRIFT
            ):
                raise RuntimeError(
                    "fused final AdaLN exceeded its startup drift bound: "
                    f"max_abs_drift={max_abs_drift:.9g}, "
                    f"mean_abs_drift={mean_abs_drift:.9g}"
                )
            logger.info(
                "Validated MiniCPM-o fused final AdaLN; max_abs_drift=%.9g, "
                "mean_abs_drift=%.9g",
                max_abs_drift,
                mean_abs_drift,
            )
        except Exception:
            self._npu_dit_fused_final_adaln_enabled = False
            logger.warning(
                "MiniCPM-o fused final AdaLN parity gate failed; retaining Addcmul",
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

    def _get_npu_dit_wide_final_adaln_steps_graph(self):
        if self._npu_dit_wide_final_adaln_steps_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_wide_final_adaln_steps_graph = torch.compile(
                _dit_wide_adaln_steps_with_final,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_wide_final_adaln_steps_graph

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

    @staticmethod
    def _validate_npu_dit_bsh_attention(
        block: nn.Module,
        hidden: torch.Tensor,
        modulation: torch.Tensor,
        candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[float, float]:
        """Validate loaded-weight BSH preamble and fused attention together."""
        attention = block.attn
        reference = _dit_attention_preamble_from_modulation(
            hidden,
            modulation,
            attention.to_q.weight,
            attention.to_q.bias,
            attention.to_k.weight,
            attention.to_k.bias,
            attention.to_v.weight,
            attention.to_v.bias,
            attention.q_norm.weight,
            attention.q_norm.bias,
            attention.k_norm.weight,
            attention.k_norm.bias,
        )
        differences: list[torch.Tensor] = []
        for bsh, bhsd in zip(candidate[1:], reference[1:], strict=True):
            expected_bsh = bhsd.transpose(1, 2).reshape_as(bsh)
            differences.append((bsh.float() - expected_bsh.float()).abs())

        _, q, k, v = candidate
        cache_width = 7
        bsh_cache = q.new_zeros((2, q.shape[0], cache_width, q.shape[-1]))
        bsh_output = q.new_empty(
            (2, q.shape[0], q.shape[1] + cache_width, q.shape[-1])
        )
        candidate_attention, _ = (
            BatchedToken2Wav._attention_from_projected_qkv_bsh_planar(
                attention,
                q,
                k,
                v,
                bsh_cache,
                bsh_output,
            )
        )
        legacy_cache = BatchedToken2Wav._legacy_att_cache_from_bsh_planar(
            bsh_cache,
            int(attention.num_heads),
            int(attention.head_dim),
        )
        reference_attention, _ = BatchedToken2Wav._attention_from_projected_qkv(
            attention,
            reference[1],
            reference[2],
            reference[3],
            legacy_cache,
        )
        differences.append(
            (candidate_attention.float() - reference_attention.float()).abs()
        )
        max_abs_drift = max(float(value.max().item()) for value in differences)
        mean_abs_drift = sum(float(value.mean().item()) for value in differences) / len(
            differences
        )
        if (
            max_abs_drift > _NPU_DIT_BSH_ATTENTION_MAX_ABS_DRIFT
            or mean_abs_drift > _NPU_DIT_BSH_ATTENTION_MEAN_ABS_DRIFT
        ):
            raise RuntimeError(
                "BSH attention exceeded its loaded-checkpoint drift bound: "
                f"max_abs_drift={max_abs_drift:.9g}, "
                f"mean_abs_drift={mean_abs_drift:.9g}"
            )
        return max_abs_drift, mean_abs_drift

    @staticmethod
    def _validate_npu_cfm_graph_attention(
        block: nn.Module,
        candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[float, float]:
        """Gate the graph-capturable attention lowering against fused BSH attention."""
        attention = block.attn
        _, q, k, v = candidate
        cache_width = 402
        cache = q.new_zeros((2, q.shape[0], cache_width, q.shape[-1]))
        fused_output = q.new_empty(
            (2, q.shape[0], q.shape[1] + cache_width, q.shape[-1])
        )
        explicit_output = torch.empty_like(fused_output)
        fused, _ = BatchedToken2Wav._attention_from_projected_qkv_bsh_planar(
            attention,
            q,
            k,
            v,
            cache,
            fused_output,
        )
        explicit, _ = BatchedToken2Wav._attention_from_projected_qkv_bsh_planar(
            attention,
            q,
            k,
            v,
            cache,
            explicit_output,
            explicit_attention=True,
        )
        difference = (explicit.float() - fused.float()).abs()
        max_abs_drift = float(difference.max().item())
        mean_abs_drift = float(difference.mean().item())
        if (
            max_abs_drift > _NPU_CFM_GRAPH_ATTN_MAX_ABS_DRIFT
            or mean_abs_drift > _NPU_CFM_GRAPH_ATTN_MEAN_ABS_DRIFT
        ):
            raise RuntimeError(
                "CFM graph attention exceeded its loaded-checkpoint drift bound: "
                f"max_abs_drift={max_abs_drift:.9g}, "
                f"mean_abs_drift={mean_abs_drift:.9g}"
            )
        return max_abs_drift, mean_abs_drift

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
        if self._npu_dit_fused_qkv_enabled:
            for estimator_block in blocks:
                attention = estimator_block.attn
                attention.register_buffer(
                    "_minicpmo_fused_qkv_weight",
                    torch.cat(
                        (
                            attention.to_q.weight.detach(),
                            attention.to_k.weight.detach(),
                            attention.to_v.weight.detach(),
                        ),
                        dim=0,
                    ).contiguous(),
                    persistent=False,
                )
                attention.register_buffer(
                    "_minicpmo_fused_qkv_bias",
                    torch.cat(
                        (
                            attention.to_q.bias.detach(),
                            attention.to_k.bias.detach(),
                            attention.to_v.bias.detach(),
                        ),
                        dim=0,
                    ).contiguous(),
                    persistent=False,
                )
        for width in self._npu_dit_graph_widths:
            try:
                graph_fn = self._get_npu_dit_preamble_graph(width)
                if graph_fn is None:
                    continue
                x = weight.new_zeros((2, width, 512))
                with torch.inference_mode():
                    result = self._call_npu_dit_preamble_graph(
                        graph_fn,
                        block,
                        x,
                        time_embedding,
                        None if wide_modulations is None else wide_modulations[:, :, 0, :],
                    )
                    if self._npu_dit_bsh_attention_enabled:
                        if wide_modulations is None:
                            raise RuntimeError(
                                "MiniCPM-o BSH attention requires loaded wide modulation"
                            )
                        max_abs_drift, mean_abs_drift = (
                            self._validate_npu_dit_bsh_attention(
                                block,
                                x,
                                wide_modulations[:, :, 0, :],
                                result,
                            )
                        )
                        if (
                            _npu_cfm_graph_enabled()
                            and width == self._npu_dit_mlp_graph_width
                        ):
                            try:
                                graph_max_abs_drift, graph_mean_abs_drift = (
                                    self._validate_npu_cfm_graph_attention(
                                        block,
                                        result,
                                    )
                                )
                            except Exception:
                                self._npu_cfm_graph_disabled = True
                                logger.warning(
                                    "MiniCPM-o explicit CFM graph attention "
                                    "parity gate failed; retaining fused BSH eager path",
                                    exc_info=True,
                                )
                torch.npu.synchronize()
                if self._npu_dit_fused_qkv_enabled:
                    logger.info(
                        "Compiled MiniCPM-o NPU DiT fused-QKV preamble graph for 2x%dx512, 8x64 heads",
                        width,
                    )
                elif self._npu_dit_bsh_attention_enabled:
                    logger.info(
                        "Compiled MiniCPM-o NPU DiT BSH attention preamble graph "
                        "for 2x%dx512, 8x64 heads; max_abs_drift=%.9g, "
                        "mean_abs_drift=%.9g",
                        width,
                        max_abs_drift,
                        mean_abs_drift,
                    )
                    if (
                        _npu_cfm_graph_enabled()
                        and not self._npu_cfm_graph_disabled
                        and width == self._npu_dit_mlp_graph_width
                    ):
                        logger.info(
                            "Validated MiniCPM-o graph-capturable explicit "
                            "attention at cache=402; max_abs_drift=%.9g, "
                            "mean_abs_drift=%.9g",
                            graph_max_abs_drift,
                            graph_mean_abs_drift,
                        )
                elif self._npu_dit_qkv_pack_enabled and width == self._npu_dit_mlp_graph_width:
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
                if self._npu_dit_bsh_attention_enabled:
                    self._npu_dit_bsh_attention_enabled = False
                    self._npu_dit_bsh_preamble_graph = None
                    logger.warning(
                        "MiniCPM-o BSH attention parity/compile gate failed at "
                        "width=%d; retrying the accepted planar BHSD path",
                        width,
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
                                (
                                    None
                                    if wide_modulations is None
                                    else wide_modulations[:, :, 0, :]
                                ),
                            )
                        torch.npu.synchronize()
                        continue
                    except Exception:
                        logger.warning(
                            "MiniCPM-o accepted planar BHSD retry failed at width=%d",
                            width,
                            exc_info=True,
                        )
                if self._npu_dit_fused_qkv_enabled:
                    self._npu_dit_fused_qkv_enabled = False
                    self._npu_dit_fused_qkv_preamble_graph = None
                    logger.warning(
                        "MiniCPM-o fused-QKV preamble compilation failed at width=%d; retrying ordinary projections",
                        width,
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
                                None if wide_modulations is None else wide_modulations[:, :, 0, :],
                            )
                        torch.npu.synchronize()
                        continue
                    except Exception:
                        logger.warning(
                            "MiniCPM-o ordinary-projection preamble retry failed at width=%d",
                            width,
                            exc_info=True,
                        )
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
                    "MiniCPM-o NPU DiT preamble graph compilation failed at "
                    "width=%d; using eager attention for that width",
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
        if self._npu_dit_bsh_attention_enabled:
            if self._npu_dit_bsh_preamble_graph is None:
                self._npu_dit_bsh_preamble_graph = torch.compile(
                    _dit_attention_preamble_bsh_from_modulation,
                    backend=torchair.get_npu_backend(compiler_config=compiler_config),
                    fullgraph=True,
                    dynamic=False,
                )
            return self._npu_dit_bsh_preamble_graph
        if self._npu_dit_fused_qkv_enabled:
            if self._npu_dit_fused_qkv_preamble_graph is None:
                self._npu_dit_fused_qkv_preamble_graph = torch.compile(
                    _dit_attention_preamble_fused_qkv_from_modulation,
                    backend=torchair.get_npu_backend(compiler_config=compiler_config),
                    fullgraph=True,
                    dynamic=False,
                )
            return self._npu_dit_fused_qkv_preamble_graph
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
        if graph_fn is self._npu_dit_fused_qkv_preamble_graph:
            if modulation is None:
                raise RuntimeError("MiniCPM-o fused-QKV preamble requires wide AdaLN modulation")
            attention = block.attn
            return graph_fn(
                hidden,
                modulation,
                attention._minicpmo_fused_qkv_weight,
                attention._minicpmo_fused_qkv_bias,
                attention.q_norm.weight,
                attention.q_norm.bias,
                attention.k_norm.weight,
                attention.k_norm.bias,
            )
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
        # Width 100 is the two-second steady streaming bucket used by the
        # single-chip latency/throughput profile.  The causal-pack and GE MLP
        # partitions are shape-polymorphic at compile time; each admitted
        # width still receives its own static executable.
        return width in (50, 64, 100) and BatchedToken2Wav._dit_conv_mlp_layout_compatible(block)

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
        width = self._npu_dit_mlp_graph_width
        hidden = weight.new_zeros((2, width, 512))
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
                    common_args = (
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
                    )
                    if self._npu_dit_fused_bf16_ffn_enabled:
                        graph_fn(
                            *common_args,
                            *self._dit_fused_bf16_ffn_args(block),
                        )
                    elif self._npu_dit_dynamic_w8a8_enabled:
                        graph_fn(
                            *common_args,
                            *self._dit_dynamic_w8a8_mlp_args(block),
                        )
                    else:
                        graph_fn(
                            *common_args,
                            block.mlp.fc1.weight,
                            block.mlp.fc1.bias,
                            block.mlp.fc2.weight,
                            block.mlp.fc2.bias,
                        )
            torch.npu.synchronize()
            if self._npu_dit_post_attn_graph_enabled:
                logger.info(
                    "Compiled MiniCPM-o NPU DiT post-attention Conv+MLP megagraph for 2x%dx512",
                    width,
                )
            else:
                logger.info(
                    "Compiled MiniCPM-o NPU DiT Conv+MLP megagraph for 2x%dx512",
                    width,
                )
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
            if self._npu_dit_fused_bf16_ffn_enabled:
                graph_partition = _dit_fused_conv_bf16_ffn_residual
            elif self._npu_dit_dynamic_w8a8_enabled:
                graph_partition = (
                    _dit_cache_major_conv_dynamic_w8a8_mlp_residual
                    if self._npu_dit_cache_major_enabled
                    else _dit_fused_conv_dynamic_w8a8_mlp_residual
                )
            elif self._npu_dit_fused_conv_linear_enabled:
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

    def _warmup_npu_dit_last_block_final_euler_graph(self) -> None:
        """Compile and validate the last-block-to-Euler producer/consumer graph."""
        if not self._npu_dit_last_block_final_euler_graph_enabled:
            return
        decoder = getattr(self.flow, "decoder", None)
        estimator = getattr(decoder, "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        final_layer = getattr(estimator, "final_layer", None)
        block = blocks[-1] if blocks else None
        if (
            block is None
            or final_layer is None
            or self._npu_dit_conv_mlp_graph_disabled
            or not self._npu_dit_final_addcmul_enabled
            or not self._dit_conv_mlp_compatible(
                block,
                self._npu_dit_mlp_graph_width,
            )
            or tuple(final_layer.norm_final.normalized_shape) != (512,)
            or final_layer.norm_final.elementwise_affine
            or float(final_layer.norm_final.eps) != 1.0e-6
            or not isinstance(final_layer.linear, nn.Linear)
            or tuple(final_layer.linear.weight.shape) != (80, 512)
            or final_layer.linear.bias is None
        ):
            self._npu_dit_last_block_final_euler_graph_enabled = False
            logger.warning(
                "MiniCPM-o last-block final Euler graph disabled: model layout is incompatible"
            )
            return

        conv1 = block.conv.block[1]
        conv_norm = block.conv.block[3]
        conv2 = block.conv.block[6]
        weight = block.mlp.fc1.weight
        width = self._npu_dit_mlp_graph_width
        hidden = torch.linspace(
            -0.125,
            0.125,
            2 * width * 512,
            device=weight.device,
            dtype=weight.dtype,
        ).reshape(2, width, 512)
        conv_input = hidden.flip(-1).contiguous()
        cnn_cache = weight.new_zeros((2, 1024, 2))
        gate_conv = weight.new_full((2, 1, 512), 0.05)
        shift_mlp = weight.new_full((2, 1, 512), 0.01)
        scale_mlp = weight.new_full((2, 1, 512), 0.02)
        gate_mlp = weight.new_full((2, 1, 512), 0.05)
        time_embedding = weight.new_full((2, 1, 512), 0.125)
        final_modulation = final_layer.adaLN_modulation(time_embedding)
        x = torch.linspace(
            -0.05,
            0.05,
            width * 80,
            device=weight.device,
            dtype=weight.dtype,
        ).reshape(1, 80, width)
        delta = weight.new_tensor(0.125)
        cfg_rate = float(decoder.inference_cfg_rate)
        base_graph = self._get_npu_dit_conv_mlp_graph()
        fused_graph = self._get_npu_dit_last_block_final_euler_graph()
        if base_graph is None or fused_graph is None:
            self._npu_dit_last_block_final_euler_graph_enabled = False
            return
        args = (
            hidden,
            conv_input,
            cnn_cache,
            gate_conv,
            shift_mlp,
            scale_mlp,
            gate_mlp,
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

        def control_replay() -> tuple[torch.Tensor, torch.Tensor]:
            control_hidden, control_cache = base_graph(*args)
            return (
                _dit_final_cfg_euler_from_modulation(
                    control_hidden,
                    final_modulation,
                    final_layer.linear.weight,
                    final_layer.linear.bias,
                    x,
                    delta,
                    cfg_rate,
                ),
                control_cache,
            )

        def candidate_replay() -> tuple[torch.Tensor, torch.Tensor]:
            return fused_graph(
                *args,
                final_modulation,
                final_layer.linear.weight,
                final_layer.linear.bias,
                x,
                delta,
                cfg_rate,
            )

        def timed_replay_us(replay: Any) -> float:
            start = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(_NPU_DIT_LAST_BLOCK_FINAL_EULER_PERF_ITERATIONS):
                replay()
            end_event.record()
            torch.npu.synchronize()
            return (
                float(start.elapsed_time(end_event))
                * 1000.0
                / _NPU_DIT_LAST_BLOCK_FINAL_EULER_PERF_ITERATIONS
            )

        try:
            with torch.inference_mode():
                expected_x, expected_cache = control_replay()
                actual_x, actual_cache = candidate_replay()
                x_drift = float((actual_x - expected_x).abs().max().item())
                x_mean_drift = float((actual_x - expected_x).abs().mean().item())
                cache_drift = float(
                    (actual_cache - expected_cache).abs().max().item()
                )
                if (
                    not torch.isfinite(actual_x).all()
                    or not torch.isfinite(actual_cache).all()
                    or x_drift > _NPU_DIT_LAST_BLOCK_FINAL_EULER_MAX_ABS_DRIFT
                    or x_mean_drift
                    > _NPU_DIT_LAST_BLOCK_FINAL_EULER_MEAN_ABS_DRIFT
                    or cache_drift > _NPU_DIT_LAST_BLOCK_FINAL_EULER_MAX_ABS_DRIFT
                ):
                    raise RuntimeError(
                        "last-block final Euler graph exceeded its startup drift bound: "
                        f"x_max_abs_drift={x_drift:.9g}, "
                        f"x_mean_abs_drift={x_mean_drift:.9g}, "
                        f"cache_max_abs_drift={cache_drift:.9g}"
                    )
            torch.npu.synchronize()
            control_trials: list[float] = []
            candidate_trials: list[float] = []
            with torch.inference_mode():
                for trial in range(
                    _NPU_DIT_LAST_BLOCK_FINAL_EULER_PERF_TRIALS
                ):
                    ordered = (
                        (
                            (control_replay, control_trials),
                            (candidate_replay, candidate_trials),
                        )
                        if trial % 2 == 0
                        else (
                            (candidate_replay, candidate_trials),
                            (control_replay, control_trials),
                        )
                    )
                    for replay, samples in ordered:
                        samples.append(timed_replay_us(replay))
            control_us = sorted(control_trials)[len(control_trials) // 2]
            candidate_us = sorted(candidate_trials)[len(candidate_trials) // 2]
            speedup = control_us / candidate_us
            saving_us = control_us - candidate_us
            if not _npu_dit_last_block_final_euler_perf_qualifies(
                control_us,
                candidate_us,
            ):
                self._npu_dit_last_block_final_euler_graph = None
                self._npu_dit_last_block_final_euler_graph_enabled = False
                logger.warning(
                    "MiniCPM-o last-block final Euler graph disabled by device-time "
                    "usefulness gate: control=%.3f us, candidate=%.3f us, "
                    "saving=%.3f us, speedup=%.4fx; required saving>=%.3f us "
                    "and speedup>=%.4fx",
                    control_us,
                    candidate_us,
                    saving_us,
                    speedup,
                    _NPU_DIT_LAST_BLOCK_FINAL_EULER_MIN_SAVING_US,
                    _NPU_DIT_LAST_BLOCK_FINAL_EULER_MIN_SPEEDUP,
                )
                return
            logger.info(
                "Compiled bounded-drift MiniCPM-o last-block final Euler graph; "
                "x_max_abs_drift=%.9g, x_mean_abs_drift=%.9g, "
                "cache_max_abs_drift=%.9g, control=%.3f us, candidate=%.3f us, "
                "saving=%.3f us, speedup=%.4fx",
                x_drift,
                x_mean_drift,
                cache_drift,
                control_us,
                candidate_us,
                saving_us,
                speedup,
            )
        except Exception:
            self._npu_dit_last_block_final_euler_graph = None
            self._npu_dit_last_block_final_euler_graph_enabled = False
            logger.warning(
                "MiniCPM-o last-block final Euler graph compilation/parity gate failed; "
                "retaining the accepted split path",
                exc_info=True,
            )

    def _get_npu_dit_last_block_final_euler_graph(self):
        if not self._npu_dit_last_block_final_euler_graph_enabled:
            return None
        if self._npu_dit_last_block_final_euler_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_last_block_final_euler_graph = torch.compile(
                _dit_fused_conv_mlp_final_euler_residual,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_last_block_final_euler_graph

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
            or (
                not self._npu_dit_bsh_attention_enabled
                and not self._npu_dit_fused_conv_pack_enabled
            )
        ):
            self._npu_dit_full_block_graph_enabled = False
            logger.warning("MiniCPM-o NPU full-block graph disabled: block or causal-pack layout is incompatible")
            return
        try:
            graph_fn = self._get_npu_dit_full_block_graph()
        except (ImportError, AttributeError, RuntimeError):
            self._npu_dit_full_block_graph_enabled = False
            logger.warning(
                "MiniCPM-o NPU full-block graph disabled: the installed "
                "vLLM-Ascend does not provide its optional converter",
                exc_info=True,
            )
            return
        hidden = weight.new_zeros((2, 50, 512))
        time_embedding = weight.new_zeros((2, 1, 512))
        cnn_cache = weight.new_zeros((2, 1024, 2))
        for cache_length in lengths:
            try:
                att_cache = weight.new_zeros(
                    (2, 2, cache_length, 512)
                    if self._npu_dit_bsh_attention_enabled
                    else (2, 8, cache_length, 128)
                )
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
                    "Compiled MiniCPM-o NPU %s full DiT block graph for "
                    "width=50, attention cache=%d",
                    "BSH standard-Conv" if self._npu_dit_bsh_attention_enabled else "legacy causal-pack",
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
                (
                    _dit_full_block_bsh_standard_conv_from_modulation
                    if self._npu_dit_bsh_attention_enabled
                    else _dit_fused_full_block
                ),
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
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv1 = block.conv.block[1]
        conv_norm = block.conv.block[3]
        conv2 = block.conv.block[6]
        if self._npu_dit_bsh_attention_enabled:
            if modulation is None:
                modulation = block.adaLN_modulation(time_embedding)
            return graph_fn(
                hidden,
                modulation,
                att_cache,
                cnn_cache,
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

    def _timeline_for(
        self,
        value: torch.Tensor,
        num_timesteps: int | None = None,
    ) -> torch.Tensor:
        steps = self.n_timesteps if num_timesteps is None else num_timesteps
        key = (value.device.type, value.device.index, value.dtype, steps)
        timeline = self._timeline_cache.get(key)
        if timeline is None:
            base = (
                self.cfm_timeline_base
                if steps == self.n_timesteps
                else 1
                - torch.cos(
                    torch.linspace(0, 1, steps + 1, dtype=torch.float32)
                    * 0.5
                    * torch.pi
                )
            )
            timeline = base.to(device=value.device, dtype=value.dtype)
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
        num_timesteps = int(timeline.shape[0]) - 1
        key = (
            timeline.device.type,
            timeline.device.index,
            timeline.dtype,
            num_timesteps,
        )
        cached = self._cfm_delta_cache.get(key)
        if cached is not None:
            return cached
        time = timeline[0]
        dt = timeline[1] - timeline[0]
        deltas: list[torch.Tensor] = []
        for step in range(num_timesteps):
            deltas.append(dt)
            time = time + dt
            if step + 1 < num_timesteps:
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
            timing = (
                self._npu_stage2_timing_enabled
                and self.speech_window.device.type == "npu"
            )
            if timing:
                torch.npu.synchronize()
                timing_start = time.perf_counter()
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
            if timing:
                torch.npu.synchronize()
                logger.info(
                    "MiniCPM-o Stage-2 timing: prompt_features=%.3fms, "
                    "speech_tokens=%d, mels=%d",
                    (time.perf_counter() - timing_start) * 1000.0,
                    int(cached.speech_tokens.shape[1]),
                    int(cached.mels.shape[1]),
                )
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
        bsh_attention: bool = False,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        planar_att = BatchedToken2Wav._is_planar_att_cache(old_att)
        old_att_len = (
            0
            if old_att is None
            else (
                int(old_att.shape[-2])
                if planar_att or bsh_attention
                else int(old_att.shape[3])
            )
        )
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        head_dim = int(block0.attn.head_dim)
        cnn_shape = (
            (depth, batch_size, cnn_width, cnn_channels)
            if cache_major
            else (depth, batch_size, cnn_channels, cnn_width)
        )
        if bsh_attention:
            att_shape = (
                depth,
                2,
                batch_size,
                old_att_len + chunk_size,
                heads * head_dim,
            )
        elif planar_att:
            att_shape = (
                depth,
                2,
                batch_size,
                heads,
                old_att_len + chunk_size,
                head_dim,
            )
        else:
            att_shape = (
                depth,
                batch_size,
                heads,
                old_att_len + chunk_size,
                head_dim * 2,
            )
        return cnn_shape, att_shape

    @classmethod
    def _estimator_buffers(
        cls,
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
        *,
        cache_major: bool = False,
        bsh_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_shape, att_shape = cls._estimator_buffer_shapes(
            estimator,
            x,
            old_att,
            cache_major=cache_major,
            bsh_attention=bsh_attention,
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
        final_modulation: torch.Tensor | None = None,
        cfm_x: torch.Tensor | None = None,
        cfm_delta: torch.Tensor | None = None,
        cfm_cfg_rate: float | None = None,
        flat_capture: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cache_major = self._is_cache_major_cnn(cnn_cache)
        planar_att = self._is_planar_att_cache(att_cache)
        first_attention = estimator.blocks[0].attn
        attention_hidden_size = int(first_attention.num_heads) * int(
            first_attention.head_dim
        )
        bsh_attention = (
            self._npu_dit_bsh_attention_enabled
            and (
                att_cache is None
                or self._is_bsh_planar_att_cache(
                    att_cache, attention_hidden_size
                )
            )
        )
        if cnn_out is None or att_out is None:
            if cnn_out is not None or att_out is not None:
                raise ValueError("cnn_out and att_out must be provided together")
            cnn_out, att_out = self._estimator_buffers(
                estimator,
                estimator_input,
                att_cache,
                cache_major=cache_major,
                bsh_attention=bsh_attention,
            )
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        graph_width = int(estimator_input.shape[2])
        use_mlp_graph = flat_capture or (
            self._npu_dit_mlp_graph_enabled
            and not self._npu_dit_mlp_graph_disabled
            and estimator_input.device.type == "npu"
            and int(estimator_input.shape[0]) == 2
            and graph_width in self._npu_dit_graph_widths
            and graph_width not in self._npu_dit_mlp_graph_disabled_widths
        )
        if use_mlp_graph:
            try:
                graph_fn = (
                    _dit_mlp_residual
                    if flat_capture
                    else self._get_npu_dit_mlp_graph()
                )
                if graph_fn is not None:
                    full_stack_cache_length = _dit_attention_cache_length(att_cache)
                    if (
                        not flat_capture
                        and graph_width == self._npu_dit_mlp_graph_width
                        and not planar_att
                        and not bsh_attention
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
                        return result, cnn_out, att_out, False
                    preamble_graph_fn = (
                        _dit_attention_preamble_bsh_from_modulation
                        if flat_capture and bsh_attention
                        else None
                    )
                    if not flat_capture and (
                        self._npu_dit_preamble_graph_enabled
                        and not self._npu_dit_preamble_graph_disabled
                        and graph_width not in self._npu_dit_preamble_graph_disabled_widths
                    ):
                        preamble_graph_fn = self._get_npu_dit_preamble_graph(graph_width)
                    conv_mlp_graph_fn = (
                        _dit_flat_capture_conv_mlp_partition(
                            fused_conv_pack=self._npu_dit_fused_conv_pack_enabled,
                            cache_major=cache_major,
                            post_attention=self._npu_dit_post_attn_graph_enabled,
                            dynamic_w8a8=self._npu_dit_dynamic_w8a8_enabled,
                            fused_bf16_ffn=self._npu_dit_fused_bf16_ffn_enabled,
                        )
                        if flat_capture
                        else None
                    )
                    last_block_final_euler_graph_fn = None
                    conv_mlp_standard_weights = False
                    full_block_graph_fn = None
                    full_block_cache_length = _dit_attention_cache_length(att_cache)
                    if not flat_capture and full_block_graph_fn is None and (
                        graph_width == self._npu_dit_mlp_graph_width
                        and not planar_att
                        and not cache_major
                        and cnn_cache is not None
                        and att_cache is not None
                        and self._npu_dit_full_block_graph_enabled
                        and full_block_cache_length in self._npu_dit_full_block_cache_buckets
                        and full_block_cache_length not in self._npu_dit_full_block_graph_disabled_lengths
                    ):
                        full_block_graph_fn = self._get_npu_dit_full_block_graph()
                    if not flat_capture and full_block_graph_fn is None and (
                        graph_width == self._npu_dit_mlp_graph_width
                        and cnn_cache is not None
                        and att_cache is not None
                        and self._npu_dit_conv_mlp_graph_enabled
                        and not self._npu_dit_conv_mlp_graph_disabled
                    ):
                        conv_mlp_graph_fn = self._get_npu_dit_conv_mlp_graph()
                        if (
                            conv_mlp_graph_fn is not None
                            and self._npu_dit_last_block_final_euler_graph_enabled
                            and final_modulation is not None
                            and cfm_x is not None
                            and cfm_delta is not None
                            and cfm_cfg_rate is not None
                        ):
                            last_block_final_euler_graph_fn = (
                                self._get_npu_dit_last_block_final_euler_graph()
                            )
                    elif not flat_capture and full_block_graph_fn is None and (
                        graph_width != self._npu_dit_mlp_graph_width
                        and self._npu_dit_prompt_conv_mlp_graph_enabled
                        and graph_width not in self._npu_dit_prompt_conv_mlp_graph_disabled_widths
                    ):
                        conv_mlp_graph_fn = self._get_npu_dit_prompt_conv_mlp_graph()
                        conv_mlp_standard_weights = conv_mlp_graph_fn is not None
                    result, cfm_updated = self._estimator_blocks_forward_chunk_mlp_graph(
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
                        final_modulation,
                        last_block_final_euler_graph_fn,
                        cfm_x,
                        cfm_delta,
                        cfm_cfg_rate,
                        flat_capture,
                    )
                    if not self._npu_dit_mlp_graph_used:
                        logger.info(
                            "MiniCPM-o NPU DiT MLP graph replay active for CFG batch=2, width=%d",
                            graph_width,
                        )
                        self._npu_dit_mlp_graph_used = True
                    if preamble_graph_fn is not None:
                        if self._npu_dit_fused_qkv_enabled and not self._npu_dit_fused_qkv_used:
                            logger.info(
                                "MiniCPM-o NPU DiT fused-QKV attention preamble graph replay active"
                            )
                            self._npu_dit_fused_qkv_used = True
                        elif (
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
                    if cfm_updated and not self._npu_dit_last_block_final_euler_graph_used:
                        logger.info(
                            "MiniCPM-o NPU last-block-to-Euler megagraph replay active"
                        )
                        self._npu_dit_last_block_final_euler_graph_used = True
                    return result, cnn_out, att_out, cfm_updated
            except Exception:
                if flat_capture:
                    raise
                self._npu_dit_mlp_graph_disabled_widths.add(graph_width)
                logger.warning(
                    "MiniCPM-o NPU DiT graph execution failed at width=%d; using eager blocks for that width",
                    graph_width,
                    exc_info=True,
                )
        if cache_major:
            old_cnn = old_cnn.transpose(-2, -1).contiguous()
            cnn_out, att_out = self._estimator_buffers(
                estimator,
                estimator_input,
                att_cache,
                bsh_attention=bsh_attention,
            )
        legacy_att_out = att_out
        if bsh_attention:
            legacy_old_att = (
                None
                if att_cache is None
                else self._legacy_att_cache_from_bsh_planar(
                    att_cache,
                    int(first_attention.num_heads),
                    int(first_attention.head_dim),
                )
            )
            _, legacy_att_shape = self._estimator_buffer_shapes(
                estimator,
                estimator_input,
                legacy_old_att,
            )
            legacy_att_out = estimator_input.new_empty(legacy_att_shape)
            old_att = (
                legacy_old_att
                if legacy_old_att is not None
                else [None] * len(estimator.blocks)
            )
        elif planar_att:
            legacy_old_att = self._legacy_att_cache_from_planar(att_cache)
            _, legacy_att_shape = self._estimator_buffer_shapes(
                estimator,
                estimator_input,
                legacy_old_att,
            )
            legacy_att_out = estimator_input.new_empty(legacy_att_shape)
            old_att = legacy_old_att
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            legacy_att_out,
        )
        if bsh_attention:
            self._copy_legacy_att_cache_to_bsh_planar(att_out, legacy_att_out)
        elif planar_att:
            self._copy_legacy_att_cache_to_planar(att_out, legacy_att_out)
        return result, cnn_out, att_out, False

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
        precomputed_final_modulation: torch.Tensor | None = None,
        last_block_final_euler_graph_fn: Any | None = None,
        cfm_x: torch.Tensor | None = None,
        cfm_delta: torch.Tensor | None = None,
        cfm_cfg_rate: float | None = None,
        flat_capture: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        """Replay enabled shape-bucketed partitions with exact eager fallbacks."""
        hidden = estimator.in_proj(estimator_input.transpose(1, 2))
        wide_modulations = precomputed_wide_modulations
        if (
            wide_modulations is None
            and self._npu_dit_wide_adaln_enabled
            and preamble_graph_fn is not None
            and (
                full_block_graph_fn is None
                or self._npu_dit_bsh_attention_enabled
            )
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
        cfm_updated = False
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
                    (
                        None
                        if wide_modulations is None
                        else wide_modulations[:, :, block_idx, :]
                    ),
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

            block_att_cache = old_att[block_idx]
            block_att_output = att_out[block_idx]
            planar_att = self._is_planar_att_cache(block_att_cache)
            bsh_attention = (
                self._npu_dit_bsh_attention_enabled
                and (
                    block_att_cache is None
                    or self._is_bsh_planar_att_cache(
                        block_att_cache,
                        int(block.attn.num_heads) * int(block.attn.head_dim),
                    )
                )
            )
            if q is None or k is None or v is None:
                legacy_att_cache = (
                    self._legacy_att_cache_from_bsh_planar(
                        block_att_cache,
                        int(block.attn.num_heads),
                        int(block.attn.head_dim),
                    )
                    if bsh_attention and block_att_cache is not None
                    else (
                        self._legacy_att_cache_from_planar(block_att_cache)
                        if planar_att
                        else block_att_cache
                    )
                )
                attention, new_att = block.attn.forward_chunk(
                    block.norm1(hidden) * (1 + scale_msa) + shift_msa,
                    legacy_att_cache,
                    None,
                )
                attention_cache_written = planar_att or bsh_attention
                if bsh_attention:
                    new_att = self._copy_legacy_att_cache_to_bsh_planar(
                        block_att_output, new_att
                    )
                elif planar_att:
                    new_att = self._copy_legacy_att_cache_to_planar(
                        block_att_output, new_att
                    )
            elif bsh_attention:
                attention, new_att = self._attention_from_projected_qkv_bsh_planar(
                    block.attn,
                    q,
                    k,
                    v,
                    block_att_cache,
                    block_att_output,
                    explicit_attention=flat_capture,
                )
                attention_cache_written = True
                if not self._npu_dit_bsh_attention_used:
                    logger.info(
                        "MiniCPM-o BSH planar K/V plus fused attention active"
                    )
                    self._npu_dit_bsh_attention_used = True
            elif planar_att:
                attention, new_att = self._attention_from_projected_qkv_planar(
                    block.attn,
                    q,
                    k,
                    v,
                    block_att_cache,
                    block_att_output,
                )
                attention_cache_written = True
                if not self._npu_cfm_planar_kv_slabs_used:
                    logger.info(
                        "MiniCPM-o contiguous planar K/V attention cache active"
                    )
                    self._npu_cfm_planar_kv_slabs_used = True
            else:
                output_cache = (
                    block_att_output
                    if self._npu_dit_attn_cache_out_enabled
                    else None
                )
                attention, new_att = BatchedToken2Wav._attention_from_projected_qkv(
                    block.attn,
                    q,
                    k,
                    v,
                    block_att_cache,
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
                        if (
                            self._npu_dit_cache_major_enabled
                            and int(hidden.shape[1])
                            == self._npu_dit_mlp_graph_width
                        )
                        else (hidden.shape[0], 1024, 2)
                    )
                conv1_weight = (
                    conv1.weight if conv_mlp_standard_weights else self._dit_conv_graph_weight(conv1)
                )
                conv2_weight = (
                    conv2.weight if conv_mlp_standard_weights else self._dit_conv_graph_weight(conv2)
                )
                use_last_block_final_euler_graph = (
                    last_block_final_euler_graph_fn is not None
                    and block_idx == len(estimator.blocks) - 1
                    and not post_attention_graph
                    and not conv_mlp_standard_weights
                    and precomputed_final_modulation is not None
                    and cfm_x is not None
                    and cfm_delta is not None
                    and cfm_cfg_rate is not None
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
                elif use_last_block_final_euler_graph:
                    final_layer = estimator.final_layer
                    try:
                        hidden, new_cnn = last_block_final_euler_graph_fn(
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
                            precomputed_final_modulation,
                            final_layer.linear.weight,
                            final_layer.linear.bias,
                            cfm_x,
                            cfm_delta,
                            cfm_cfg_rate,
                        )
                        cfm_updated = True
                    except Exception:
                        self._npu_dit_last_block_final_euler_graph_enabled = False
                        self._npu_dit_last_block_final_euler_graph = None
                        logger.warning(
                            "MiniCPM-o last-block final Euler graph replay failed; "
                            "retaining the accepted Conv+MLP and final path",
                            exc_info=True,
                        )
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
                else:
                    common_args = (
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
                    )
                    if (
                        self._npu_dit_fused_bf16_ffn_enabled
                        and not conv_mlp_standard_weights
                    ):
                        hidden, new_cnn = conv_mlp_graph_fn(
                            *common_args,
                            *self._dit_fused_bf16_ffn_args(block),
                        )
                        if not self._npu_dit_fused_bf16_ffn_used:
                            logger.info(
                                "MiniCPM-o A2 native fused BF16 DiT FFN active"
                            )
                            self._npu_dit_fused_bf16_ffn_used = True
                    elif (
                        self._npu_dit_dynamic_w8a8_enabled
                        and not conv_mlp_standard_weights
                    ):
                        hidden, new_cnn = conv_mlp_graph_fn(
                            *common_args,
                            *self._dit_dynamic_w8a8_mlp_args(block),
                        )
                        if not self._npu_dit_dynamic_w8a8_used:
                            logger.info(
                                "MiniCPM-o selective dynamic W8A8 DiT MLP graph active"
                            )
                            self._npu_dit_dynamic_w8a8_used = True
                    else:
                        hidden, new_cnn = conv_mlp_graph_fn(
                            *common_args,
                            block.mlp.fc1.weight,
                            block.mlp.fc1.bias,
                            block.mlp.fc2.weight,
                            block.mlp.fc2.bias,
                        )
            cnn_out[block_idx].copy_(new_cnn)
            if not attention_cache_written:
                att_out[block_idx, :, :, : new_att.shape[2], :].copy_(new_att)

        if cfm_updated:
            return hidden, True
        if precomputed_final_modulation is None:
            hidden = estimator.final_layer(hidden, time_embedding)
        elif flat_capture:
            final_layer = estimator.final_layer
            hidden = _dit_final_from_modulation(
                hidden,
                precomputed_final_modulation,
                final_layer.norm_final,
                final_layer.linear,
            )
        elif (
            self._npu_dit_fused_final_adaln_enabled
            and hidden.shape == (2, 50, 512)
        ):
            final_layer = estimator.final_layer
            try:
                hidden = _dit_final_from_modulation_fused_npu(
                    hidden,
                    precomputed_final_modulation,
                    final_layer.linear,
                )
                if not self._npu_dit_fused_final_adaln_used:
                    logger.info("MiniCPM-o fused final AdaLN active")
                    self._npu_dit_fused_final_adaln_used = True
            except Exception:
                self._npu_dit_fused_final_adaln_enabled = False
                logger.warning(
                    "MiniCPM-o fused final AdaLN failed; retaining Addcmul",
                    exc_info=True,
                )
                if self._npu_dit_final_addcmul_enabled:
                    hidden = _dit_final_from_modulation_addcmul(
                        hidden,
                        precomputed_final_modulation,
                        final_layer.norm_final,
                        final_layer.linear,
                    )
                else:
                    hidden = _dit_final_from_modulation(
                        hidden,
                        precomputed_final_modulation,
                        final_layer.norm_final,
                        final_layer.linear,
                    )
        elif self._npu_dit_final_addcmul_enabled:
            final_layer = estimator.final_layer
            try:
                hidden = _dit_final_from_modulation_addcmul(
                    hidden,
                    precomputed_final_modulation,
                    final_layer.norm_final,
                    final_layer.linear,
                )
                if not self._npu_dit_final_addcmul_used:
                    logger.info("MiniCPM-o final Addcmul replay active")
                    self._npu_dit_final_addcmul_used = True
            except Exception:
                self._npu_dit_final_addcmul_enabled = False
                logger.warning(
                    "MiniCPM-o final Addcmul replay failed; retaining canonical AdaLN",
                    exc_info=True,
                )
                hidden = _dit_final_from_modulation(
                    hidden,
                    precomputed_final_modulation,
                    final_layer.norm_final,
                    final_layer.linear,
                )
        else:
            final_layer = estimator.final_layer
            hidden = _dit_final_from_modulation(
                hidden,
                precomputed_final_modulation,
                final_layer.norm_final,
                final_layer.linear,
            )
        return hidden.transpose(1, 2), False

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

    @staticmethod
    def _is_planar_att_cache(cache: torch.Tensor | None) -> bool:
        """Whether cache uses ``[..., K/V, CFG, heads, time, head_dim]``."""
        return (
            isinstance(cache, torch.Tensor)
            and cache.ndim in {5, 6, 7}
            and int(cache.shape[-5]) == 2
        )

    @staticmethod
    def _is_bsh_planar_att_cache(
        cache: torch.Tensor | None,
        hidden_size: int = 512,
    ) -> bool:
        """Whether cache uses ``[..., K/V, CFG, time, hidden]``."""
        return (
            isinstance(cache, torch.Tensor)
            and cache.ndim in {4, 5, 6}
            and int(cache.shape[-4]) == 2
            and int(cache.shape[-1]) == hidden_size
            and not BatchedToken2Wav._is_planar_att_cache(cache)
        )

    @staticmethod
    def _legacy_att_cache_from_bsh_planar(
        cache: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Convert sequence-major planar K/V to CosyVoice's packed BHSD cache."""
        key = cache.select(-4, 0)
        value = cache.select(-4, 1)

        def to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                *tensor.shape[:-1], num_heads, head_dim
            ).transpose(-3, -2)

        return torch.cat((to_bhsd(key), to_bhsd(value)), dim=-1)

    @staticmethod
    def _copy_legacy_att_cache_to_bsh_planar(
        destination: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Copy CosyVoice packed BHSD K/V into sequence-major planes."""
        key, value = source.chunk(2, dim=-1)

        def to_bsh(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.transpose(-3, -2).flatten(-2)

        destination.select(-4, 0).copy_(to_bsh(key))
        destination.select(-4, 1).copy_(to_bsh(value))
        return destination

    @staticmethod
    def _legacy_att_cache_from_planar(cache: torch.Tensor) -> torch.Tensor:
        if not BatchedToken2Wav._is_planar_att_cache(cache):
            return cache
        key = cache.select(-5, 0)
        value = cache.select(-5, 1)
        return torch.cat((key, value), dim=-1)

    @staticmethod
    def _copy_legacy_att_cache_to_planar(
        destination: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        key, value = source.chunk(2, dim=-1)
        destination.select(-5, 0).copy_(key)
        destination.select(-5, 1).copy_(value)
        return destination

    @staticmethod
    def _attention_from_projected_qkv_planar(
        attention_module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_cache: torch.Tensor | None,
        output_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append and attend with independently contiguous K and V planes."""
        output_k = output_cache.select(0, 0)
        output_v = output_cache.select(0, 1)
        if att_cache is None:
            output_k.copy_(k)
            output_v.copy_(v)
        else:
            cache_k = att_cache.select(0, 0)
            cache_v = att_cache.select(0, 1)
            torch.cat((k, cache_k), dim=2, out=output_k)
            torch.cat((v, cache_v), dim=2, out=output_v)
        hidden = F.scaled_dot_product_attention(q, output_k, output_v)
        batch_size, _, width, _ = hidden.shape
        hidden = hidden.transpose(1, 2).reshape(batch_size, width, -1)
        hidden = attention_module.proj(hidden)
        hidden = attention_module.proj_drop(hidden)
        return hidden, output_cache

    @staticmethod
    def _attention_from_projected_qkv_bsh_planar(
        attention_module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_cache: torch.Tensor | None,
        output_cache: torch.Tensor,
        *,
        explicit_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append planar BSH K/V and run attention without BSH/BHSD transposes."""
        output_k = output_cache.select(0, 0)
        output_v = output_cache.select(0, 1)
        if att_cache is None:
            output_k.copy_(k)
            output_v.copy_(v)
        else:
            cache_k = att_cache.select(0, 0)
            cache_v = att_cache.select(0, 1)
            torch.cat((k, cache_k), dim=1, out=output_k)
            torch.cat((v, cache_v), dim=1, out=output_v)

        batch, query_width, hidden_size = q.shape
        num_heads = int(attention_module.num_heads)
        head_dim = int(attention_module.head_dim)
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                "DiT BSH attention hidden size mismatch: "
                f"hidden={hidden_size}, heads={num_heads}, head_dim={head_dim}"
            )
        if explicit_attention:
            # ``npu_fusion_attention`` launches an auxiliary stream and cannot
            # be enclosed by raw NPUGraph on the competition CANN 9.0 image.
            # Keep accumulation in FP32 so this graph-only lowering stays
            # within the loaded-checkpoint parity gate above.
            query = q.float().reshape(
                batch, query_width, num_heads, head_dim
            ).transpose(1, 2)
            key = output_k.float().reshape(
                batch, output_k.shape[1], num_heads, head_dim
            ).transpose(1, 2)
            value = output_v.float().reshape(
                batch, output_v.shape[1], num_heads, head_dim
            ).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
            probabilities = torch.softmax(scores, dim=-1)
            hidden = torch.matmul(probabilities, value)
            hidden = hidden.transpose(1, 2).reshape(
                batch, query_width, hidden_size
            ).to(dtype=q.dtype)
        elif q.device.type == "npu":
            import torch_npu

            hidden = torch_npu.npu_fusion_attention(
                query=q,
                key=output_k,
                value=output_v,
                head_num=num_heads,
                input_layout="BSH",
                scale=head_dim**-0.5,
                keep_prob=1.0,
                pre_tockens=2147483647,
                next_tockens=2147483647,
                sparse_mode=0,
            )[0]
        else:
            # Exact semantic fallback used by CPU tests and unsupported
            # platforms. Production NPU execution takes the BSH FIA branch.
            query = q.reshape(batch, query_width, num_heads, head_dim).transpose(1, 2)
            key = output_k.reshape(
                batch, output_k.shape[1], num_heads, head_dim
            ).transpose(1, 2)
            value = output_v.reshape(
                batch, output_v.shape[1], num_heads, head_dim
            ).transpose(1, 2)
            hidden = F.scaled_dot_product_attention(query, key, value)
            hidden = hidden.transpose(1, 2).reshape(batch, query_width, hidden_size)
        hidden = attention_module.proj(hidden)
        hidden = attention_module.proj_drop(hidden)
        return hidden, output_cache

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
        num_timesteps = int(timeline.shape[0]) - 1
        embedder = estimator.t_embedder
        frequency_size = getattr(embedder, "frequency_embedding_size", None)
        scale = getattr(embedder, "scale", None)
        mlp = getattr(embedder, "mlp", None)
        if not isinstance(frequency_size, int) or scale is None or not isinstance(mlp, nn.Module):
            return torch.stack(
                [
                    embedder(timeline[step].expand(cfg_batch_size)).unsqueeze(1)
                    for step in range(num_timesteps)
                ]
            )

        key = (
            id(embedder),
            cfg_batch_size,
            timeline.device.type,
            timeline.device.index,
            timeline.dtype,
            num_timesteps,
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
        for step in range(num_timesteps):
            arguments = (time * float(scale))[:, None] * frequencies[None]
            sinusoidal = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
            if frequency_size % 2:
                sinusoidal = torch.cat((sinusoidal, torch.zeros_like(sinusoidal[:, :1])), dim=-1)
            embeddings.append(mlp(sinusoidal).unsqueeze(1))
            time = time + dt
            if step + 1 < num_timesteps:
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
        cnn_output: torch.Tensor | None = None,
        att_output: torch.Tensor | None = None,
        flat_capture: bool = False,
        num_timesteps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        steps = self.n_timesteps if num_timesteps is None else num_timesteps
        if not 1 <= steps <= self.n_timesteps:
            raise ValueError(
                f"num_timesteps must be in [1, {self.n_timesteps}], got {steps}"
            )
        reduced_steps = steps != self.n_timesteps
        batch_size = int(mu.shape[0])
        integration_dtype = (
            self._npu_cfm_integration_dtype
            if self._npu_dit_mixed_precision_enabled
            else mu.dtype
        )
        estimator_dtype = (
            self._npu_dit_compute_dtype
            if self._npu_dit_mixed_precision_enabled
            else mu.dtype
        )
        offset = int(att_cache.shape[-2]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        x = (
            decoder.rand_noise[:, :, offset:end]
            .to(dtype=integration_dtype)
            .expand(batch_size, -1, -1)
            .clone()
        )
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
        estimator_mu = mu.to(dtype=estimator_dtype)
        estimator_speakers = speakers.to(dtype=estimator_dtype)
        estimator_cond = cond.to(dtype=estimator_dtype)
        estimator_timeline = self._timeline_for(estimator_mu, steps)
        integration_reference = mu.to(dtype=integration_dtype)
        integration_timeline = self._timeline_for(integration_reference, steps)
        mu_cfg = self._cfg_pair("mu", estimator_mu, zero_unconditional=True)
        speakers_cfg = self._cfg_pair(
            "speakers",
            estimator_speakers,
            zero_unconditional=True,
        )
        cond_cfg = self._cfg_pair("cond", estimator_cond, zero_unconditional=True)
        time_embeddings = self._estimator_time_embeddings(
            estimator,
            estimator_timeline,
            batch_size * 2,
        )
        wide_modulation_steps: torch.Tensor | None = None
        final_modulation_steps: torch.Tensor | None = None
        if self._npu_dit_wide_adaln_enabled:
            if reduced_steps and self._npu_dit_wide_final_adaln_enabled:
                wide_modulation_steps, final_modulation_steps = (
                    _dit_wide_adaln_steps_with_final(
                        time_embeddings,
                        self._npu_dit_wide_final_adaln_weight,
                        self._npu_dit_wide_final_adaln_bias,
                    )
                )
            elif reduced_steps:
                wide_modulation_steps = _dit_wide_adaln_steps(
                    time_embeddings,
                    self._npu_dit_wide_adaln_weight,
                    self._npu_dit_wide_adaln_bias,
                )
            elif flat_capture and self._npu_dit_wide_final_adaln_enabled:
                wide_modulation_steps, final_modulation_steps = (
                    _dit_wide_adaln_steps_with_final(
                        time_embeddings,
                        self._npu_dit_wide_final_adaln_weight,
                        self._npu_dit_wide_final_adaln_bias,
                    )
                )
            elif flat_capture:
                wide_modulation_steps = _dit_wide_adaln_steps(
                    time_embeddings,
                    self._npu_dit_wide_adaln_weight,
                    self._npu_dit_wide_adaln_bias,
                )
            elif self._npu_dit_wide_final_adaln_enabled:
                try:
                    wide_modulation_steps, final_modulation_steps = (
                        self._get_npu_dit_wide_final_adaln_steps_graph()(
                            time_embeddings,
                            self._npu_dit_wide_final_adaln_weight,
                            self._npu_dit_wide_final_adaln_bias,
                        )
                    )
                except Exception:
                    self._npu_dit_wide_final_adaln_enabled = False
                    self._npu_dit_wide_final_adaln_steps_graph = None
                    logger.warning(
                        "MiniCPM-o all-step final AdaLN replay failed; retaining "
                        "block-only wide AdaLN",
                        exc_info=True,
                    )
            if wide_modulation_steps is None:
                try:
                    wide_modulation_steps = self._get_npu_dit_wide_adaln_steps_graph()(
                        time_embeddings,
                        self._npu_dit_wide_adaln_weight,
                        self._npu_dit_wide_adaln_bias,
                    )
                except Exception:
                    self._npu_dit_wide_adaln_enabled = False
                    self._npu_dit_wide_adaln_steps_graph = None
                    wide_modulation_steps = None
                    logger.warning(
                        "MiniCPM-o all-step wide AdaLN replay failed; using per-block projections",
                        exc_info=True,
                    )
            if wide_modulation_steps is not None and not self._npu_dit_wide_adaln_steps_used:
                logger.info(
                    "MiniCPM-o all-step wide AdaLN replay active for %d CFM steps x "
                    "16 blocks%s",
                    steps,
                    " plus final layer" if final_modulation_steps is not None else "",
                )
                self._npu_dit_wide_adaln_steps_used = True
        # The default keeps ODE recurrence in FP32. The experimental homogeneous
        # BF16 mode instead removes per-step estimator-boundary casts and casts
        # the completed mel once before the FP32 HiFT boundary.
        deltas = self._cfm_deltas_for(integration_timeline)
        if (cnn_output is None) != (att_output is None):
            raise ValueError("cnn_output and att_output must be provided together")
        direct_cache_output = (
            not reduced_steps
            and (
                cnn_output is not None
                or (
                    self._npu_cfm_stacked_cache_out_enabled
                    and mu.device.type == "npu"
                )
            )
        )
        stacked_cnn_out = cnn_output
        stacked_att_out = att_output
        if direct_cache_output:
            first_old_cnn = working_cnn_cache[0] if working_cnn_cache is not None else None
            first_old_att = att_cache[0] if att_cache is not None else None
            first_attention = estimator.blocks[0].attn
            bsh_attention = (
                self._npu_dit_bsh_attention_enabled
                and (
                    first_old_att is None
                    or self._is_bsh_planar_att_cache(
                        first_old_att,
                        int(first_attention.num_heads)
                        * int(first_attention.head_dim),
                    )
                )
            )
            cnn_shape, att_shape = self._estimator_buffer_shapes(
                estimator,
                mu_cfg,
                first_old_att,
                cache_major=self._is_cache_major_cnn(first_old_cnn),
                bsh_attention=bsh_attention,
            )
            expected_cnn = (steps, *cnn_shape)
            expected_att = (steps, *att_shape)
            if stacked_cnn_out is None:
                stacked_cnn_out = mu_cfg.new_empty(expected_cnn)
                stacked_att_out = mu_cfg.new_empty(expected_att)
            elif (
                tuple(stacked_cnn_out.shape) != expected_cnn
                or tuple(stacked_att_out.shape) != expected_att
            ):
                raise ValueError(
                    "fixed CFM output slab shape mismatch: "
                    f"expected cnn={expected_cnn}, att={expected_att}; "
                    f"got cnn={tuple(stacked_cnn_out.shape)}, "
                    f"att={tuple(stacked_att_out.shape)}"
                )
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        for step in range(steps):
            cache_step = min(
                self.n_timesteps - 1,
                step * self.n_timesteps // steps,
            )
            old_cnn = (
                working_cnn_cache[cache_step]
                if working_cnn_cache is not None
                else None
            )
            old_att = att_cache[cache_step] if att_cache is not None else None
            estimate, step_cnn, step_att, cfm_updated = self._estimator_step(
                estimator,
                x=self._cfg_pair(
                    "x",
                    x.to(dtype=estimator_dtype),
                    zero_unconditional=False,
                ),
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
                final_modulation=(
                    final_modulation_steps[step]
                    if final_modulation_steps is not None
                    else None
                ),
                cfm_x=x,
                cfm_delta=deltas[step],
                cfm_cfg_rate=float(decoder.inference_cfg_rate),
                flat_capture=flat_capture,
            )
            if cfm_updated:
                x = estimate.to(dtype=integration_dtype)
            else:
                conditional, unconditional = estimate.to(
                    dtype=integration_dtype
                ).split(batch_size, dim=0)
                velocity = (
                    (1.0 + decoder.inference_cfg_rate) * conditional
                    - decoder.inference_cfg_rate * unconditional
                )
                x = x + deltas[step] * velocity
            if stacked_cnn_out is None:
                next_cnn.append(step_cnn)
                next_att.append(step_att)
        if stacked_cnn_out is not None:
            stacked_cnn = stacked_cnn_out
            stacked_att = stacked_att_out
        elif reduced_steps:
            # The fixed serving ABI owns one cache slot per full CFM timestep.
            # Map each reduced solver cache to its nearest following full slot
            # so subsequent CFM6 chunks retain stable six-slot tensor shapes.
            full_to_reduced = tuple(
                min(steps - 1, step * steps // self.n_timesteps)
                for step in range(self.n_timesteps)
            )
            stacked_cnn = torch.stack(
                [next_cnn[index] for index in full_to_reduced]
            )
            stacked_att = torch.stack(
                [next_att[index] for index in full_to_reduced]
            )
        else:
            stacked_cnn = torch.stack(next_cnn)
            stacked_att = torch.stack(next_att)
        if direct_cache_output and not self._npu_cfm_stacked_cache_out_used:
            logger.info("MiniCPM-o NPU direct stacked CFM cache output active")
            self._npu_cfm_stacked_cache_out_used = True
        # A compile or replay failure disables the graph inside
        # ``_estimator_step``. Return to the canonical layout in that case so
        # subsequent chunks do not pay two compatibility transposes.
        retain_cache_major = use_cache_major and not self._npu_dit_conv_mlp_graph_disabled
        if retain_cache_major != self._is_cache_major_cnn(stacked_cnn):
            stacked_cnn = stacked_cnn.transpose(-2, -1).contiguous()
        return x.to(dtype=mu.dtype), stacked_cnn, stacked_att

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

    @staticmethod
    def _npu_cfm_graph_slot_count() -> int:
        """Return the number of fixed-address output sets per steady graph.

        The width-50 path retains the proven two-slot default. Wider chunks
        can select one slot to avoid capturing a second, very large attention
        workspace when Stage 2 executes requests synchronously.
        """
        raw = os.environ.get("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_SLOTS", "2")
        try:
            return max(1, min(2, int(raw)))
        except ValueError:
            logger.warning(
                "Invalid VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_SLOTS=%r; using 2",
                raw,
            )
            return 2

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        cnn_output: torch.Tensor | None = None,
        att_output: torch.Tensor | None = None,
        steady_graph: bool = False,
        num_timesteps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            num_timesteps is not None
            and num_timesteps != self.n_timesteps
        ):
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                # Reduced first-packet execution returns a six-slot cache by
                # remapping its solver caches; fixed graph destinations remain
                # reserved for the full six-step steady path.
                cnn_output=None,
                att_output=None,
                num_timesteps=num_timesteps,
            )
        if (
            not _npu_cfm_graph_enabled()
            or mu.device.type != "npu"
            or self._npu_cfm_graph_disabled
        ):
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                cnn_output=cnn_output,
                att_output=att_output,
            )

        graph_phase = _npu_cfm_graph_phase(
            fixed_kv_slabs=self._npu_cfm_fixed_kv_slabs_enabled,
            cache_fill_lengths=self._npu_cfm_cache_fill_graph_lengths,
            steady_graph=steady_graph,
            width=int(mu.shape[2]),
            cache_length=(
                None if att_cache is None else int(att_cache.shape[-2])
            ),
            has_cache_outputs=cnn_output is not None and att_output is not None,
        )
        # Prompt CFM and variable-width tails remain eager. The opt-in
        # cache-fill policy admits only explicitly configured fixed width-50
        # shapes before cache-402 steady state.
        if graph_phase is None:
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                cnn_output=cnn_output,
                att_output=att_output,
            )

        inputs = (mu, speakers, cond, cnn_cache, att_cache, cnn_output, att_output)
        flat_capture = (
            self._npu_cfm_fixed_kv_slabs_enabled
            and self._npu_dit_bsh_attention_enabled
            and graph_phase in {"cache-fill", "steady"}
        )
        key = tuple(self._optional_tensor_signature(value) for value in inputs)
        entry = self._npu_cfm_graphs.get(key)
        # Cache-fill shapes execute once per request, so one persistent output
        # set is safe. Steady state uses the configured single or ping-pong
        # output policy; the proven width-50 default remains two slots.
        slot_count = (
            self._npu_cfm_graph_slot_count()
            if graph_phase == "steady"
            else 1
        )
        if entry is not None and len(entry["slots"]) >= slot_count:
            self._npu_cfm_graphs.move_to_end(key)
            slot_index = int(entry["next_slot"])
            slot = entry["slots"][slot_index]
            entry["next_slot"] = (slot_index + 1) % slot_count
            # The final two buffers are graph destinations, not inputs. Copying
            # their old contents was a full-cache HBM round trip every replay.
            for static, current in zip(slot["inputs"][:5], inputs[:5], strict=True):
                if static is not None and current is not None:
                    static.copy_(current)
            slot["graph"].replay()
            if graph_phase not in self._npu_cfm_graph_replay_phases:
                logger.info(
                    "MiniCPM-o NPU CFM graph replay active: "
                    "phase=%s, slots=%d, mu=%s, attention_cache=%s",
                    graph_phase,
                    slot_count,
                    tuple(mu.shape),
                    None if att_cache is None else tuple(att_cache.shape),
                )
                self._npu_cfm_graph_replay_phases.add(graph_phase)
                self._npu_cfm_graph_replay_used = True
            if self._npu_cfm_fixed_kv_slabs_enabled:
                return slot["outputs"]
            return tuple(output.clone() for output in slot["outputs"])

        if entry is None:
            entry = {
                "slots": [],
                "next_slot": 0,
                "pool": torch.npu.graph_pool_handle(),
            }
            self._npu_cfm_graphs[key] = entry
        static_inputs = tuple(
            (
                value.clone()
                if index < 5
                else torch.empty_like(value)
            )
            if value is not None
            else None
            for index, value in enumerate(inputs)
        )
        (
            static_mu,
            static_speakers,
            static_cond,
            static_cnn,
            static_att,
            static_cnn_output,
            static_att_output,
        ) = static_inputs
        try:
            with torch.inference_mode():
                self._decode_cfm_eager(
                    static_mu,
                    static_speakers,
                    static_cond,
                    cnn_cache=static_cnn,
                    att_cache=static_att,
                    cnn_output=static_cnn_output,
                    att_output=static_att_output,
                    flat_capture=flat_capture,
                )
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.inference_mode(), torch.npu.graph(
                graph,
                pool=entry["pool"],
            ):
                outputs = self._decode_cfm_eager(
                    static_mu,
                    static_speakers,
                    static_cond,
                    cnn_cache=static_cnn,
                    att_cache=static_att,
                    cnn_output=static_cnn_output,
                    att_output=static_att_output,
                    flat_capture=flat_capture,
                )
            graph.replay()
        except Exception as exc:
            self._npu_cfm_graph_disabled = True
            self._npu_cfm_graphs.clear()
            if flat_capture:
                raise RuntimeError(
                    "MiniCPM-o steady CFM NPUGraph capture failed; restart "
                    "Stage 2 with VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH=0. "
                    "A failed CANN capture can leave the allocator state "
                    "unsafe for eager fallback in the current process."
                ) from exc
            logger.warning(
                "MiniCPM-o NPU CFM graph capture failed; using eager Code2Wav",
                exc_info=True,
            )
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                cnn_output=cnn_output,
                att_output=att_output,
            )

        entry["slots"].append({
            "graph": graph,
            "inputs": static_inputs,
            "outputs": outputs,
        })
        if graph_phase not in self._npu_cfm_graph_capture_phases:
            logger.info(
                "MiniCPM-o NPU CFM graph captured: "
                "phase=%s, slot=%d/%d, mu=%s, attention_cache=%s",
                graph_phase,
                len(entry["slots"]),
                slot_count,
                tuple(mu.shape),
                None if att_cache is None else tuple(att_cache.shape),
            )
            self._npu_cfm_graph_capture_phases.add(graph_phase)
            self._npu_cfm_graph_capture_used = True
        entry["next_slot"] = len(entry["slots"]) % slot_count
        max_graphs = self._npu_cfm_graph_cache_limit()
        while len(self._npu_cfm_graphs) > max_graphs:
            self._npu_cfm_graphs.popitem(last=False)
        if self._npu_cfm_fixed_kv_slabs_enabled:
            return outputs
        return tuple(output.clone() for output in outputs)

    def _split_flow_cache(
        self,
        cache: dict[str, torch.Tensor],
        batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        estimator_att = cache.get("estimator_att_cache")
        planar_att = self._is_planar_att_cache(estimator_att)
        bsh_attention = (
            self._npu_dit_bsh_attention_enabled
            and self._is_bsh_planar_att_cache(estimator_att)
        )
        if batch_size == 1 and (
            self._npu_single_request_cache_passthrough_enabled
            or planar_att
            or bsh_attention
        ):
            if not self._npu_single_request_cache_passthrough_used:
                logger.info("MiniCPM-o NPU single-request cache passthrough active")
                self._npu_single_request_cache_passthrough_used = True
            return [{name: value.detach() for name, value in cache.items()}]
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            if bsh_attention:
                estimator_att_cache = torch.cat(
                    (
                        cache["estimator_att_cache"][:, :, :, row : row + 1],
                        cache["estimator_att_cache"][
                            :, :, :, batch_size + row : batch_size + row + 1
                        ],
                    ),
                    dim=3,
                ).detach()
            else:
                estimator_att_cache = torch.cat(
                    (
                        cache["estimator_att_cache"][:, :, row : row + 1],
                        cache["estimator_att_cache"][
                            :, :, batch_size + row : batch_size + row + 1
                        ],
                    ),
                    dim=2,
                ).detach()
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
                    "estimator_att_cache": estimator_att_cache,
                }
            )
        return result

    def _stack_flow_cache(self, states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        single_slabs = states[0].estimator_kv_slabs if len(states) == 1 else None
        if len(states) == 1 and (
            self._npu_single_request_cache_passthrough_enabled
            or (single_slabs is not None and single_slabs.bsh_attention)
            or self._is_planar_att_cache(
                states[0].flow_cache.get("estimator_att_cache")
            )
        ):
            return states[0].flow_cache
        flows: list[dict[str, torch.Tensor]] = []
        for state in states:
            flow = state.flow_cache
            slabs = state.estimator_kv_slabs
            if (
                (slabs is not None and slabs.bsh_attention)
                or self._is_bsh_planar_att_cache(
                    flow.get("estimator_att_cache")
                )
            ):
                flow = dict(flow)
                attention = getattr(
                    getattr(self.flow.decoder.estimator.blocks[0], "attn", None),
                    "num_heads",
                    8,
                )
                head_dim = getattr(
                    getattr(self.flow.decoder.estimator.blocks[0], "attn", None),
                    "head_dim",
                    64,
                )
                flow["estimator_att_cache"] = self._legacy_att_cache_from_bsh_planar(
                    flow["estimator_att_cache"], int(attention), int(head_dim)
                )
            elif self._is_planar_att_cache(flow.get("estimator_att_cache")):
                flow = dict(flow)
                flow["estimator_att_cache"] = (
                    self._legacy_att_cache_from_planar(
                        flow["estimator_att_cache"]
                    )
                )
            flows.append(flow)
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

    @staticmethod
    def clone_prompt_state(state: BatchedToken2WavState) -> BatchedToken2WavState:
        """Materialize a request-owned copy of a cached prompt state.

        Prompt setup is deterministic for a given reference-audio fingerprint,
        but every live decode mutates its rolling attention/CNN slabs.  Keep one
        immutable template and clone only its tensors instead of rerunning the
        Conformer and prompt CFM for every request.
        """
        slabs = state.estimator_kv_slabs
        if slabs is None:
            flow_cache = {
                name: value.detach().clone()
                for name, value in state.flow_cache.items()
            }
            cloned_slabs = None
        else:
            retained = slabs.retained.detach().clone()
            append = torch.empty_like(slabs.append)
            cnn_banks = tuple(
                bank.detach().clone()
                if index == slabs.active_cnn_bank
                else torch.empty_like(bank)
                for index, bank in enumerate(slabs.cnn_banks)
            )
            cloned_slabs = FixedEstimatorKVSlabs(
                retained=retained,
                append=append,
                cnn_banks=cnn_banks,
                prompt_length=slabs.prompt_length,
                logical_length=slabs.logical_length,
                active_cnn_bank=slabs.active_cnn_bank,
                planar=slabs.planar,
                bsh_attention=slabs.bsh_attention,
                cnn_cache_major=slabs.cnn_cache_major,
            )
            flow_cache = {
                name: value.detach().clone()
                for name, value in state.flow_cache.items()
                if name not in {"estimator_att_cache", "estimator_cnn_cache"}
            }
            flow_cache["estimator_att_cache"] = retained[
                ..., : slabs.logical_length, :
            ]
            flow_cache["estimator_cnn_cache"] = cnn_banks[
                slabs.active_cnn_bank
            ]

        return BatchedToken2WavState(
            flow_cache=flow_cache,
            hift_cache={
                name: value.detach().clone()
                for name, value in state.hift_cache.items()
            },
            estimator_kv_slabs=cloned_slabs,
        )

    @staticmethod
    def _make_fixed_estimator_kv_slabs(
        cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        prompt_length: int,
        *,
        steady_width: int = 50,
        planar: bool = False,
        bsh_attention: bool = False,
        cnn_cache_major: bool = False,
    ) -> FixedEstimatorKVSlabs | None:
        if bsh_attention and not BatchedToken2Wav._is_bsh_planar_att_cache(cache):
            if BatchedToken2Wav._is_planar_att_cache(cache):
                key = cache.select(-5, 0)
                value = cache.select(-5, 1)
            else:
                key, value = cache.chunk(2, dim=-1)
            key = key.transpose(-3, -2).flatten(-2)
            value = value.transpose(-3, -2).flatten(-2)
            cache = torch.stack((key, value), dim=2).contiguous()
            planar = True
        elif (
            not bsh_attention
            and planar
            and not BatchedToken2Wav._is_planar_att_cache(cache)
        ):
            key, value = cache.chunk(2, dim=-1)
            cache = torch.stack((key, value), dim=2).contiguous()
        logical_length = int(cache.shape[-2])
        if logical_length != prompt_length:
            logger.warning(
                "MiniCPM-o fixed KV slabs disabled for request: prompt=%d, cache=%d",
                prompt_length,
                logical_length,
            )
            return None
        # Keep the exact prompt+100 rolling history policy, but size the
        # separate output workspace for one complete configured steady
        # append.  Width-50 retains the historical prompt+150 allocation;
        # width-100 needs prompt+200 so graph destinations never change
        # address or fall back to a dynamic output on every large chunk.
        append_width = max(50, int(steady_width))
        retained_shape = (*cache.shape[:-2], prompt_length + 100, cache.shape[-1])
        append_shape = (
            *cache.shape[:-2],
            prompt_length + 100 + append_width,
            cache.shape[-1],
        )
        retained = cache.new_empty(retained_shape)
        append = cache.new_empty(append_shape)
        retained[..., :logical_length, :].copy_(cache)
        if cnn_cache_major and not BatchedToken2Wav._is_cache_major_cnn(cnn_cache):
            cnn_cache = cnn_cache.transpose(-2, -1).contiguous()
        cnn_banks = (cnn_cache.clone(), torch.empty_like(cnn_cache))
        return FixedEstimatorKVSlabs(
            retained=retained,
            append=append,
            cnn_banks=cnn_banks,
            prompt_length=prompt_length,
            logical_length=logical_length,
            active_cnn_bank=0,
            planar=planar,
            bsh_attention=bsh_attention,
            cnn_cache_major=cnn_cache_major,
        )

    @staticmethod
    def _advance_fixed_estimator_kv_slabs(
        slabs: FixedEstimatorKVSlabs,
        att_output: torch.Tensor,
        cnn_output: torch.Tensor,
    ) -> FixedEstimatorKVSlabs:
        output_length = int(att_output.shape[-2])
        retained_length = min(output_length, slabs.prompt_length + 100)
        if output_length <= retained_length:
            slabs.retained[..., :output_length, :].copy_(att_output)
        else:
            prompt = slabs.prompt_length
            slabs.retained[..., :prompt, :].copy_(att_output[..., :prompt, :])
            slabs.retained[..., prompt : prompt + 100, :].copy_(
                att_output[..., -100:, :]
            )
        next_cnn_bank = 1 - slabs.active_cnn_bank
        cnn_bank = slabs.cnn_banks[next_cnn_bank]
        if slabs.cnn_cache_major != BatchedToken2Wav._is_cache_major_cnn(
            cnn_output
        ):
            cnn_output = cnn_output.transpose(-2, -1).contiguous()
        if cnn_bank.data_ptr() != cnn_output.data_ptr():
            cnn_bank.copy_(cnn_output)
        return FixedEstimatorKVSlabs(
            retained=slabs.retained,
            append=slabs.append,
            cnn_banks=slabs.cnn_banks,
            prompt_length=slabs.prompt_length,
            logical_length=retained_length,
            active_cnn_bank=next_cnn_bank,
            planar=slabs.planar,
            bsh_attention=slabs.bsh_attention,
            cnn_cache_major=slabs.cnn_cache_major,
        )

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
        timing = self._npu_stage2_timing_enabled and prompt_tokens.device.type == "npu"
        if timing:
            torch.npu.synchronize()
            timing_start = time.perf_counter()
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            if timing:
                torch.npu.synchronize()
                timing_encode = time.perf_counter()
            estimator_hidden = hidden
            estimator_prompt_mels = prompt_mels
            prompt_cache_limit = self._npu_prompt_cache_max_frames
            if (
                prompt_cache_limit is not None
                and int(hidden.shape[1]) > prompt_cache_limit
            ):
                # Preserve full prompt tokenization, Conformer execution and
                # speaker embedding. Only the suffix admitted to the costly
                # DiT estimator K/V cache is bounded. The retained hidden rows
                # have already attended to the complete causal prompt, while
                # the aligned mel suffix keeps estimator conditioning exact
                # within the retained region.
                estimator_hidden = hidden[:, -prompt_cache_limit:, :]
                estimator_prompt_mels = prompt_mels[:, -prompt_cache_limit:, :]
                if not self._npu_prompt_cache_limit_used:
                    logger.info(
                        "MiniCPM-o bounded DiT prompt cache active: full=%d, "
                        "retained_suffix=%d; full Conformer and speaker paths preserved",
                        int(hidden.shape[1]),
                        prompt_cache_limit,
                    )
                    self._npu_prompt_cache_limit_used = True
            _, estimator_cnn, estimator_att = self._decode_cfm(
                estimator_hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                estimator_prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
                num_timesteps=self.prompt_cfm_timesteps,
            )
            if timing:
                torch.npu.synchronize()
                timing_cfm = time.perf_counter()
            if (
                self.prompt_cfm_timesteps != self.n_timesteps
                and not self._npu_prompt_cfm_used
            ):
                logger.info(
                    "MiniCPM-o prompt-cache CFM solver active: steps=%d/%d, "
                    "prompt_width=%d; live steady chunks retain the full solver",
                    self.prompt_cfm_timesteps,
                    self.n_timesteps,
                    int(estimator_hidden.shape[1]),
                )
                self._npu_prompt_cfm_used = True
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        states: list[BatchedToken2WavState] = []
        prompt_length = int(estimator_hidden.shape[1])
        for row in split:
            slabs = (
                self._make_fixed_estimator_kv_slabs(
                    row["estimator_att_cache"],
                    row["estimator_cnn_cache"],
                    prompt_length,
                    steady_width=self._npu_dit_mlp_graph_width,
                    planar=self._npu_cfm_planar_kv_slabs_enabled,
                    bsh_attention=self._npu_dit_bsh_attention_enabled,
                    cnn_cache_major=(
                        self._npu_dit_cache_major_enabled
                        and row["estimator_cnn_cache"].device.type == "npu"
                    ),
                )
                if self._npu_cfm_fixed_kv_slabs_enabled and batch_size == 1
                else None
            )
            if slabs is not None:
                row = dict(row)
                row["estimator_att_cache"] = slabs.retained[
                    ..., : slabs.logical_length, :
                ]
                row["estimator_cnn_cache"] = slabs.cnn_banks[
                    slabs.active_cnn_bank
                ]
            states.append(BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
                estimator_kv_slabs=slabs,
            ))
        if timing:
            torch.npu.synchronize()
            timing_state = time.perf_counter()
            logger.info(
                "MiniCPM-o Stage-2 timing: prompt_setup encode=%.3fms, "
                "cfm=%.3fms, state=%.3fms, total=%.3fms, width=%d",
                (timing_encode - timing_start) * 1000.0,
                (timing_cfm - timing_encode) * 1000.0,
                (timing_state - timing_cfm) * 1000.0,
                (timing_state - timing_start) * 1000.0,
                int(estimator_hidden.shape[1]),
            )
        return states

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
        first_audio_chunk = all(
            int(state.hift_cache["mel"].shape[-1]) == 0 for state in states
        )
        timing = (
            self._npu_stage2_timing_enabled
            and first_audio_chunk
            and tokens.device.type == "npu"
        )
        if timing:
            torch.npu.synchronize()
            timing_start = time.perf_counter()
        cfm_timesteps = (
            self.initial_cfm_timesteps
            if first_audio_chunk and not last_chunk
            else self.n_timesteps
        )
        flow_cache = self._stack_flow_cache(states)
        fixed_slabs = (
            states[0].estimator_kv_slabs
            if batch_size == 1 and self._npu_cfm_fixed_kv_slabs_enabled
            else None
        )
        projected_speakers = features.projected_speaker_embedding.expand(batch_size, -1)
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            if timing:
                torch.npu.synchronize()
                timing_encode = time.perf_counter()
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            cnn_output = None
            att_output = None
            steady_graph = False
            if fixed_slabs is not None:
                output_length = fixed_slabs.logical_length + int(hidden.shape[1])
                append_capacity = int(fixed_slabs.append.shape[-2])
                fixed_output_layout_compatible = (
                    not fixed_slabs.cnn_cache_major
                    or int(hidden.shape[1]) == self._npu_dit_mlp_graph_width
                )
                if output_length <= append_capacity and fixed_output_layout_compatible:
                    att_output = fixed_slabs.append[..., :output_length, :]
                    cnn_output = fixed_slabs.cnn_banks[
                        1 - fixed_slabs.active_cnn_bank
                    ]
                    steady_graph = (
                        int(hidden.shape[1])
                        == self._npu_dit_mlp_graph_width
                        and fixed_slabs.logical_length
                        == fixed_slabs.prompt_length + 100
                    )
                elif (
                    output_length > append_capacity
                    and not self._npu_cfm_fixed_kv_tail_fallback_used
                ):
                    logger.info(
                        "MiniCPM-o fixed estimator KV tail uses eager dynamic "
                        "output: required=%d, append=%d",
                        output_length,
                        append_capacity,
                    )
                    self._npu_cfm_fixed_kv_tail_fallback_used = True
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
                cnn_output=cnn_output,
                att_output=att_output,
                steady_graph=steady_graph,
                num_timesteps=cfm_timesteps,
            )
            if timing:
                torch.npu.synchronize()
                timing_cfm = time.perf_counter()
            if (
                cfm_timesteps != self.n_timesteps
                and not self._npu_initial_cfm_used
            ):
                logger.info(
                    "MiniCPM-o first-packet CFM solver active: steps=%d/%d, "
                    "codec_frames=%d; subsequent chunks retain the full solver",
                    cfm_timesteps,
                    self.n_timesteps,
                    int(tokens.shape[1]),
                )
                self._npu_initial_cfm_used = True

        conformer_prompt_len = int(features.mels.shape[1])
        estimator_prompt_len = (
            fixed_slabs.prompt_length
            if fixed_slabs is not None
            else min(
                conformer_prompt_len,
                self._npu_prompt_cache_max_frames or conformer_prompt_len,
            )
        )
        if fixed_slabs is not None:
            fixed_slabs = self._advance_fixed_estimator_kv_slabs(
                fixed_slabs, estimator_att, estimator_cnn
            )
            estimator_att = fixed_slabs.retained[
                ..., : fixed_slabs.logical_length, :
            ]
            estimator_cnn = fixed_slabs.cnn_banks[
                fixed_slabs.active_cnn_bank
            ]
            if not self._npu_cfm_fixed_kv_slabs_used:
                logger.info(
                    "MiniCPM-o fixed estimator KV slabs active: retained=%d, "
                    "append=%d, planar=%s, bsh_attention=%s, cnn_cache_major=%s",
                    int(fixed_slabs.retained.shape[-2]),
                    int(fixed_slabs.append.shape[-2]),
                    fixed_slabs.planar,
                    fixed_slabs.bsh_attention,
                    fixed_slabs.cnn_cache_major,
                )
                self._npu_cfm_fixed_kv_slabs_used = True
        elif estimator_att.shape[-2] > estimator_prompt_len + 100:
            estimator_att = torch.cat(
                (
                    estimator_att[..., :estimator_prompt_len, :],
                    estimator_att[..., -100:, :],
                ),
                dim=-2,
            )
        if conformer_att.shape[3] > conformer_prompt_len + 100:
            conformer_att = torch.cat(
                (
                    conformer_att[..., :conformer_prompt_len, :],
                    conformer_att[..., -100:, :],
                ),
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
        if timing:
            torch.npu.synchronize()
            timing_hift = time.perf_counter()
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
                estimator_kv_slabs=fixed_slabs if row == 0 else None,
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        if timing:
            torch.npu.synchronize()
            timing_state = time.perf_counter()
            logger.info(
                "MiniCPM-o Stage-2 timing: first_chunk encode=%.3fms, "
                "cfm=%.3fms, hift=%.3fms, state=%.3fms, total=%.3fms, width=%d",
                (timing_encode - timing_start) * 1000.0,
                (timing_cfm - timing_encode) * 1000.0,
                (timing_hift - timing_cfm) * 1000.0,
                (timing_state - timing_hift) * 1000.0,
                (timing_state - timing_start) * 1000.0,
                int(tokens.shape[1]),
            )
        return audios, next_states
