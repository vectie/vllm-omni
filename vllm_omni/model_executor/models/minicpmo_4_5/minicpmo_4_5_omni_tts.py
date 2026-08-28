# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Continuously generate request-aligned discrete audio-code deltas
"""

import json
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.experimental.fullduplex.engine.intermediate import get_tts_handoff
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_REPETITION_WINDOW = 16
_MIN_AUDIO_TOKENS = 64
_MAX_AUDIO_TOKENS = 2048
_AUDIO_TOKENS_PER_TEXT_TOKEN = 10
# Codec-token sampling happens inside the model; vLLM sampling parameters
# only choose the Talker's binary continue/stop row.
_CODEC_SEED = 42
_CODEC_TEMPERATURE = 0.8
_CODEC_TOP_K = 25
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_CODEC_MIN_TOKENS = 50
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26
_NPU_BOUNDED_CODEC_SAMPLER_ENV = "VLLM_OMNI_MINICPMO45_NPU_BOUNDED_CODEC_SAMPLER"
_NPU_CODEC_SAMPLER_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"
_NPU_FUSED_CODEC_SAMPLER_ENV = "VLLM_OMNI_MINICPMO45_NPU_FUSED_CODEC_SAMPLER"
_NPU_BATCHED_CODEC_OUTPUT_ENV = "VLLM_OMNI_MINICPMO45_NPU_BATCHED_CODEC_OUTPUT"
_NPU_DEFERRED_CHUNK_EOS_ENV = "VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS"
_DIRECT_STOP_SAMPLER_ENV = "VLLM_OMNI_MINICPMO45_DIRECT_STOP_SAMPLER"
_CODEC_CHUNK_FRAMES_ENV = "VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES"
_INITIAL_CODEC_CHUNK_FRAMES_ENV = "VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"
_NPU_TALKER_STATIC_W8A8_CALIBRATION_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_TALKER_STATIC_W8A8_CALIBRATION"
)
_NPU_TALKER_STATIC_W8A8_ENV = "VLLM_OMNI_MINICPMO45_NPU_TALKER_STATIC_W8A8"
_NPU_TALKER_STATIC_W8A8_TARGETS_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_TALKER_STATIC_W8A8_TARGETS"
)
_NPU_TALKER_STATIC_W8A8_HEADROOM_ENV = (
    "VLLM_OMNI_MINICPMO45_NPU_TALKER_STATIC_W8A8_HEADROOM"
)


def _talker_static_w8a8_suffixes(targets: str) -> tuple[str, ...]:
    target_map = {
        "qkv": "self_attn.qkv_proj",
        "gate_up": "mlp.gate_up_proj",
    }
    requested = tuple(part.strip() for part in targets.split(",") if part.strip())
    unknown = sorted(set(requested) - target_map.keys())
    if not requested or unknown:
        raise ValueError(
            f"Invalid {_NPU_TALKER_STATIC_W8A8_TARGETS_ENV}={targets!r}; "
            "expected qkv, gate_up, or both"
        )
    return tuple(target_map[target] for target in requested)


def _quantize_talker_static_w8a8_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an Ascend-ready ``[in,out]`` INT8 matrix and output scales."""
    if weight.ndim != 2:
        raise ValueError(
            f"Talker static W8A8 requires a matrix, got shape={tuple(weight.shape)}"
        )
    source = weight.detach().to(dtype=torch.float32)
    scale = source.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
    quantized = torch.round(source / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return quantized.transpose(0, 1).contiguous(), scale.contiguous()


class _TalkerStaticW8A8LinearMethod(QuantizeMethodBase):
    """Fixed-scale W8A8 method for post-load Talker conversion.

    Unlike the rejected dynamic candidate, this path performs no per-token
    reduction.  ``vllm.quantize`` is intentionally graph-visible so
    vLLM-Ascend can fuse an upstream RMSNorm with the fixed quantizer.
    """

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        del layer, args, kwargs
        raise RuntimeError("Talker static W8A8 is installed after weight loading")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        del layer

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_shape = (*x.shape[:-1], layer.weight.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        quantized_x = torch.ops.vllm.quantize(
            x_2d,
            layer.aclnn_input_scale,
            layer.aclnn_input_scale_reciprocal,
            layer.aclnn_input_offset,
        )
        quant_bias = layer.quant_bias if bias is None else bias
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=x.dtype,
        )
        return output.reshape(output_shape)


def _prepare_talker_static_w8a8_calibration(
    model: nn.Module,
    targets: str,
) -> dict[str, torch.Tensor]:
    """Attach graph-captured max collectors to selected projection inputs."""
    collectors: dict[str, torch.Tensor] = {}
    selected_suffixes = _talker_static_w8a8_suffixes(targets)
    for name, layer in model.named_modules():
        if not name.endswith(selected_suffixes):
            continue
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
            raise ValueError(f"Talker W8A8 calibration target {name!r} has no matrix weight")
        absmax = torch.zeros((), device=weight.device, dtype=torch.float32)
        collectors[name] = absmax

        def collect_input_absmax(
            module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            destination: torch.Tensor = absmax,
        ) -> None:
            del module
            if inputs and isinstance(inputs[0], torch.Tensor):
                observed = inputs[0].detach().to(dtype=torch.float32).abs().amax()
                destination.copy_(torch.maximum(destination, observed))

        layer.register_forward_pre_hook(collect_input_absmax)
    if not collectors:
        raise ValueError("Talker static W8A8 calibration found no eligible projections")
    return collectors


def _load_talker_static_w8a8_scales(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    targets = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(targets, dict) or not targets:
        raise ValueError(f"Talker static W8A8 calibration {path!r} has no targets")
    scales: dict[str, float] = {}
    for name, value in targets.items():
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"Talker static W8A8 calibration for {name!r} is not positive")
        scales[str(name)] = value
    return scales


def _prepare_talker_static_w8a8(
    model: nn.Module,
    calibration_path: str,
    targets: str,
    headroom: float,
) -> tuple[int, int]:
    """Convert calibrated Talker projections to fixed-scale Ascend W8A8."""
    if headroom < 1.0:
        raise ValueError("Talker static W8A8 headroom must be at least 1.0")
    calibrated_absmax = _load_talker_static_w8a8_scales(calibration_path)
    selected_suffixes = _talker_static_w8a8_suffixes(targets)
    converted = 0
    parameter_bytes = 0
    for name, layer in model.named_modules():
        if not name.endswith(selected_suffixes):
            continue
        if name not in calibrated_absmax:
            raise ValueError(f"Talker static W8A8 calibration is missing {name!r}")
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
            raise ValueError(f"Talker static W8A8 target {name!r} has no matrix weight")
        quantized, weight_scale = _quantize_talker_static_w8a8_weight(weight)
        input_scale_value = calibrated_absmax[name] * headroom / 127.0
        input_scale = torch.full(
            (weight.shape[1],),
            input_scale_value,
            device=weight.device,
            dtype=weight.dtype,
        )
        weight.requires_grad_(False)
        weight.data = quantized
        layer.register_parameter(
            "aclnn_input_scale",
            nn.Parameter(input_scale, requires_grad=False),
        )
        layer.register_parameter(
            "aclnn_input_scale_reciprocal",
            nn.Parameter(input_scale.reciprocal(), requires_grad=False),
        )
        layer.register_parameter(
            "aclnn_input_offset",
            nn.Parameter(torch.zeros_like(input_scale), requires_grad=False),
        )
        layer.register_parameter(
            "deq_scale",
            nn.Parameter(weight_scale * input_scale_value, requires_grad=False),
        )
        layer.register_parameter(
            "quant_bias",
            nn.Parameter(
                torch.zeros(weight.shape[-1], device=weight.device, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.quant_method = _TalkerStaticW8A8LinearMethod()
        custom_op = getattr(layer, "custom_op", None)
        if custom_op is not None:
            custom_op.update_attrs()
        converted += 1
        parameter_bytes += sum(
            parameter.numel() * parameter.element_size()
            for parameter in (
                weight,
                layer.aclnn_input_scale,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                layer.deq_scale,
                layer.quant_bias,
            )
        )
    if converted == 0:
        raise ValueError("Talker static W8A8 found no eligible projections")
    return converted, parameter_bytes


def _max_audio_tokens(condition_tokens: int) -> int:
    """Bound codec generation with a conservative text-length estimate.

    EOS is masked for the first 50 steps, so a direct ``text_tokens * 10``
    limit can terminate short responses before EOS is eligible. The 2048
    ceiling matches the checkpoint's native generation default and keeps the
    sequence within the Talker's 4096-position context.
    """
    return max(
        _MIN_AUDIO_TOKENS,
        min(_MAX_AUDIO_TOKENS, condition_tokens * _AUDIO_TOKENS_PER_TEXT_TOKEN),
    )


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Match MiniCPMTTS' frequency-aware repetition penalty."""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
    vocab_ids = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.long)
    frequencies = torch.sum(recent[:, None] == vocab_ids, dim=0).to(dtype=logits.dtype)
    return _apply_repetition_penalty_from_frequencies(logits, frequencies, penalty=penalty)


def _apply_repetition_penalty_from_frequencies(
    logits: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    penalty: float,
) -> torch.Tensor:
    """Apply a precomputed frequency penalty without NPU host fallbacks."""
    if penalty == 1.0:
        return logits
    alpha = torch.pow(torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


def _bounded_top_k_top_p_candidates(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-vocabulary top-p while sorting only final top-k candidates.

    The checkpoint applies top-p before top-k.  Sorting all 6,562 codec logits
    for that ordering is especially expensive in single-token Ascend decode.
    Tokens outside the final top-k are discarded anyway: their aggregate
    probability mass is sufficient to compute the exact top-p cutoff for the
    retained candidates.  This reduces the sort and multinomial domains to 25
    values with the checkpoint defaults while preserving the candidate
    probabilities (apart from top-k boundary ties).
    """
    vocab_size = int(logits.shape[-1])
    keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
    candidate_logits, candidate_ids = torch.topk(logits, keep, dim=-1)
    if top_p is None or not 0.0 < top_p < 1.0:
        return candidate_logits, candidate_ids

    max_logits = logits.amax(dim=-1, keepdim=True)
    total_mass = torch.exp(logits - max_logits).sum(dim=-1, keepdim=True)
    candidate_mass = torch.exp(candidate_logits - max_logits)
    outside_mass = (total_mass - candidate_mass.sum(dim=-1, keepdim=True)).clamp_min_(0.0)

    # topk returns descending values; top-p's released warper accumulates from
    # the low-probability end and always retains at least the final three.
    candidate_logits = candidate_logits.flip(-1)
    candidate_ids = candidate_ids.flip(-1)
    candidate_mass = candidate_mass.flip(-1)
    cumulative = (outside_mass + candidate_mass.cumsum(dim=-1)) / total_mass
    remove = cumulative <= (1.0 - float(top_p))
    remove[..., -min_tokens_to_keep:] = False
    return candidate_logits.masked_fill(remove, float("-inf")), candidate_ids


def _bounded_codec_distribution(
    hidden_state: torch.Tensor,
    frequencies: torch.Tensor,
    weight: torch.Tensor,
    penalty: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float | None,
    eos_id: int,
    mask_eos: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact bounded codec distribution for one Talker token.

    This pure distribution builder is used by correctness tests and by the
    exact eager sampling path. The optimized Ascend path uses the same values
    with inverse-CDF sampling in ``_graphable_codec_sample``.
    """
    logits = F.linear(hidden_state, weight).float() / temperature
    alpha = torch.pow(penalty, frequencies)
    logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    if mask_eos:
        logits[..., eos_id] = float("-inf")
    candidate_logits, candidate_ids = _bounded_top_k_top_p_candidates(
        logits,
        top_k=top_k,
        top_p=top_p,
        min_tokens_to_keep=3,
    )
    return torch.softmax(candidate_logits, dim=-1), candidate_ids


def _graphable_codec_sample(
    hidden_state: torch.Tensor,
    frequencies: torch.Tensor,
    weight: torch.Tensor,
    penalty: torch.Tensor,
    uniform: torch.Tensor,
    mask_eos: torch.Tensor,
    expired: torch.Tensor,
    vocab_ids: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float | None,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one codec token without the Ascend AICPU multinomial kernel.

    Inverse-CDF sampling is distribution-equivalent to ``multinomial`` for a
    single draw.  The explicit uniform is a static graph input so request-local
    RNG remains outside capture, while the head, filters, draw and 16-token
    frequency-window update become one fixed-shape executable.
    """
    logits = F.linear(hidden_state, weight).float() / temperature
    alpha = torch.pow(penalty, frequencies)
    logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    eos_value = torch.where(
        mask_eos,
        logits.new_full(logits[..., eos_id].shape, float("-inf")),
        logits[..., eos_id],
    )
    logits[..., eos_id] = eos_value
    candidate_logits, candidate_ids = _bounded_top_k_top_p_candidates(
        logits,
        top_k=top_k,
        top_p=top_p,
        min_tokens_to_keep=3,
    )
    probabilities = torch.softmax(candidate_logits, dim=-1)
    sampled_position = torch.sum(
        probabilities.cumsum(dim=-1) < uniform,
        dim=-1,
        keepdim=True,
    ).clamp_max_(probabilities.shape[-1] - 1)
    sampled = candidate_ids.gather(-1, sampled_position)
    next_frequencies = frequencies + (vocab_ids == sampled).to(frequencies.dtype)
    next_frequencies = next_frequencies - (
        (expired >= 0) & (vocab_ids == expired)
    ).to(frequencies.dtype)
    return sampled, next_frequencies


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._batch_stop_logits: torch.Tensor | None = None
        self._batch_stop_token_ids: torch.Tensor | None = None
        self._stop_logits_constants: tuple[torch.Tensor, torch.Tensor] | None = None
        self._stop_token_constants: tuple[torch.Tensor, torch.Tensor] | None = None
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._request_repetition_frequencies: dict[str, torch.Tensor] = {}
        self._deferred_cleanup_ids: set[str] = set()
        self._static_w8a8_calibration_path: str | None = None
        self._static_w8a8_collectors: dict[str, torch.Tensor] = {}
        self._npu_codec_sampler_graphs: dict[bool, dict[str, Any]] = {}
        self._npu_codec_sampler_graph_pool: Any | None = None
        self._npu_codec_sampler_graph_disabled = False
        # The single-request competition path can fold codec-head sampling
        # into the already captured Talker executable. Python stages its
        # request-local RNG/window state into fixed-address buffers before
        # replay; make_omni_output consumes the graph-owned result afterward.
        self._fused_codec_sampler_enabled = _env_enabled(
            _NPU_FUSED_CODEC_SAMPLER_ENV,
            default=False,
        )
        self._fused_codec_sampler_prepared = False
        self._fused_codec_sampler_request_id: str | None = None
        # Code2Wav consumes codec chunks, not Talker's per-token hidden row.
        # Batch codec scalars on-device so the NPU runner performs one D2H per
        # publishable chunk instead of one D2H for every autoregressive step.
        self.batched_codec_output = _env_enabled(
            _NPU_BATCHED_CODEC_OUTPUT_ENV,
            default=False,
        )
        # In the sparse chunk transport path, an EOS decision is not visible
        # downstream until the next publish boundary anyway.  Defer its scalar
        # D2H read to that boundary so steady Talker decode does not serialize
        # the NPU and Python once per codec token.  The boundary copy is also
        # used to keep EOS and any speculative post-EOS codes out of the
        # sequence seen by Code2Wav.
        self.deferred_chunk_eos = self.batched_codec_output and _env_enabled(
            _NPU_DEFERRED_CHUNK_EOS_ENV,
            default=False,
        )
        # The Talker samples codec IDs internally. Its vLLM-visible two-token
        # head is only a deterministic continue/stop control channel. Reuse
        # that decision instead of running the generic logits-processor and
        # sampler stack a second time on every codec step.
        self.direct_stop_sampler = _env_enabled(
            _DIRECT_STOP_SAMPLER_ENV,
            default=False,
        )
        self.omni_pooler_payload_include_hidden = not self.batched_codec_output
        self._request_transport_codes: dict[str, list[torch.Tensor]] = {}
        self._request_transport_chunks: dict[str, int] = {}

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            self._codec_seed = int(getattr(tts_config, "seed", _CODEC_SEED))
            self._codec_temperature = float(getattr(tts_config, "temperature", _CODEC_TEMPERATURE))
            self._codec_top_k = int(getattr(tts_config, "top_k", _CODEC_TOP_K))
            self._codec_top_p = float(getattr(tts_config, "top_p", _CODEC_TOP_P))
            self._codec_repetition_penalty = float(getattr(tts_config, "repetition_penalty", _CODEC_REPETITION_PENALTY))
            self._codec_min_tokens = int(getattr(tts_config, "min_new_tokens", _CODEC_MIN_TOKENS))
        else:
            self._tts_config = None

        if tts_config is not None:
            self.register_buffer(
                "_codec_vocab_ids",
                torch.arange(self._num_audio_tokens, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_frequencies",
                torch.zeros((1, self._num_audio_tokens), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_uniform",
                torch.full((1, 1), 0.5, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_mask_eos",
                torch.ones((1,), dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_expired",
                torch.full((1, 1), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_penalty",
                torch.full((1,), self._codec_repetition_penalty, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_sampled",
                torch.zeros((1, 1), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_fused_codec_next_frequencies",
                torch.zeros((1, self._num_audio_tokens), dtype=torch.float32),
                persistent=False,
            )

        self.has_preprocess = True
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
        }
        self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # Cross-process stage transport serializes CPU tensors as lists.
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            request_id = str(info_dict.get("request_id", "0"))
            meta = info_dict.get("meta")
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            duplex_boundary = isinstance(meta, dict) and (
                bool(meta.get("turn_start", False)) or bool(meta.get("turn_end", False))
            )
            if native_duplex:
                max_tokens = _DUPLEX_CODEC_TOKENS_PER_CHUNK
                min_tokens = 0 if duplex_boundary else _DUPLEX_CODEC_TOKENS_PER_CHUNK
            else:
                max_tokens = _max_audio_tokens(int(token_ids.numel()))
                min_tokens = self._codec_min_tokens
            state = {
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "finished": empty_condition,
            }
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            self._request_repetition_frequencies.pop(request_id, None)
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            if state.get("finished"):
                # A request that finished before sampling any code can still be
                # scheduled for decode steps while sampling min_tokens masks the
                # stop token. make_omni_output ignores its hidden states, so any
                # shape-correct embedding will do.
                weight = self.emb_code[0].weight
                return input_ids, weight.new_zeros((span_len, weight.shape[1])), {}
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def _repetition_frequencies(
        self,
        request_id: str,
        history: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        frequencies = self._request_repetition_frequencies.get(request_id)
        if (
            frequencies is not None
            and frequencies.device == logits.device
            and frequencies.dtype == logits.dtype
            and frequencies.shape[-1] == logits.shape[-1]
        ):
            return frequencies
        recent = history.reshape(-1)[-_REPETITION_WINDOW:].to(device=logits.device, dtype=torch.long)
        if recent.numel() == 0:
            frequencies = logits.new_zeros(logits.shape)
        else:
            vocab_ids = self._codec_vocab_ids.to(device=logits.device)
            frequencies = torch.sum(recent[:, None] == vocab_ids, dim=0, keepdim=True).to(dtype=logits.dtype)
        self._request_repetition_frequencies[request_id] = frequencies
        return frequencies

    def _advance_repetition_frequencies(
        self,
        request_id: str,
        history: torch.Tensor,
        sampled: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> None:
        """Advance the 16-code frequency window using device-native compares."""
        vocab_ids = self._codec_vocab_ids.to(device=frequencies.device)
        next_frequencies = frequencies + (vocab_ids == sampled).to(dtype=frequencies.dtype)
        if history.numel() >= _REPETITION_WINDOW:
            expired = history.reshape(-1)[-_REPETITION_WINDOW].to(device=frequencies.device)
            next_frequencies = next_frequencies - (vocab_ids == expired).to(dtype=frequencies.dtype)
        self._request_repetition_frequencies[request_id] = next_frequencies

    def prepare_fused_codec_sampler_inputs(
        self,
        *,
        model_intermediate_buffer: list[Any] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        request_sample_eligible: list[bool] | None = None,
        **_: Any,
    ) -> bool:
        """Stage one batch-1 request into the Talker graph's sampler slabs.

        This deliberately supports only the stable single-request decode shape.
        Any prefill, compaction, or batched case falls back to the standalone
        sampler without changing request RNG or repetition state.
        """
        self._fused_codec_sampler_prepared = False
        self._fused_codec_sampler_request_id = None
        if not getattr(self, "_fused_codec_sampler_enabled", False):
            return False
        infos = model_intermediate_buffer or []
        spans = request_token_spans or []
        eligible = request_sample_eligible or []
        if len(infos) != 1 or len(spans) != 1 or eligible != [True]:
            return False
        info = infos[0]
        if not isinstance(info, dict):
            return False
        start, end = spans[0]
        # Full-decode capture owns exactly one Talker row. The final prefill
        # chunk can also be sample-eligible, but its wider hidden output does
        # not execute the fused branch in forward; staging it would advance
        # RNG and later consume a stale output slab.
        if int(end) - int(start) != 1:
            return False
        request_id = str(info.get("request_id", 0))
        state = self._request_audio_states.get(request_id)
        if not isinstance(state, dict) or state.get("finished"):
            return False
        codes = state.get("codes")
        if not isinstance(codes, torch.Tensor):
            codes = (info.get("audio_codes", {}) or {}).get("accumulated")
        if not isinstance(codes, torch.Tensor):
            codes = torch.empty(
                0,
                dtype=torch.long,
                device=self._fused_codec_frequencies.device,
            )
        else:
            codes = codes.to(
                device=self._fused_codec_frequencies.device,
                dtype=torch.long,
            ).reshape(-1)

        frequencies = self._repetition_frequencies(
            request_id,
            codes,
            self._fused_codec_frequencies,
        )
        if frequencies.data_ptr() != self._fused_codec_frequencies.data_ptr():
            self._fused_codec_frequencies.copy_(frequencies)
        # Keep request state bound to the stable graph input address.
        self._request_repetition_frequencies[request_id] = self._fused_codec_frequencies
        step = int(state.get("step", 0))
        min_tokens = int(state.get("min_tokens", self._codec_min_tokens))
        self._fused_codec_mask_eos.fill_(step < min_tokens)
        self._fused_codec_expired.fill_(-1)
        if codes.numel() >= _REPETITION_WINDOW:
            self._fused_codec_expired.copy_(
                codes[-_REPETITION_WINDOW].reshape(1, 1)
            )
        self._fused_codec_uniform.uniform_(
            0.0,
            1.0,
            generator=self._request_generator(
                request_id,
                self._fused_codec_uniform.device,
            ),
        )
        self._fused_codec_sampler_request_id = request_id
        self._fused_codec_sampler_prepared = True
        return True

    def _consume_fused_codec_sample(self, request_id: str) -> torch.Tensor | None:
        if (
            not getattr(self, "_fused_codec_sampler_prepared", False)
            or self._fused_codec_sampler_request_id != request_id
        ):
            return None
        self._fused_codec_sampler_prepared = False
        self._fused_codec_sampler_request_id = None
        self._fused_codec_frequencies.copy_(self._fused_codec_next_frequencies)
        self._request_repetition_frequencies[request_id] = self._fused_codec_frequencies
        return self._fused_codec_sampled.reshape(())

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
    ) -> torch.Tensor:
        eos_id = self._num_audio_tokens - 1
        request_states = getattr(self, "_request_audio_states", {})
        state = request_states.get(request_id)
        min_tokens = (
            int(state.get("min_tokens", self._codec_min_tokens)) if isinstance(state, dict) else self._codec_min_tokens
        )
        mask_eos = step < min_tokens
        if (
            hidden_state.device.type == "npu"
            and not getattr(self, "_npu_codec_sampler_graphs", {})
            and not getattr(self, "_npu_codec_sampler_graph_disabled", False)
            and _env_enabled(_NPU_CODEC_SAMPLER_GRAPH_ENV, default=False)
        ):
            # vLLM-Ascend captures the Talker backbone after load_weights().
            # Capture this continuation lazily on the first real execution
            # stream so the backbone output and the inverse-CDF continuation
            # share one ordered stream.
            self._prepare_npu_codec_sampler_graphs(hidden_state)
        graph_entry = getattr(self, "_npu_codec_sampler_graphs", {}).get(mask_eos)
        if graph_entry is not None and hidden_state.device.type == "npu":
            frequencies = self._repetition_frequencies(
                request_id,
                history,
                graph_entry["frequencies"],
            )
            graph_entry["hidden"].copy_(hidden_state)
            graph_entry["frequencies"].copy_(frequencies)
            graph_entry["mask_eos"].fill_(mask_eos)
            graph_entry["expired"].fill_(-1)
            if history.numel() >= _REPETITION_WINDOW:
                graph_entry["expired"].copy_(
                    history.reshape(-1)[-_REPETITION_WINDOW].reshape(1, 1)
                )
            graph_entry["uniform"].uniform_(
                0.0,
                1.0,
                generator=self._request_generator(request_id, hidden_state.device),
            )
            graph_entry["graph"].replay()
            graph_sampled, next_frequencies = graph_entry["outputs"]
            sampled = graph_entry["sampled"]
            sampled.copy_(graph_sampled)
            frequencies.copy_(next_frequencies)
            if not graph_entry["runtime_validated"]:
                valid_sample = bool(
                    ((sampled >= 0) & (sampled < self._num_audio_tokens)).all().item()
                ) and bool(
                    torch.isfinite(frequencies).all().item()
                )
                if not valid_sample:
                    raise RuntimeError(
                        "captured codec sampler produced invalid runtime state"
                    )
                graph_entry["runtime_validated"] = True
            return sampled.reshape(())

        logits = self.head_code[0](hidden_state).float() / self._codec_temperature
        frequencies = self._repetition_frequencies(request_id, history, logits)
        logits = _apply_repetition_penalty_from_frequencies(
            logits,
            frequencies,
            penalty=self._codec_repetition_penalty,
        )
        if mask_eos:
            logits[..., eos_id] = float("-inf")
        bounded_sampler = (
            logits.device.type == "npu"
            and self._codec_top_k > 0
            and self._codec_top_k < logits.shape[-1]
            and _env_enabled(_NPU_BOUNDED_CODEC_SAMPLER_ENV, default=True)
        )
        if bounded_sampler:
            candidate_logits, candidate_ids = _bounded_top_k_top_p_candidates(
                logits,
                top_k=self._codec_top_k,
                top_p=self._codec_top_p,
                min_tokens_to_keep=3,
            )
            probabilities = torch.softmax(candidate_logits, dim=-1)
            sampled_position = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self._request_generator(request_id, probabilities.device),
            )
            sampled = candidate_ids.gather(-1, sampled_position).reshape(())
        else:
            logits = _apply_top_k_top_p(
                logits,
                top_k=self._codec_top_k,
                top_p=self._codec_top_p,
                min_tokens_to_keep=3,
            )
            probabilities = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self._request_generator(request_id, probabilities.device),
            ).reshape(())
        self._advance_repetition_frequencies(request_id, history, sampled, frequencies)
        return sampled

    def _prepare_npu_codec_sampler_graphs(
        self,
        runtime_hidden: torch.Tensor | None = None,
    ) -> None:
        """Capture the distribution-equivalent fixed-shape Talker sampler."""
        if not _env_enabled(_NPU_CODEC_SAMPLER_GRAPH_ENV, default=False):
            return
        weight = self.head_code[0].weight
        if weight.device.type != "npu" or self._npu_codec_sampler_graph_disabled:
            return
        if self._codec_top_k <= 0 or self._codec_top_k >= self._num_audio_tokens:
            logger.warning(
                "MiniCPM-o NPU codec sampler graph requires bounded top-k; retaining eager sampling"
            )
            return

        graphs: dict[bool, dict[str, Any]] = {}
        pool = torch.npu.graph_pool_handle()
        penalty = torch.full(
            (1,),
            self._codec_repetition_penalty,
            device=weight.device,
            dtype=torch.float32,
        )
        if runtime_hidden is not None:
            if runtime_hidden.shape != (1, weight.shape[1]):
                logger.warning(
                    "MiniCPM-o NPU codec sampler graph requires hidden shape (1, %d), got %s",
                    int(weight.shape[1]),
                    tuple(runtime_hidden.shape),
                )
                return
            hidden_template = runtime_hidden.detach().to(dtype=weight.dtype).clone()
        else:
            generator = torch.Generator(device="cpu").manual_seed(20260826)
            hidden_cpu = torch.randn(
                (1, weight.shape[1]),
                generator=generator,
                dtype=torch.float32,
            )
            hidden_template = hidden_cpu.to(device=weight.device, dtype=weight.dtype)
        frequency_template = torch.zeros(
            (1, self._num_audio_tokens),
            device=weight.device,
            dtype=torch.float32,
        )
        try:
            static_hidden = hidden_template.clone()
            static_frequencies = frequency_template.clone()
            static_uniform = torch.full(
                (1, 1),
                0.5,
                device=weight.device,
                dtype=torch.float32,
            )
            static_mask_eos = torch.ones(
                (1,),
                device=weight.device,
                dtype=torch.bool,
            )
            static_expired = torch.full(
                (1, 1),
                -1,
                device=weight.device,
                dtype=torch.long,
            )
            vocab_ids = self._codec_vocab_ids.to(device=weight.device).reshape(1, -1)
            with torch.inference_mode():
                eager_outputs = tuple(
                    value.clone()
                    for value in _graphable_codec_sample(
                        static_hidden,
                        static_frequencies,
                        weight,
                        penalty,
                        static_uniform,
                        static_mask_eos,
                        static_expired,
                        vocab_ids,
                        temperature=self._codec_temperature,
                        top_k=self._codec_top_k,
                        top_p=self._codec_top_p,
                        eos_id=self._num_audio_tokens - 1,
                    )
                )
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.inference_mode(), torch.npu.graph(graph, pool=pool):
                outputs = _graphable_codec_sample(
                    static_hidden,
                    static_frequencies,
                    weight,
                    penalty,
                    static_uniform,
                    static_mask_eos,
                    static_expired,
                    vocab_ids,
                    temperature=self._codec_temperature,
                    top_k=self._codec_top_k,
                    top_p=self._codec_top_p,
                    eos_id=self._num_audio_tokens - 1,
                )
            graph.replay()
            torch.npu.synchronize()
            if not all(
                torch.equal(actual, expected)
                for actual, expected in zip(outputs, eager_outputs)
            ):
                raise RuntimeError("captured codec sample did not match eager execution")
            entry = {
                "graph": graph,
                "hidden": static_hidden,
                "frequencies": static_frequencies,
                "uniform": static_uniform,
                "mask_eos": static_mask_eos,
                "expired": static_expired,
                "outputs": outputs,
                "sampled": torch.empty_like(outputs[0]),
                "runtime_validated": False,
            }
            graphs = {True: entry, False: entry}
        except Exception:
            self._npu_codec_sampler_graph_disabled = True
            self._npu_codec_sampler_graphs = {}
            logger.warning(
                "MiniCPM-o NPU codec sampler graph capture failed; retaining exact eager sampling",
                exc_info=True,
            )
            return

        self._npu_codec_sampler_graph_pool = pool
        self._npu_codec_sampler_graphs = graphs
        logger.info(
            "MiniCPM-o inverse-CDF Talker codec sampler NPUGraph active: hidden=%d, vocab=%d, top_k=%d",
            int(weight.shape[1]),
            self._num_audio_tokens,
            self._codec_top_k,
        )

    def _sampled_code_is_eos(
        self,
        sampled: torch.Tensor,
        *,
        step: int,
        min_tokens: int,
        reached_limit: bool,
    ) -> bool:
        """Synchronize the sampled code only when EOS can affect control flow.

        ``_sample_audio_code`` masks EOS while ``step < min_tokens``.  Reading
        the scalar back to Python in that interval therefore cannot change the
        result, but on NPU it creates a full device/host synchronization after
        every Talker token.  The max-token boundary is terminal regardless of
        the sampled value and can skip the readback as well.
        """
        if reached_limit or step < min_tokens:
            return False
        return int(sampled.item()) == self._num_audio_tokens - 1

    def _transport_codec_delta(
        self,
        request_id: str,
        delta: torch.Tensor,
        *,
        finished: bool,
        native_duplex: bool,
    ) -> torch.Tensor:
        """Coalesce one-code NPU outputs into the chunks Code2Wav consumes."""
        if not getattr(self, "batched_codec_output", False) or native_duplex:
            return delta

        pending_by_request = getattr(self, "_request_transport_codes", None)
        if pending_by_request is None:
            pending_by_request = {}
            self._request_transport_codes = pending_by_request
        chunks_by_request = getattr(self, "_request_transport_chunks", None)
        if chunks_by_request is None:
            chunks_by_request = {}
            self._request_transport_chunks = chunks_by_request

        pending = pending_by_request.setdefault(request_id, [])
        if delta.numel():
            # The codec sampler graph reuses its output address. Own each code
            # until the current output slab is published.
            pending.append(delta.reshape(-1).clone())

        chunk_index = chunks_by_request.get(request_id, 0)
        default_chunk = max(1, int(os.environ.get(_CODEC_CHUNK_FRAMES_ENV, "25")))
        initial_chunk = max(
            1,
            int(os.environ.get(_INITIAL_CODEC_CHUNK_FRAMES_ENV, str(default_chunk))),
        )
        threshold = initial_chunk if chunk_index == 0 else default_chunk
        pending_count = sum(int(item.numel()) for item in pending)
        if not finished and pending_count < threshold:
            return delta.new_empty((0, 1))
        if not pending:
            return delta.new_empty((0, 1))

        output = torch.cat(pending).reshape(1, -1)
        pending.clear()
        chunks_by_request[request_id] = chunk_index + 1
        return output

    def _transport_codec_delta_with_deferred_eos(
        self,
        request_id: str,
        sampled: torch.Tensor,
        *,
        step: int,
        min_tokens: int,
        reached_limit: bool,
    ) -> tuple[torch.Tensor, bool]:
        """Publish one codec slab and reconcile EOS once per chunk.

        Samples are retained on-device until the normal Code2Wav boundary.
        At that boundary a single vector read replaces one scalar read after
        every eligible Talker token.  If EOS occurred inside the slab, only
        the prefix before it is published; later speculative samples are
        discarded together with the terminal request state.
        """
        pending = self._request_transport_codes.setdefault(request_id, [])
        if not reached_limit:
            # The fused sampler graph reuses this output address on replay.
            pending.append(sampled.reshape(-1).clone())

        chunk_index = self._request_transport_chunks.get(request_id, 0)
        default_chunk = max(1, int(os.environ.get(_CODEC_CHUNK_FRAMES_ENV, "25")))
        initial_chunk = max(
            1,
            int(os.environ.get(_INITIAL_CODEC_CHUNK_FRAMES_ENV, str(default_chunk))),
        )
        threshold = initial_chunk if chunk_index == 0 else default_chunk
        if not reached_limit and len(pending) < threshold:
            return sampled.new_empty((0, 1)), False
        if not pending:
            return sampled.new_empty((0, 1)), False

        output = torch.cat(pending).reshape(1, -1)
        is_eos = False
        # EOS is masked for all samples whose zero-based step is below
        # min_tokens.  Avoid even the chunk readback at those early boundaries.
        if reached_limit or step >= min_tokens:
            eos_id = self._num_audio_tokens - 1
            host_codes = output.reshape(-1).tolist()
            try:
                eos_offset = host_codes.index(eos_id)
            except ValueError:
                eos_offset = -1
            if eos_offset >= 0:
                output = output[:, :eos_offset]
                is_eos = True

        pending.clear()
        self._request_transport_chunks[request_id] = chunk_index + 1
        return output, is_eos

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)

        stop_rows: list[torch.Tensor] = []
        codec_deltas: list[torch.Tensor] = []
        terminal_flags: list[torch.Tensor] = []
        finished_rows: list[bool] = []
        output_request_ids: list[str] = []
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)

        def append_stop_control(stop: bool) -> None:
            finished_rows.append(stop)
            if not self.direct_stop_sampler:
                stop_rows.append(
                    hidden.new_tensor([float("-inf"), 0.0] if stop else [0.0, float("-inf")])
                )

        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            request_id = str(info_dict.get("request_id", index))
            output_request_ids.append(request_id)
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                append_stop_control(False)
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                append_stop_control(False)
                continue
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                append_stop_control(True)
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                append_stop_control(False)
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            sampled = self._consume_fused_codec_sample(request_id)
            if sampled is None:
                sampled = self._sample_audio_code(
                    hidden[end - 1 : end],
                    codes,
                    request_id,
                    step,
                )
            min_tokens = int(state.get("min_tokens", self._codec_min_tokens))
            state["step"] = step + 1
            reached_limit = int(state["step"]) >= int(state.get("max_tokens", 2048))
            defer_eos = (
                getattr(self, "deferred_chunk_eos", False)
                and getattr(self, "batched_codec_output", False)
                and not native_duplex
            )
            if defer_eos:
                is_eos = False
            else:
                is_eos = self._sampled_code_is_eos(
                    sampled,
                    step=step,
                    min_tokens=min_tokens,
                    reached_limit=reached_limit,
                )
            # MiniCPMTTS.generate_chunk consumes the max-token boundary sample
            # but returns only codes that were fed into retained KV state.
            # Deferred EOS may feed a few terminal-tail samples speculatively;
            # the transport boundary trims all of them from observable output.
            if (defer_eos or not is_eos) and not reached_limit:
                codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                delta = sampled.reshape(1, 1)
            else:
                delta = empty_delta
            if defer_eos:
                delta, is_eos = self._transport_codec_delta_with_deferred_eos(
                    request_id,
                    sampled,
                    step=step,
                    min_tokens=min_tokens,
                    reached_limit=reached_limit,
                )
                finished = is_eos or reached_limit
            else:
                finished = is_eos or reached_limit
                delta = self._transport_codec_delta(
                    request_id,
                    delta,
                    finished=finished,
                    native_duplex=native_duplex,
                )
            state["finished"] = finished
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas.append(delta)
            terminal_flags.append(torch.tensor(finished, dtype=torch.bool))
            append_stop_control(finished)

        self._batch_stop_token_ids = None
        if self.direct_stop_sampler and finished_rows:
            logits_constants = self._stop_logits_constants
            token_constants = self._stop_token_constants
            if (
                logits_constants is None
                or logits_constants[0].device != hidden.device
                or logits_constants[0].dtype != hidden.dtype
                or token_constants is None
                or token_constants[0].device != hidden.device
            ):
                logits_rows = hidden.new_tensor(
                    [[0.0, float("-inf")], [float("-inf"), 0.0]],
                )
                token_rows = hidden.new_tensor([[0], [1]], dtype=torch.int32)
                logits_constants = (logits_rows[0:1], logits_rows[1:2])
                token_constants = (token_rows[0:1], token_rows[1:2])
                self._stop_logits_constants = logits_constants
                self._stop_token_constants = token_constants
            if len(finished_rows) == 1:
                # The competition profile is max_num_seqs=1. Returning an
                # immutable pair of resident views makes compute_logits and
                # sample allocation- and kernel-free after their first use.
                stop_index = int(finished_rows[0])
                self._batch_stop_logits = logits_constants[stop_index]
                self._batch_stop_token_ids = token_constants[stop_index]
            else:
                self._batch_stop_logits = hidden.new_tensor(
                    [
                        [float("-inf"), 0.0] if stop else [0.0, float("-inf")]
                        for stop in finished_rows
                    ]
                )
                self._batch_stop_token_ids = hidden.new_tensor(
                    finished_rows,
                    dtype=torch.int32,
                ).reshape(-1, 1)
        else:
            self._batch_stop_logits = torch.stack(stop_rows, dim=0) if stop_rows else hidden.new_empty((0, 2))
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        meta_outputs = {"finished": terminal_flags}
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        if getattr(self, "batched_codec_output", False):
            emit_indices = [
                index
                for index, delta in enumerate(codec_deltas)
                if delta.numel() or finished_rows[index]
            ]
            multimodal_outputs["codes"]["audio"] = [
                codec_deltas[index] for index in emit_indices
            ]
            sparse_meta = {
                key: [values[index] for index in emit_indices]
                for key, values in meta_outputs.items()
            }
            sparse_meta["req_id"] = [output_request_ids[index] for index in emit_indices]
            sparse_meta["sparse_audio"] = ["1"]
            multimodal_outputs["meta"] = sparse_meta
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)
        self._export_static_w8a8_calibration()

    def _export_static_w8a8_calibration(self) -> None:
        path = self._static_w8a8_calibration_path
        collectors = self._static_w8a8_collectors
        if not path or not collectors:
            return
        payload = {
            "format": "minicpmo45-talker-static-w8a8-v1",
            "targets": {
                name: float(absmax.item())
                for name, absmax in sorted(collectors.items())
            },
        }
        temporary_path = f"{path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        transport_codes = getattr(self, "_request_transport_codes", {})
        transport_chunks = getattr(self, "_request_transport_chunks", {})
        for request_id in self._deferred_cleanup_ids:
            self._request_generators.pop(request_id, None)
            request_audio_states.pop(request_id, None)
            self._request_repetition_frequencies.pop(request_id, None)
            transport_codes.pop(request_id, None)
            transport_chunks.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        hidden_states = self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if (
            getattr(self, "_fused_codec_sampler_enabled", False)
            and hidden_states.shape[0] == 1
        ):
            sampled, next_frequencies = _graphable_codec_sample(
                hidden_states[-1:],
                self._fused_codec_frequencies,
                self.head_code[0].weight,
                self._fused_codec_penalty,
                self._fused_codec_uniform,
                self._fused_codec_mask_eos,
                self._fused_codec_expired,
                self._codec_vocab_ids,
                temperature=self._codec_temperature,
                top_k=self._codec_top_k,
                top_p=self._codec_top_p,
                eos_id=self._num_audio_tokens - 1,
            )
            # These fixed-address stores are observable outputs of the outer
            # graph. They let post-forward Python consume the draw without a
            # second ACL graph launch or per-token output clone.
            self._fused_codec_sampled.copy_(sampled)
            self._fused_codec_next_frequencies.copy_(next_frequencies)
        return hidden_states

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._batch_stop_logits is None:
            return torch.zeros(
                hidden_states.shape[0],
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def sample(self, logits, sampling_metadata):
        stop_token_ids = self._batch_stop_token_ids
        self._batch_stop_token_ids = None
        if (
            self.direct_stop_sampler
            and stop_token_ids is not None
            and stop_token_ids.shape[0] == logits.shape[0]
            and getattr(sampling_metadata, "max_num_logprobs", None) is None
            and not getattr(sampling_metadata, "logprob_token_ids", None)
        ):
            return SamplerOutput(
                sampled_token_ids=stop_token_ids,
                logprobs_tensors=None,
            )
        return Sampler()(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        calibration_path = os.environ.get(
            _NPU_TALKER_STATIC_W8A8_CALIBRATION_ENV,
            "",
        ).strip()
        static_w8a8_path = os.environ.get(
            _NPU_TALKER_STATIC_W8A8_ENV,
            "",
        ).strip()
        if calibration_path and static_w8a8_path:
            raise ValueError(
                "Talker static W8A8 calibration and inference cannot be enabled together"
            )
        static_targets = os.environ.get(
            _NPU_TALKER_STATIC_W8A8_TARGETS_ENV,
            "gate_up",
        )
        if calibration_path:
            self._static_w8a8_calibration_path = calibration_path
            self._static_w8a8_collectors = _prepare_talker_static_w8a8_calibration(
                self.tts_model,
                static_targets,
            )
            logger.info(
                "MiniCPM-o Talker static W8A8 calibration active: "
                "%d projections -> %s",
                len(self._static_w8a8_collectors),
                calibration_path,
            )
        elif static_w8a8_path:
            headroom = float(
                os.environ.get(_NPU_TALKER_STATIC_W8A8_HEADROOM_ENV, "1.05")
            )
            converted, parameter_bytes = _prepare_talker_static_w8a8(
                self.tts_model,
                static_w8a8_path,
                static_targets,
                headroom,
            )
            logger.info(
                "MiniCPM-o Talker selective static W8A8 active: "
                "%d projections, %.2f MiB persistent parameters, headroom=%.3f",
                converted,
                parameter_bytes / (1024 * 1024),
                headroom,
            )

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
