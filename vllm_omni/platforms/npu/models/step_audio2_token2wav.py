# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for Step-Audio2 / MiniCPM Token2Wav.

Ascend-specific workarounds that must not live in the shared GPU model file:

1. HiFT sine-source downsample — replace the failing 480x ``linear1d``
   downsample with its exact midpoint form while keeping HiFT on NPU.
2. CosyVoice2 DiT SDPA — expand the DiT attention mask and let the adapter
   probe fused attention with a sticky MATH fallback for older CANN stacks.
3. HiFT weight norm — optionally freeze inference-only normalized convolution
   weights after checkpoint loading instead of recomputing them per chunk.
4. HiFT F0 graph — replay the fixed steady-state five-convolution feature
   stack as one TorchAir graph while retaining the original eager classifier.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_ensure_models_loaded = None
_original_forward = None
_original_stream_chunk_for = None

_HIFT_MATERIALIZE_WEIGHT_NORM_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_MATERIALIZE_WEIGHT_NORM"
_HIFT_F0_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH"
_HIFT_F0_GRAPH_WIDTH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_WIDTH"
_HIFT_F0_GRAPH_BUCKETS_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_BUCKETS"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _hift_f0_graph_width() -> int:
    raw = os.environ.get(_HIFT_F0_GRAPH_WIDTH_ENV, "58")
    try:
        width = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {_HIFT_F0_GRAPH_WIDTH_ENV}={raw!r}") from exc
    if width <= 0:
        raise ValueError(f"{_HIFT_F0_GRAPH_WIDTH_ENV} must be positive, got {width}")
    return width


def _hift_f0_graph_buckets() -> tuple[int, ...]:
    """Return additional static F0 widths beside the steady-state width."""
    raw = os.environ.get(_HIFT_F0_GRAPH_BUCKETS_ENV, "")
    if not raw.strip():
        return ()
    widths: list[int] = []
    for value in raw.split(","):
        try:
            width = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid {_HIFT_F0_GRAPH_BUCKETS_ENV}={raw!r}") from exc
        if width <= 0:
            raise ValueError(f"{_HIFT_F0_GRAPH_BUCKETS_ENV} widths must be positive, got {width}")
        if width not in widths:
            widths.append(width)
    return tuple(widths)


def _ensure_torchair_broadcast_alias() -> None:
    """Repair TorchAir's broadcast converter import order in vLLM workers."""
    module = importlib.import_module(
        "torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast"
    )
    if not hasattr(module, "op_broadcast"):
        broadcast = torch.ops.npu_define.broadcast
        module.op_broadcast = getattr(broadcast, "default", broadcast)


def _hift_f0_features(
    value: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    w4: torch.Tensor,
    b4: torch.Tensor,
) -> torch.Tensor:
    """Exact five-Conv feature region of flashcosyvoice's F0 predictor."""
    value = F.elu(F.conv1d(value, w0, b0, padding=1))
    value = F.elu(F.conv1d(value, w1, b1, padding=1))
    value = F.elu(F.conv1d(value, w2, b2, padding=1))
    value = F.elu(F.conv1d(value, w3, b3, padding=1))
    return F.elu(F.conv1d(value, w4, b4, padding=1))


def _f0_predictor_with_npu_graph(self, value: torch.Tensor) -> torch.Tensor:
    shape = tuple(value.shape)
    expected_prefix = (1, int(self._step_audio2_npu_f0_input_channels))
    graph_widths = self._step_audio2_npu_f0_graph_widths
    if value.device.type != "npu" or shape[:2] != expected_prefix or shape[2] not in graph_widths:
        if value.device.type == "npu" and not getattr(self, "_step_audio2_npu_f0_graph_fallback_logged", False):
            logger.info(
                "HiFT F0 graph falling back for runtime shape %s; compiled widths are %s",
                shape,
                graph_widths,
            )
            self._step_audio2_npu_f0_graph_fallback_logged = True
        return self._step_audio2_original_forward(value)

    replayed_widths = self._step_audio2_npu_f0_graph_replayed_widths
    if shape[2] not in replayed_widths:
        logger.info("HiFT F0 feature graph replay active for runtime shape %s", shape)
        replayed_widths.add(shape[2])
    features = self._step_audio2_npu_f0_graph(value, *self._step_audio2_npu_f0_graph_weights)
    # Keep the checkpoint's original per-timestep Linear outside the graph.
    # TorchAir 8.5 lowers an equivalent 1x1 Conv faster, but its accumulation
    # order moves F0 by up to 0.36 Hz. This form stays bit-exact to upstream.
    return torch.abs(self.classifier(features.transpose(1, 2)).squeeze(-1))


def materialize_hift_weight_norm_for_npu(hift: torch.nn.Module) -> int:
    """Replace HiFT inference-time weight-norm parametrizations with weights.

    PyTorch's parametrization API recomputes every normalized convolution
    weight on every attribute access. HiFT is immutable in serving, so the
    effective weights can be stored once after checkpoint loading. Only the
    standard ``_WeightNorm`` parametrization is removed; unrelated or mixed
    parametrizations are left untouched.
    """
    if getattr(hift, "_step_audio2_npu_weight_norm_materialized", False):
        return 0

    materialized = 0
    for module in hift.modules():
        if not parametrize.is_parametrized(module, "weight"):
            continue
        weight_parametrizations = module.parametrizations.weight
        if not weight_parametrizations or any(
            type(item).__name__ != "_WeightNorm" for item in weight_parametrizations
        ):
            continue
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        materialized += 1

    hift._step_audio2_npu_weight_norm_materialized = True
    logger.info("Materialized %d HiFT weight-norm parametrizations for Ascend NPU", materialized)
    return materialized


def prepare_hift_f0_graph_for_npu(
    hift: torch.nn.Module,
    *,
    width: int,
    extra_widths: tuple[int, ...] = (),
) -> bool:
    """Compile fixed-width HiFT F0 feature stacks for Ascend inference.

    Only the five Conv+ELU feature layers enter GE. The classifier remains the
    upstream eager Linear so the optimization changes neither its operation
    nor its accumulation order. Non-steady widths and batches use the original
    bound method.
    """
    predictor = getattr(hift, "f0_predictor", None)
    if predictor is None:
        raise TypeError("expected HiFT f0_predictor")
    if getattr(predictor, "_step_audio2_npu_f0_graph_patched", False):
        return True

    condnet = getattr(predictor, "condnet", None)
    classifier = getattr(predictor, "classifier", None)
    convolutions = [layer for layer in condnet or () if isinstance(layer, torch.nn.Conv1d)]
    if (
        len(convolutions) != 5
        or not isinstance(classifier, torch.nn.Linear)
        or any(tuple(layer.kernel_size) != (3,) or tuple(layer.padding) != (1,) for layer in convolutions)
        or any(layer.bias is None for layer in convolutions)
    ):
        raise TypeError("expected the biased flashcosyvoice five-Conv F0 predictor layout")
    if convolutions[0].weight.device.type != "npu":
        raise ValueError("HiFT F0 graph requires NPU-resident weights")

    # Graph parameters must be immutable. This removes only the five standard
    # weight-norm parametrizations owned by the F0 feature stack.
    materialize_hift_weight_norm_for_npu(predictor)
    weights: tuple[torch.Tensor, ...] = tuple(
        tensor
        for layer in convolutions
        for tensor in (layer.weight, layer.bias)
    )

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    graph = torch.compile(
        _hift_f0_features,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    widths = tuple(dict.fromkeys((width, *extra_widths)))
    for graph_width in widths:
        if graph_width <= 0:
            raise ValueError(f"HiFT F0 graph widths must be positive, got {graph_width}")
        sample = convolutions[0].weight.new_zeros(
            (1, int(convolutions[0].in_channels), graph_width)
        )
        with torch.inference_mode():
            expected_features = predictor.condnet(sample)
            actual_features = graph(sample, *weights)
        torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
        torch.npu.synchronize()
        logger.info(
            "Compiled HiFT F0 feature graph for Ascend NPU: batch=1 width=%d",
            graph_width,
        )

    predictor._step_audio2_original_forward = predictor.forward
    predictor._step_audio2_npu_f0_graph = graph
    predictor._step_audio2_npu_f0_graph_weights = weights
    predictor._step_audio2_npu_f0_graph_widths = widths
    predictor._step_audio2_npu_f0_graph_replayed_widths = set()
    predictor._step_audio2_npu_f0_input_channels = int(convolutions[0].in_channels)
    predictor.forward = MethodType(_f0_predictor_with_npu_graph, predictor)
    predictor._step_audio2_npu_f0_graph_patched = True
    return True


def _linear_downsample_even_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Match ``F.interpolate(..., mode="linear")`` for an even integer scale.

    With ``align_corners=False``, every output location for an even integer
    downsample lies exactly halfway between two source samples. Selecting and
    averaging those samples avoids Ascend/pytorch#150's ``linear1d`` kernel.
    """
    if scale <= 0 or scale % 2:
        raise ValueError(f"scale must be a positive even integer, got {scale}")
    if x.shape[-1] % scale:
        raise ValueError(f"input length {x.shape[-1]} must be divisible by scale {scale}")

    left = scale // 2 - 1
    right = scale // 2
    return (x[..., left::scale] + x[..., right::scale]) * 0.5


def _run_original_f02sine_on_cpu(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Run the unmodified ``_f02sine`` without invoking NPU ``linear1d``."""
    output_device = f0_values.device
    output = self._step_audio2_original_f02sine(f0_values.cpu())
    return output.to(output_device)


def _f02sine_with_npu_safe_downsample(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Use the exact NPU midpoint path, with a narrow CPU fallback."""
    if getattr(self, "flag_for_pulse", False):
        return _run_original_f02sine_on_cpu(self, f0_values)

    upsample_scale = self.upsample_scale
    if upsample_scale <= 0:
        raise ValueError(f"upsample_scale must be positive, got {upsample_scale}")

    scale = int(upsample_scale)
    midpoint_supported = scale == upsample_scale and scale % 2 == 0 and f0_values.shape[1] % scale == 0
    if not midpoint_supported:
        return _run_original_f02sine_on_cpu(self, f0_values)

    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

    rad_values = _linear_downsample_even_scale(rad_values.transpose(1, 2), scale).transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
    phase = F.interpolate(
        phase.transpose(1, 2) * self.upsample_scale,
        scale_factor=self.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    return torch.sin(phase)


def patch_step_audio2_hift_for_npu(hift: torch.nn.Module) -> None:
    """Patch the non-causal Step-Audio2 HiFT implementation for Ascend.

    The ``flashcosyvoice.SineGen2`` instantiated by Step-Audio2 1.0.0 is
    non-causal and reduces a full-rate phase tensor by ``1 / 480`` before
    restoring it to the waveform rate. Ascend's ``upsample_linear1d`` kernel
    can raise an AIVector UB-address exception (ACL 507015) for that reduction.

    The exact midpoint form keeps the common path on NPU. Unsupported or pulse
    configurations delegate only ``_f02sine`` to CPU, preserving upstream
    behavior without restoring the old whole-HiFT CPU offload.
    """
    if getattr(hift, "_step_audio2_npu_downsample_patched", False):
        return

    try:
        sine_gen = hift.m_source.l_sin_gen
        original_f02sine = sine_gen._f02sine
    except AttributeError as exc:
        raise TypeError("expected a Step-Audio2 flashcosyvoice HiFT with m_source.l_sin_gen._f02sine") from exc

    if getattr(sine_gen, "causal", False):
        raise ValueError("the Step-Audio2 NPU HiFT patch only supports non-causal SineGen2")

    sine_gen._step_audio2_original_f02sine = original_f02sine
    sine_gen._f02sine = MethodType(_f02sine_with_npu_safe_downsample, sine_gen)
    hift._step_audio2_npu_downsample_patched = True
    logger.info("Patched Step-Audio2 HiFT linear downsample for Ascend NPU")


@contextmanager
def npu_token2wav_sdpa_context() -> Iterator[None]:
    """Install the CosyVoice mask/fused-attention adapter for this process."""
    try:
        from vllm_omni.platforms.npu.models.cosyvoice2_dit_attn import (
            apply_cosyvoice2_dit_attn_npu_patch,
        )
    except ImportError as exc:
        logger.debug("CosyVoice2 NPU attention adapter unavailable: %s", exc)
        yield
        return

    try:
        apply_cosyvoice2_dit_attn_npu_patch()
    except Exception as exc:
        # Patching remains best-effort for optional Token2Wav dependencies,
        # but exceptions raised by the actual model invocation must propagate.
        logger.warning("Unable to install CosyVoice2 NPU attention adapter: %s", exc)
    yield


def _patched_ensure_models_loaded(self) -> None:
    assert _original_ensure_models_loaded is not None
    was_loaded = self._models_loaded
    _original_ensure_models_loaded(self)
    if was_loaded or self.device.type != "npu" or self._hift is None:
        return
    patch_step_audio2_hift_for_npu(self._hift)
    if _env_flag_enabled(_HIFT_MATERIALIZE_WEIGHT_NORM_ENV):
        materialize_hift_weight_norm_for_npu(self._hift)
    if _env_flag_enabled(_HIFT_F0_GRAPH_ENV):
        try:
            prepare_hift_f0_graph_for_npu(
                self._hift,
                width=_hift_f0_graph_width(),
                extra_widths=_hift_f0_graph_buckets(),
            )
        except Exception:
            logger.warning("HiFT F0 graph compilation failed; using eager predictor", exc_info=True)


def _patched_forward(self, generated_speech_tokens, prompt_wav, return_bytes=True):
    assert _original_forward is not None
    if self.device.type != "npu":
        return _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)
    with npu_token2wav_sdpa_context():
        return _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)


def _patched_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state):
    assert _original_stream_chunk_for is not None
    if self.device.type != "npu":
        return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)
    with npu_token2wav_sdpa_context():
        return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)


def apply_step_audio2_token2wav_npu_patch() -> None:
    """Monkey-patch StepAudio2Token2WavCore for Ascend NPU.

    Import is deferred and optional: platform bootstrap (e.g. resolving
    ``current_omni_platform`` from rotary embedding) must not require
    Token2Wav optional deps such as ``librosa``.
    """
    global _PATCHED, _original_ensure_models_loaded, _original_forward, _original_stream_chunk_for
    if _PATCHED:
        return

    try:
        from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
            StepAudio2Token2WavCore,
        )
    except ImportError as e:
        logger.debug("step_audio2 token2wav deps unavailable; skip NPU patch: %s", e)
        return

    _original_ensure_models_loaded = StepAudio2Token2WavCore._ensure_models_loaded
    _original_forward = StepAudio2Token2WavCore.forward
    _original_stream_chunk_for = StepAudio2Token2WavCore.stream_chunk_for

    StepAudio2Token2WavCore._ensure_models_loaded = _patched_ensure_models_loaded  # type: ignore[method-assign]
    StepAudio2Token2WavCore.forward = _patched_forward  # type: ignore[method-assign]
    StepAudio2Token2WavCore.stream_chunk_for = _patched_stream_chunk_for  # type: ignore[method-assign]

    _PATCHED = True
    logger.debug("Applied NPU patch for StepAudio2Token2WavCore")
