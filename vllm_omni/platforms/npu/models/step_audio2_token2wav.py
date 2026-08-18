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
   stack as one TorchAir graph, with an opt-in exact-Linear classifier graph.
5. HiFT residual graphs — replay one selected fixed-shape vocoder stage as
   three fused residual-block graphs while keeping upsampling, source fusion,
   and ISTFT visible to the surrounding eager execution.
6. HiFT fixed ISTFT — replace the steady 16-point complex ISTFT with a static
   real inverse transform and four-way overlap-add graph.
7. HiFT window residency — keep the immutable Hann window on NPU instead of
   copying it from CPU in every STFT and ISTFT call.
8. HiFT harmonic residency — keep SineGen2's immutable harmonic multiplier on
   NPU instead of constructing it on CPU and copying it for every audio chunk.
9. HiFT source-noise scratch — preserve SourceModuleHnNSF2's auxiliary random
   draw while reusing its storage when MiniCPM-o discards that return value.
"""

from __future__ import annotations

import importlib
import math
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
_HIFT_F0_CLASSIFIER_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_CLASSIFIER_GRAPH"
_HIFT_F0_GRAPH_WIDTH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_WIDTH"
_HIFT_F0_GRAPH_BUCKETS_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_BUCKETS"
_HIFT_RESBLOCK_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_RESBLOCK_GRAPH"
_HIFT_RESBLOCK_GRAPH_STAGE_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_RESBLOCK_GRAPH_STAGE"
_HIFT_RESBLOCK_GRAPH_MEL_WIDTH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_RESBLOCK_GRAPH_MEL_WIDTH"
_HIFT_FIXED_ISTFT_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_FIXED_ISTFT_GRAPH"
_HIFT_FIXED_ISTFT_GRAPH_MEL_WIDTH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_FIXED_ISTFT_GRAPH_MEL_WIDTH"
_HIFT_RESIDENT_HARMONICS_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_RESIDENT_HARMONICS"
_HIFT_SOURCE_NOISE_SCRATCH_ENV = "VLLM_OMNI_MINICPMO45_NPU_HIFT_SOURCE_NOISE_SCRATCH"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _place_hift_stft_window(hift: torch.nn.Module, device: torch.device) -> bool:
    """Place HiFT's unregistered immutable Hann window on its compute device."""
    window = getattr(hift, "stft_window", None)
    if not isinstance(window, torch.Tensor):
        raise TypeError("expected HiFT stft_window tensor")
    if window.device == device:
        return False
    hift.stft_window = window.to(device=device)
    logger.info("Placed HiFT STFT window on %s", device)
    return True


def _sinegen_forward_with_harmonics(
    sine_gen: torch.nn.Module,
    f0: torch.Tensor,
    harmonics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run flashcosyvoice SineGen2 with a supplied harmonic multiplier."""
    fn = torch.multiply(f0, harmonics)
    sine_waves = sine_gen._f02sine(fn) * sine_gen.sine_amp
    uv = sine_gen._f02uv(f0)
    noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
    noise = noise_amp * torch.randn_like(sine_waves)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


def _sinegen_with_resident_harmonics(
    self,
    f0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    harmonics = self._step_audio2_npu_harmonics
    if f0.device != harmonics.device or f0.dtype != harmonics.dtype:
        return self._step_audio2_original_forward(f0)
    return _sinegen_forward_with_harmonics(self, f0, harmonics)


def prepare_hift_resident_harmonics_for_npu(
    hift: torch.nn.Module,
    device: torch.device,
) -> bool:
    """Cache SineGen2's immutable harmonic multiplier on its NPU device."""
    try:
        sine_gen = hift.m_source.l_sin_gen
        harmonic_num = sine_gen.harmonic_num
        original_forward = sine_gen.forward
    except AttributeError as exc:
        raise TypeError("expected a Step-Audio2 flashcosyvoice HiFT with m_source.l_sin_gen.forward") from exc

    if getattr(sine_gen, "_step_audio2_npu_harmonics_patched", False):
        return False
    if not isinstance(harmonic_num, int) or harmonic_num < 0:
        raise ValueError(f"invalid SineGen2 harmonic_num={harmonic_num!r}")

    # Preserve flashcosyvoice's exact FloatTensor construction and shape while
    # paying for the host allocation and device transfer only once.
    harmonics = torch.FloatTensor([[range(1, harmonic_num + 2)]]).to(device)
    sine_gen._step_audio2_original_forward = original_forward
    sine_gen._step_audio2_npu_harmonics = harmonics
    sine_gen.forward = MethodType(_sinegen_with_resident_harmonics, sine_gen)
    sine_gen._step_audio2_npu_harmonics_patched = True
    logger.info("Placed HiFT SineGen2 harmonics on %s", device)
    return True


def _source_module_with_noise_scratch(
    self,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SourceModuleHnNSF2 while reusing its discarded noise storage.

    MiniCPM-o consumes only the first return value from ``m_source``. Keep the
    exact upstream random draw and arithmetic for compatibility, but place the
    auxiliary result in a shape-specific scratch tensor instead of allocating
    a new full-waveform tensor for every audio chunk.
    """
    with torch.no_grad():
        sine_wavs, uv, _ = self.l_sin_gen(value)
    sine_merge = self.l_tanh(self.l_linear(sine_wavs))

    key = (
        tuple(uv.shape),
        uv.dtype,
        uv.device,
        tuple(uv.stride()),
    )
    noise = self._step_audio2_npu_source_noise_scratch.get(key)
    if noise is None:
        noise = torch.empty_like(uv)
        self._step_audio2_npu_source_noise_scratch[key] = noise
    torch.randn(uv.shape, out=noise)
    # Retain upstream's two scalar operations and their rounding order.
    noise.mul_(self.sine_amp)
    noise.div_(3)

    replayed_shapes = self._step_audio2_npu_source_noise_scratch_replayed_shapes
    if key not in replayed_shapes:
        logger.info(
            "HiFT source-noise scratch active for shape=%s dtype=%s device=%s",
            tuple(uv.shape),
            uv.dtype,
            uv.device,
        )
        replayed_shapes.add(key)
    return sine_merge, noise, uv


def prepare_hift_source_noise_scratch_for_npu(hift: torch.nn.Module) -> bool:
    """Reuse MiniCPM-o's discarded HiFT source-noise result storage.

    This is intentionally installed on ``m_source`` rather than the upstream
    class: other flashcosyvoice callers may retain the auxiliary noise tensor,
    whereas MiniCPM-o immediately discards it. The random operation and scalar
    multiplication remain unchanged, so output values and RNG advancement are
    identical to upstream for each invocation.
    """
    source = getattr(hift, "m_source", None)
    if source is None:
        raise TypeError("expected a Step-Audio2 flashcosyvoice HiFT m_source")
    if getattr(source, "_step_audio2_npu_source_noise_scratch_patched", False):
        return False
    try:
        original_forward = source.forward
        source.l_sin_gen
        source.l_linear
        source.l_tanh
        float(source.sine_amp)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("expected a flashcosyvoice SourceModuleHnNSF2 layout") from exc

    source._step_audio2_original_forward = original_forward
    source._step_audio2_npu_source_noise_scratch = {}
    source._step_audio2_npu_source_noise_scratch_replayed_shapes = set()
    source.forward = MethodType(_source_module_with_noise_scratch, source)
    source._step_audio2_npu_source_noise_scratch_patched = True
    logger.info("Enabled MiniCPM-o HiFT source-noise scratch reuse")
    return True


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


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _hift_resblock_graph_stage() -> int:
    raw = os.environ.get(_HIFT_RESBLOCK_GRAPH_STAGE_ENV, "0")
    try:
        stage = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {_HIFT_RESBLOCK_GRAPH_STAGE_ENV}={raw!r}") from exc
    if stage < 0:
        raise ValueError(f"{_HIFT_RESBLOCK_GRAPH_STAGE_ENV} must be non-negative, got {stage}")
    return stage


def _conv_transpose_output_width(layer: torch.nn.Module, width: int) -> int:
    """Return the exact 1-D transposed-convolution output width."""
    kernel = int(layer.kernel_size[0])
    stride = int(layer.stride[0])
    padding = int(layer.padding[0])
    dilation = int(layer.dilation[0])
    output_padding = int(layer.output_padding[0])
    return (width - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def _hift_resblock_stage_shape(
    hift: torch.nn.Module,
    *,
    mel_width: int,
    stage: int,
) -> tuple[int, int, int]:
    """Derive the fixed residual input shape from the checkpoint modules."""
    upsamples = getattr(hift, "ups", None)
    if upsamples is None or stage >= len(upsamples):
        raise ValueError(f"HiFT residual graph stage {stage} is outside the upsample stack")
    width = mel_width
    channels = int(hift.conv_pre.out_channels)
    for index, upsample in enumerate(upsamples):
        width = _conv_transpose_output_width(upsample, width)
        channels = int(upsample.out_channels)
        if index == len(upsamples) - 1:
            # flashcosyvoice pads the final upsample by one frame before the
            # source/residual branches.
            width += 1
        if index == stage:
            return (1, channels, width)
    raise AssertionError("unreachable HiFT residual graph stage")


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


def _hift_f0_predictor(
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
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
) -> torch.Tensor:
    """Exact flashcosyvoice F0 predictor, including its original Linear."""
    features = _hift_f0_features(value, w0, b0, w1, b1, w2, b2, w3, b3, w4, b4)
    # TorchAir 8.5 otherwise drops the transpose while inferring MatMul's
    # K-axis (58 instead of 512). Materialize only the view's layout; the
    # classifier operation and values remain identical to upstream.
    classifier_input = features.transpose(1, 2).contiguous()
    prediction = F.linear(classifier_input, classifier_weight, classifier_bias)
    return torch.abs(prediction.squeeze(-1))


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
        logger.info(
            "HiFT F0 %s graph replay active for runtime shape %s",
            "predictor" if self._step_audio2_npu_f0_classifier_in_graph else "feature",
            shape,
        )
        replayed_widths.add(shape[2])
    output = self._step_audio2_npu_f0_graph(value, *self._step_audio2_npu_f0_graph_weights)
    if self._step_audio2_npu_f0_classifier_in_graph:
        return output
    # Keep the checkpoint's original per-timestep Linear outside the graph.
    # TorchAir 8.5 lowers an equivalent 1x1 Conv faster, but its accumulation
    # order moves F0 by up to 0.36 Hz. This form stays bit-exact to upstream.
    return torch.abs(self.classifier(output.transpose(1, 2)).squeeze(-1))


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
    include_classifier: bool = False,
) -> bool:
    """Compile fixed-width HiFT F0 feature stacks for Ascend inference.

    The default boundary contains the five Conv+ELU feature layers. The
    experimental full boundary also contains the checkpoint's original
    per-timestep Linear and absolute value; unlike the rejected 1x1-Conv
    substitution, this preserves the source operation. Non-steady widths and
    batches use the original bound method.
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
    feature_weights: tuple[torch.Tensor, ...] = tuple(
        tensor
        for layer in convolutions
        for tensor in (layer.weight, layer.bias)
    )
    if include_classifier:
        if classifier.bias is None:
            raise TypeError("expected the biased flashcosyvoice F0 classifier")
        graph_function = _hift_f0_predictor
        weights = (*feature_weights, classifier.weight, classifier.bias)
    else:
        graph_function = _hift_f0_features
        weights = feature_weights

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    graph = torch.compile(
        graph_function,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    widths = tuple(dict.fromkeys((width, *extra_widths)))
    for graph_width in widths:
        if graph_width <= 0:
            raise ValueError(f"HiFT F0 graph widths must be positive, got {graph_width}")
        shape = (1, int(convolutions[0].in_channels), graph_width)
        sample = (
            convolutions[0].weight.new_full(shape, 0.125)
            if include_classifier
            else convolutions[0].weight.new_zeros(shape)
        )
        with torch.inference_mode():
            expected = predictor(sample) if include_classifier else predictor.condnet(sample)
            actual = graph(sample, *weights)
        tolerance = 1e-5 if include_classifier else 0.0
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.npu.synchronize()
        logger.info(
            "Compiled HiFT F0 %s graph for Ascend NPU: batch=1 width=%d max_abs_drift=%.8g",
            "predictor" if include_classifier else "feature",
            graph_width,
            float(torch.max(torch.abs(actual - expected)).item()),
        )

    predictor._step_audio2_original_forward = predictor.forward
    predictor._step_audio2_npu_f0_graph = graph
    predictor._step_audio2_npu_f0_graph_weights = weights
    predictor._step_audio2_npu_f0_classifier_in_graph = include_classifier
    predictor._step_audio2_npu_f0_graph_widths = widths
    predictor._step_audio2_npu_f0_graph_replayed_widths = set()
    predictor._step_audio2_npu_f0_input_channels = int(convolutions[0].in_channels)
    predictor.forward = MethodType(_f0_predictor_with_npu_graph, predictor)
    predictor._step_audio2_npu_f0_graph_patched = True
    return True


def _resblock_with_npu_graph(self, value: torch.Tensor) -> torch.Tensor:
    """Replay one exact-shape HiFT residual graph with an eager fallback."""
    if (
        value.device.type != "npu"
        or tuple(value.shape) != self._step_audio2_npu_resblock_graph_shape
        or self._step_audio2_npu_resblock_graph_disabled
    ):
        return self._step_audio2_original_forward(value)
    try:
        output = self._step_audio2_npu_resblock_graph(value)
    except Exception:
        self._step_audio2_npu_resblock_graph_disabled = True
        logger.warning(
            "HiFT residual-block graph failed for shape %s; using eager execution",
            tuple(value.shape),
            exc_info=True,
        )
        return self._step_audio2_original_forward(value)
    if not self._step_audio2_npu_resblock_graph_replayed:
        logger.info(
            "HiFT residual-block graph replay active for shape %s",
            tuple(value.shape),
        )
        self._step_audio2_npu_resblock_graph_replayed = True
    return output


def prepare_hift_resblock_graph_for_npu(
    hift: torch.nn.Module,
    *,
    stage: int,
    mel_width: int,
) -> int:
    """Compile the three residual blocks in one fixed HiFT upsample stage.

    Each flashcosyvoice residual block contains three
    ``Snake -> Conv1d -> Snake -> Conv1d -> add`` sequences. Compiling at the
    block boundary removes the intervening Python/operator launch chain while
    leaving ConvTranspose1d, source injection, and ISTFT outside the graph.
    Unsupported shapes retain the package's original bound method.
    """
    if getattr(hift, "_step_audio2_npu_resblock_graph_patched", False):
        return 0
    if mel_width <= 0:
        raise ValueError(f"HiFT residual graph mel width must be positive, got {mel_width}")

    num_upsamples = int(getattr(hift, "num_upsamples", -1))
    num_kernels = int(getattr(hift, "num_kernels", -1))
    residuals = getattr(hift, "resblocks", None)
    if (
        num_upsamples <= 0
        or num_kernels <= 0
        or residuals is None
        or len(residuals) != num_upsamples * num_kernels
    ):
        raise TypeError("expected the flashcosyvoice staged HiFT residual layout")
    if stage < 0 or stage >= num_upsamples:
        raise ValueError(f"HiFT residual graph stage {stage} must be in [0, {num_upsamples})")

    shape = _hift_resblock_stage_shape(hift, mel_width=mel_width, stage=stage)
    first = residuals[stage * num_kernels]
    parameter = next(first.parameters(), None)
    if parameter is None or parameter.device.type != "npu":
        raise ValueError("HiFT residual graph requires NPU-resident weights")

    # Graph parameters must be immutable, and fetching a parametrized weight
    # from inside every residual convolution would recreate the work this
    # optimization is intended to remove.
    materialize_hift_weight_norm_for_npu(hift)

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    prepared: list[tuple[torch.nn.Module, object, object]] = []
    for block_index in range(stage * num_kernels, (stage + 1) * num_kernels):
        block = residuals[block_index]
        original_forward = block.forward
        graph = torch.compile(
            original_forward,
            backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
            fullgraph=True,
            dynamic=False,
        )
        # Exercise the periodic Snake path as well as convolution bias during
        # the exactness gate; an all-zero sample would under-test the graph.
        sample = parameter.new_full(shape, 0.125)
        with torch.inference_mode():
            expected = original_forward(sample)
            actual = graph(sample)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.npu.synchronize()

        prepared.append((block, original_forward, graph))

    # Install only after every sibling compiled and passed the exactness gate.
    # A failure in one block must leave the whole stage on its original path.
    for block, original_forward, graph in prepared:
        block._step_audio2_original_forward = original_forward
        block._step_audio2_npu_resblock_graph = graph
        block._step_audio2_npu_resblock_graph_shape = shape
        block._step_audio2_npu_resblock_graph_disabled = False
        block._step_audio2_npu_resblock_graph_replayed = False
        block.forward = MethodType(_resblock_with_npu_graph, block)

    hift._step_audio2_npu_resblock_graph_patched = True
    hift._step_audio2_npu_resblock_graph_stage = stage
    hift._step_audio2_npu_resblock_graph_shape = shape
    logger.info(
        "Compiled %d HiFT residual-block graphs for stage=%d shape=%s",
        len(prepared),
        stage,
        shape,
    )
    return len(prepared)


def _hift_fixed_istft(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    real_weight: torch.Tensor,
    imag_weight: torch.Tensor,
    window: torch.Tensor,
    envelope: torch.Tensor,
) -> torch.Tensor:
    """Evaluate MiniCPM-o's fixed 16-point ISTFT without complex tensors."""
    magnitude = torch.clip(magnitude, max=1e2)
    real = (magnitude * torch.cos(phase)).permute(0, 2, 1).contiguous()
    imag = (magnitude * torch.sin(phase)).permute(0, 2, 1).contiguous()
    frames = F.linear(real.reshape(-1, 9), real_weight)
    frames = frames + F.linear(imag.reshape(-1, 9), imag_weight)
    frames = frames.reshape(real.shape[0], real.shape[1], 16)

    weighted = frames.reshape(frames.shape[0], frames.shape[1], 4, 4)
    weighted = weighted * window.reshape(1, 1, 4, 4)
    width = frames.shape[1]
    parts: list[torch.Tensor] = []
    for quarter in range(4):
        padded = F.pad(weighted[:, :, quarter, :], (0, 0, 3, 3))
        start = 5 - quarter
        parts.append(padded[:, start : start + width - 1, :])
    output = (parts[0] + parts[1] + parts[2] + parts[3]) / envelope
    return output.reshape(output.shape[0], -1)


def _hift_fixed_istft_constants(
    *,
    width: int,
    window: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the real inverse-DFT weights and exact centered OLA envelope."""
    if width <= 1:
        raise ValueError(f"HiFT ISTFT graph width must exceed one, got {width}")
    if window.numel() != 16:
        raise ValueError(f"HiFT ISTFT graph requires a 16-sample window, got {window.numel()}")

    real_weight = torch.empty((16, 9), dtype=torch.float32)
    imag_weight = torch.empty((16, 9), dtype=torch.float32)
    for sample in range(16):
        for frequency in range(9):
            scale = 1.0 if frequency in (0, 8) else 2.0
            angle = 2.0 * math.pi * frequency * sample / 16
            real_weight[sample, frequency] = scale * math.cos(angle) / 16
            imag_weight[sample, frequency] = -scale * math.sin(angle) / 16

    window_cpu = window.detach().to(device="cpu", dtype=torch.float32).reshape(4, 4)
    envelope = torch.zeros((width - 1, 4), dtype=torch.float32)
    for output_block, block in enumerate(range(2, width + 1)):
        for quarter in range(4):
            frame = block - quarter
            if 0 <= frame < width:
                envelope[output_block] += window_cpu[quarter].square()

    return (
        real_weight.to(device=device, dtype=dtype),
        imag_weight.to(device=device, dtype=dtype),
        window_cpu.reshape(16).to(device=device, dtype=dtype),
        envelope.reshape(1, width - 1, 4).to(device=device, dtype=dtype),
    )


def _istft_with_npu_graph(
    self,
    magnitude: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Replay the exact-shape fixed ISTFT graph, otherwise use upstream."""
    if (
        magnitude.device.type != "npu"
        or tuple(magnitude.shape) != self._step_audio2_npu_istft_graph_shape
        or tuple(phase.shape) != self._step_audio2_npu_istft_graph_shape
        or magnitude.dtype != self._step_audio2_npu_istft_graph_dtype
        or phase.dtype != self._step_audio2_npu_istft_graph_dtype
        or self._step_audio2_npu_istft_graph_disabled
    ):
        return self._step_audio2_original_istft(magnitude, phase)
    try:
        output = self._step_audio2_npu_istft_graph(
            magnitude,
            phase,
            *self._step_audio2_npu_istft_graph_constants,
        )
    except Exception:
        self._step_audio2_npu_istft_graph_disabled = True
        logger.warning(
            "HiFT fixed ISTFT graph failed for shape %s; using eager ISTFT",
            tuple(magnitude.shape),
            exc_info=True,
        )
        return self._step_audio2_original_istft(magnitude, phase)
    if not self._step_audio2_npu_istft_graph_replayed:
        logger.info("HiFT fixed ISTFT graph replay active for shape %s", tuple(magnitude.shape))
        self._step_audio2_npu_istft_graph_replayed = True
    return output


def prepare_hift_fixed_istft_graph_for_npu(
    hift: torch.nn.Module,
    *,
    mel_width: int,
) -> bool:
    """Compile MiniCPM-o's steady 16-point, hop-4 HiFT inverse transform."""
    if getattr(hift, "_step_audio2_npu_istft_graph_patched", False):
        return True
    params = getattr(hift, "istft_params", None)
    if not isinstance(params, dict) or params.get("n_fft") != 16 or params.get("hop_len") != 4:
        raise TypeError("HiFT fixed ISTFT graph requires n_fft=16 and hop_len=4")
    original_istft = getattr(hift, "_istft", None)
    window = getattr(hift, "stft_window", None)
    parameter = next(hift.parameters(), None)
    if original_istft is None or not isinstance(window, torch.Tensor):
        raise TypeError("expected HiFT _istft and stft_window")
    if parameter is None or parameter.device.type != "npu":
        raise ValueError("HiFT fixed ISTFT graph requires NPU-resident weights")

    stage = int(getattr(hift, "num_upsamples", 0)) - 1
    width = _hift_resblock_stage_shape(hift, mel_width=mel_width, stage=stage)[2]
    shape = (1, 9, width)
    constants = _hift_fixed_istft_constants(
        width=width,
        window=window,
        device=parameter.device,
        dtype=parameter.dtype,
    )

    from torch_npu.dynamo import torchair

    _ensure_torchair_broadcast_alias()
    graph = torch.compile(
        _hift_fixed_istft,
        backend=torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()),
        fullgraph=True,
        dynamic=False,
    )
    values = 9 * width
    magnitude = torch.linspace(
        0.1,
        4.0,
        steps=values,
        device=parameter.device,
        dtype=parameter.dtype,
    ).reshape(shape)
    phase = torch.linspace(
        -1.0,
        1.0,
        steps=values,
        device=parameter.device,
        dtype=parameter.dtype,
    ).reshape(shape)
    with torch.inference_mode():
        expected = original_istft(magnitude, phase)
        actual = graph(magnitude, phase, *constants)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.npu.synchronize()

    hift._step_audio2_original_istft = original_istft
    hift._step_audio2_npu_istft_graph = graph
    hift._step_audio2_npu_istft_graph_constants = constants
    hift._step_audio2_npu_istft_graph_shape = shape
    hift._step_audio2_npu_istft_graph_dtype = parameter.dtype
    hift._step_audio2_npu_istft_graph_disabled = False
    hift._step_audio2_npu_istft_graph_replayed = False
    hift._istft = MethodType(_istft_with_npu_graph, hift)
    hift._step_audio2_npu_istft_graph_patched = True
    logger.info("Compiled HiFT fixed ISTFT graph for shape=%s", shape)
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
    try:
        _place_hift_stft_window(self._hift, self.device)
    except Exception:
        logger.warning("Unable to keep HiFT STFT window on NPU; using per-call copies", exc_info=True)
    if _env_flag_enabled(_HIFT_RESIDENT_HARMONICS_ENV):
        try:
            prepare_hift_resident_harmonics_for_npu(self._hift, self.device)
        except Exception:
            logger.warning(
                "Unable to keep HiFT harmonics on NPU; using per-call copies",
                exc_info=True,
            )
    if _env_flag_enabled(_HIFT_SOURCE_NOISE_SCRATCH_ENV):
        try:
            prepare_hift_source_noise_scratch_for_npu(self._hift)
        except Exception:
            logger.warning(
                "Unable to enable HiFT source-noise scratch; using per-call allocation",
                exc_info=True,
            )
    if _env_flag_enabled(_HIFT_MATERIALIZE_WEIGHT_NORM_ENV):
        materialize_hift_weight_norm_for_npu(self._hift)
    if _env_flag_enabled(_HIFT_F0_GRAPH_ENV):
        try:
            prepare_hift_f0_graph_for_npu(
                self._hift,
                width=_hift_f0_graph_width(),
                extra_widths=_hift_f0_graph_buckets(),
                include_classifier=_env_flag_enabled(_HIFT_F0_CLASSIFIER_GRAPH_ENV),
            )
        except Exception:
            logger.warning("HiFT F0 graph compilation failed; using eager predictor", exc_info=True)
    if _env_flag_enabled(_HIFT_RESBLOCK_GRAPH_ENV):
        try:
            prepare_hift_resblock_graph_for_npu(
                self._hift,
                stage=_hift_resblock_graph_stage(),
                mel_width=_positive_int_env(_HIFT_RESBLOCK_GRAPH_MEL_WIDTH_ENV, 58),
            )
        except Exception:
            logger.warning("HiFT residual-block graph compilation failed; using eager blocks", exc_info=True)
    if _env_flag_enabled(_HIFT_FIXED_ISTFT_GRAPH_ENV):
        try:
            prepare_hift_fixed_istft_graph_for_npu(
                self._hift,
                mel_width=_positive_int_env(_HIFT_FIXED_ISTFT_GRAPH_MEL_WIDTH_ENV, 58),
            )
        except Exception:
            logger.warning("HiFT fixed ISTFT graph compilation failed; using eager ISTFT", exc_info=True)


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
