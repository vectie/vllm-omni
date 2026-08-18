# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Step-Audio2 Ascend HiFT patch."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    fake_logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = lambda _name: fake_logger
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_vllm_logger)

    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "vllm_omni").is_dir())
    path = root / "vllm_omni" / "platforms" / "npu" / "models" / "step_audio2_token2wav.py"
    spec = importlib.util.spec_from_file_location("test_step_audio2_token2wav_npu_patch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_f02sine(sine_gen, f0_values: torch.Tensor) -> torch.Tensor:
    rad_values = (f0_values / sine_gen.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    rad_values = F.interpolate(
        rad_values.transpose(1, 2),
        scale_factor=1 / sine_gen.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
    phase = F.interpolate(
        phase.transpose(1, 2) * sine_gen.upsample_scale,
        scale_factor=sine_gen.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    return torch.sin(phase)


def test_even_scale_downsample_matches_linear_interpolate(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    x = torch.randn(2, 3, 960)

    expected = F.interpolate(x, scale_factor=1 / 4, mode="linear")
    actual = module._linear_downsample_even_scale(x, 4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("shape", "scale"), [((1, 2, 12), 3), ((1, 2, 10), 4)])
def test_even_scale_downsample_rejects_unsupported_shapes(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    scale: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    with pytest.raises(ValueError):
        module._linear_downsample_even_scale(torch.randn(shape), scale)


def test_hift_patch_is_exact_idempotent_and_stays_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSineGen:
        sampling_rate = 24000
        upsample_scale = 4
        flag_for_pulse = False

        def __init__(self):
            self.original_calls = 0

        def _f02sine(self, f0_values):
            self.original_calls += 1
            return _reference_f02sine(self, f0_values)

    class FakeHiFT:
        def __init__(self):
            self.m_source = SimpleNamespace(l_sin_gen=FakeSineGen())

        def to(self, *_args, **_kwargs):
            raise AssertionError("HiFT must stay on its accelerator")

    hift = FakeHiFT()
    module.patch_step_audio2_hift_for_npu(hift)
    patched_method = hift.m_source.l_sin_gen._f02sine
    module.patch_step_audio2_hift_for_npu(hift)
    assert hift.m_source.l_sin_gen._f02sine is patched_method

    f0_values = torch.rand(1, 16, 3)
    torch.manual_seed(7)
    expected = _reference_f02sine(hift.m_source.l_sin_gen, f0_values)
    torch.manual_seed(7)
    actual = patched_method(f0_values)

    assert hift.m_source.l_sin_gen.original_calls == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("flag_for_pulse", "upsample_scale", "input_length"),
    [
        (True, 4, 16),
        (False, 3, 12),
        (False, 4.5, 18),
        (False, 4, 10),
    ],
)
def test_hift_patch_delegates_unsupported_cases_to_original_f02sine_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    flag_for_pulse: bool,
    upsample_scale: float,
    input_length: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    calls: list[str] = []

    class FakeSineGen:
        def __init__(self):
            self.flag_for_pulse = flag_for_pulse
            self.upsample_scale = upsample_scale

        def _f02sine(self, f0_values):
            calls.append(f0_values.device.type)
            return f0_values + 1

    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=FakeSineGen()))
    module.patch_step_audio2_hift_for_npu(hift)

    f0_values = torch.randn(1, input_length, 3)
    output = hift.m_source.l_sin_gen._f02sine(f0_values)

    assert calls == ["cpu"]
    assert output.device == f0_values.device
    torch.testing.assert_close(output, f0_values + 1)


@pytest.mark.parametrize("upsample_scale", [0, -2])
def test_hift_patch_rejects_non_positive_scale(
    monkeypatch: pytest.MonkeyPatch,
    upsample_scale: int,
) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSineGen:
        flag_for_pulse = False

        def __init__(self):
            self.upsample_scale = upsample_scale

        def _f02sine(self, f0_values):
            return f0_values

    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=FakeSineGen()))
    module.patch_step_audio2_hift_for_npu(hift)

    with pytest.raises(ValueError, match="upsample_scale must be positive"):
        hift.m_source.l_sin_gen._f02sine(torch.randn(1, 16, 3))


def test_hift_patch_rejects_causal_sinegen(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeCausalSineGen:
        causal = True

        def _f02sine(self, f0_values):
            return f0_values

    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=FakeCausalSineGen()))

    with pytest.raises(ValueError, match="only supports non-causal SineGen2"):
        module.patch_step_audio2_hift_for_npu(hift)


def test_hift_patch_reports_incompatible_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)

    with pytest.raises(TypeError, match=r"m_source\.l_sin_gen\._f02sine"):
        module.patch_step_audio2_hift_for_npu(SimpleNamespace())


def test_hift_weight_norm_materialization_is_exact_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    hift = torch.nn.Sequential(
        torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 4, 3)),
        torch.nn.SiLU(),
        torch.nn.utils.parametrizations.weight_norm(torch.nn.ConvTranspose1d(4, 2, 4, stride=2)),
    ).eval()
    sample = torch.randn(2, 3, 16)
    expected = hift(sample)

    assert module.materialize_hift_weight_norm_for_npu(hift) == 2
    assert module.materialize_hift_weight_norm_for_npu(hift) == 0
    assert not parametrize.is_parametrized(hift[0], "weight")
    assert not parametrize.is_parametrized(hift[2], "weight")
    torch.testing.assert_close(hift(sample), expected, rtol=0, atol=0)


def test_hift_weight_norm_materialization_preserves_unrelated_parametrization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)

    class AbsWeight(torch.nn.Module):
        def forward(self, weight):
            return weight.abs()

    layer = torch.nn.Conv1d(2, 2, 1)
    parametrize.register_parametrization(layer, "weight", AbsWeight())
    hift = torch.nn.Sequential(layer)

    assert module.materialize_hift_weight_norm_for_npu(hift) == 0
    assert parametrize.is_parametrized(layer, "weight")


def test_hift_resblock_stage_shapes_match_flashcosyvoice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    hift = SimpleNamespace(
        conv_pre=SimpleNamespace(out_channels=512),
        ups=torch.nn.ModuleList(
            (
                torch.nn.ConvTranspose1d(512, 256, 16, stride=8, padding=4),
                torch.nn.ConvTranspose1d(256, 128, 11, stride=5, padding=3),
                torch.nn.ConvTranspose1d(128, 64, 7, stride=3, padding=2),
            )
        ),
    )

    assert module._hift_resblock_stage_shape(hift, mel_width=58, stage=0) == (1, 256, 464)
    assert module._hift_resblock_stage_shape(hift, mel_width=58, stage=1) == (1, 128, 2320)
    assert module._hift_resblock_stage_shape(hift, mel_width=58, stage=2) == (1, 64, 6961)


def test_hift_fixed_istft_matches_torch_istft(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    width = 17
    window = torch.hann_window(16, periodic=True)
    constants = module._hift_fixed_istft_constants(
        width=width,
        window=window,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    raw = torch.linspace(-2.0, 2.0, steps=18 * width).reshape(1, 18, width)
    magnitude = torch.exp(raw[:, :9, :])
    phase = torch.sin(raw[:, 9:, :])
    real = torch.clip(magnitude, max=1e2) * torch.cos(phase)
    imag = torch.clip(magnitude, max=1e2) * torch.sin(phase)

    expected = torch.istft(torch.complex(real, imag), 16, 4, 16, window=window)
    actual = module._hift_fixed_istft(magnitude, phase, *constants)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_hift_stft_window_placement_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    hift = SimpleNamespace(stft_window=torch.arange(16, dtype=torch.float32))
    target = torch.device("meta")

    assert module._place_hift_stft_window(hift, target) is True
    assert hift.stft_window.device == target
    assert module._place_hift_stft_window(hift, target) is False


def test_hift_stft_window_placement_rejects_missing_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    with pytest.raises(TypeError, match="stft_window tensor"):
        module._place_hift_stft_window(SimpleNamespace(stft_window=None), torch.device("cpu"))


def test_hift_resident_harmonics_is_exact_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSineGen:
        harmonic_num = 3
        sine_amp = 0.1
        noise_std = 0.003

        def _f02sine(self, value):
            return torch.sin(value)

        def _f02uv(self, value):
            return (value > 0).to(torch.float32)

        def forward(self, f0):
            harmonics = torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
            return module._sinegen_forward_with_harmonics(self, f0, harmonics)

    sine_gen = FakeSineGen()
    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=sine_gen))
    f0 = torch.linspace(0, 440, steps=48).reshape(1, 12, 4)[:, :, :1]

    torch.manual_seed(7)
    expected = sine_gen.forward(f0)
    assert module.prepare_hift_resident_harmonics_for_npu(hift, torch.device("cpu"))
    patched_forward = sine_gen.forward
    assert not module.prepare_hift_resident_harmonics_for_npu(hift, torch.device("cpu"))
    assert sine_gen.forward is patched_forward
    torch.manual_seed(7)
    actual = sine_gen.forward(f0)

    assert sine_gen._step_audio2_npu_harmonics.shape == (1, 1, 4)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_hift_resident_harmonics_falls_back_for_other_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    calls: list[torch.dtype] = []

    class FakeSineGen:
        harmonic_num = 1

        def forward(self, f0):
            calls.append(f0.dtype)
            return (f0, f0, f0)

    sine_gen = FakeSineGen()
    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=sine_gen))
    module.prepare_hift_resident_harmonics_for_npu(hift, torch.device("cpu"))
    f0 = torch.ones(1, 8, 1, dtype=torch.float64)

    actual = sine_gen.forward(f0)

    assert calls == [torch.float64]
    assert all(actual_tensor is f0 for actual_tensor in actual)


def test_hift_source_noise_scratch_is_exact_reuses_storage_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSineGen(torch.nn.Module):
        def forward(self, value):
            uv = (value > 0).to(value.dtype)
            return torch.sin(value), uv, torch.zeros_like(value)

    class FakeSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l_sin_gen = FakeSineGen()
            self.l_linear = torch.nn.Linear(1, 1)
            self.l_tanh = torch.nn.Tanh()
            self.sine_amp = 0.1

        def forward(self, value):
            with torch.no_grad():
                sine_wavs, uv, _ = self.l_sin_gen(value)
            sine_merge = self.l_tanh(self.l_linear(sine_wavs))
            noise = torch.randn_like(uv) * self.sine_amp / 3
            return sine_merge, noise, uv

    source = FakeSource().eval()
    hift = SimpleNamespace(m_source=source)
    value = torch.linspace(-1, 1, steps=24).reshape(1, 24, 1)

    torch.manual_seed(17)
    expected = source(value)
    expected_next = torch.randn(7)

    assert module.prepare_hift_source_noise_scratch_for_npu(hift)
    patched_forward = source.forward
    assert not module.prepare_hift_source_noise_scratch_for_npu(hift)
    assert source.forward is patched_forward

    torch.manual_seed(17)
    actual = source(value)
    actual_next = torch.randn(7)
    first_pointer = actual[1].data_ptr()
    actual_noise = actual[1].clone()
    second = source(value)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_noise, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)
    assert second[1].data_ptr() == first_pointer
    assert len(source._step_audio2_npu_source_noise_scratch) == 1


def test_hift_source_noise_scratch_keeps_separate_shape_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSource:
        sine_amp = 0.1
        l_sin_gen = staticmethod(lambda value: (value, torch.ones_like(value), value))
        l_linear = staticmethod(lambda value: value)
        l_tanh = staticmethod(torch.tanh)

        def forward(self, value):
            return value, torch.randn_like(value) * self.sine_amp / 3, value

    source = FakeSource()
    module.prepare_hift_source_noise_scratch_for_npu(SimpleNamespace(m_source=source))

    first = source.forward(torch.ones(1, 8, 1))[1]
    second = source.forward(torch.ones(1, 12, 1))[1]
    third = source.forward(torch.ones(1, 8, 1))[1]

    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() == third.data_ptr()
    assert len(source._step_audio2_npu_source_noise_scratch) == 2


@pytest.mark.parametrize(("width", "window_size"), [(1, 16), (17, 15)])
def test_hift_fixed_istft_constants_reject_invalid_layout(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    window_size: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    with pytest.raises(ValueError):
        module._hift_fixed_istft_constants(
            width=width,
            window=torch.ones(window_size),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_hift_fixed_istft_uses_original_off_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    hift = SimpleNamespace(
        _step_audio2_original_istft=lambda magnitude, phase: calls.append((magnitude, phase))
        or magnitude[:, 0],
    )
    magnitude = torch.ones(1, 9, 17)
    phase = torch.zeros_like(magnitude)

    actual = module._istft_with_npu_graph(hift, magnitude, phase)

    assert calls == [(magnitude, phase)]
    torch.testing.assert_close(actual, magnitude[:, 0])


def test_hift_resblock_graph_uses_eager_fallback_off_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    calls: list[str] = []

    block = SimpleNamespace(
        _step_audio2_npu_resblock_graph_shape=(1, 4, 8),
        _step_audio2_npu_resblock_graph_disabled=False,
        _step_audio2_npu_resblock_graph_replayed=False,
        _step_audio2_original_forward=lambda value: calls.append("eager") or value + 1,
        _step_audio2_npu_resblock_graph=lambda _value: (_ for _ in ()).throw(
            AssertionError("CPU must not enter the NPU graph")
        ),
    )
    value = torch.zeros(1, 4, 8)

    output = module._resblock_with_npu_graph(block, value)

    assert calls == ["eager"]
    torch.testing.assert_close(output, value + 1)


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("2", 2)])
def test_hift_resblock_graph_stage_env(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_RESBLOCK_GRAPH_STAGE_ENV, value)
    assert module._hift_resblock_graph_stage() == expected


@pytest.mark.parametrize("value", ["-1", "bad"])
def test_hift_resblock_graph_stage_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_RESBLOCK_GRAPH_STAGE_ENV, value)
    with pytest.raises(ValueError):
        module._hift_resblock_graph_stage()


@pytest.mark.parametrize(("value", "expected"), [("58", 58), ("1", 1)])
def test_hift_resblock_graph_mel_width_env(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_RESBLOCK_GRAPH_MEL_WIDTH_ENV, value)
    assert module._positive_int_env(module._HIFT_RESBLOCK_GRAPH_MEL_WIDTH_ENV, 58) == expected


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_hift_resblock_graph_mel_width_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_RESBLOCK_GRAPH_MEL_WIDTH_ENV, value)
    with pytest.raises(ValueError):
        module._positive_int_env(module._HIFT_RESBLOCK_GRAPH_MEL_WIDTH_ENV, 58)


def test_hift_f0_feature_partition_matches_sequential_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    layers: list[torch.nn.Module] = []
    channels = (3, 4, 4, 4, 4, 4)
    for in_channels, out_channels in zip(channels, channels[1:]):
        layers.extend(
            (
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.ELU(),
            )
        )
    condnet = torch.nn.Sequential(*layers).eval()
    convolutions = [layer for layer in condnet if isinstance(layer, torch.nn.Conv1d)]
    value = torch.randn(1, 3, 11)

    expected = condnet(value)
    actual = module._hift_f0_features(
        value,
        *(tensor for layer in convolutions for tensor in (layer.weight, layer.bias)),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("value", "expected"), [("58", 58), ("1", 1)])
def test_hift_f0_graph_width_env(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_F0_GRAPH_WIDTH_ENV, value)
    assert module._hift_f0_graph_width() == expected


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_hift_f0_graph_width_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_F0_GRAPH_WIDTH_ENV, value)
    with pytest.raises(ValueError):
        module._hift_f0_graph_width()


def test_hift_f0_graph_buckets_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_F0_GRAPH_BUCKETS_ENV, "50,58,50")
    assert module._hift_f0_graph_buckets() == (50, 58)


@pytest.mark.parametrize("value", ["0", "-1", "bad", "50,bad"])
def test_hift_f0_graph_buckets_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_F0_GRAPH_BUCKETS_ENV, value)
    with pytest.raises(ValueError):
        module._hift_f0_graph_buckets()


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("yes", True), ("0", False)])
def test_hift_weight_norm_materialization_env_flag(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    module = _load_patch_module(monkeypatch)
    monkeypatch.setenv(module._HIFT_MATERIALIZE_WEIGHT_NORM_ENV, value)
    assert module._env_flag_enabled(module._HIFT_MATERIALIZE_WEIGHT_NORM_ENV) is expected
