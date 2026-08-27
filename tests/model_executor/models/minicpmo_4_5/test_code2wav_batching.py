import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    _dit_attention_cache_length,
    _dit_attention_preamble,
    _dit_attention_preamble_bsh_from_modulation,
    _dit_attention_preamble_from_modulation,
    _dit_attention_preamble_fused_qkv_from_modulation,
    _dit_attention_preamble_qkv_pack,
    _dit_cache_major_conv_dynamic_w8a8_mlp_residual,
    _dit_cache_major_conv_mlp_residual,
    _dit_cache_major_post_attention_conv_mlp_residual,
    _dit_conv_mlp_residual,
    _dit_explicit_attention,
    _dit_final_cfg_euler_from_modulation,
    _dit_final_from_modulation,
    _dit_final_from_modulation_addcmul,
    _dit_flat_capture_conv_mlp_partition,
    _dit_fused_conv_block_mlp_residual,
    _dit_fused_conv_dynamic_w8a8_mlp_residual,
    _dit_fused_conv_linear_mlp_residual,
    _dit_fused_conv_mlp_residual,
    _dit_fused_full_block,
    _dit_fused_full_block_bsh_from_modulation,
    _dit_full_block_bsh_standard_conv_from_modulation,
    _dit_mlp_residual,
    _dit_wide_adaln_steps,
    _dit_wide_adaln_steps_with_final,
    _npu_cfm_fixed_kv_slabs_enabled,
    _npu_cfm_cache_fill_graph_enabled,
    _npu_cfm_cache_fill_graph_lengths,
    _npu_cfm_graph_enabled,
    _npu_cfm_graph_phase,
    _npu_cfm_integration_dtype,
    _npu_initial_cfm_timesteps,
    _npu_prompt_cfm_timesteps,
    _npu_prompt_cache_max_frames,
    _npu_cfm_planar_kv_slabs_enabled,
    _npu_cfm_stacked_cache_out_enabled,
    _npu_dit_attn_cache_out_enabled,
    _npu_dit_bsh_attention_enabled,
    _npu_dit_cache_major_enabled,
    _npu_dit_compute_dtype,
    _npu_dit_conv_mlp_graph_enabled,
    _npu_dit_final_addcmul_enabled,
    _npu_dit_full_block_cache_buckets,
    _npu_dit_full_block_graph_enabled,
    _npu_dit_full_stack_graph_enabled,
    _npu_dit_fused_conv_block_enabled,
    _npu_dit_fused_conv_linear_enabled,
    _npu_dit_fused_conv_pack_enabled,
    _npu_dit_fused_final_adaln_enabled,
    _npu_dit_fused_qkv_enabled,
    _npu_dit_graph_buckets,
    _npu_dit_last_block_final_euler_graph_enabled,
    _npu_dit_last_block_final_euler_perf_qualifies,
    _npu_matmul_hf32_enabled,
    _npu_dit_mlp_graph_enabled,
    _npu_dit_mlp_graph_width,
    _npu_dit_post_attn_graph_enabled,
    _npu_dit_preamble_graph_enabled,
    _npu_dit_prompt_conv_mlp_graph_enabled,
    _npu_dit_qkv_pack_enabled,
    _npu_dit_wide_adaln_enabled,
    _npu_dit_wide_final_adaln_enabled,
    _npu_single_request_cache_passthrough_enabled,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _npu_optimized_defaults_enabled,
    _resolve_token2wav_n_timesteps,
    _with_npu_optimized_defaults,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_flat_capture_forces_explicit_attention_only_for_bsh_cache():
    """Raw NPUGraph capture must not enter the fused-attention aux stream."""
    source = textwrap.dedent(
        inspect.getsource(
            BatchedToken2Wav._estimator_blocks_forward_chunk_mlp_graph
        )
    )
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    bsh_calls = [
        call
        for call in calls
        if call.func.attr == "_attention_from_projected_qkv_bsh_planar"
    ]
    planar_calls = [
        call
        for call in calls
        if call.func.attr == "_attention_from_projected_qkv_planar"
    ]

    assert len(bsh_calls) == 1
    explicit = {
        keyword.arg: keyword.value for keyword in bsh_calls[0].keywords
    }["explicit_attention"]
    assert isinstance(explicit, ast.Name)
    assert explicit.id == "flat_capture"
    assert len(planar_calls) == 1
    assert all(
        keyword.arg != "explicit_attention"
        for keyword in planar_calls[0].keywords
    )


@pytest.mark.parametrize(
    ("steady", "width", "cache_length", "has_outputs", "expected"),
    [
        (False, 50, 302, True, "cache-fill"),
        (True, 50, 402, True, "steady"),
        (False, 50, 352, True, None),
        (False, 44, 302, True, None),
        (False, 50, None, False, None),
    ],
)
def test_fixed_slab_cfm_graph_phase_is_shape_strict(
    steady: bool,
    width: int,
    cache_length: int | None,
    has_outputs: bool,
    expected: str | None,
):
    assert _npu_cfm_graph_phase(
        fixed_kv_slabs=True,
        cache_fill_lengths=(302,),
        steady_graph=steady,
        width=width,
        cache_length=cache_length,
        has_cache_outputs=has_outputs,
    ) == expected


def test_fixed_kv_slabs_reserve_one_configured_steady_append():
    attention = torch.zeros((6, 16, 2, 2, 302, 512))
    cnn = torch.zeros((6, 16, 2, 1024, 2))

    slabs = BatchedToken2Wav._make_fixed_estimator_kv_slabs(
        attention,
        cnn,
        302,
        steady_width=100,
    )

    assert slabs is not None
    assert slabs.retained.shape[-2] == 402
    assert slabs.append.shape[-2] == 502


def test_dit_conv_mlp_admits_static_width_100():
    conv_layers = nn.ModuleList([nn.Identity() for _ in range(7)])
    conv_layers[1] = nn.Conv1d(512, 512, kernel_size=3)
    conv_layers[6] = nn.Conv1d(512, 512, kernel_size=3)
    block = SimpleNamespace(
        training=False,
        conv=SimpleNamespace(block=conv_layers),
        norm2=nn.LayerNorm(512, eps=1e-6, elementwise_affine=False),
        mlp=SimpleNamespace(
            fc1=nn.Linear(512, 2048),
            fc2=nn.Linear(2048, 512),
        ),
    )

    assert BatchedToken2Wav._dit_conv_mlp_compatible(block, 100)


def test_cfm_graph_slot_count_is_bounded(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_SLOTS", raising=False)
    assert BatchedToken2Wav._npu_cfm_graph_slot_count() == 2

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_SLOTS", "1")
    assert BatchedToken2Wav._npu_cfm_graph_slot_count() == 1

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_SLOTS", "99")
    assert BatchedToken2Wav._npu_cfm_graph_slot_count() == 2


def test_cache_fill_cfm_graph_is_opt_in(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS",
        raising=False,
    )
    assert not _npu_cfm_cache_fill_graph_enabled()
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH", "true"
    )
    assert _npu_cfm_cache_fill_graph_enabled()
    assert _npu_cfm_cache_fill_graph_lengths() == (302,)
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS",
        "302,352,302",
    )
    assert _npu_cfm_cache_fill_graph_lengths() == (302, 352)


def test_flat_capture_preserves_cache_major_conv_partition():
    assert (
        _dit_flat_capture_conv_mlp_partition(
            fused_conv_pack=True,
            cache_major=True,
            post_attention=False,
        )
        is _dit_cache_major_conv_mlp_residual
    )
    assert (
        _dit_flat_capture_conv_mlp_partition(
            fused_conv_pack=True,
            cache_major=True,
            post_attention=True,
        )
        is _dit_cache_major_post_attention_conv_mlp_residual
    )
    assert (
        _dit_flat_capture_conv_mlp_partition(
            fused_conv_pack=True,
            cache_major=False,
            post_attention=False,
        )
        is _dit_fused_conv_mlp_residual
    )
    assert (
        _dit_flat_capture_conv_mlp_partition(
            fused_conv_pack=False,
            cache_major=True,
            post_attention=False,
        )
        is None
    )


def test_flat_capture_embeds_dynamic_w8a8_without_changing_cache_abi():
    assert (
        _dit_flat_capture_conv_mlp_partition(
            fused_conv_pack=True,
            cache_major=False,
            post_attention=False,
            dynamic_w8a8=True,
        )
        is _dit_fused_conv_dynamic_w8a8_mlp_residual
    )
    assert (
        _dit_flat_capture_conv_mlp_partition(
            fused_conv_pack=True,
            cache_major=True,
            post_attention=False,
            dynamic_w8a8=True,
        )
        is _dit_cache_major_conv_dynamic_w8a8_mlp_residual
    )


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.last_chunk_calls: list[bool] = []

    def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
        batch, length, _ = xs.shape
        self.calls.append(batch)
        self.last_chunk_calls.append(last_chunk)
        old_length = 0 if att_cache is None else att_cache.shape[3]
        output = xs[:, : max(1, length - 1)]
        cnn = xs[:, :1, :].transpose(1, 2).contiguous()
        marker = xs[:, 0, 0].reshape(1, batch, 1, 1, 1)
        att = marker.expand(1, batch, 1, old_length + output.shape[1], 1).clone()
        return output, cnn, att


class _FakeBlock:
    def __init__(self):
        conv1 = SimpleNamespace(causal_padding=(1, 0))
        self.conv = SimpleNamespace(
            in_channels=1,
            out_channels=1,
            block=[None, conv1],
        )
        self.attn = SimpleNamespace(num_heads=1, head_dim=1)


class _FakeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_FakeBlock()]
        self.cfg_batches: list[int] = []
        self.speaker_order: list[list[float]] = []

    def t_embedder(self, time):
        return time[:, None]

    def blocks_forward_chunk(
        self,
        inputs,
        time,
        mask,
        cnn_cache,
        att_cache,
        cnn_out,
        att_out,
    ):
        del time, mask, cnn_cache, att_cache
        self.cfg_batches.append(inputs.shape[0])
        self.speaker_order.append(inputs[:, 2, 0].tolist())
        marker = inputs[:, 1, 0]
        cnn_out.copy_(marker.reshape(1, -1, 1, 1).expand_as(cnn_out))
        att_out.copy_(marker.reshape(1, -1, 1, 1, 1).expand_as(att_out))
        return inputs[:, 1:2]


class _CosyVoiceStyleTimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frequency_embedding_size = 4
        self.scale = 1000
        self.mlp = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 3))
        self.calls = 0

    def forward(self, time):
        self.calls += 1
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -torch.log(time.new_tensor(10000.0))
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / half
        )
        arguments = (time * self.scale)[:, None] * frequencies[None]
        return self.mlp(torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1))


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _FakeEstimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.zeros(1, 1, 100), persistent=False)


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.encoder_proj = nn.Identity()
        self.decoder = _FakeDecoder()
        self.spk_embed_affine_layer = _CountingSpeakerProjection()

    def input_embedding(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1)


class _CountingSpeakerProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class _FakeHiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def forward(self, mel, source):
        del source
        self.calls.append(mel.shape[0])
        speech = mel[:, 0].repeat_interleave(3, dim=1)
        generated_source = speech[:, None]
        return speech, generated_source


class _FakeToken2Wav:
    def __init__(self):
        self.flow = _FakeFlow()
        self.hift = _FakeHiFT()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.hamming_window(4, periodic=False)
        self.prompt_calls = 0

    def _prepare_prompt(self, prompt_wav):
        del prompt_wav
        self.prompt_calls += 1
        return (
            torch.tensor([[5, 6]], dtype=torch.long),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1, 1),
            torch.ones(1, 4, 1),
            torch.tensor([4], dtype=torch.int32),
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("sequential stream fallback must never be called")

    def __call__(self, *args, **kwargs):
        raise AssertionError("sequential __call__ fallback must never be called")


def _config(minimum: int = 1):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config={
                "extra": {
                    "code2wav_min_batch_size": minimum,
                    "prompt_cache_id": "shared",
                    "prompt_wav": "/fake/prompt.wav",
                }
            },
        )
    )


def _model():
    token2wav = _FakeToken2Wav()
    backend = BatchedToken2Wav(token2wav)
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    model.backend = backend
    return model, token2wav


def test_token2wav_step_count_defaults_to_checkpoint_quality() -> None:
    assert _resolve_token2wav_n_timesteps({}) == 10
    assert _resolve_token2wav_n_timesteps({"token2wav_n_timesteps": 8}) == 8


def test_initial_cfm_solver_is_opt_in_and_range_checked(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS",
        raising=False,
    )
    assert _npu_initial_cfm_timesteps(6) == 6

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS",
        "4",
    )
    assert _npu_initial_cfm_timesteps(6) == 4

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS",
        "7",
    )
    with pytest.raises(ValueError, match=r"integer in \[1, 6\]"):
        _npu_initial_cfm_timesteps(6)


def test_prompt_cfm_solver_is_opt_in_and_range_checked(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS",
        raising=False,
    )
    assert _npu_prompt_cfm_timesteps(6) == 6

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS",
        "2",
    )
    assert _npu_prompt_cfm_timesteps(6) == 2

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS",
        "0",
    )
    with pytest.raises(ValueError, match=r"integer in \[1, 6\]"):
        _npu_prompt_cfm_timesteps(6)


def test_prompt_cache_limit_is_opt_in_and_range_checked(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CACHE_MAX_FRAMES",
        raising=False,
    )
    assert _npu_prompt_cache_max_frames() is None

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CACHE_MAX_FRAMES",
        "150",
    )
    assert _npu_prompt_cache_max_frames() == 150

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CACHE_MAX_FRAMES",
        "0",
    )
    with pytest.raises(ValueError, match="positive integer"):
        _npu_prompt_cache_max_frames()


def test_prompt_cache_limit_preserves_full_conformer_state(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CACHE_MAX_FRAMES",
        "2",
    )
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    assert states[0].flow_cache["estimator_att_cache"].shape[-2] == 2
    assert states[0].flow_cache["conformer_att_cache"].shape[3] == 4
    assert prompt.mels.shape[1] == 4


def test_reduced_prompt_solver_restores_full_cache_abi(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS",
        "1",
    )
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    assert len(token2wav.flow.decoder.estimator.cfg_batches) == 1
    assert states[0].flow_cache["estimator_att_cache"].shape[0] == 2
    assert states[0].flow_cache["estimator_cnn_cache"].shape[0] == 2


def test_reduced_first_packet_solver_restores_full_cache_abi(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS",
        "1",
    )
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)
    calls_after_prompt = len(token2wav.flow.decoder.estimator.cfg_batches)

    _, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )
    assert len(token2wav.flow.decoder.estimator.cfg_batches) == (
        calls_after_prompt + 1
    )
    assert states[0].flow_cache["estimator_att_cache"].shape[0] == 2
    assert states[0].flow_cache["estimator_cnn_cache"].shape[0] == 2

    adapter.decode_batch(
        torch.tensor([[12, 13]]),
        prompt,
        states,
        last_chunk=False,
    )
    assert len(token2wav.flow.decoder.estimator.cfg_batches) == (
        calls_after_prompt + 3
    )


def test_npu_optimized_defaults_activate_only_qualified_cfm6(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS",
        raising=False,
    )
    resolved = _with_npu_optimized_defaults({}, is_npu=True)

    assert resolved == {"token2wav_n_timesteps": 6}


def test_npu_dual_chip_planar_defaults_exclude_rejected_deep_paths(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS",
        raising=False,
    )
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_PLANAR_DEFAULTS",
        "1",
    )
    resolved = _with_npu_optimized_defaults({}, is_npu=True)

    assert resolved["token2wav_n_timesteps"] == 6
    assert resolved["npu_dit_compute_dtype"] == "bf16"
    assert resolved["npu_cfm_integration_dtype"] == "bf16"
    assert resolved["npu_cfm_fixed_kv_slabs"] is True
    assert resolved["npu_cfm_planar_kv_slabs"] is True
    assert resolved["npu_dit_mlp_graph_width"] == 50
    assert resolved["npu_dit_graph_buckets"] == [20, 302]
    assert "npu_dit_bsh_attention" not in resolved
    assert "npu_cfm_graph" not in resolved


def test_npu_aggressive_experiments_are_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS",
        raising=False,
    )
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_AGGRESSIVE_EXPERIMENTS",
        "1",
    )
    resolved = _with_npu_optimized_defaults({}, is_npu=True)

    assert resolved["token2wav_n_timesteps"] == 6
    assert resolved["npu_dit_compute_dtype"] == "bf16"
    assert resolved["npu_cfm_integration_dtype"] == "bf16"
    assert resolved["npu_cfm_fixed_kv_slabs"] is True
    assert resolved["npu_cfm_planar_kv_slabs"] is True
    assert resolved["npu_dit_bsh_attention"] is True
    assert resolved["npu_dit_fused_conv_pack"] is True


def test_npu_optimized_defaults_preserve_explicit_config_and_cpu(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS",
        raising=False,
    )
    explicit = {
        "token2wav_n_timesteps": 8,
        "npu_dit_compute_dtype": "fp32",
        "npu_dit_bsh_attention": False,
    }
    resolved = _with_npu_optimized_defaults(explicit, is_npu=True)
    assert resolved["token2wav_n_timesteps"] == 8
    assert resolved["npu_dit_compute_dtype"] == "fp32"
    assert resolved["npu_dit_bsh_attention"] is False
    assert _with_npu_optimized_defaults({}, is_npu=False) == {}


def test_npu_optimized_defaults_have_master_opt_out(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS",
        "0",
    )
    assert _npu_optimized_defaults_enabled(is_npu=True) is False
    assert _with_npu_optimized_defaults({}, is_npu=True) == {}


def test_npu_optimized_defaults_reject_invalid_master_switch(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS",
        "sometimes",
    )
    with pytest.raises(ValueError, match="NPU_OPTIMIZED_DEFAULTS"):
        _npu_optimized_defaults_enabled(is_npu=True)


def test_npu_dit_compute_dtype_config_and_environment(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_COMPUTE_DTYPE",
        raising=False,
    )
    assert _npu_dit_compute_dtype() == torch.float32
    assert _npu_dit_compute_dtype("bf16") == torch.bfloat16
    assert _npu_dit_compute_dtype("torch.bfloat16") == torch.bfloat16

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_COMPUTE_DTYPE",
        "fp32",
    )
    assert _npu_dit_compute_dtype("bf16") == torch.float32

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_COMPUTE_DTYPE",
        "fp16",
    )
    with pytest.raises(ValueError, match="NPU_DIT_COMPUTE_DTYPE"):
        _npu_dit_compute_dtype()


def test_npu_matmul_hf32_config_and_environment(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_MATMUL_HF32",
        raising=False,
    )
    assert _npu_matmul_hf32_enabled() is False
    assert _npu_matmul_hf32_enabled(True) is True
    assert _npu_matmul_hf32_enabled("on") is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_MATMUL_HF32",
        "0",
    )
    assert _npu_matmul_hf32_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_MATMUL_HF32",
        "sometimes",
    )
    with pytest.raises(ValueError, match="NPU_MATMUL_HF32"):
        _npu_matmul_hf32_enabled()


def test_npu_cfm_integration_dtype_config_and_environment(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_INTEGRATION_DTYPE",
        raising=False,
    )
    assert _npu_cfm_integration_dtype() == torch.float32
    assert _npu_cfm_integration_dtype("bf16") == torch.bfloat16

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_INTEGRATION_DTYPE",
        "fp32",
    )
    assert _npu_cfm_integration_dtype("bf16") == torch.float32

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_INTEGRATION_DTYPE",
        "fp16",
    )
    with pytest.raises(ValueError, match="NPU_CFM_INTEGRATION_DTYPE"):
        _npu_cfm_integration_dtype()


def test_selective_bf16_dit_keeps_integration_and_hift_fp32() -> None:
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    # CPU tests cannot activate the NPU-only constructor branch. Exercise the
    # dtype boundary directly with the same state used after a successful NPU
    # estimator conversion.
    adapter._npu_dit_compute_dtype = torch.bfloat16
    adapter._npu_cfm_integration_dtype = torch.float32
    adapter._npu_dit_mixed_precision_enabled = True

    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)
    state = states[0]
    assert state.flow_cache["conformer_cnn_cache"].dtype == torch.float32
    assert state.flow_cache["estimator_cnn_cache"].dtype == torch.bfloat16
    assert state.flow_cache["estimator_att_cache"].dtype == torch.bfloat16
    assert state.hift_cache["mel"].dtype == torch.float32

    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )
    assert audios[0].dtype == torch.float32
    assert states[0].flow_cache["estimator_cnn_cache"].dtype == torch.bfloat16
    assert states[0].hift_cache["speech"].dtype == torch.float32


def test_homogeneous_bf16_cfm_casts_once_before_hift() -> None:
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    adapter.flow.decoder.estimator.to(dtype=torch.bfloat16)
    adapter._npu_dit_compute_dtype = torch.bfloat16
    adapter._npu_cfm_integration_dtype = torch.bfloat16
    adapter._npu_dit_mixed_precision_enabled = True

    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)
    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert audios[0].dtype == torch.float32
    assert states[0].flow_cache["estimator_cnn_cache"].dtype == torch.bfloat16
    assert states[0].flow_cache["estimator_att_cache"].dtype == torch.bfloat16
    assert states[0].hift_cache["mel"].dtype == torch.float32


def test_token2wav_step_count_environment_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS", "6")
    assert _resolve_token2wav_n_timesteps({"token2wav_n_timesteps": 8}) == 6


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_invalid_token2wav_step_count_is_rejected(monkeypatch, value: str) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS", value)
    with pytest.raises(ValueError, match="MiniCPM-o Code2Wav"):
        _resolve_token2wav_n_timesteps({})


def _info(
    request_id: str,
    chunk_seq: int,
    codes: list[int],
    *,
    last_chunk: bool = False,
    cache_epoch: int = 0,
):
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long)},
        "meta": {
            "request_id": request_id,
            "chunk_seq": chunk_seq,
            "cache_epoch": cache_epoch,
            "last_chunk": last_chunk,
            "prompt_cache_id": "shared",
        },
    }


def _forward(model, infos, placeholder_counts=None, request_ids=None):
    placeholder_counts = placeholder_counts or [1] * len(infos)
    input_ids = torch.zeros(sum(placeholder_counts), dtype=torch.long)
    return model(
        input_ids=input_ids,
        seq_token_counts=placeholder_counts,
        runtime_additional_information=infos,
        request_ids=request_ids,
    )


def test_adapter_runs_true_batch_cfg_and_splits_request_caches():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.prompt_calls == 1
    assert token2wav.flow.spk_embed_affine_layer.calls == 1
    assert token2wav.flow.encoder.calls == [2, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [4, 4, 4, 4]
    assert all(order == [1.0, 1.0, 0.0, 0.0] for order in token2wav.flow.decoder.estimator.speaker_order)
    assert token2wav.hift.calls == [2]
    assert len(audios) == 2
    cache0 = states[0].flow_cache["estimator_cnn_cache"]
    cache1 = states[1].flow_cache["estimator_cnn_cache"]
    assert cache0.data_ptr() != cache1.data_ptr()
    assert cache0[0, 0, 0, 0, 0].item() == 10
    assert cache1[0, 0, 0, 0, 0].item() == 20


def test_adapter_caches_projected_speaker_for_all_stream_chunks():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    expected = F.normalize(prompt.speaker_embedding, dim=1)

    assert torch.equal(prompt.projected_speaker_embedding, expected)
    assert token2wav.flow.spk_embed_affine_layer.calls == 1

    states = adapter.setup_batch(prompt, 1)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=False,
    )
    adapter.decode_batch(
        torch.tensor([[12, 13]]),
        prompt,
        states,
        last_chunk=True,
    )

    assert token2wav.flow.spk_embed_affine_layer.calls == 1
    assert adapter.prepare_prompt("shared", "/fake/prompt.wav") is prompt
    assert token2wav.flow.spk_embed_affine_layer.calls == 1


def test_adapter_reuses_timeline_and_cfg_workspaces():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    timeline = adapter._timeline_for(value)
    assert adapter._timeline_for(value) is timeline

    duplicated = adapter._cfg_pair("test", value, zero_unconditional=False)
    duplicated_ptr = duplicated.data_ptr()
    torch.testing.assert_close(duplicated[:2], value)
    torch.testing.assert_close(duplicated[2:], value)

    zeroed = adapter._cfg_pair("test", value + 1, zero_unconditional=True)
    assert zeroed.data_ptr() == duplicated_ptr
    torch.testing.assert_close(zeroed[:2], value + 1)
    torch.testing.assert_close(zeroed[2:], torch.zeros_like(value))


def test_adapter_caches_cfm_deltas_with_original_recurrence():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    timeline = adapter._timeline_for(torch.zeros(1, dtype=torch.float32))

    time = timeline[0]
    dt = timeline[1] - timeline[0]
    expected = []
    for step in range(adapter.n_timesteps):
        expected.append(dt)
        time = time + dt
        if step + 1 < adapter.n_timesteps:
            dt = timeline[step + 2] - time

    actual = adapter._cfm_deltas_for(timeline)
    cached = adapter._cfm_deltas_for(timeline)

    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    assert cached is actual
    torch.testing.assert_close(actual.sum(), timeline[-1] - timeline[0])


def test_adapter_caches_cosyvoice_timestep_embeddings_without_forward_calls():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    estimator = SimpleNamespace(t_embedder=_CosyVoiceStyleTimestepEmbedder())
    timeline = adapter._timeline_for(torch.zeros(1, dtype=torch.float32))

    expected = torch.stack(
        [estimator.t_embedder(timeline[step].expand(2)).unsqueeze(1) for step in range(adapter.n_timesteps)]
    )
    calls_before_cache = estimator.t_embedder.calls
    actual = adapter._estimator_time_embeddings(estimator, timeline, 2)
    cached = adapter._estimator_time_embeddings(estimator, timeline, 2)

    torch.testing.assert_close(actual, expected)
    assert cached is actual
    assert estimator.t_embedder.calls == calls_before_cache


def test_dit_mlp_residual_matches_eager_block_math():
    torch.manual_seed(7)
    x = torch.randn(2, 5, 4)
    shift = torch.randn(2, 1, 4)
    scale = torch.randn(2, 1, 4)
    gate = torch.randn(2, 1, 4)
    fc1 = nn.Linear(4, 12)
    fc2 = nn.Linear(12, 4)

    normalized = F.layer_norm(x, (4,), eps=1e-6)
    expected = x + gate * fc2(F.gelu(fc1(normalized * (1 + scale) + shift), approximate="tanh"))
    actual = _dit_mlp_residual(
        x,
        shift,
        scale,
        gate,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [20, 50, 302])
def test_dit_attention_preamble_matches_eager_block_math(width: int):
    torch.manual_seed(11)
    x = torch.randn(2, width, 512)
    time_embedding = torch.randn(2, 1, 512)
    adaln = nn.Linear(512, 9 * 512)
    to_q = nn.Linear(512, 512)
    to_k = nn.Linear(512, 512)
    to_v = nn.Linear(512, 512)
    q_norm = nn.LayerNorm(64)
    k_norm = nn.LayerNorm(64)

    modulation = adaln(F.silu(time_embedding))
    shift_msa, scale_msa = modulation.chunk(9, dim=-1)[:2]
    hidden = F.layer_norm(x, (512,), eps=1e-6) * (1 + scale_msa) + shift_msa
    q = q_norm(to_q(hidden).reshape(2, width, 8, 64).transpose(1, 2))
    k = k_norm(to_k(hidden).reshape(2, width, 8, 64).transpose(1, 2))
    v = to_v(hidden).reshape(2, width, 8, 64).transpose(1, 2)

    actual = _dit_attention_preamble(
        x,
        time_embedding,
        adaln.weight,
        adaln.bias,
        to_q.weight,
        to_q.bias,
        to_k.weight,
        to_k.bias,
        to_v.weight,
        to_v.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )

    for result, expected in zip(actual, (modulation, q, k, v), strict=True):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    supplied = _dit_attention_preamble_from_modulation(
        x,
        modulation,
        to_q.weight,
        to_q.bias,
        to_k.weight,
        to_k.bias,
        to_v.weight,
        to_v.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )
    for result, expected in zip(supplied, actual, strict=True):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_dit_attention_preamble_qkv_pack_matches_standard_partition(monkeypatch):
    torch.manual_seed(17)

    def qkv_pack(q, k, v):
        return tuple(value.reshape(2, 50, 8, 64).transpose(1, 2) for value in (q, k, v))

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_qkv_pack",
        qkv_pack,
        raising=False,
    )
    x = torch.randn(2, 50, 512)
    time_embedding = torch.randn(2, 1, 512)
    adaln = nn.Linear(512, 9 * 512)
    to_q = nn.Linear(512, 512)
    to_k = nn.Linear(512, 512)
    to_v = nn.Linear(512, 512)
    q_norm = nn.LayerNorm(64)
    k_norm = nn.LayerNorm(64)
    arguments = (
        x,
        time_embedding,
        adaln.weight,
        adaln.bias,
        to_q.weight,
        to_q.bias,
        to_k.weight,
        to_k.bias,
        to_v.weight,
        to_v.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )

    actual = _dit_attention_preamble_qkv_pack(*arguments)
    expected = _dit_attention_preamble(*arguments)

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("width", [20, 50, 302])
def test_dit_attention_preamble_fused_qkv_matches_standard_partition(width: int):
    torch.manual_seed(19)
    x = torch.randn(2, width, 512)
    modulation = torch.randn(2, 1, 9 * 512)
    to_q = nn.Linear(512, 512)
    to_k = nn.Linear(512, 512)
    to_v = nn.Linear(512, 512)
    q_norm = nn.LayerNorm(64)
    k_norm = nn.LayerNorm(64)
    qkv_weight = torch.cat((to_q.weight, to_k.weight, to_v.weight), dim=0)
    qkv_bias = torch.cat((to_q.bias, to_k.bias, to_v.bias), dim=0)

    actual = _dit_attention_preamble_fused_qkv_from_modulation(
        x,
        modulation,
        qkv_weight,
        qkv_bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )
    expected = _dit_attention_preamble_from_modulation(
        x,
        modulation,
        to_q.weight,
        to_q.bias,
        to_k.weight,
        to_k.bias,
        to_v.weight,
        to_v.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("width", [20, 50])
def test_dit_attention_preamble_bsh_matches_standard_partition(width: int):
    torch.manual_seed(23)
    x = torch.randn(2, width, 512)
    modulation = torch.randn(2, 1, 9 * 512)
    to_q = nn.Linear(512, 512)
    to_k = nn.Linear(512, 512)
    to_v = nn.Linear(512, 512)
    q_norm = nn.LayerNorm(64)
    k_norm = nn.LayerNorm(64)

    actual = _dit_attention_preamble_bsh_from_modulation(
        x,
        modulation,
        to_q.weight,
        to_q.bias,
        to_k.weight,
        to_k.bias,
        to_v.weight,
        to_v.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )
    expected = _dit_attention_preamble_from_modulation(
        x,
        modulation,
        to_q.weight,
        to_q.bias,
        to_k.weight,
        to_k.bias,
        to_v.weight,
        to_v.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for bsh, bhsd in zip(actual[1:], expected[1:], strict=True):
        reference = bhsd.transpose(1, 2).reshape(2, width, 512)
        torch.testing.assert_close(bsh, reference, rtol=0, atol=0)


def test_attention_from_projected_qkv_matches_cached_sdpa_math():
    torch.manual_seed(12)
    attention = SimpleNamespace(proj=nn.Linear(512, 512), proj_drop=nn.Identity())
    q = torch.randn(2, 8, 50, 64)
    k = torch.randn(2, 8, 50, 64)
    v = torch.randn(2, 8, 50, 64)
    cache = torch.randn(2, 8, 7, 128)
    cached_k, cached_v = cache.chunk(2, dim=3)
    full_k = torch.cat((k, cached_k), dim=2)
    full_v = torch.cat((v, cached_v), dim=2)
    expected = F.scaled_dot_product_attention(q, full_k, full_v)
    expected = attention.proj(expected.transpose(1, 2).reshape(2, 50, 512))

    actual, new_cache = BatchedToken2Wav._attention_from_projected_qkv(attention, q, k, v, cache)
    output_cache = torch.empty_like(new_cache)
    direct, direct_cache = BatchedToken2Wav._attention_from_projected_qkv(
        attention,
        q,
        k,
        v,
        cache,
        output_cache=output_cache,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(new_cache, torch.cat((full_k, full_v), dim=3), rtol=0, atol=0)
    torch.testing.assert_close(direct, expected, rtol=0, atol=0)
    torch.testing.assert_close(direct_cache, new_cache, rtol=0, atol=0)
    assert direct_cache is output_cache


def test_planar_attention_from_projected_qkv_matches_legacy_cache_math():
    torch.manual_seed(17)
    attention = SimpleNamespace(proj=nn.Linear(512, 512), proj_drop=nn.Identity())
    q = torch.randn(2, 8, 50, 64)
    k = torch.randn(2, 8, 50, 64)
    v = torch.randn(2, 8, 50, 64)
    legacy_cache = torch.randn(2, 8, 7, 128)
    cached_k, cached_v = legacy_cache.chunk(2, dim=-1)
    planar_cache = torch.stack((cached_k, cached_v), dim=0).contiguous()
    planar_output = torch.empty(2, 2, 8, 57, 64)

    expected, expected_cache = BatchedToken2Wav._attention_from_projected_qkv(
        attention, q, k, v, legacy_cache
    )
    actual, actual_cache = (
        BatchedToken2Wav._attention_from_projected_qkv_planar(
            attention, q, k, v, planar_cache, planar_output
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        BatchedToken2Wav._legacy_att_cache_from_planar(actual_cache),
        expected_cache,
        rtol=0,
        atol=0,
    )
    assert actual_cache is planar_output
    assert actual_cache.select(0, 0).is_contiguous()
    assert actual_cache.select(0, 1).is_contiguous()


def test_bsh_planar_attention_matches_legacy_cache_math():
    torch.manual_seed(29)
    attention = SimpleNamespace(
        num_heads=8,
        head_dim=64,
        proj=nn.Linear(512, 512),
        proj_drop=nn.Identity(),
    )
    q_bhsd = torch.randn(2, 8, 50, 64)
    k_bhsd = torch.randn(2, 8, 50, 64)
    v_bhsd = torch.randn(2, 8, 50, 64)
    legacy_cache = torch.randn(2, 8, 7, 128)
    cached_k, cached_v = legacy_cache.chunk(2, dim=-1)

    def to_bsh(value: torch.Tensor) -> torch.Tensor:
        return value.transpose(1, 2).reshape(2, value.shape[2], 512)

    bsh_cache = torch.stack((to_bsh(cached_k), to_bsh(cached_v)), dim=0)
    bsh_output = torch.empty(2, 2, 57, 512)
    expected, expected_cache = BatchedToken2Wav._attention_from_projected_qkv(
        attention, q_bhsd, k_bhsd, v_bhsd, legacy_cache
    )
    actual, actual_cache = (
        BatchedToken2Wav._attention_from_projected_qkv_bsh_planar(
            attention,
            to_bsh(q_bhsd),
            to_bsh(k_bhsd),
            to_bsh(v_bhsd),
            bsh_cache,
            bsh_output,
        )
    )
    explicit_output = torch.empty_like(bsh_output)
    explicit, explicit_cache = (
        BatchedToken2Wav._attention_from_projected_qkv_bsh_planar(
            attention,
            to_bsh(q_bhsd),
            to_bsh(k_bhsd),
            to_bsh(v_bhsd),
            bsh_cache,
            explicit_output,
            explicit_attention=True,
        )
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(explicit, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        BatchedToken2Wav._legacy_att_cache_from_bsh_planar(
            actual_cache, 8, 64
        ),
        expected_cache,
        rtol=0,
        atol=0,
    )
    assert actual_cache is bsh_output
    assert explicit_cache is explicit_output
    assert actual_cache.select(0, 0).is_contiguous()
    assert actual_cache.select(0, 1).is_contiguous()


def test_attention_from_projected_qkv_rejects_bad_output_cache_shape():
    attention = SimpleNamespace(proj=nn.Linear(512, 512), proj_drop=nn.Identity())
    q = torch.randn(2, 8, 50, 64)

    with pytest.raises(ValueError, match="output cache shape mismatch"):
        BatchedToken2Wav._attention_from_projected_qkv(
            attention,
            q,
            q,
            q,
            None,
            output_cache=torch.empty(2, 8, 49, 128),
        )


@pytest.mark.parametrize("width", [20, 50, 302])
def test_dit_conv_mlp_residual_matches_partition_math(width: int):
    torch.manual_seed(13)
    hidden = torch.randn(2, width, 512)
    conv_input = torch.randn(2, width, 512)
    cache = torch.randn(2, 1024, 2)
    gate_conv = torch.randn(2, 1, 512)
    shift_mlp = torch.randn(2, 1, 512)
    scale_mlp = torch.randn(2, 1, 512)
    gate_mlp = torch.randn(2, 1, 512)
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)

    cache1, cache2 = cache.split((512, 512), dim=1)
    first_input = torch.cat((cache1, conv_input.transpose(1, 2)), dim=2)
    convolution = conv1(first_input).transpose(1, 2)
    convolution = F.mish(conv_norm(convolution))
    second_input = torch.cat((cache2, convolution.transpose(1, 2)), dim=2)
    convolution = conv2(second_input).transpose(1, 2)
    expected_hidden = hidden + gate_conv * convolution
    expected_hidden = _dit_mlp_residual(
        expected_hidden,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    expected_cache = torch.cat((first_input[:, :, -2:], second_input[:, :, -2:]), dim=1)

    actual_hidden, actual_cache = _dit_conv_mlp_residual(
        hidden,
        conv_input,
        cache,
        gate_conv,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        conv1.weight,
        conv1.bias,
        conv_norm.weight,
        conv_norm.bias,
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )

    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [50, 64])
def test_dit_fused_conv_mlp_residual_matches_partition_math(monkeypatch, frames: int):
    torch.manual_seed(14)

    def causal_pack(x, cache):
        history = torch.cat((cache, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [
                history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1)
                for offset in range(frames)
            ],
            dim=1,
        ).reshape(2 * frames, 1536)
        return packed, x[:, -2:, :].transpose(1, 2).contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_pack",
        causal_pack,
        raising=False,
    )
    hidden = torch.randn(2, frames, 512)
    conv_input = torch.randn(2, frames, 512)
    cache = torch.randn(2, 1024, 2)
    modulations = [torch.randn(2, 1, 512) for _ in range(4)]
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)
    common = (*modulations, conv_norm.weight, conv_norm.bias)
    standard = _dit_conv_mlp_residual(
        hidden,
        conv_input,
        cache,
        *common[:4],
        conv1.weight,
        conv1.bias,
        *common[4:],
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    fused = _dit_fused_conv_mlp_residual(
        hidden,
        conv_input,
        cache,
        *common[:4],
        conv1.weight.permute(0, 2, 1).reshape(512, 1536),
        conv1.bias,
        *common[4:],
        conv2.weight.permute(0, 2, 1).reshape(512, 1536),
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    torch.testing.assert_close(fused[0], standard[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[1], standard[1], rtol=0, atol=0)


def test_dit_cache_major_conv_mlp_residual_matches_partition_math(monkeypatch):
    torch.manual_seed(15)

    def causal_pack(x, cache):
        channel_major = cache.transpose(1, 2)
        history = torch.cat((channel_major, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
            dim=1,
        ).reshape(100, 1536)
        return packed, x[:, -2:, :].contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_pack",
        causal_pack,
        raising=False,
    )
    hidden = torch.randn(2, 50, 512)
    conv_input = torch.randn(2, 50, 512)
    channel_major_cache = torch.randn(2, 1024, 2)
    cache_major = channel_major_cache.transpose(1, 2).contiguous()
    modulations = [torch.randn(2, 1, 512) for _ in range(4)]
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)
    common = (*modulations, conv_norm.weight, conv_norm.bias)
    standard = _dit_conv_mlp_residual(
        hidden,
        conv_input,
        channel_major_cache,
        *common[:4],
        conv1.weight,
        conv1.bias,
        *common[4:],
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    fused = _dit_cache_major_conv_mlp_residual(
        hidden,
        conv_input,
        cache_major,
        *common[:4],
        conv1.weight.permute(0, 2, 1).reshape(512, 1536),
        conv1.bias,
        *common[4:],
        conv2.weight.permute(0, 2, 1).reshape(512, 1536),
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    torch.testing.assert_close(fused[0], standard[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[1].transpose(1, 2), standard[1], rtol=0, atol=0)


def test_dit_cache_major_post_attention_conv_mlp_matches_split_math(monkeypatch):
    torch.manual_seed(16)

    def causal_pack(x, cache):
        channel_major = cache.transpose(1, 2)
        history = torch.cat((channel_major, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
            dim=1,
        ).reshape(100, 1536)
        return packed, x[:, -2:, :].contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_pack",
        causal_pack,
        raising=False,
    )
    hidden = torch.randn(2, 50, 512)
    attention = torch.randn(2, 50, 512)
    cache_major = torch.randn(2, 2, 1024)
    modulations = [torch.randn(2, 1, 512) for _ in range(7)]
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)
    gate_msa, shift_conv, scale_conv, gate_conv, shift_mlp, scale_mlp, gate_mlp = modulations
    residual = hidden + gate_msa * attention
    conv_input = F.layer_norm(residual, (512,), eps=1e-6)
    conv_input = conv_input * (1 + scale_conv) + shift_conv
    common = (
        cache_major,
        gate_conv,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        conv1.weight.permute(0, 2, 1).reshape(512, 1536),
        conv1.bias,
        conv_norm.weight,
        conv_norm.bias,
        conv2.weight.permute(0, 2, 1).reshape(512, 1536),
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    split = _dit_cache_major_conv_mlp_residual(residual, conv_input, *common)
    fused = _dit_cache_major_post_attention_conv_mlp_residual(
        hidden,
        attention,
        cache_major,
        gate_msa,
        shift_conv,
        scale_conv,
        *common[1:],
    )
    torch.testing.assert_close(fused[0], split[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[1], split[1], rtol=0, atol=0)


def test_dit_fused_conv_block_mlp_residual_matches_partition_math(monkeypatch):
    torch.manual_seed(15)

    def causal_pack(x, cache):
        history = torch.cat((cache, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
            dim=1,
        ).reshape(100, 1536)
        return packed, x[:, -2:, :].transpose(1, 2).contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_pack",
        causal_pack,
        raising=False,
    )
    hidden = torch.randn(2, 50, 512)
    conv_input = torch.randn(2, 50, 512)
    cache = torch.randn(2, 1024, 2)
    modulations = [torch.randn(2, 1, 512) for _ in range(4)]
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)
    common = (*modulations, conv_norm.weight, conv_norm.bias)
    standard = _dit_conv_mlp_residual(
        hidden,
        conv_input,
        cache,
        *common[:4],
        conv1.weight,
        conv1.bias,
        *common[4:],
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    fused = _dit_fused_conv_block_mlp_residual(
        hidden,
        conv_input,
        cache,
        *common[:4],
        conv1.weight.permute(0, 2, 1).reshape(512, 1536),
        conv1.bias,
        *common[4:],
        conv2.weight.permute(0, 2, 1).reshape(512, 1536),
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    torch.testing.assert_close(fused[0], standard[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[1], standard[1], rtol=0, atol=0)


def test_dit_fused_conv_linear_mlp_residual_matches_partition_math(monkeypatch):
    torch.manual_seed(16)

    def causal_linear(x, cache, weight, bias):
        history = torch.cat((cache, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
            dim=1,
        ).reshape(100, 1536)
        projected = F.linear(packed, weight, bias).reshape(2, 50, 512)
        return projected, x[:, -2:, :].transpose(1, 2).contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_linear",
        causal_linear,
        raising=False,
    )
    hidden = torch.randn(2, 50, 512)
    conv_input = torch.randn(2, 50, 512)
    cache = torch.randn(2, 1024, 2)
    modulations = [torch.randn(2, 1, 512) for _ in range(4)]
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)
    common = (*modulations, conv_norm.weight, conv_norm.bias)
    standard = _dit_conv_mlp_residual(
        hidden,
        conv_input,
        cache,
        *common[:4],
        conv1.weight,
        conv1.bias,
        *common[4:],
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    fused = _dit_fused_conv_linear_mlp_residual(
        hidden,
        conv_input,
        cache,
        *common[:4],
        conv1.weight.permute(0, 2, 1).reshape(512, 1536),
        conv1.bias,
        *common[4:],
        conv2.weight.permute(0, 2, 1).reshape(512, 1536),
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    torch.testing.assert_close(fused[0], standard[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[1], standard[1], rtol=0, atol=0)


def test_dit_fused_full_block_matches_split_partition_math(monkeypatch):
    torch.manual_seed(17)

    def causal_pack(x, cache):
        history = torch.cat((cache, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [history[:, :, offset : offset + 3].transpose(1, 2).reshape(2, -1) for offset in range(50)],
            dim=1,
        ).reshape(100, 1536)
        return packed, x[:, -2:, :].transpose(1, 2).contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_pack",
        causal_pack,
        raising=False,
    )
    hidden = torch.randn(2, 50, 512)
    time_embedding = torch.randn(2, 1, 512)
    att_cache = torch.randn(2, 8, 4, 128)
    cnn_cache = torch.randn(2, 1024, 2)
    adaln = nn.Linear(512, 9 * 512)
    q_proj = nn.Linear(512, 512)
    k_proj = nn.Linear(512, 512)
    v_proj = nn.Linear(512, 512)
    q_norm = nn.LayerNorm(64)
    k_norm = nn.LayerNorm(64)
    out_proj = nn.Linear(512, 512)
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)

    modulation, q, k, v = _dit_attention_preamble(
        hidden,
        time_embedding,
        adaln.weight,
        adaln.bias,
        q_proj.weight,
        q_proj.bias,
        k_proj.weight,
        k_proj.bias,
        v_proj.weight,
        v_proj.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )
    old_k, old_v = att_cache.chunk(2, dim=3)
    k = torch.cat((k, old_k), dim=2)
    v = torch.cat((v, old_v), dim=2)
    expected_att = torch.cat((k, v), dim=3)
    attention = F.scaled_dot_product_attention(q, k, v)
    attention = out_proj(attention.transpose(1, 2).reshape(2, 50, 512))
    modulations = modulation.chunk(9, dim=-1)
    expected_hidden = hidden + modulations[2] * attention
    conv_input = F.layer_norm(expected_hidden, (512,), eps=1e-6)
    conv_input = conv_input * (1 + modulations[7]) + modulations[6]
    expected_hidden, expected_cnn = _dit_conv_mlp_residual(
        expected_hidden,
        conv_input,
        cnn_cache,
        modulations[8],
        modulations[3],
        modulations[4],
        modulations[5],
        conv1.weight,
        conv1.bias,
        conv_norm.weight,
        conv_norm.bias,
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    actual_hidden, actual_cnn, actual_att = _dit_fused_full_block(
        hidden,
        time_embedding,
        att_cache,
        cnn_cache,
        adaln.weight,
        adaln.bias,
        q_proj.weight,
        q_proj.bias,
        k_proj.weight,
        k_proj.bias,
        v_proj.weight,
        v_proj.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
        out_proj.weight,
        out_proj.bias,
        conv1.weight.permute(0, 2, 1).reshape(512, 1536),
        conv1.bias,
        conv_norm.weight,
        conv_norm.bias,
        conv2.weight.permute(0, 2, 1).reshape(512, 1536),
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_cnn, expected_cnn, rtol=0, atol=0)
    torch.testing.assert_close(actual_att, expected_att, rtol=0, atol=0)


def test_dit_fused_full_block_bsh_matches_split_partition_math(monkeypatch):
    torch.manual_seed(23)

    def causal_pack(x, cache):
        history = torch.cat((cache, x.transpose(1, 2)), dim=2)
        packed = torch.stack(
            [
                history[:, :, offset : offset + 3]
                .transpose(1, 2)
                .reshape(2, -1)
                for offset in range(50)
            ],
            dim=1,
        ).reshape(100, 1536)
        return packed, x[:, -2:, :].transpose(1, 2).contiguous()

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_minicpmo_causal_conv_pack",
        causal_pack,
        raising=False,
    )
    hidden = torch.randn(2, 50, 512)
    modulation = torch.randn(2, 1, 9 * 512)
    att_cache = torch.randn(2, 2, 4, 512)
    cnn_cache = torch.randn(2, 1024, 2)
    q_proj = nn.Linear(512, 512)
    k_proj = nn.Linear(512, 512)
    v_proj = nn.Linear(512, 512)
    q_norm = nn.LayerNorm(64)
    k_norm = nn.LayerNorm(64)
    out_proj = nn.Linear(512, 512)
    conv1 = nn.Conv1d(512, 512, 3)
    conv_norm = nn.LayerNorm(512)
    conv2 = nn.Conv1d(512, 512, 3)
    fc1 = nn.Linear(512, 2048)
    fc2 = nn.Linear(2048, 512)

    _, q, k, v = _dit_attention_preamble_bsh_from_modulation(
        hidden,
        modulation,
        q_proj.weight,
        q_proj.bias,
        k_proj.weight,
        k_proj.bias,
        v_proj.weight,
        v_proj.bias,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
    )
    full_k = torch.cat((k, att_cache[0]), dim=1)
    full_v = torch.cat((v, att_cache[1]), dim=1)
    expected_att = torch.stack((full_k, full_v), dim=0)
    query = q.reshape(2, 50, 8, 64).transpose(1, 2)
    key = full_k.reshape(2, 54, 8, 64).transpose(1, 2)
    value = full_v.reshape(2, 54, 8, 64).transpose(1, 2)
    attention = F.scaled_dot_product_attention(query, key, value)
    attention = out_proj(attention.transpose(1, 2).reshape(2, 50, 512))
    modulations = modulation.chunk(9, dim=-1)
    expected_hidden = hidden + modulations[2] * attention
    conv_input = F.layer_norm(expected_hidden, (512,), eps=1e-6)
    conv_input = conv_input * (1 + modulations[7]) + modulations[6]
    expected_hidden, expected_cnn = _dit_conv_mlp_residual(
        expected_hidden,
        conv_input,
        cnn_cache,
        modulations[8],
        modulations[3],
        modulations[4],
        modulations[5],
        conv1.weight,
        conv1.bias,
        conv_norm.weight,
        conv_norm.bias,
        conv2.weight,
        conv2.bias,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
    )
    actual_hidden, actual_cnn, actual_att = (
        _dit_fused_full_block_bsh_from_modulation(
            hidden,
            modulation,
            att_cache,
            cnn_cache,
            q_proj.weight,
            q_proj.bias,
            k_proj.weight,
            k_proj.bias,
            v_proj.weight,
            v_proj.bias,
            q_norm.weight,
            q_norm.bias,
            k_norm.weight,
            k_norm.bias,
            out_proj.weight,
            out_proj.bias,
            conv1.weight.permute(0, 2, 1).reshape(512, 1536),
            conv1.bias,
            conv_norm.weight,
            conv_norm.bias,
            conv2.weight.permute(0, 2, 1).reshape(512, 1536),
            conv2.bias,
            fc1.weight,
            fc1.bias,
            fc2.weight,
            fc2.bias,
        )
    )
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_cnn, expected_cnn, rtol=0, atol=0)
    torch.testing.assert_close(actual_att, expected_att, rtol=0, atol=0)
    standard_hidden, standard_cnn, standard_att = (
        _dit_full_block_bsh_standard_conv_from_modulation(
            hidden,
            modulation,
            att_cache,
            cnn_cache,
            q_proj.weight,
            q_proj.bias,
            k_proj.weight,
            k_proj.bias,
            v_proj.weight,
            v_proj.bias,
            q_norm.weight,
            q_norm.bias,
            k_norm.weight,
            k_norm.bias,
            out_proj.weight,
            out_proj.bias,
            conv1.weight,
            conv1.bias,
            conv_norm.weight,
            conv_norm.bias,
            conv2.weight,
            conv2.bias,
            fc1.weight,
            fc1.bias,
            fc2.weight,
            fc2.bias,
        )
    )
    torch.testing.assert_close(
        standard_hidden, expected_hidden, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(standard_cnn, expected_cnn, rtol=0, atol=0)
    torch.testing.assert_close(standard_att, expected_att, rtol=0, atol=0)


@pytest.mark.parametrize(("value", "expected"), [(None, 50), ("64", 64)])
def test_npu_dit_mlp_graph_width(monkeypatch, value: str | None, expected: int):
    if value is None:
        monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH", value)
    assert _npu_dit_mlp_graph_width() == expected


def test_npu_dit_mlp_graph_config_is_used_without_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH", raising=False)
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH", raising=False)

    assert _npu_dit_mlp_graph_enabled(True) is True
    assert _npu_dit_mlp_graph_width(64) == 64


def test_npu_dit_graph_buckets_support_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_GRAPH_BUCKETS", raising=False)
    assert _npu_dit_graph_buckets([20, 302, 20]) == (20, 302)

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_GRAPH_BUCKETS", "302,20")
    assert _npu_dit_graph_buckets([64]) == (302, 20)


def test_npu_dit_full_block_config_and_cache_buckets(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_GRAPH", raising=False)
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_STACK_GRAPH", raising=False)
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS", raising=False)
    assert _npu_dit_full_block_graph_enabled(True) is True
    assert _npu_dit_full_stack_graph_enabled(True) is True
    assert _npu_dit_full_block_cache_buckets([302, 352, 302]) == (302, 352)

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_GRAPH", "0")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_STACK_GRAPH", "0")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FULL_BLOCK_CACHE_BUCKETS", "402,352")
    assert _npu_dit_full_block_graph_enabled(True) is False
    assert _npu_dit_full_stack_graph_enabled(True) is False
    assert _npu_dit_full_block_cache_buckets([302]) == (402, 352)


def test_dit_attention_cache_length_reads_sequence_axis():
    cache = torch.empty(57, 2, 8, 302, 128)
    assert _dit_attention_cache_length(cache) == 302
    assert _dit_attention_cache_length(None) == 0


def test_dit_explicit_attention_matches_sdpa_at_head_dim_64():
    torch.manual_seed(23)
    query = torch.randn(2, 8, 5, 64)
    key = torch.randn(2, 8, 11, 64)
    value = torch.randn(2, 8, 11, 64)
    expected = F.scaled_dot_product_attention(query, key, value)
    actual = _dit_explicit_attention(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("value", ["0,20", "-1", "20,bad"])
def test_invalid_npu_dit_graph_buckets_are_rejected(monkeypatch, value: str):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_GRAPH_BUCKETS", value)
    with pytest.raises(ValueError, match="NPU_DIT_GRAPH_BUCKETS"):
        _npu_dit_graph_buckets()


def test_npu_dit_preamble_graph_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_PREAMBLE_GRAPH", raising=False)
    assert _npu_dit_preamble_graph_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_PREAMBLE_GRAPH", "off")
    assert _npu_dit_preamble_graph_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_PREAMBLE_GRAPH", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_PREAMBLE_GRAPH"):
        _npu_dit_preamble_graph_enabled()


def test_npu_dit_wide_adaln_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_ADALN", raising=False)
    assert _npu_dit_wide_adaln_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_ADALN", "off")
    assert _npu_dit_wide_adaln_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_ADALN", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_WIDE_ADALN"):
        _npu_dit_wide_adaln_enabled()


def test_npu_dit_wide_final_adaln_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_FINAL_ADALN",
        raising=False,
    )
    assert _npu_dit_wide_final_adaln_enabled(True) is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_FINAL_ADALN",
        "off",
    )
    assert _npu_dit_wide_final_adaln_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_WIDE_FINAL_ADALN",
        "sometimes",
    )
    with pytest.raises(ValueError, match="NPU_DIT_WIDE_FINAL_ADALN"):
        _npu_dit_wide_final_adaln_enabled()


def test_npu_dit_final_addcmul_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_FINAL_ADDCMUL",
        raising=False,
    )
    assert _npu_dit_final_addcmul_enabled(True) is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_FINAL_ADDCMUL",
        "off",
    )
    assert _npu_dit_final_addcmul_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_FINAL_ADDCMUL",
        "sometimes",
    )
    with pytest.raises(ValueError, match="NPU_DIT_FINAL_ADDCMUL"):
        _npu_dit_final_addcmul_enabled()


def test_npu_dit_fused_final_adaln_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_FINAL_ADALN",
        raising=False,
    )
    assert _npu_dit_fused_final_adaln_enabled(True) is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_FINAL_ADALN",
        "off",
    )
    assert _npu_dit_fused_final_adaln_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_FINAL_ADALN",
        "sometimes",
    )
    with pytest.raises(ValueError, match="NPU_DIT_FUSED_FINAL_ADALN"):
        _npu_dit_fused_final_adaln_enabled()


def test_dit_final_addcmul_matches_canonical_adaln():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 8)
    modulation = torch.randn(2, 1, 16)
    norm = nn.LayerNorm(8, elementwise_affine=False, eps=1e-6)
    output = nn.Linear(8, 3)

    expected = _dit_final_from_modulation(hidden, modulation, norm, output)
    actual = _dit_final_from_modulation_addcmul(
        hidden,
        modulation,
        norm,
        output,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_dit_final_cfg_euler_matches_split_path():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 512)
    modulation = torch.randn(2, 1, 1024)
    output = nn.Linear(512, 80)
    norm = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
    x = torch.randn(1, 80, 5)
    delta = torch.tensor(0.125)
    cfg_rate = 0.7

    estimate = _dit_final_from_modulation_addcmul(
        hidden,
        modulation,
        norm,
        output,
    ).transpose(1, 2)
    conditional, unconditional = estimate.chunk(2, dim=0)
    expected = x + delta * (
        (1.0 + cfg_rate) * conditional - cfg_rate * unconditional
    )
    actual = _dit_final_cfg_euler_from_modulation(
        hidden,
        modulation,
        output.weight,
        output.bias,
        x,
        delta,
        cfg_rate,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_dit_wide_adaln_steps_preserves_step_and_block_axes():
    time_embeddings = torch.empty(6, 2, 1, 512, device="meta")
    packed_weight = torch.empty(16 * 9 * 512, 512, device="meta")
    packed_bias = torch.empty(16 * 9 * 512, device="meta")

    actual = _dit_wide_adaln_steps(
        time_embeddings,
        packed_weight,
        packed_bias,
    )

    assert actual.shape == (6, 2, 1, 16, 9 * 512)


def test_dit_wide_adaln_steps_with_final_splits_projection_rows():
    time_embeddings = torch.empty(6, 2, 1, 512, device="meta")
    output_width = 16 * 9 * 512 + 2 * 512
    packed_weight = torch.empty(output_width, 512, device="meta")
    packed_bias = torch.empty(output_width, device="meta")

    blocks, final = _dit_wide_adaln_steps_with_final(
        time_embeddings,
        packed_weight,
        packed_bias,
    )

    assert blocks.shape == (6, 2, 1, 16, 9 * 512)
    assert final.shape == (6, 2, 1, 2 * 512)


def test_npu_dit_conv_mlp_graph_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_CONV_MLP_GRAPH", raising=False)
    assert _npu_dit_conv_mlp_graph_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_CONV_MLP_GRAPH", "off")
    assert _npu_dit_conv_mlp_graph_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_CONV_MLP_GRAPH", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_CONV_MLP_GRAPH"):
        _npu_dit_conv_mlp_graph_enabled()


def test_npu_dit_last_block_final_euler_graph_config_and_environment(monkeypatch):
    name = "VLLM_OMNI_MINICPMO45_NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH"
    monkeypatch.delenv(name, raising=False)
    assert _npu_dit_last_block_final_euler_graph_enabled(True) is True

    monkeypatch.setenv(name, "off")
    assert _npu_dit_last_block_final_euler_graph_enabled(True) is False

    monkeypatch.setenv(name, "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH"):
        _npu_dit_last_block_final_euler_graph_enabled()


@pytest.mark.parametrize(
    ("control_us", "candidate_us", "expected"),
    [
        (500.0, 250.0, True),
        (358.774, 298.390, False),
        (500.0, 350.0, False),
        (0.0, 0.0, False),
    ],
)
def test_npu_dit_last_block_final_euler_perf_gate(
    control_us,
    candidate_us,
    expected,
):
    assert (
        _npu_dit_last_block_final_euler_perf_qualifies(
            control_us,
            candidate_us,
        )
        is expected
    )


def test_npu_dit_prompt_conv_mlp_graph_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_PROMPT_CONV_MLP_GRAPH", raising=False)
    assert _npu_dit_prompt_conv_mlp_graph_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_PROMPT_CONV_MLP_GRAPH", "off")
    assert _npu_dit_prompt_conv_mlp_graph_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_PROMPT_CONV_MLP_GRAPH", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_PROMPT_CONV_MLP_GRAPH"):
        _npu_dit_prompt_conv_mlp_graph_enabled()


def test_npu_dit_fused_conv_pack_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK", raising=False)
    assert _npu_dit_fused_conv_pack_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK", "off")
    assert _npu_dit_fused_conv_pack_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_FUSED_CONV_PACK"):
        _npu_dit_fused_conv_pack_enabled()


def test_npu_dit_cache_major_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR", raising=False)
    assert _npu_dit_cache_major_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR", "off")
    assert _npu_dit_cache_major_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_CACHE_MAJOR"):
        _npu_dit_cache_major_enabled()


def test_npu_dit_post_attn_graph_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_POST_ATTN_GRAPH", raising=False)
    assert _npu_dit_post_attn_graph_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_POST_ATTN_GRAPH", "off")
    assert _npu_dit_post_attn_graph_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_POST_ATTN_GRAPH", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_POST_ATTN_GRAPH"):
        _npu_dit_post_attn_graph_enabled()


def test_npu_dit_qkv_pack_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_QKV_PACK", raising=False)
    assert _npu_dit_qkv_pack_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_QKV_PACK", "off")
    assert _npu_dit_qkv_pack_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_QKV_PACK", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_QKV_PACK"):
        _npu_dit_qkv_pack_enabled()


def test_npu_dit_fused_qkv_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_QKV", raising=False)
    assert _npu_dit_fused_qkv_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_QKV", "off")
    assert _npu_dit_fused_qkv_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_QKV", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_FUSED_QKV"):
        _npu_dit_fused_qkv_enabled()


def test_npu_dit_attn_cache_out_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_ATTN_CACHE_OUT", raising=False)
    assert _npu_dit_attn_cache_out_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_ATTN_CACHE_OUT", "off")
    assert _npu_dit_attn_cache_out_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_ATTN_CACHE_OUT", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_ATTN_CACHE_OUT"):
        _npu_dit_attn_cache_out_enabled()


def test_npu_cfm_stacked_cache_out_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_CFM_STACKED_CACHE_OUT", raising=False)
    assert _npu_cfm_stacked_cache_out_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_STACKED_CACHE_OUT", "off")
    assert _npu_cfm_stacked_cache_out_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_STACKED_CACHE_OUT", "sometimes")
    with pytest.raises(ValueError, match="NPU_CFM_STACKED_CACHE_OUT"):
        _npu_cfm_stacked_cache_out_enabled()


def test_npu_cfm_fixed_kv_slabs_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_FIXED_KV_SLABS", raising=False
    )
    assert _npu_cfm_fixed_kv_slabs_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_FIXED_KV_SLABS", "off")
    assert _npu_cfm_fixed_kv_slabs_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_FIXED_KV_SLABS", "sometimes"
    )
    with pytest.raises(ValueError, match="NPU_CFM_FIXED_KV_SLABS"):
        _npu_cfm_fixed_kv_slabs_enabled()


def test_npu_cfm_planar_kv_slabs_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_PLANAR_KV_SLABS", raising=False
    )
    assert _npu_cfm_planar_kv_slabs_enabled(True) is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_PLANAR_KV_SLABS", "off"
    )
    assert _npu_cfm_planar_kv_slabs_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_CFM_PLANAR_KV_SLABS", "sometimes"
    )
    with pytest.raises(ValueError, match="NPU_CFM_PLANAR_KV_SLABS"):
        _npu_cfm_planar_kv_slabs_enabled()


def test_npu_dit_bsh_attention_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_BSH_ATTENTION", raising=False
    )
    assert _npu_dit_bsh_attention_enabled(True) is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_BSH_ATTENTION", "off"
    )
    assert _npu_dit_bsh_attention_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_DIT_BSH_ATTENTION", "sometimes"
    )
    with pytest.raises(ValueError, match="NPU_DIT_BSH_ATTENTION"):
        _npu_dit_bsh_attention_enabled()


def test_npu_single_request_cache_passthrough_config_and_environment(monkeypatch):
    monkeypatch.delenv(
        "VLLM_OMNI_MINICPMO45_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH",
        raising=False,
    )
    assert _npu_single_request_cache_passthrough_enabled(True) is True

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH",
        "off",
    )
    assert _npu_single_request_cache_passthrough_enabled(True) is False

    monkeypatch.setenv(
        "VLLM_OMNI_MINICPMO45_NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH",
        "sometimes",
    )
    with pytest.raises(ValueError, match="NPU_SINGLE_REQUEST_CACHE_PASSTHROUGH"):
        _npu_single_request_cache_passthrough_enabled()


def test_npu_dit_fused_conv_block_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_BLOCK", raising=False)
    assert _npu_dit_fused_conv_block_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_BLOCK", "off")
    assert _npu_dit_fused_conv_block_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_BLOCK", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_FUSED_CONV_BLOCK"):
        _npu_dit_fused_conv_block_enabled()


def test_npu_dit_fused_conv_linear_config_and_environment(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_LINEAR", raising=False)
    assert _npu_dit_fused_conv_linear_enabled(True) is True

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_LINEAR", "off")
    assert _npu_dit_fused_conv_linear_enabled(True) is False

    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_LINEAR", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_FUSED_CONV_LINEAR"):
        _npu_dit_fused_conv_linear_enabled()


def test_npu_dit_mlp_graph_environment_overrides_profile(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH", "off")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH", "25")

    assert _npu_dit_mlp_graph_enabled(True) is False
    assert _npu_dit_mlp_graph_width(50) == 25


def test_invalid_npu_dit_mlp_graph_switch_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH", "sometimes")
    with pytest.raises(ValueError, match="NPU_DIT_MLP_GRAPH"):
        _npu_dit_mlp_graph_enabled()


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_npu_dit_mlp_graph_width_is_rejected(monkeypatch, value: str):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH", value)
    with pytest.raises(ValueError, match="NPU_DIT_MLP_GRAPH_WIDTH"):
        _npu_dit_mlp_graph_width()


@pytest.mark.parametrize(("value", "expected"), [("0", 1), ("8", 8), ("bad", 4)])
def test_npu_cfm_graph_cache_limit_is_bounded(monkeypatch, value: str, expected: int):
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE", value)
    assert BatchedToken2Wav._npu_cfm_graph_cache_limit() == expected


def test_npu_cfm_graph_env_is_explicit(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH", raising=False)
    assert _npu_cfm_graph_enabled() is False
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH", "yes")
    assert _npu_cfm_graph_enabled() is True
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH", "0")
    assert _npu_cfm_graph_enabled() is False


def test_fade_in_out_limits_overlap_to_available_previous_audio():
    speech = torch.arange(6, dtype=torch.float32).reshape(1, -1)
    previous = torch.full((1, 3), 2.0)
    window = torch.hamming_window(8, periodic=False)

    actual = BatchedToken2Wav._fade_in_out(speech, previous, window)

    expected = speech.clone()
    expected[..., :3] = speech[..., :3] * window[:3] + previous * window[-3:]
    torch.testing.assert_close(actual, expected)


def test_estimator_cache_stack_split_round_trip_preserves_cfg_rows():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    stacked = adapter._stack_flow_cache(states)
    assert stacked["estimator_cnn_cache"].shape[2] == 4
    assert stacked["estimator_att_cache"].shape[2] == 4
    restored = adapter._split_flow_cache(stacked, 2)
    for original, round_tripped in zip(states, restored, strict=True):
        torch.testing.assert_close(
            round_tripped["estimator_cnn_cache"],
            original.flow_cache["estimator_cnn_cache"],
        )
        torch.testing.assert_close(
            round_tripped["estimator_att_cache"],
            original.flow_cache["estimator_att_cache"],
        )


def test_bsh_estimator_cache_split_and_stack_use_cfg_axis():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    # Match the real MiniCPM-o DiT attention geometry used by BSH detection
    # and by the conversion back to the canonical BHSD cache.
    adapter.flow.decoder.estimator.blocks[0].attn.num_heads = 8
    adapter.flow.decoder.estimator.blocks[0].attn.head_dim = 64
    adapter._npu_dit_bsh_attention_enabled = True
    cache = {
        "conformer_cnn_cache": torch.randn(2, 4, 3),
        "conformer_att_cache": torch.randn(2, 2, 4, 3),
        "estimator_cnn_cache": torch.randn(6, 16, 4, 1024, 2),
        "estimator_att_cache": torch.randn(6, 16, 2, 4, 5, 512),
    }

    split = adapter._split_flow_cache(cache, 2)
    for row, flow in enumerate(split):
        expected_bsh = torch.cat(
            (
                cache["estimator_att_cache"][:, :, :, row : row + 1],
                cache["estimator_att_cache"][:, :, :, 2 + row : 3 + row],
            ),
            dim=3,
        )
        torch.testing.assert_close(flow["estimator_att_cache"], expected_bsh)

    states = [
        BatchedToken2WavState(flow_cache=flow, hift_cache={}) for flow in split
    ]
    stacked = adapter._stack_flow_cache(states)
    expected_legacy = BatchedToken2Wav._legacy_att_cache_from_bsh_planar(
        cache["estimator_att_cache"], 8, 64
    )
    torch.testing.assert_close(stacked["estimator_att_cache"], expected_legacy)


def test_single_request_cache_passthrough_preserves_storage():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        npu_single_request_cache_passthrough=True,
    )
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    stacked = adapter._stack_flow_cache(states)
    split = adapter._split_flow_cache(stacked, 1)

    for name, value in states[0].flow_cache.items():
        assert stacked[name].data_ptr() == value.data_ptr()
        assert split[0][name].data_ptr() == value.data_ptr()


def test_single_request_cache_passthrough_is_exact_across_chunks():
    control = BatchedToken2Wav(_FakeToken2Wav())
    candidate = BatchedToken2Wav(
        _FakeToken2Wav(),
        npu_single_request_cache_passthrough=True,
    )
    control_prompt = control.prepare_prompt("shared", "/fake/prompt.wav")
    candidate_prompt = candidate.prepare_prompt("shared", "/fake/prompt.wav")
    control_states = control.setup_batch(control_prompt, 1)
    candidate_states = candidate.setup_batch(candidate_prompt, 1)

    for tokens, last_chunk in (
        (torch.tensor([[10, 11]]), False),
        (torch.tensor([[12, 13]]), False),
        (torch.tensor([[14]]), True),
    ):
        control_audio, control_states = control.decode_batch(
            tokens,
            control_prompt,
            control_states,
            last_chunk=last_chunk,
        )
        candidate_audio, candidate_states = candidate.decode_batch(
            tokens,
            candidate_prompt,
            candidate_states,
            last_chunk=last_chunk,
        )
        assert torch.equal(control_audio[0], candidate_audio[0])
        for cache_name in ("flow_cache", "hift_cache"):
            control_cache = getattr(control_states[0], cache_name)
            candidate_cache = getattr(candidate_states[0], cache_name)
            assert control_cache.keys() == candidate_cache.keys()
            for name in control_cache:
                assert torch.equal(control_cache[name], candidate_cache[name])


def test_fixed_estimator_kv_slabs_are_exact_and_keep_addresses():
    control = BatchedToken2Wav(_FakeToken2Wav())
    candidate = BatchedToken2Wav(
        _FakeToken2Wav(),
        npu_cfm_fixed_kv_slabs=True,
    )
    control_prompt = control.prepare_prompt("shared", "/fake/prompt.wav")
    candidate_prompt = candidate.prepare_prompt("shared", "/fake/prompt.wav")
    control_states = control.setup_batch(control_prompt, 1)
    candidate_states = candidate.setup_batch(candidate_prompt, 1)
    slabs = candidate_states[0].estimator_kv_slabs
    assert slabs is not None
    retained_address = slabs.retained.data_ptr()
    append_address = slabs.append.data_ptr()
    cnn_addresses = tuple(bank.data_ptr() for bank in slabs.cnn_banks)

    for tokens, last_chunk in (
        (torch.tensor([[10, 11]]), False),
        (torch.tensor([[12, 13]]), False),
        (torch.tensor([[14, 15]]), False),
        (torch.tensor([[16]]), True),
    ):
        control_audio, control_states = control.decode_batch(
            tokens,
            control_prompt,
            control_states,
            last_chunk=last_chunk,
        )
        candidate_audio, candidate_states = candidate.decode_batch(
            tokens,
            candidate_prompt,
            candidate_states,
            last_chunk=last_chunk,
        )
        assert torch.equal(control_audio[0], candidate_audio[0])
        for name in control_states[0].flow_cache:
            assert torch.equal(
                control_states[0].flow_cache[name],
                candidate_states[0].flow_cache[name],
            )
        slabs = candidate_states[0].estimator_kv_slabs
        assert slabs is not None
        assert slabs.retained.data_ptr() == retained_address
        assert slabs.append.data_ptr() == append_address
        assert tuple(bank.data_ptr() for bank in slabs.cnn_banks) == cnn_addresses


def test_fixed_estimator_slabs_keep_cnn_cache_major_across_eager_tail():
    attention = torch.randn(6, 16, 2, 8, 4, 128)
    cnn = torch.randn(6, 16, 2, 1024, 2)
    slabs = BatchedToken2Wav._make_fixed_estimator_kv_slabs(
        attention,
        cnn,
        4,
        cnn_cache_major=True,
    )

    assert slabs is not None
    assert slabs.cnn_cache_major is True
    assert slabs.cnn_banks[0].shape[-2:] == (2, 1024)
    torch.testing.assert_close(slabs.cnn_banks[0], cnn.transpose(-2, -1))

    eager_tail_cnn = torch.randn_like(cnn)
    advanced = BatchedToken2Wav._advance_fixed_estimator_kv_slabs(
        slabs,
        slabs.append[..., :5, :],
        eager_tail_cnn,
    )
    assert advanced.cnn_cache_major is True
    torch.testing.assert_close(
        advanced.cnn_banks[advanced.active_cnn_bank],
        eager_tail_cnn.transpose(-2, -1),
    )


def test_bsh_estimator_kv_slabs_preserve_legacy_cache_and_addresses():
    torch.manual_seed(31)
    attention = torch.randn(6, 16, 2, 8, 4, 128)
    cnn = torch.randn(6, 16, 2, 1024, 2)
    slabs = BatchedToken2Wav._make_fixed_estimator_kv_slabs(
        attention,
        cnn,
        4,
        bsh_attention=True,
    )

    assert slabs is not None
    assert slabs.planar is True
    assert slabs.bsh_attention is True
    assert slabs.retained.shape == (6, 16, 2, 2, 104, 512)
    assert slabs.append.shape == (6, 16, 2, 2, 154, 512)
    retained_address = slabs.retained.data_ptr()
    append_address = slabs.append.data_ptr()
    restored = BatchedToken2Wav._legacy_att_cache_from_bsh_planar(
        slabs.retained[..., :4, :], 8, 64
    )
    torch.testing.assert_close(restored, attention, rtol=0, atol=0)

    # The accepted profile also carries the older planar flag. Once the first
    # prompt decode already emitted BSH storage, that flag must not reinterpret
    # the 512-wide sequence-major axis as another packed K/V axis.
    bsh_source = slabs.retained[..., :4, :].clone()
    inherited_planar_slabs = BatchedToken2Wav._make_fixed_estimator_kv_slabs(
        bsh_source,
        cnn,
        4,
        planar=True,
        bsh_attention=True,
    )
    assert inherited_planar_slabs is not None
    assert inherited_planar_slabs.retained.shape == (6, 16, 2, 2, 104, 512)
    torch.testing.assert_close(
        inherited_planar_slabs.retained[..., :4, :],
        bsh_source,
        rtol=0,
        atol=0,
    )

    output = slabs.append[..., :6, :]
    output.copy_(torch.randn_like(output))
    advanced = BatchedToken2Wav._advance_fixed_estimator_kv_slabs(
        slabs,
        output,
        torch.randn_like(cnn),
    )
    assert advanced.bsh_attention is True
    assert advanced.retained.data_ptr() == retained_address
    assert advanced.append.data_ptr() == append_address


def test_planar_estimator_kv_slabs_are_exact_and_keep_addresses():
    control = BatchedToken2Wav(_FakeToken2Wav())
    candidate = BatchedToken2Wav(
        _FakeToken2Wav(),
        npu_cfm_planar_kv_slabs=True,
    )
    control_prompt = control.prepare_prompt("shared", "/fake/prompt.wav")
    candidate_prompt = candidate.prepare_prompt("shared", "/fake/prompt.wav")
    control_states = control.setup_batch(control_prompt, 1)
    candidate_states = candidate.setup_batch(candidate_prompt, 1)
    slabs = candidate_states[0].estimator_kv_slabs
    assert slabs is not None
    assert slabs.planar is True
    assert slabs.retained.ndim == 7
    assert slabs.retained[0, 0].select(0, 0).is_contiguous()
    assert slabs.retained[0, 0].select(0, 1).is_contiguous()
    retained_address = slabs.retained.data_ptr()
    append_address = slabs.append.data_ptr()

    for tokens, last_chunk in (
        (torch.tensor([[10, 11]]), False),
        (torch.tensor([[12, 13]]), False),
        (torch.tensor([[14, 15]]), False),
        (torch.tensor([[16]]), True),
    ):
        control_audio, control_states = control.decode_batch(
            tokens, control_prompt, control_states, last_chunk=last_chunk
        )
        candidate_audio, candidate_states = candidate.decode_batch(
            tokens, candidate_prompt, candidate_states, last_chunk=last_chunk
        )
        assert torch.equal(control_audio[0], candidate_audio[0])
        for name in control_states[0].flow_cache:
            candidate_value = candidate_states[0].flow_cache[name]
            if name == "estimator_att_cache":
                candidate_value = candidate._legacy_att_cache_from_planar(
                    candidate_value
                )
            assert torch.equal(
                control_states[0].flow_cache[name], candidate_value
            )
        slabs = candidate_states[0].estimator_kv_slabs
        assert slabs is not None
        assert slabs.retained.data_ptr() == retained_address
        assert slabs.append.data_ptr() == append_address


def test_planar_estimator_kv_slabs_use_exact_dynamic_tail_on_overflow():
    control = BatchedToken2Wav(_FakeToken2Wav())
    candidate = BatchedToken2Wav(
        _FakeToken2Wav(),
        npu_cfm_planar_kv_slabs=True,
    )
    control_prompt = control.prepare_prompt("shared", "/fake/prompt.wav")
    candidate_prompt = candidate.prepare_prompt("shared", "/fake/prompt.wav")
    control_states = control.setup_batch(control_prompt, 1)
    candidate_state = candidate.setup_batch(candidate_prompt, 1)[0]
    slabs = candidate_state.estimator_kv_slabs
    assert slabs is not None
    limited_slabs = type(slabs)(
        retained=slabs.retained,
        append=slabs.append[..., : slabs.logical_length, :],
        cnn_banks=slabs.cnn_banks,
        prompt_length=slabs.prompt_length,
        logical_length=slabs.logical_length,
        active_cnn_bank=slabs.active_cnn_bank,
        planar=slabs.planar,
    )
    candidate_states = [
        type(candidate_state)(
            flow_cache=candidate_state.flow_cache,
            hift_cache=candidate_state.hift_cache,
            estimator_kv_slabs=limited_slabs,
        )
    ]

    control_audio, control_states = control.decode_batch(
        torch.tensor([[10, 11]]),
        control_prompt,
        control_states,
        last_chunk=True,
    )
    candidate_audio, candidate_states = candidate.decode_batch(
        torch.tensor([[10, 11]]),
        candidate_prompt,
        candidate_states,
        last_chunk=True,
    )

    assert torch.equal(control_audio[0], candidate_audio[0])
    assert candidate._npu_cfm_fixed_kv_tail_fallback_used is True
    for name in control_states[0].flow_cache:
        candidate_value = candidate_states[0].flow_cache[name]
        if name == "estimator_att_cache":
            candidate_value = candidate._legacy_att_cache_from_planar(
                candidate_value
            )
        assert torch.equal(control_states[0].flow_cache[name], candidate_value)


def test_fixed_estimator_kv_slabs_compact_exact_newest_first_history():
    adapter = BatchedToken2Wav(
        _FakeToken2Wav(),
        npu_cfm_fixed_kv_slabs=True,
    )
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    state = adapter.setup_batch(prompt, 1)[0]
    slabs = state.estimator_kv_slabs
    assert slabs is not None
    output = slabs.append
    frame_values = torch.arange(output.shape[4], dtype=output.dtype)
    output.copy_(frame_values.reshape(1, 1, 1, 1, -1, 1))
    cnn_output = torch.full_like(slabs.cnn_banks[1], 7)

    advanced = adapter._advance_fixed_estimator_kv_slabs(
        slabs, output, cnn_output
    )

    expected = torch.cat((frame_values[:4], frame_values[-100:]))
    torch.testing.assert_close(
        advanced.retained[0, 0, 0, 0, : advanced.logical_length, 0],
        expected,
    )
    assert advanced.logical_length == 104
    assert advanced.active_cnn_bank == 1
    assert torch.equal(advanced.cnn_banks[1], cnn_output)


def test_model_preserves_output_slots_and_prefers_runtime_codes():
    model, token2wav = _model()
    output = _forward(
        model,
        [_info("a", 0, [10, 11]), _info("b", 0, [20, 21])],
        placeholder_counts=[3, 1],
    )

    audios = output.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert len(output.multimodal_outputs["sr"]) == 2
    assert all(sr.item() == 24000 for sr in output.multimodal_outputs["sr"])
    assert all(audio.dtype == torch.float32 for audio in audios)
    # Fake CFM uses two Euler steps whose deltas sum to one. Its conditional
    # row is mu and its unconditional row is zero, so CFG produces 1.7 * mu.
    torch.testing.assert_close(audios[0][0], torch.tensor(1.7 * 10))
    torch.testing.assert_close(audios[1][0], torch.tensor(1.7 * 20))
    assert token2wav.flow.encoder.calls[-1] == 2


def test_code2wav_projects_duplex_metadata_to_final_audio_output():
    model, token2wav = _model()
    segment = _info("duplex", 0, [10, 11])
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    segment["meta"].update(
        {
            "duplex_epoch": 3,
            "duplex_turn_id": 7,
            "llm_output_text_utf8": segment_text_utf8,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    segment_output = _forward(model, [segment])

    assert segment_output.multimodal_outputs["meta.turn_end"][0].item() is False
    # A Talker unit boundary only drains pending codec tokens. The official
    # streaming path keeps Token2wav open until the assistant turn ends.
    assert token2wav.flow.encoder.last_chunk_calls[-1] is False
    assert "duplex" in model._states

    final = _info("duplex", 1, [12, 13], last_chunk=True)
    final["meta"].update(segment["meta"])
    final["meta"]["chunk_seq"] = 1
    final["meta"]["last_chunk"] = True
    final["meta"]["turn_end"] = True
    output = _forward(model, [final])

    payload = output.multimodal_outputs
    assert "meta" not in payload
    assert payload["meta.duplex_epoch"][0].item() == 3
    assert payload["meta.duplex_turn_id"][0].item() == 7
    torch.testing.assert_close(
        payload["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert payload["meta.tts_is_last_chunk"][0].item() is True
    assert payload["meta.turn_end"][0].item() is True
    assert token2wav.flow.encoder.last_chunk_calls[-1] is True
    assert "duplex" not in model._states


def test_initial_empty_segment_marker_initializes_stream_without_audio():
    model, token2wav = _model()
    boundary = _info("duplex", 0, [])
    boundary["meta"].update(
        {
            "code_flat_numel": 0,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    output = _forward(model, [boundary])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert "duplex" in model._states
    assert token2wav.hift.calls == []

    resumed = _info(
        "duplex",
        1,
        [4218, 4218, 4218, 10, 11, 12, 13, 14],
    )
    output = _forward(model, [resumed])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert "duplex" in model._states


def test_shared_runtime_prompt_recreates_missing_file_before_second_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    first = _info("voice-a", 0, [10, 11])
    first["codes"]["ref"] = reference
    first["meta"]["ref_audio_sr"] = 16000
    first["meta"].pop("prompt_cache_id")
    _forward(model, [first], request_ids=["internal-a"])

    prompt_key = model._request_prompt_keys["internal-a"]
    prompt_path = Path(model._runtime_prompts[prompt_key].path)
    prompt_path.unlink()

    second = _info("voice-b", 0, [12, 13])
    second["codes"]["ref"] = reference
    second["meta"]["ref_audio_sr"] = 16000
    second["meta"].pop("prompt_cache_id")
    _forward(model, [second], request_ids=["internal-b"])

    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-a", "internal-b"}

    model.on_requests_finished(["internal-a"])
    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-b"}

    model.on_requests_finished(["internal-b"])
    assert not prompt_path.exists()
    assert prompt_key not in model._runtime_prompts


def test_runtime_prompt_write_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.sf.write",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated write failure"):
        model._materialize_runtime_prompt(reference, 16000)

    assert len(model._runtime_prompts) == 1
    entry = next(iter(model._runtime_prompts.values()))
    assert not Path(entry.path).exists()
    assert list(Path(entry.path).parent.iterdir()) == []


def test_runtime_prompt_files_are_isolated_between_model_instances(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    first_model, _ = _model()
    second_model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def runtime_ref_info(request_id: str):
        info = _info(request_id, 0, [10, 11])
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    _forward(first_model, [runtime_ref_info("voice-a")], request_ids=["internal-a"])
    _forward(second_model, [runtime_ref_info("voice-b")], request_ids=["internal-b"])

    first_key = first_model._request_prompt_keys["internal-a"]
    second_key = second_model._request_prompt_keys["internal-b"]
    first_path = Path(first_model._runtime_prompts[first_key].path)
    second_path = Path(second_model._runtime_prompts[second_key].path)
    assert first_key == second_key
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()

    first_model.on_requests_finished(["internal-a"])
    assert not first_path.exists()
    assert second_path.is_file()

    second_model.on_requests_finished(["internal-b"])
    assert not second_path.exists()


def test_mixed_final_exact_buckets_keep_order_and_release_only_final_states():
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    output = _forward(
        model,
        [
            _info("a", 1, [11, 12]),
            _info("c", 1, [31, 32, 33], last_chunk=True),
            _info("b", 1, [21, 22]),
            _info("d", 1, [41, 42, 43], last_chunk=True),
        ],
    )

    audios = output.multimodal_outputs["model_outputs"]
    window = torch.hamming_window(4, periodic=False)
    overlap_scale = 1.7 * (window[0] + window[2])
    expected = torch.tensor([1, 3, 2, 4], dtype=torch.float32) * overlap_scale
    actual = torch.stack([audio[0] for audio in audios])
    torch.testing.assert_close(actual, expected)
    assert set(model._states) == {"a", "b"}


def test_empty_final_sentinel_emits_empty_and_releases_state_without_compute():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    hift_calls = list(token2wav.hift.calls)
    output = _forward(
        model,
        [
            _info("a", 1, [], last_chunk=True),
            _info("b", 1, [], last_chunk=True),
        ],
    )

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}
    assert token2wav.hift.calls == hift_calls


def test_empty_final_ignores_generation_scheduler_placeholder_token():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    infos = [_info("a", 1, [], last_chunk=True), _info("b", 1, [], last_chunk=True)]
    for info in infos:
        info.pop("codes")
        info["meta"]["code_flat_numel"] = 0

    output = _forward(model, infos, placeholder_counts=[1, 1])

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}


@pytest.mark.parametrize(
    "info",
    [
        # The runner injects the engine request id on every step (GPU
        # _preprocess, NPU _gather_runtime_additional_information)...
        {"request_id": "a", "meta": {"request_id": "a"}},
        # ...but a pre-warm step can also reach the model with nothing at all.
        {},
    ],
)
def test_prewarm_placeholder_step_emits_silence_without_touching_state(info):
    # async-chunk pre-warm submits Stage 2 with a reserved placeholder prompt.
    # If it gets scheduled before the first codec window lands, those reserved
    # tokens must neither be vocoded nor held to the codec payload contract.
    model, token2wav = _model()

    output = _forward(model, [info], request_ids=["a"])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._states == {}
    assert token2wav.hift.calls == []


def test_npu_prepare_event_builds_state_and_first_chunk_reuses_it(mocker):
    model, _ = _model()
    model._prompt_prewarm_enabled = True
    setup = mocker.spy(model.backend, "setup_batch")
    prepare = {
        "meta": {
            "request_id": "a",
            "lifecycle_event": "prepare",
            "lifecycle_generation": 0,
        }
    }

    output = _forward(model, [prepare], request_ids=["a"])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._states["a"].chunk_seq is None
    assert setup.call_count == 1

    first = _forward(model, [_info("a", 0, [1, 2])], request_ids=["a"])

    assert first.multimodal_outputs["model_outputs"][0].numel() > 0
    assert model._states["a"].chunk_seq == 0
    assert setup.call_count == 1


def test_prompt_state_cache_reuses_setup_without_aliasing(monkeypatch):
    model, _ = _model()
    model._prompt_prewarm_enabled = True
    model._prompt_state_cache_enabled = True
    setup_calls = 0
    original_setup = model.backend.setup_batch

    def counted_setup(*args, **kwargs):
        nonlocal setup_calls
        setup_calls += 1
        return original_setup(*args, **kwargs)

    monkeypatch.setattr(model.backend, "setup_batch", counted_setup)

    def prepare(request_id: str):
        return {
            "meta": {
                "request_id": request_id,
                "lifecycle_event": "prepare",
                "lifecycle_generation": 0,
            }
        }

    _forward(model, [prepare("a")], request_ids=["a"])
    _forward(model, [prepare("b")], request_ids=["b"])

    assert setup_calls == 1
    assert len(model._prompt_state_templates) == 1
    state_a = model._states["a"].token2wav
    state_b = model._states["b"].token2wav
    for name in state_a.flow_cache:
        assert state_a.flow_cache[name].data_ptr() != state_b.flow_cache[name].data_ptr()
    with torch.inference_mode():
        state_a.flow_cache["conformer_att_cache"].fill_(123)
    assert not torch.equal(
        state_a.flow_cache["conformer_att_cache"],
        state_b.flow_cache["conformer_att_cache"],
    )

    first = _forward(model, [_info("a", 0, [1, 2])], request_ids=["a"])

    assert first.multimodal_outputs["model_outputs"][0].numel() > 0
    assert model._states["a"].chunk_seq == 0
    assert setup_calls == 1


def test_prepare_event_carries_exact_reference_and_avoids_default_prompt_setup(mocker):
    model, _ = _model()
    model._prompt_prewarm_enabled = True
    setup = mocker.spy(model.backend, "setup_batch")
    materialize = mocker.spy(model, "_materialize_runtime_prompt")
    reference = torch.linspace(-0.1, 0.1, 160)
    prepare = {
        "codes": {"ref": reference},
        "meta": {
            "request_id": "voice-a",
            "lifecycle_event": "prepare",
            "lifecycle_generation": 0,
            "ref_audio_sr": 16000,
        },
    }

    _forward(model, [prepare], request_ids=["voice-a"])

    state = model._states["voice-a"]
    assert state.chunk_seq is None
    assert state.prompt_cache_id.startswith("runtime-ref-")
    prompt_key = model._request_prompt_keys["voice-a"]
    assert model._runtime_prompts[prompt_key].fingerprint == prompt_key
    assert setup.call_count == 1

    first = _info("voice-a", 0, [1, 2])
    first["codes"]["ref"] = reference
    first["meta"].pop("prompt_cache_id")
    first["meta"]["ref_audio_sr"] = 16000
    _forward(model, [first], request_ids=["voice-a"])

    assert model._states["voice-a"].chunk_seq == 0
    assert model._request_prompt_keys["voice-a"] == prompt_key
    assert setup.call_count == 1
    assert materialize.call_count == 1


def test_prompt_cache_id_includes_prompt_content_fingerprint(tmp_path):
    model, _ = _model()
    prompt_path = tmp_path / "voice.wav"
    prompt_path.write_bytes(b"first-prompt")
    first = _info("a", 0, [1, 2])
    first["meta"]["prompt_wav"] = str(prompt_path)
    _forward(model, [first], request_ids=["a"])
    first_cache_id = model._states["a"].prompt_cache_id

    prompt_path.write_bytes(b"second-prompt-with-different-content")
    second = _info("b", 0, [3, 4])
    second["meta"]["prompt_wav"] = str(prompt_path)
    _forward(model, [second], request_ids=["b"])

    assert model._states["b"].prompt_cache_id != first_cache_id


def test_finished_request_tombstone_drops_late_first_chunk_without_recreating_state():
    model, token2wav = _model()
    _forward(model, [_info("external-a", 0, [1, 2])], request_ids=["internal-a"])
    model.on_requests_finished(["internal-a"])
    hift_calls = list(token2wav.hift.calls)

    output = _forward(
        model,
        [_info("external-a", 0, [3, 4])],
        request_ids=["internal-a"],
    )

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert "internal-a" not in model._states
    assert token2wav.hift.calls == hift_calls


def test_finished_request_tombstones_are_bounded_oldest_first():
    model, _ = _model()
    model._terminal_state_limit = 2

    model.on_requests_finished(["a", "b", "c"])

    assert list(model._terminal_state_ids) == ["b", "c"]


def test_lifecycle_generation_must_match_cache_epoch():
    model, _ = _model()
    info = _info("a", 0, [1, 2], cache_epoch=1)
    info["meta"]["lifecycle_generation"] = 0

    with pytest.raises(RuntimeError, match="lifecycle_generation_mismatch"):
        _forward(model, [info])


def test_metadata_only_payload_still_decodes_codec_from_prompt_tokens():
    # The connector strips 1-D codec tensors out of additional_information and
    # leaves them in the prompt tokens, so a real chunk reaches the model as
    # producer metadata plus input ids. It must still be vocoded.
    model, _ = _model()
    info = {
        "request_id": "a",
        "meta": {
            "request_id": "a",
            "chunk_seq": 0,
            "code_flat_numel": 2,
            "prompt_cache_id": "shared",
        },
    }

    output = _forward(model, [info], placeholder_counts=[2])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert set(model._states) == {"a"}


def test_non_final_chunk_shorter_than_lookahead_window_is_rejected():
    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    with pytest.raises(RuntimeError, match="chunk_below_lookahead_window"):
        adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=False)

    # The final chunk is zero-padded by the encoder, so it stays decodable.
    audios, _ = adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=True)
    assert len(audios) == 1


def test_forward_builds_backend_when_weight_loading_was_skipped(monkeypatch):
    # load_format=dummy never calls load_weights(), so Stage 2 would otherwise
    # reach its first request with no Token2wav assets at all.
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    token2wav = _FakeToken2Wav()
    builds = 0

    def build_backend():
        nonlocal builds
        builds += 1
        model.backend = BatchedToken2Wav(token2wav)

    monkeypatch.setattr(model, "_build_backend", build_backend)

    output = _forward(model, [_info("a", 0, [10, 11])])
    _forward(model, [_info("a", 1, [12, 13])])

    assert builds == 1
    assert output.multimodal_outputs["model_outputs"][0].numel() > 0


@pytest.mark.parametrize(
    ("info", "reason"),
    [
        (_info("a", 0, [1, 2], cache_epoch=-1), "negative_stream_position"),
        (_info("a", 0, [1, 2]), "stale_or_reordered_chunk"),
        (_info("a", 2, [1, 2]), "stale_or_reordered_chunk"),
    ],
)
def test_stale_epoch_and_reordered_chunks_are_rejected(info, reason):
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])

    with pytest.raises(RuntimeError, match=reason):
        _forward(model, [info, _info("b", 1, [3, 4])])


def test_singleton_and_mixed_shape_buckets_use_same_batched_backend_without_fallback():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    output = _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8, 9])])

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    # Exact-shape buckets execute independently but both use the same vectorized
    # adapter; there is no Token2wav.stream/__call__ fallback.
    assert token2wav.hift.calls[-2:] == [1, 1]


def test_backend_failure_does_not_commit_any_request_state(monkeypatch):
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    before = dict(model._states)
    original = model.backend.decode_batch
    call_count = 0

    def fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "decode_batch", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _forward(
            model,
            [
                _info("a", 1, [5, 6]),
                _info("b", 1, [7, 8]),
                _info("c", 1, [9, 10, 11]),
                _info("d", 1, [12, 13, 14]),
            ],
        )
    assert call_count == 2
    assert model._states == before


def test_cleanup_and_profile_output_are_aligned():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    model.on_requests_finished(["a"])
    assert set(model._states) == {"b"}

    profile = model(
        input_ids=torch.zeros(5, dtype=torch.long),
        seq_token_counts=[2, 3],
    )
    assert [audio.numel() for audio in profile.multimodal_outputs["model_outputs"]] == [0, 0]
    assert set(model._states) == {"b"}


def test_cleanup_uses_generation_runner_internal_request_ids():
    model, _ = _model()
    _forward(
        model,
        [_info("external-a", 0, [1, 2]), _info("external-b", 0, [3, 4])],
        request_ids=["internal-a", "internal-b"],
    )

    model.on_requests_finished(["internal-a"])

    assert set(model._states) == {"internal-b"}


def test_reference_voice_and_duplex_metadata_follow_request_lifecycle():
    model, _ = _model()
    first = _info("voice-a", 0, [1, 2])
    first["codes"]["ref"] = torch.linspace(-0.1, 0.1, 160)
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    first["meta"].update(
        ref_audio_sr=16000,
        llm_output_text_utf8=segment_text_utf8,
        duplex_turn_id=7,
        duplex_epoch=3,
    )
    first["meta"].pop("prompt_cache_id")

    output = _forward(model, [first])
    prompt_key = model._request_prompt_keys["voice-a"]
    prompt = model._runtime_prompts[prompt_key]
    prompt_cache_id, prompt_wav = prompt.cache_id, prompt.path
    assert prompt_cache_id.startswith("runtime-ref-")
    assert Path(prompt_wav).is_file()
    torch.testing.assert_close(
        output.multimodal_outputs["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert output.multimodal_outputs["meta.duplex_turn_id"][0].item() == 7
    assert output.multimodal_outputs["meta.duplex_epoch"][0].item() == 3

    final = _info("voice-a", 1, [3, 4], last_chunk=True)
    final["meta"].pop("prompt_cache_id")
    final["meta"]["tts_is_last_chunk"] = True
    output = _forward(model, [final])

    assert output.multimodal_outputs["meta.tts_is_last_chunk"][0].item() is True
    assert model._request_prompt_keys["voice-a"] == prompt_key
    model.on_requests_finished(["voice-a"])
    assert "voice-a" not in model._request_prompt_keys
    assert prompt_key not in model._runtime_prompts
    assert not Path(prompt_wav).exists()
    assert (prompt_cache_id, prompt_wav) not in model.backend._prompt_features
