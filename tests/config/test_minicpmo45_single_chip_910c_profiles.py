# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_profile(name: str):
    path = Path(get_deploy_config_path(name))
    return _apply_platform_overrides(load_deploy_config(path), platform="npu")


@pytest.mark.parametrize(
    "name",
    [
        "minicpmo_4_5_1npu_910c_cfm6_canonical_layout_experimental.yaml",
        "minicpmo_4_5_1npu_910c_cfm6_canonical_qkv_dma_experimental.yaml",
        "minicpmo_4_5_1npu_910c_cfm6_canonical_cfm_graph_experimental.yaml",
        "minicpmo_4_5_1npu_910c_cfm6_canonical_w8a8_mlp_experimental.yaml",
    ],
)
def test_single_chip_profiles_keep_all_stages_on_device_zero(name):
    deploy = _load_profile(name)

    assert [stage.devices for stage in deploy.stages] == ["0", "0", "0"]
    assert deploy.connectors is not None
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    assert extra["token2wav_n_timesteps"] == 6
    assert extra["codec_chunk_frames"] == 25
    assert extra["codec_left_context_frames"] == 3
    assert extra["raw_tensor_shm"] is False
    assert extra["shm_event_notifications"] is True


def test_single_chip_canonical_profile_is_fp32_fixed_planar_cache_major():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_910c_cfm6_canonical_layout_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]

    assert "npu_dit_compute_dtype" not in extra
    assert extra["npu_dit_cache_major"] is True
    assert extra["npu_dit_fused_conv_pack"] is True
    assert extra["npu_dit_wide_adaln"] is False
    assert extra["npu_single_request_cache_passthrough"] is True
    assert extra["npu_cfm_fixed_kv_slabs"] is True
    assert extra["npu_cfm_planar_kv_slabs"] is True


def test_a2_evaluator_compat_profile_changes_capacity_not_model_numerics():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_evaluator_compat.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]

    assert stage0.max_num_seqs == 1
    assert stage0.max_model_len == 8192
    assert stage0.engine_extras["kv_cache_memory_bytes"] == 1342177280
    assert stage0.skip_mm_profiling is True
    assert stage1.max_num_seqs == 1
    assert stage1.engine_extras["kv_cache_memory_bytes"] == 268435456
    assert "token2wav_n_timesteps" not in extra
    assert "initial_codec_chunk_frames" not in extra
    assert not any(key.startswith("npu_") for key in extra)


def test_a2_evaluator_fia_bucket_candidate_targets_only_talker():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_experimental.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert (
        stage1.engine_extras["additional_config"]["fia_graph_seq_len_bucket_size"]
        == 16
    )
    assert "fia_graph_seq_len_bucket_size" not in stage0.engine_extras["additional_config"]
    assert "fia_graph_seq_len_bucket_size" not in stage2.engine_extras["additional_config"]


def test_a2_evaluator_async_sampler_candidate_is_explicitly_talker_only():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_sampler_graph_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    additional_config = stage1.engine_extras["additional_config"]
    assert additional_config["fia_graph_seq_len_bucket_size"] == 16
    assert additional_config["enable_fia_bucket_async_replay"] is True
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert "VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH" not in (stage0.env or {})
    assert "VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH" not in (stage2.env or {})


def test_a2_evaluator_fused_distribution_candidate_is_talker_only():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_fused_distribution_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert (
        stage1.env["VLLM_OMNI_MINICPMO45_NPU_FUSED_CODEC_DISTRIBUTION"]
        == "1"
    )
    assert "VLLM_OMNI_MINICPMO45_NPU_FUSED_CODEC_DISTRIBUTION" not in (
        stage0.env or {}
    )
    assert "VLLM_OMNI_MINICPMO45_NPU_FUSED_CODEC_DISTRIBUTION" not in (
        stage2.env or {}
    )


def test_a2_evaluator_sampler_sync_diagnostic_is_explicitly_talker_only():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_sampler_graph_sync_diagnostic.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_SYNC"] == "1"
    assert "VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_SYNC" not in (stage0.env or {})
    assert "VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_SYNC" not in (stage2.env or {})


def test_a2_evaluator_async_safe_profile_does_not_enable_rejected_sampler():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_experimental.yaml"
    )

    for stage in deploy.stages:
        assert "VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH" not in (stage.env or {})
        assert "VLLM_OMNI_MINICPMO45_NPU_FUSED_CODEC_DISTRIBUTION" not in (
            stage.env or {}
        )


def test_a2_evaluator_fia_bucket32_candidate_targets_only_talker():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_evaluator_fia_bucket32_experimental.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.engine_extras["additional_config"]["fia_graph_seq_len_bucket_size"] == 32
    assert "fia_graph_seq_len_bucket_size" not in stage0.engine_extras["additional_config"]
    assert "fia_graph_seq_len_bucket_size" not in stage2.engine_extras["additional_config"]


def test_a2_evaluator_fia_bucket16_slotfast_targets_only_talker():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_slotfast_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.engine_extras["additional_config"]["fia_graph_seq_len_bucket_size"] == 16
    assert stage1.env["VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH"] == "1"
    assert "VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH" not in (stage0.env or {})
    assert "VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH" not in (stage2.env or {})


def test_a2_evaluator_metacache_targets_only_talker():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_slotfast_metacache_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.engine_extras["additional_config"]["fia_graph_seq_len_bucket_size"] == 16
    assert stage1.env["VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH"] == "1"
    assert stage1.env["VLLM_ASCEND_DIRTY_BLOCK_TABLE_COMMIT"] == "1"
    assert stage1.env["VLLM_ASCEND_SINGLE_REQUEST_DECODE_METADATA_CACHE"] == "1"
    assert stage1.env["VLLM_ASCEND_SINGLE_REQUEST_DECODE_SCALAR_STAGING"] == "1"
    for name in (
        "VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH",
        "VLLM_ASCEND_DIRTY_BLOCK_TABLE_COMMIT",
        "VLLM_ASCEND_SINGLE_REQUEST_DECODE_METADATA_CACHE",
        "VLLM_ASCEND_SINGLE_REQUEST_DECODE_SCALAR_STAGING",
    ):
        assert name not in (stage0.env or {})
        assert name not in (stage2.env or {})


def test_a2_evaluator_fia_bucket16_profiler_targets_only_talker():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_talker_profile.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.profiler_config is not None
    assert stage1.profiler_config.profiler == "torch"
    assert stage0.profiler_config is None
    assert stage2.profiler_config is None


def test_a2_evaluator_stable_fia_v2_candidate_targets_only_talker():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_evaluator_fia_v2_stable_experimental.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.engine_extras["additional_config"]["enable_stable_fia_v2_graph_inputs"] is True
    assert "enable_stable_fia_v2_graph_inputs" not in stage0.engine_extras["additional_config"]
    assert "enable_stable_fia_v2_graph_inputs" not in stage2.engine_extras["additional_config"]


def test_single_chip_qkv_candidate_adds_only_explicit_qkv_pack():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_910c_cfm6_canonical_qkv_dma_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]

    assert extra["npu_dit_qkv_pack"] is True
    assert extra["npu_dit_cache_major"] is True
    assert extra["npu_dit_wide_adaln"] is False
    assert extra["npu_cfm_planar_kv_slabs"] is True


def test_single_chip_cfm_graph_has_one_static_graph_entry():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_910c_cfm6_canonical_cfm_graph_experimental.yaml"
    )
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE"] == "1"


def test_a2_cache_fill_candidate_keeps_first_packet_and_steady_shapes():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_cache_fill_experimental.yaml"
    )
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"
    assert stage2.env[
        "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH"
    ] == "1"
    assert stage2.env[
        "VLLM_OMNI_MINICPMO45_NPU_CFM_CACHE_FILL_GRAPH_LENGTHS"
    ] == "302"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE"] == "2"


def test_single_chip_w8a8_candidate_quantizes_only_dit_mlp():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_910c_cfm6_canonical_w8a8_mlp_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]

    assert extra["npu_dit_dynamic_w8a8"] is True
    assert "npu_dit_compute_dtype" not in extra
    assert extra["npu_dit_cache_major"] is True
    assert extra["npu_dit_conv_mlp_graph"] is True
    assert extra["npu_cfm_planar_kv_slabs"] is True


def test_a2_fused_bf16_ffn_candidate_keeps_retained_outer_graph():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_fused_ffn_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["npu_dit_fused_bf16_ffn"] is True
    assert extra["npu_dit_compute_dtype"] == "bf16"
    assert extra["npu_dit_cache_major"] is False
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_talker_sampler_graph_is_scoped_to_stage_one():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_graph_experimental.yaml"
    )
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_low_ttfp_profile_reuses_width_twenty_first_packet_graph():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["codec_chunk_frames"] == 25
    assert extra["initial_codec_chunk_frames"] == 25
    assert 20 in extra["npu_dit_graph_buckets"]
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "10"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_first_packet_cfm4_retains_cfm6_after_initial_chunk():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_cfm4_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["token2wav_n_timesteps"] == 6
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "10"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "4"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_minimum_useful_first_packet_precompiles_width_ten():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_i5_cfm4_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["codec_chunk_frames"] == 25
    assert extra["token2wav_n_timesteps"] == 6
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "5"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_DIT_GRAPH_BUCKETS"] == "10,20,302"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "4"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_width_ten_eager_fallback_does_not_admit_failed_graph_bucket():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_i5_eager_cfm4_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "5"
    assert 10 not in extra["npu_dit_graph_buckets"]
    assert "VLLM_OMNI_MINICPMO45_NPU_DIT_GRAPH_BUCKETS" not in stage2.env
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "4"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_prompt_cfm2_retains_exact_speaker_and_cfm6_steady_policy():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt2_cfm4_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["token2wav_n_timesteps"] == 6
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "10"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS"] == "2"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "4"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_minimum_solver_first_path_still_retains_cfm6_steady_policy():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["token2wav_n_timesteps"] == 6
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


@pytest.mark.parametrize(
    ("name", "prompt_steps", "initial_steps"),
    [
        (
            "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt2_cfm4_fused_ffn_experimental.yaml",
            "2",
            "4",
        ),
        (
            "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_experimental.yaml",
            "1",
            "1",
        ),
    ],
)
def test_a2_low_ttfp_fused_ffn_profiles_compose_first_and_steady_paths(
    name: str,
    prompt_steps: str,
    initial_steps: str,
):
    deploy = _load_profile(name)
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["token2wav_n_timesteps"] == 6
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "10"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS"] == prompt_steps
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == initial_steps
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_BF16_FFN"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_prompt1_fused_ffn_disables_overlapping_single_chip_cpu_binding():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_experimental.yaml"
    )

    assert all(
        stage.engine_extras["additional_config"]["enable_cpu_binding"] is False
        for stage in deploy.stages
    )


def test_a2_chunk50_profile_uses_static_width_100_after_first_packet():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_chunk50_experimental.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["initial_codec_chunk_frames"] == 10
    assert extra["codec_chunk_frames"] == 50
    assert extra["npu_dit_mlp_graph_width"] == 100
    assert extra["npu_dit_graph_buckets"] == [20, 50, 302]
    assert extra["token2wav_n_timesteps"] == 6
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_BF16_FFN"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_SLOTS"] == "1"


def test_a2_chunk50_eager_diagnostic_changes_only_outer_cfm_capture():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_chunk50_eager_diagnostic.yaml"
    )
    extra = deploy.connectors["connector_of_shared_memory"]["extra"]
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert extra["initial_codec_chunk_frames"] == 10
    assert extra["codec_chunk_frames"] == 50
    assert extra["npu_dit_mlp_graph_width"] == 100
    assert extra["npu_cfm_fixed_kv_slabs"] is True
    assert extra["npu_dit_conv_mlp_graph"] is True
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "0"


def test_a2_talker_static_kernel_is_scoped_to_fixed_shape_stage_one():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_static_kernel_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    stage1_compile = stage1.engine_extras["additional_config"][
        "ascend_compilation_config"
    ]
    assert stage1_compile["enable_npugraph_ex"] is True
    assert stage1_compile["enable_static_kernel"] is True
    assert stage1_compile["fuse_norm_quant"] is False
    assert stage1.engine_extras["additional_config"]["enable_cpu_binding"] is False
    assert stage0.engine_extras["additional_config"]["ascend_compilation_config"].get(
        "enable_static_kernel", False
    ) is False
    assert stage0.engine_extras["additional_config"]["enable_cpu_binding"] is False
    assert stage2.engine_extras["additional_config"]["enable_cpu_binding"] is False


def test_a2_talker_nz_preformats_only_stage_one_bf16_weights():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_nz_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.engine_extras["additional_config"]["weight_nz_mode"] == 2
    assert stage1.engine_extras["additional_config"]["enable_cpu_binding"] is False
    assert "VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH" not in stage1.env
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_CODEC_SAMPLER_GRAPH"] == "1"
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "10"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_BATCHED_CODEC_OUTPUT"] == "1"
    assert stage1.env["VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES"] == "25"
    assert "weight_nz_mode" not in stage0.engine_extras["additional_config"]
    assert stage0.engine_extras["additional_config"]["enable_cpu_binding"] is False
    assert stage2.engine_extras["additional_config"]["enable_cpu_binding"] is False


def test_a2_talker_stable_pa_removes_per_layer_graph_rebinding():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_nz_stable_pa_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    additional_config = stage1.engine_extras["additional_config"]
    assert additional_config["pa_shape_list"] == [1]
    assert additional_config["enable_stable_pa_graph_inputs"] is True
    assert additional_config["weight_nz_mode"] == 2
    assert "pa_shape_list" not in stage0.engine_extras["additional_config"]
    assert "pa_shape_list" not in stage2.engine_extras["additional_config"]


@pytest.mark.parametrize(
    ("name", "steady_steps"),
    [
        (
            "minicpmo_4_5_1npu_a2_cfm4_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_nz_stable_pa_prompt_state_cache_experimental.yaml",
            "4",
        ),
        (
            "minicpmo_4_5_1npu_a2_cfm3_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_nz_stable_pa_prompt_state_cache_experimental.yaml",
            "3",
        ),
    ],
)
def test_a2_static_reduced_cfm_profiles_rebuild_the_complete_solver_abi(
    name: str,
    steady_steps: str,
):
    deploy = _load_profile(name)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "10"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_BATCHED_CODEC_OUTPUT"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS"] == steady_steps
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_PROMPT_CFM_TIMESTEPS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_INITIAL_CFM_TIMESTEPS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_BF16_FFN"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_CODE2WAV_PROMPT_STATE_CACHE"] == "1"


def test_a2_talker_nz_profiler_is_isolated_from_cfm_graph_process():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm6_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_nz_profile.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage0.profiler_config is None
    assert stage1.profiler_config["profiler"] == "torch"
    assert stage1.profiler_config["torch_profiler_record_shapes"] is True
    assert stage1.profiler_config["torch_profiler_with_memory"] is False
    assert stage1.profiler_config["torch_profiler_with_stack"] is False
    assert stage2.profiler_config is None
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"


def test_a2_cfm3_current_profiler_isolates_talker_stage():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm3_bf16_bsh_cfm_graph_hf32_talker_sampler_low_ttfp_prompt1_cfm1_fused_ffn_talker_nz_stable_pa_prompt_state_cache_profile.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage0.profiler_config is None
    assert stage1.profiler_config["profiler"] == "torch"
    assert stage1.profiler_config["torch_profiler_record_shapes"] is True
    assert stage2.profiler_config is None
    assert stage2.env["VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS"] == "3"


def test_a2_cfm3_deferred_eos_is_scoped_to_sparse_talker_transport():
    deploy = _load_profile(
        "minicpmo_4_5_1npu_a2_cfm3_deferred_chunk_eos_experimental.yaml"
    )
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert "VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS" not in (stage0.env or {})
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS"] == "1"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_BATCHED_CODEC_OUTPUT"] == "1"
    assert stage1.env["VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES"] == "25"
    assert "VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS" not in (stage2.env or {})
    assert stage2.env["VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS"] == "3"


def test_a2_cfm3_deferred_eos_i5_changes_only_initial_transport_boundary():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_cfm3_deferred_eos_i5_experimental.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert "VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES" not in (stage0.env or {})
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "5"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS"] == "1"
    assert "VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES" not in (stage2.env or {})
    assert stage2.env["VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS"] == "3"


def test_a2_cfm2_i5_rebuilds_only_the_steady_solver_width():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_cfm2_deferred_eos_i5_experimental.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert "VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS" not in (stage0.env or {})
    assert stage1.env["VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"] == "5"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS"] == "1"
    assert stage2.env["VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS"] == "2"


def test_a2_cfm3_deferred_eos_profiler_keeps_stage_two_unprofiled():
    deploy = _load_profile("minicpmo_4_5_1npu_a2_cfm3_deferred_eos_profile.yaml")
    stage0 = next(stage for stage in deploy.stages if stage.stage_id == 0)
    stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
    stage2 = next(stage for stage in deploy.stages if stage.stage_id == 2)

    assert stage0.profiler_config is None
    assert stage1.profiler_config["profiler"] == "torch"
    assert stage1.env["VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS"] == "1"
    assert stage2.profiler_config is None
    assert stage2.env["VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH"] == "1"
