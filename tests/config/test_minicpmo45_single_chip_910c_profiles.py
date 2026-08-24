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

