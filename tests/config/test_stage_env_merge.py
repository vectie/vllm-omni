# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.config.stage_config import _deep_merge_stage


def test_stage_env_deep_merge_preserves_inherited_flags():
    base = {
        "stage_id": 2,
        "env": {
            "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH": "1",
            "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_WIDTH": "58",
        },
    }
    overlay = {
        "stage_id": 2,
        "env": {"VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_BUCKETS": "50"},
    }

    merged = _deep_merge_stage(base, overlay)

    assert merged["env"] == {
        "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH": "1",
        "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_WIDTH": "58",
        "VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_BUCKETS": "50",
    }
