from types import SimpleNamespace

import numpy as np
import torch

from vllm_omni.platforms.npu.worker import npu_model_runner
from vllm_omni.platforms.npu.worker.npu_model_runner import (
    _init_context_parallel_profile_batch,
    _profiling_chunk_config,
    _update_full_graph_params_compat,
)


class _FakeQueryLens:
    def __init__(self):
        self.cpu = torch.zeros(4, dtype=torch.int32)
        self.copy_calls = 0

    def copy_to_gpu(self):
        self.copy_calls += 1


class _FakeManager:
    def __init__(self, query_lens_name: str):
        self.calls = []
        setattr(self, query_lens_name, _FakeQueryLens())

    def init_batch_info(self, *args):
        self.calls.append(args)


def _runner(**kwargs):
    values = {
        "input_batch": SimpleNamespace(
            num_computed_tokens_cpu=torch.tensor([3, 4], dtype=torch.int32),
            num_prompt_tokens=torch.tensor([5, 6], dtype=torch.int32),
        ),
        "speculative_config": object(),
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_profile_batch_uses_current_dcp_manager_api():
    manager = _FakeManager("query_lens_full")
    runner = _runner(use_dcp=True, dcp_manager=manager)
    tokens = np.array([7, 8], dtype=np.int32)

    _init_context_parallel_profile_batch(runner, tokens, 2)

    assert len(manager.calls) == 1
    torch.testing.assert_close(manager.query_lens_full.cpu[:2], torch.tensor([7, 8], dtype=torch.int32))
    assert manager.query_lens_full.copy_calls == 1


def test_profile_batch_falls_back_to_legacy_pcp_manager_api():
    manager = _FakeManager("query_lens_pcp_full")
    runner = _runner(use_cp=True, pcp_manager=manager)
    tokens = np.array([9, 10], dtype=np.int32)

    _init_context_parallel_profile_batch(runner, tokens, 2)

    assert len(manager.calls) == 1
    torch.testing.assert_close(manager.query_lens_pcp_full.cpu[:2], torch.tensor([9, 10], dtype=torch.int32))
    assert manager.query_lens_pcp_full.copy_calls == 1


def test_profile_batch_is_noop_without_context_parallelism():
    _init_context_parallel_profile_batch(_runner(), np.array([1], dtype=np.int32), 1)


def test_profiling_chunk_config_accepts_nested_and_legacy_locations():
    nested = SimpleNamespace(need_timing=True)
    legacy = SimpleNamespace(need_timing=False)

    config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=nested))
    assert _profiling_chunk_config(config) is nested
    assert _profiling_chunk_config(SimpleNamespace(profiling_chunk_config=legacy)) is legacy


def test_profiling_chunk_config_is_optional():
    assert _profiling_chunk_config(SimpleNamespace()) is None


class _FakeGraphRunner:
    def __init__(self):
        self.calls = []

    def _update_full_graph_params_if_needed(self, *args):
        self.calls.append(args)


def test_full_graph_update_passes_positions_to_current_api(monkeypatch):
    monkeypatch.setattr(npu_model_runner, "_FULL_GRAPH_UPDATE_ACCEPTS_POSITIONS", True)
    runner = _FakeGraphRunner()
    context = object()
    positions = torch.tensor([1, 2], dtype=torch.int64)

    _update_full_graph_params_compat(runner, context, 2, positions)

    assert runner.calls == [(context, 2, positions)]


def test_full_graph_update_supports_legacy_api(monkeypatch):
    monkeypatch.setattr(npu_model_runner, "_FULL_GRAPH_UPDATE_ACCEPTS_POSITIONS", False)
    runner = _FakeGraphRunner()
    context = object()

    _update_full_graph_params_compat(runner, context, 2, None)

    assert runner.calls == [(context, 2)]
