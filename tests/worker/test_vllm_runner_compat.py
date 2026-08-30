# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from types import SimpleNamespace

from vllm_omni.worker import vllm_runner_compat


def test_import_runner_symbol_uses_current_location(monkeypatch):
    sentinel = object()
    requested = []

    def fake_import(module_name):
        requested.append(module_name)
        return SimpleNamespace(IntermediateTensors=sentinel)

    monkeypatch.setattr(vllm_runner_compat, "import_module", fake_import)

    resolved = vllm_runner_compat._import_runner_symbol(
        "IntermediateTensors",
        "vllm.sequence",
    )

    assert resolved is sentinel
    assert requested == ["vllm.sequence"]


def test_import_runner_symbol_falls_back_to_vllm_025_location(monkeypatch):
    sentinel = object()
    requested = []

    def fake_import(module_name):
        requested.append(module_name)
        if module_name == "vllm.v1.outputs":
            return SimpleNamespace()
        return SimpleNamespace(EMPTY_MODEL_RUNNER_OUTPUT=sentinel)

    monkeypatch.setattr(vllm_runner_compat, "import_module", fake_import)

    resolved = vllm_runner_compat._import_runner_symbol(
        "EMPTY_MODEL_RUNNER_OUTPUT",
        "vllm.v1.outputs",
    )

    assert resolved is sentinel
    assert requested == [
        "vllm.v1.outputs",
        "vllm.v1.worker.gpu_model_runner",
    ]


def test_module_imports_with_vllm_025_symbol_layout(monkeypatch):
    intermediate_tensors = object()
    empty_output = object()
    monkeypatch.setitem(sys.modules, "vllm.sequence", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        SimpleNamespace(
            IntermediateTensors=intermediate_tensors,
            EMPTY_MODEL_RUNNER_OUTPUT=empty_output,
        ),
    )

    spec = importlib.util.spec_from_file_location(
        "vllm_runner_compat_v025_test",
        vllm_runner_compat.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.IntermediateTensors is intermediate_tensors
    assert module.EMPTY_MODEL_RUNNER_OUTPUT is empty_output
