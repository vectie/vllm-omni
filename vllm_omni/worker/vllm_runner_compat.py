# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner symbols that moved between the supported vLLM release images."""

from importlib import import_module
from typing import Any


def _import_runner_symbol(symbol: str, current_module: str) -> Any:
    """Resolve a runner symbol from vLLM 0.26 or its vLLM 0.25 location."""
    try:
        return getattr(import_module(current_module), symbol)
    except (ImportError, AttributeError):
        legacy_module = import_module("vllm.v1.worker.gpu_model_runner")
        return getattr(legacy_module, symbol)


IntermediateTensors = _import_runner_symbol("IntermediateTensors", "vllm.sequence")
EMPTY_MODEL_RUNNER_OUTPUT = _import_runner_symbol(
    "EMPTY_MODEL_RUNNER_OUTPUT",
    "vllm.v1.outputs",
)

