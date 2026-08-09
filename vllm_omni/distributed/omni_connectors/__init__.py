# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Any

from .connectors.base import OmniConnectorBase
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector
from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec, OmniTransferConfig
from .utils.initialization import (
    build_stage_connectors,
    get_connectors_config_for_stage,
    get_stage_connector_config,
    initialize_connectors_from_config,
    initialize_orchestrator_connectors,
    load_omni_transfer_config,
)

_LAZY_CONNECTORS = {
    "MooncakeStoreConnector": (
        "vllm_omni.distributed.omni_connectors.connectors.mooncake_store_connector",
        "MooncakeStoreConnector",
    ),
    "MooncakeTransferEngineConnector": (
        "vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector",
        "MooncakeTransferEngineConnector",
    ),
    "MoriTransferEngineConnector": (
        "vllm_omni.distributed.omni_connectors.connectors.mori_transfer_engine_connector",
        "MoriTransferEngineConnector",
    ),
    "YuanrongTransferEngineConnector": (
        "vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector",
        "YuanrongTransferEngineConnector",
    ),
}


def __getattr__(name: str) -> Any:
    lookup_name = "MooncakeStoreConnector" if name == "MooncakeConnector" else name
    target = _LAZY_CONNECTORS.get(lookup_name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    try:
        value = getattr(importlib.import_module(module_name), attribute_name)
    except ImportError:
        value = None
    globals()[lookup_name] = value
    if name == "MooncakeConnector":
        globals()[name] = value
    return value


__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    # Base classes and implementations
    "OmniConnectorBase",
    # Factory
    "OmniConnectorFactory",
    # Specific implementations
    "MooncakeConnector",  # compat alias → MooncakeStoreConnector
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
    "initialize_connectors_from_config",
    "get_connectors_config_for_stage",
    # Manager helpers
    "initialize_orchestrator_connectors",
    "get_stage_connector_config",
    "build_stage_connectors",
]
