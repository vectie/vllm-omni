# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from . import omni_connectors as _omni_connectors
from .omni_connectors import (
    ConnectorSpec,
    OmniConnectorBase,
    OmniConnectorFactory,
    OmniTransferConfig,
    SharedMemoryConnector,
    YuanrongConnector,
    load_omni_transfer_config,
)

_LAZY_CONNECTOR_EXPORTS = {
    "MooncakeConnector",
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "YuanrongTransferEngineConnector",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_CONNECTOR_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_omni_connectors, name)
    globals()[name] = value
    return value


__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    # Connectors
    "OmniConnectorBase",
    "OmniConnectorFactory",
    "MooncakeConnector",  # compat alias
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
]
