import os
import subprocess
import sys
from types import SimpleNamespace


def test_distributed_import_does_not_load_optional_native_connectors() -> None:
    code = """
import sys
import vllm_omni.distributed

optional_modules = (
    "vllm_omni.distributed.omni_connectors.connectors.mooncake_store_connector",
    "vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector",
    "vllm_omni.distributed.omni_connectors.connectors.mori_transfer_engine_connector",
    "vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector",
)
loaded = [name for name in optional_modules if name in sys.modules]
if loaded:
    raise SystemExit(f"optional connector modules loaded eagerly: {loaded}")
"""
    env = os.environ.copy()
    env.setdefault("VLLM_PLUGINS", "none")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_optional_connector_export_resolves_on_access(monkeypatch) -> None:
    from vllm_omni.distributed import omni_connectors

    sentinel = object()
    fake_module = SimpleNamespace(MooncakeStoreConnector=sentinel)
    monkeypatch.delitem(omni_connectors.__dict__, "MooncakeStoreConnector", raising=False)
    monkeypatch.delitem(omni_connectors.__dict__, "MooncakeConnector", raising=False)
    monkeypatch.setattr(omni_connectors.importlib, "import_module", lambda _name: fake_module)

    assert omni_connectors.MooncakeStoreConnector is sentinel
    monkeypatch.delitem(omni_connectors.__dict__, "MooncakeStoreConnector", raising=False)
    assert omni_connectors.MooncakeConnector is sentinel
