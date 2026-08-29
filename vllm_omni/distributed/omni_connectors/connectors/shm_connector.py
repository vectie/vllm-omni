# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import hashlib
import os
import select
import socket
import struct
from dataclasses import fields, is_dataclass
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import msgspec
import torch

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_TENSOR_SHM_MAGIC = b"OMNITEN1"
_TENSOR_SHM_PREFIX = struct.Struct("!8sQQ")
_TENSOR_SHM_MARKER = "__omni_raw_shm_tensor__"
_TENSOR_SHM_ALIGNMENT = 64


def _align_up(value: int, alignment: int = _TENSOR_SHM_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _extract_raw_tensors(value: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        index = len(tensors)
        tensors.append(tensor)
        return {
            _TENSOR_SHM_MARKER: index,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
        }
    if isinstance(value, dict):
        return {key: _extract_raw_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_extract_raw_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return {"__omni_raw_shm_tuple__": [_extract_raw_tensors(item, tensors) for item in value]}
    if is_dataclass(value) and not isinstance(value, type):
        # ``dataclasses.asdict`` deep-copies leaves, which is both expensive
        # and unsafe for some Tensor subclasses. Preserve the wire contract
        # (Struct/dataclass -> dict) without copying the tensor twice.
        return {field.name: _extract_raw_tensors(getattr(value, field.name), tensors) for field in fields(value)}
    if isinstance(value, msgspec.Struct):
        # Inter-stage payloads are normally OmniPayloadStruct instances.
        # The receiver intentionally observes a type-erased dict, matching
        # the existing serialized connector contract.
        return {
            name: _extract_raw_tensors(field_value, tensors)
            for name in value.__struct_fields__
            if (field_value := getattr(value, name)) is not None
        }
    return value


def _restore_raw_tensors(value: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(value, dict):
        if _TENSOR_SHM_MARKER in value:
            return tensors[int(value[_TENSOR_SHM_MARKER])]
        if "__omni_raw_shm_tuple__" in value:
            return tuple(_restore_raw_tensors(item, tensors) for item in value["__omni_raw_shm_tuple__"])
        return {key: _restore_raw_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_raw_tensors(item, tensors) for item in value]
    return value


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        extra = config.get("extra", config)
        self._raw_tensor_shm = bool(extra.get("raw_tensor_shm", False)) if isinstance(extra, dict) else False
        self._event_notifications = (
            bool(extra.get("shm_event_notifications", False)) if isinstance(extra, dict) else False
        )
        namespace = str(extra.get("shm_notification_namespace", "default")) if isinstance(extra, dict) else "default"
        namespace_digest = hashlib.sha256(namespace.encode()).hexdigest()[:16]
        self._notify_prefix = f"/dev/shm/vllm_omni_{namespace_digest}"
        self._notify_socket: socket.socket | None = None
        self._notify_path: str | None = None
        if self._event_notifications and int(self.stage_id) >= 0:
            self._notify_path = f"{self._notify_prefix}_{self.stage_id}.sock"
            try:
                os.unlink(self._notify_path)
            except FileNotFoundError:
                pass
            self._notify_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._notify_socket.setblocking(False)
            self._notify_socket.bind(self._notify_path)
        self._pending_keys: set[str] = set()
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "raw_tensor_puts": 0,
        }

    @property
    def event_notifications_enabled(self) -> bool:
        """Whether this connector has a usable receive notification socket."""
        return self._notify_socket is not None

    @staticmethod
    def _create_segment(name: str, size: int) -> shm_pkg.SharedMemory:
        try:
            return shm_pkg.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            existing = shm_pkg.SharedMemory(name=name)
            existing.close()
            existing.unlink()
            return shm_pkg.SharedMemory(create=True, size=size, name=name)

    def _put_raw_tensors(self, name: str, data: Any) -> tuple[dict[str, Any], int] | None:
        tensors: list[torch.Tensor] = []
        header_obj = _extract_raw_tensors(data, tensors)
        if not tensors:
            return None
        header = self.serialize_obj(header_obj)
        data_start = _align_up(_TENSOR_SHM_PREFIX.size + len(header))
        tensor_sizes = [tensor.numel() * tensor.element_size() for tensor in tensors]
        offsets: list[int] = []
        total_size = data_start
        for nbytes in tensor_sizes:
            total_size = _align_up(total_size)
            offsets.append(total_size)
            total_size += nbytes
        segment = self._create_segment(name, total_size)
        try:
            prefix = _TENSOR_SHM_PREFIX.pack(_TENSOR_SHM_MAGIC, len(header), data_start)
            segment.buf[: len(prefix)] = prefix
            segment.buf[_TENSOR_SHM_PREFIX.size : _TENSOR_SHM_PREFIX.size + len(header)] = header
            for tensor, nbytes, offset in zip(tensors, tensor_sizes, offsets, strict=True):
                if nbytes:
                    raw = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
                    segment.buf[offset : offset + nbytes] = raw
                    raw.release()
        finally:
            segment.close()
        return {"name": name, "size": total_size, "format": "tensor-v1"}, total_size

    def _notify_stage(self, to_stage: str) -> None:
        if not self._event_notifications:
            return
        target = f"{self._notify_prefix}_{to_stage}.sock"
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sender.setblocking(False)
            sender.sendto(b"1", target)
        except (FileNotFoundError, BlockingIOError, ConnectionRefusedError):
            # Startup races retain the existing key-addressed polling fallback.
            pass
        finally:
            sender.close()

    def wait_for_data(self, timeout: float) -> bool:
        """Wait for a cross-process stage notification when configured."""
        if self._notify_socket is None:
            return False
        ready, _, _ = select.select([self._notify_socket], [], [], timeout)
        if not ready:
            return False
        try:
            while self._notify_socket.recv(4096):
                pass
        except BlockingIOError:
            pass
        return True

    def _read_raw_tensors(self, handle: dict[str, Any]) -> Any:
        segment = shm_pkg.SharedMemory(name=handle["name"])
        try:
            magic, header_len, data_start = _TENSOR_SHM_PREFIX.unpack(bytes(segment.buf[: _TENSOR_SHM_PREFIX.size]))
            if magic != _TENSOR_SHM_MAGIC:
                raise ValueError("shared-memory payload is not tensor-v1")
            header_begin = _TENSOR_SHM_PREFIX.size
            header = self.deserialize_obj(bytes(segment.buf[header_begin : header_begin + header_len]))
            descriptors: list[dict[str, Any]] = []

            def collect(value: Any) -> None:
                if isinstance(value, dict):
                    if _TENSOR_SHM_MARKER in value:
                        descriptors.append(value)
                    else:
                        for item in value.values():
                            collect(item)
                elif isinstance(value, list):
                    for item in value:
                        collect(item)

            collect(header)
            descriptors.sort(key=lambda item: int(item[_TENSOR_SHM_MARKER]))
            indices = [int(item[_TENSOR_SHM_MARKER]) for item in descriptors]
            if indices != list(range(len(descriptors))):
                raise ValueError(f"invalid tensor-v1 descriptor indices: {indices}")
            tensors: list[torch.Tensor] = []
            offset = int(data_start)
            for descriptor in descriptors:
                offset = _align_up(offset)
                dtype = getattr(torch, descriptor["dtype"])
                numel = int(descriptor["numel"])
                nbytes = numel * torch.empty((), dtype=dtype).element_size()
                if offset + nbytes > segment.size:
                    raise ValueError("tensor-v1 descriptor exceeds shared-memory payload")
                if numel:
                    source = torch.frombuffer(
                        segment.buf,
                        dtype=dtype,
                        count=numel,
                        offset=offset,
                    ).reshape(descriptor["shape"])
                    tensors.append(source.clone())
                    del source
                else:
                    tensors.append(torch.empty(descriptor["shape"], dtype=dtype))
                offset += nbytes
            return _restore_raw_tensors(header, tensors)
        finally:
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _is_raw_tensor_handle(handle: dict[str, Any]) -> bool:
        if handle.get("format") == "tensor-v1":
            return True
        try:
            segment = shm_pkg.SharedMemory(name=handle["name"])
            try:
                return bytes(segment.buf[: len(_TENSOR_SHM_MAGIC)]) == _TENSOR_SHM_MAGIC
            finally:
                segment.close()
        except (FileNotFoundError, ValueError):
            return False

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                raw_result = self._put_raw_tensors(put_key, data) if self._raw_tensor_shm else None
                if raw_result is not None:
                    meta, size = raw_result
                    self._metrics["raw_tensor_puts"] += 1
                else:
                    payload = self.serialize_obj(data)
                    size = len(payload)
                    meta = shm_write_bytes(payload, name=put_key)
                fcntl.flock(lockf, fcntl.LOCK_UN)

            # meta contains {'name': ..., 'size': ...}
            metadata = {"shm": meta, "size": size}
            self._pending_keys.add(put_key)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            self._notify_stage(to_stage)

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict[str, Any]) -> tuple[Any, int] | None:
        deserialized = False
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                if self._is_raw_tensor_handle(shm_handle):
                    obj = self._read_raw_tensors(shm_handle)
                else:
                    data_bytes = shm_read_bytes(shm_handle)
                    obj = self.deserialize_obj(data_bytes)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            result = (obj, int(shm_handle.get("size", 0)))
            deserialized = True
            return result
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            if deserialized:
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass

    def _get_by_key(self, get_key: str) -> tuple[Any, int] | None:
        """Read a SHM segment addressed purely by *get_key*."""
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle)
            if result is not None:
                self._pending_keys.discard(get_key)
            return result
        except FileNotFoundError:
            return None
        except ValueError as e:
            # A receiver can observe a newly-created POSIX SHM object before
            # the writer has finished sizing it. Treat that as "not ready yet"
            # so async polling can retry without a traceback.
            if "empty file" in str(e):
                return None
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        except Exception:
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        finally:
            if shm:
                shm.close()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        if metadata is not None:
            if isinstance(metadata, dict) and get_key in metadata:
                metadata = metadata.get(get_key)

            if isinstance(metadata, dict) and "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                if result is not None:
                    self._pending_keys.discard(get_key)
            else:
                # Missing or non-SHM metadata falls back to key-based lookup.
                result = self._get_by_key(get_key)
        else:
            result = self._get_by_key(get_key)

        if result is not None:
            self._metrics["gets"] += 1
        return result

    def cleanup(self, request_id: str) -> None:
        """Best-effort cleanup of unconsumed SHM segments for *request_id*.

        Matches pending keys where *request_id* appears as the full key,
        as a ``_``-delimited prefix, or as a ``_``-delimited suffix.
        If ``get()`` was never called, we unlink it here so /dev/shm
        doesn't leak.
        """
        stale = [
            k
            for k in self._pending_keys
            if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
        ]
        for key in stale:
            self._pending_keys.discard(key)
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
                logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments."""
        for key in list(self._pending_keys):
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
            except Exception:
                pass
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        self._pending_keys.clear()
        if self._notify_socket is not None:
            self._notify_socket.close()
            self._notify_socket = None
        if self._notify_path is not None:
            try:
                os.unlink(self._notify_path)
            except FileNotFoundError:
                pass

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", **self._metrics}
