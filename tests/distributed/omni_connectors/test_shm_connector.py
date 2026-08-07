# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedMemoryConnector focusing on TP / CFG / metadata fallback."""

import os

import pytest
import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def connector():
    c = SharedMemoryConnector({})
    yield c
    c.close()


# ── Key-based read (the fundamental SHM path) ────────────────────────


class TestKeyBasedReadWrite:
    def test_put_then_get_by_key(self, connector):
        data = {"hello": "world", "n": 42}
        ok, size, meta = connector.put("s0", "s1", "test_key_1", data)
        assert ok
        assert size > 0
        assert "shm" in meta
        assert "test_key_1" in connector._pending_keys

        result = connector.get("s0", "s1", "test_key_1", metadata=None)
        assert result is not None
        obj, rsize = result
        assert obj == data
        assert rsize == size
        assert "test_key_1" not in connector._pending_keys
        assert connector._metrics["gets"] == 1

    def test_tensor_payload_removes_lock_file(self, connector):
        key = "tensor_payload"
        payload = torch.ones(2, 2)
        ok, _, metadata = connector.put("s0", "s1", key, payload)
        assert ok

        result = connector.get("s0", "s1", key, metadata=metadata)

        assert result is not None
        assert torch.equal(result[0], payload)
        assert not os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock")

    def test_raw_tensor_payload_round_trip_by_key(self):
        connector = SharedMemoryConnector({"extra": {"raw_tensor_shm": True}})
        key = "raw_tensor_payload"
        payload = {
            "codes": {"audio": torch.arange(12, dtype=torch.long).reshape(3, 4)},
            "meta": {"finished": torch.tensor(False), "request_id": "r0"},
        }
        try:
            ok, size, metadata = connector.put("s1", "s2", key, payload)
            assert ok
            assert size > payload["codes"]["audio"].numel() * payload["codes"]["audio"].element_size()
            assert metadata["shm"]["format"] == "tensor-v1"

            result = connector.get("s1", "s2", key)
            assert result is not None
            restored, restored_size = result
            assert torch.equal(restored["codes"]["audio"], payload["codes"]["audio"])
            assert torch.equal(restored["meta"]["finished"], payload["meta"]["finished"])
            assert restored["meta"]["request_id"] == "r0"
            assert restored_size == size
            assert connector._metrics["raw_tensor_puts"] == 1
        finally:
            connector.close()

    def test_raw_msgspec_payload_round_trip_with_mixed_alignment(self):
        connector = SharedMemoryConnector({"extra": {"raw_tensor_shm": True}})
        key = "raw_struct_mixed_alignment"
        payload = OmniPayloadStruct(
            codes=CodesStruct(audio=torch.arange(6, dtype=torch.int64).reshape(2, 3)),
            # A one-byte tensor before int64 would expose an unaligned-offset
            # bug if every raw tensor did not receive its own alignment.
            meta=MetaStruct(finished=torch.tensor(False), request_id="r-struct"),
        )
        # Struct field order puts codes before meta today; use a plain dict to
        # force the one-byte tensor to precede the int64 tensor as well.
        mixed = {"flag": torch.tensor(True), "payload": payload}
        try:
            ok, _, metadata = connector.put("s1", "s2", key, mixed)
            assert ok
            assert metadata["shm"]["format"] == "tensor-v1"

            result = connector.get("s1", "s2", key)
            assert result is not None
            restored, _ = result
            assert torch.equal(restored["flag"], mixed["flag"])
            assert torch.equal(restored["payload"]["codes"]["audio"], payload.codes.audio)
            assert torch.equal(restored["payload"]["meta"]["finished"], payload.meta.finished)
            assert restored["payload"]["meta"]["request_id"] == "r-struct"
            assert connector._metrics["raw_tensor_puts"] == 1
        finally:
            connector.close()

    def test_event_notification_disabled_reports_unavailable(self, connector):
        assert connector.event_notifications_enabled is False
        assert connector.wait_for_data(0.001) is False

    def test_event_notification_wakes_target_stage(self):
        namespace = f"pytest-{os.getpid()}"
        sender = SharedMemoryConnector(
            {
                "stage_id": 1,
                "extra": {
                    "shm_event_notifications": True,
                    "shm_notification_namespace": namespace,
                },
            }
        )
        receiver = SharedMemoryConnector(
            {
                "stage_id": 2,
                "extra": {
                    "shm_event_notifications": True,
                    "shm_notification_namespace": namespace,
                },
            }
        )
        try:
            ok, _, _ = sender.put("1", "2", "notify_target_stage", {"ready": True})
            assert ok
            assert receiver.wait_for_data(0.1)
            assert receiver.get("1", "2", "notify_target_stage") is not None
        finally:
            sender.close()
            receiver.close()

    def test_falsey_payload_removes_lock_file(self, connector):
        key = "falsey_payload"
        ok, _, metadata = connector.put("s0", "s1", key, 0)
        assert ok

        result = connector.get("s0", "s1", key, metadata=metadata)

        assert result is not None
        assert result[0] == 0
        assert not os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock")

    def test_get_nonexistent_key_returns_none(self, connector):
        result = connector.get("s0", "s1", "no_such_key_xyz", metadata=None)
        assert result is None

    def test_get_empty_shm_race_returns_none(self, connector, monkeypatch):
        def raise_empty_file(*args, **kwargs):
            raise ValueError("cannot mmap an empty file")

        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_pkg.SharedMemory",
            raise_empty_file,
        )

        result = connector.get("s0", "s1", "not_ready_yet", metadata=None)

        assert result is None

    def test_rank_aware_keys_independent(self, connector):
        """Each TP rank writes/reads its own key — simulates homogeneous TP."""
        payloads = {}
        for rank in range(4):
            key = f"req1_s0_0_{rank}_{rank}"
            data = {"rank": rank, "values": list(range(rank, rank + 3))}
            ok, _, _ = connector.put("s0", "s1", key, data)
            assert ok
            payloads[rank] = data

        for rank in range(4):
            key = f"req1_s0_0_{rank}_{rank}"
            result = connector.get("s0", "s1", key, metadata=None)
            assert result is not None
            obj, _ = result
            assert obj == payloads[rank]


# ── Metadata fallback behaviour ──────────────────────────────────────


class TestMetadataFallback:
    def test_rdma_style_metadata_falls_back_to_key(self, connector):
        """source_host/source_port metadata should be ignored; key read used."""
        data = {"payload": True}
        connector.put("s0", "s1", "fb_key_1", data)

        rdma_meta = {"source_host": "10.0.0.1", "source_port": 12345}
        result = connector.get("s0", "s1", "fb_key_1", metadata=rdma_meta)
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_non_dict_metadata_falls_back_to_key(self, connector):
        data = {"val": 99}
        connector.put("s0", "s1", "fb_key_2", data)

        result = connector.get("s0", "s1", "fb_key_2", metadata="not_a_dict")
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_empty_dict_metadata_falls_back_to_key(self, connector):
        data = {"x": 1}
        connector.put("s0", "s1", "fb_key_3", data)

        result = connector.get("s0", "s1", "fb_key_3", metadata={})
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_shm_handle_metadata_still_works(self, connector):
        """When metadata contains a proper 'shm' handle, use it directly."""
        data = {"direct": True}
        ok, size, meta = connector.put("s0", "s1", "shm_direct_1", data)
        assert ok
        result = connector.get("s0", "s1", "shm_direct_1", metadata=meta)
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_metadata_keyed_by_request_id(self, connector):
        """Metadata wrapped as {get_key: actual_meta} should be unwrapped."""
        data = {"wrapped": True}
        ok, size, meta = connector.put("s0", "s1", "wrap_key", data)
        assert ok
        wrapped = {"wrap_key": meta}
        result = connector.get("s0", "s1", "wrap_key", metadata=wrapped)
        assert result is not None
        obj, _ = result
        assert obj == data


# ── Heterogeneous TP multi-key read ──────────────────────────────────


class TestHeteroTPMultiKey:
    def test_receiver_reads_multiple_sender_keys(self, connector):
        """Simulates from_tp=2 -> to_tp=1: receiver reads 2 keys and merges."""
        for sender_rank in range(2):
            key = f"req1_s0_0_{sender_rank}_0"
            data = {"sender": sender_rank, "shard": [sender_rank * 10]}
            connector.put("s0", "s1", key, data)

        shards = []
        for sender_rank in range(2):
            key = f"req1_s0_0_{sender_rank}_0"
            result = connector.get("s0", "s1", key, metadata=None)
            assert result is not None
            obj, _ = result
            shards.append(obj)

        assert len(shards) == 2
        assert shards[0]["sender"] == 0
        assert shards[1]["sender"] == 1

    def test_sender_writes_multiple_receiver_keys(self, connector):
        """Simulates from_tp=1 -> to_tp=2: sender writes 2 sliced keys."""
        for recv_rank in range(2):
            key = f"req1_s0_0_0_{recv_rank}"
            data = {"target": recv_rank, "slice": list(range(recv_rank, recv_rank + 2))}
            connector.put("s0", "s1", key, data)

        for recv_rank in range(2):
            key = f"req1_s0_0_0_{recv_rank}"
            result = connector.get("s0", "s1", key, metadata=None)
            assert result is not None
            obj, _ = result
            assert obj["target"] == recv_rank


# ── Cleanup ──────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_removes_unconsumed_segment(self, connector):
        data = {"leak": True}
        connector.put("s0", "s1", "cleanup_req_42", data)
        assert "cleanup_req_42" in connector._pending_keys

        connector.cleanup("req_42")
        assert "cleanup_req_42" not in connector._pending_keys

        result = connector.get("s0", "s1", "cleanup_req_42", metadata=None)
        assert result is None

    def test_cleanup_noop_for_consumed_segment(self, connector):
        data = {"consumed": True}
        connector.put("s0", "s1", "consumed_req_99", data)
        connector.get("s0", "s1", "consumed_req_99", metadata=None)

        connector.cleanup("req_99")
        assert "consumed_req_99" not in connector._pending_keys

    def test_close_cleans_all_pending(self, connector):
        for i in range(3):
            connector.put("s0", "s1", f"close_test_{i}", {"i": i})

        assert len(connector._pending_keys) == 3
        connector.close()
        assert len(connector._pending_keys) == 0
