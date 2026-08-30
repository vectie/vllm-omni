# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _apply_repetition_penalty,
    _apply_repetition_penalty_from_frequencies,
    _apply_top_k_top_p,
    _bounded_codec_distribution,
    _bounded_top_k_top_p_candidates,
    _graphable_advance_codec_state,
    _graphable_codec_distribution,
    _graphable_codec_sample,
    _load_talker_static_w8a8_scales,
    _max_audio_tokens,
    _prepare_talker_static_w8a8_calibration,
    _restore_weight_norm_weight,
    _talker_static_w8a8_suffixes,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.ones(2, 4)


def test_wrapper_always_delegates_talker_to_native_ar_path() -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.talker = _FakeNativeTalker()

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.arange(2),
        model_intermediate_buffer=[{"request_id": "req"}],
    )

    assert output.shape == (2, 4)
    assert model.talker.forward_kwargs["model_intermediate_buffer"][0]["request_id"] == "req"


def test_wrapper_llm_preprocess_embeds_plain_requests(monkeypatch) -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "llm"
    input_ids = torch.tensor([11, 12])
    expected = torch.ones((2, 4))
    monkeypatch.setattr(model, "get_input_embeddings", lambda _: expected)

    returned_ids, embeds, updates = model.preprocess(input_ids)

    assert returned_ids is input_ids
    assert embeds is expected
    assert updates == {}


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._batch_stop_logits = None
    talker._batch_stop_token_ids = None
    talker._stop_logits_constants = None
    talker._stop_token_constants = None
    talker.direct_stop_sampler = False
    talker.batched_codec_output = False
    talker.deferred_chunk_eos = False
    talker._request_transport_codes = {}
    talker._request_transport_chunks = {}
    talker._request_generators = {}
    talker._request_audio_states = {}
    talker._request_repetition_frequencies = {}
    talker._request_codec_rings = {}
    talker._deferred_cleanup_ids = set()
    talker._static_w8a8_calibration_path = None
    talker._static_w8a8_collectors = {}
    talker._codec_vocab_ids = torch.arange(8)
    talker._codec_min_tokens = 50
    talker._codec_seed = 42
    talker._fused_codec_sampler_enabled = False
    talker._fused_codec_distribution_enabled = False
    talker._fused_codec_distribution_disabled = False
    talker._fused_codec_distribution_validated_steps = set()
    talker._fixed_codec_ring_enabled = False
    talker._graph_codec_state_enabled = False
    talker._graph_codec_state_request_id = None
    talker._fused_codec_mask_eos_value = None
    talker._fused_codec_sample_position = torch.zeros((1, 1), dtype=torch.long)
    talker._fused_codec_pending_sample = torch.full((1,), -1, dtype=torch.long)
    talker._fused_codec_sampler_prepared = False
    talker._fused_codec_sampler_request_id = None
    return talker


def test_talker_static_w8a8_target_validation() -> None:
    assert _talker_static_w8a8_suffixes("gate_up") == ("mlp.gate_up_proj",)
    assert _talker_static_w8a8_suffixes("qkv,gate_up") == (
        "self_attn.qkv_proj",
        "mlp.gate_up_proj",
    )
    with pytest.raises(ValueError, match="expected qkv, gate_up"):
        _talker_static_w8a8_suffixes("down")


def test_talker_static_w8a8_calibration_collects_projection_inputs(tmp_path) -> None:
    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.qkv_proj = nn.Linear(3, 4, bias=False)
            self.mlp = nn.Module()
            self.mlp.gate_up_proj = nn.Linear(3, 6, bias=False)

    model = nn.Module()
    model.layers = nn.ModuleList([_Block()])
    collectors = _prepare_talker_static_w8a8_calibration(model, "gate_up")
    model.layers[0].mlp.gate_up_proj(torch.tensor([[1.0, -3.5, 2.0]]))

    assert collectors.keys() == {"layers.0.mlp.gate_up_proj"}
    assert collectors["layers.0.mlp.gate_up_proj"].item() == 3.5

    calibration = tmp_path / "talker-scales.json"
    talker = _make_talker()
    talker._static_w8a8_calibration_path = str(calibration)
    talker._static_w8a8_collectors = collectors
    talker.on_requests_finished(["calibration-request"])

    assert _load_talker_static_w8a8_scales(str(calibration)) == {
        "layers.0.mlp.gate_up_proj": 3.5,
    }


def test_fused_codec_sampler_stages_fixed_request_state() -> None:
    talker = _make_talker()
    talker._fused_codec_sampler_enabled = True
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_uniform = torch.full((1, 1), 0.5)
    talker._fused_codec_mask_eos = torch.ones(1, dtype=torch.bool)
    talker._fused_codec_expired = torch.full((1, 1), -1, dtype=torch.long)
    history = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6])
    talker._request_audio_states["req-fused"] = {
        "codes": history,
        "step": 4,
        "min_tokens": 5,
    }

    prepared = talker.prepare_fused_codec_sampler_inputs(
        model_intermediate_buffer=[{"request_id": "req-fused"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )

    assert prepared is True
    assert talker._fused_codec_sampler_request_id == "req-fused"
    assert talker._fused_codec_sampler_prepared is True
    assert talker._fused_codec_mask_eos.item() is True
    assert talker._fused_codec_expired.item() == 0
    expected = torch.bincount(history, minlength=8).float().reshape(1, -1)
    assert torch.equal(talker._fused_codec_frequencies, expected)
    assert talker._request_repetition_frequencies["req-fused"].data_ptr() == (
        talker._fused_codec_frequencies.data_ptr()
    )


def test_fused_codec_sampler_does_not_stage_final_prefill_chunk() -> None:
    talker = _make_talker()
    talker._fused_codec_sampler_enabled = True
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_uniform = torch.full((1, 1), 0.5)
    talker._fused_codec_mask_eos = torch.ones(1, dtype=torch.bool)
    talker._fused_codec_expired = torch.full((1, 1), -1, dtype=torch.long)
    talker._request_audio_states["req-prefill"] = {
        "codes": torch.empty(0, dtype=torch.long),
        "step": 0,
        "min_tokens": 5,
    }

    prepared = talker.prepare_fused_codec_sampler_inputs(
        model_intermediate_buffer=[{"request_id": "req-prefill"}],
        request_token_spans=[(0, 12)],
        request_sample_eligible=[True],
    )

    assert prepared is False
    assert talker._fused_codec_sampler_prepared is False
    assert talker._request_generators == {}


def test_fused_codec_distribution_staging_does_not_advance_rng() -> None:
    talker = _make_talker()
    talker._fused_codec_distribution_enabled = True
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_uniform = torch.full((1, 1), 0.5)
    talker._fused_codec_mask_eos = torch.ones(1, dtype=torch.bool)
    talker._fused_codec_expired = torch.full((1, 1), -1, dtype=torch.long)
    talker._request_audio_states["req-distribution"] = {
        "codes": torch.tensor([1, 2]),
        "step": 2,
        "min_tokens": 5,
    }

    assert talker.prepare_fused_codec_sampler_inputs(
        model_intermediate_buffer=[{"request_id": "req-distribution"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )

    assert talker._request_generators == {}
    assert talker._fused_codec_mask_eos.item() is True


def test_graph_codec_state_stages_invariant_controls_only_when_needed() -> None:
    talker = _make_talker()
    talker._fused_codec_distribution_enabled = True
    talker._graph_codec_state_enabled = True
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_history_slab = torch.full((16,), -1, dtype=torch.long)
    talker._fused_codec_pending_sample = torch.full((1,), -1, dtype=torch.long)
    talker._fused_codec_mask_eos = torch.ones(1, dtype=torch.bool)
    talker._fused_codec_expired = torch.tensor([[123]], dtype=torch.long)
    state = {"codes": torch.tensor([1]), "step": 2, "min_tokens": 5}
    talker._request_audio_states["req-controls"] = state

    assert talker.prepare_fused_codec_sampler_inputs(
        model_intermediate_buffer=[{"request_id": "req-controls"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )
    assert talker._fused_codec_mask_eos_value is True
    # The graph-owned FIFO does not consume the legacy expired-code slab.
    assert talker._fused_codec_expired.item() == 123

    state["step"] = 5
    assert talker.prepare_fused_codec_sampler_inputs(
        model_intermediate_buffer=[{"request_id": "req-controls"}],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )
    assert talker._fused_codec_mask_eos_value is False
    assert talker._fused_codec_mask_eos.item() is False


def test_fused_codec_distribution_keeps_native_multinomial_mapping() -> None:
    talker = _make_talker()
    talker._codec_temperature = 0.8
    talker._codec_top_k = 5
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._fused_codec_distribution_enabled = True
    talker._fused_codec_sampler_prepared = True
    talker._fused_codec_sampler_request_id = "req-distribution"
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_penalty = torch.tensor([1.05])
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    hidden = torch.tensor([[0.5, -0.25, 0.75, 1.0]])
    history = torch.empty(0, dtype=torch.long)
    state = {"step": 0, "min_tokens": 2, "max_tokens": 64}
    talker._request_audio_states["req-distribution"] = state
    probabilities, candidate_ids = _bounded_codec_distribution(
        hidden,
        talker._fused_codec_frequencies,
        talker.head_code[0].weight,
        talker._fused_codec_penalty,
        temperature=0.8,
        top_k=5,
        top_p=0.85,
        eos_id=7,
        mask_eos=True,
    )
    talker._fused_codec_probabilities = probabilities.clone()
    talker._fused_codec_candidate_ids = candidate_ids.clone()

    reference_generator = torch.Generator().manual_seed(42)
    expected_position = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=reference_generator,
    )
    expected = candidate_ids.gather(-1, expected_position).reshape(())

    sampled = talker._consume_fused_codec_distribution(
        hidden,
        history,
        "req-distribution",
        0,
    )

    assert torch.equal(sampled, expected)
    assert talker._fused_codec_distribution_validated_steps == {0}
    assert torch.equal(
        talker._request_generators["req-distribution"].get_state(),
        reference_generator.get_state(),
    )
    assert talker._request_repetition_frequencies["req-distribution"][0, int(expected)] == 1


def test_graph_codec_state_samples_into_fixed_native_output_slabs() -> None:
    talker = _make_talker()
    talker._codec_temperature = 0.8
    talker._codec_top_k = 5
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._fused_codec_distribution_enabled = True
    talker._graph_codec_state_enabled = True
    talker._fused_codec_sampler_prepared = True
    talker._fused_codec_sampler_request_id = "req-fixed-native"
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_penalty = torch.tensor([1.05])
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    hidden = torch.tensor([[0.5, -0.25, 0.75, 1.0]])
    state = {"step": 2, "min_tokens": 2, "max_tokens": 64}
    talker._request_audio_states["req-fixed-native"] = state
    probabilities, candidate_ids = _bounded_codec_distribution(
        hidden,
        talker._fused_codec_frequencies,
        talker.head_code[0].weight,
        talker._fused_codec_penalty,
        temperature=0.8,
        top_k=5,
        top_p=0.85,
        eos_id=7,
        mask_eos=False,
    )
    talker._fused_codec_probabilities = probabilities.clone()
    talker._fused_codec_candidate_ids = candidate_ids.clone()
    output_ptr = talker._fused_codec_pending_sample.data_ptr()
    reference_generator = torch.Generator().manual_seed(42)
    expected_position = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=reference_generator,
    )
    expected = candidate_ids.gather(-1, expected_position).reshape(())

    sampled = talker._consume_fused_codec_distribution(
        hidden,
        torch.empty(0, dtype=torch.long),
        "req-fixed-native",
        2,
    )

    assert torch.equal(sampled, expected)
    assert sampled.data_ptr() == output_ptr
    assert torch.equal(
        talker._request_generators["req-fixed-native"].get_state(),
        reference_generator.get_state(),
    )


def test_fused_codec_distribution_fails_closed_on_stale_graph_output() -> None:
    talker = _make_talker()
    talker._codec_temperature = 0.8
    talker._codec_top_k = 5
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._fused_codec_distribution_enabled = True
    talker._fused_codec_sampler_prepared = True
    talker._fused_codec_sampler_request_id = "req-stale"
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_penalty = torch.tensor([1.05])
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    hidden = torch.tensor([[0.5, -0.25, 0.75, 1.0]])
    state = {"step": 0, "min_tokens": 2, "max_tokens": 64}
    talker._request_audio_states["req-stale"] = state
    eager_probabilities, eager_ids = _bounded_codec_distribution(
        hidden,
        talker._fused_codec_frequencies,
        talker.head_code[0].weight,
        talker._fused_codec_penalty,
        temperature=0.8,
        top_k=5,
        top_p=0.85,
        eos_id=7,
        mask_eos=True,
    )
    talker._fused_codec_probabilities = torch.roll(eager_probabilities, 1, dims=-1)
    talker._fused_codec_candidate_ids = eager_ids.clone()
    reference_generator = torch.Generator().manual_seed(42)
    expected_position = torch.multinomial(
        eager_probabilities,
        num_samples=1,
        generator=reference_generator,
    )
    expected = eager_ids.gather(-1, expected_position).reshape(())

    sampled = talker._consume_fused_codec_distribution(
        hidden,
        torch.empty(0, dtype=torch.long),
        "req-stale",
        0,
    )

    assert torch.equal(sampled, expected)
    assert talker._fused_codec_distribution_disabled is True
    assert torch.equal(
        talker._request_generators["req-stale"].get_state(),
        reference_generator.get_state(),
    )


def test_make_output_consumes_fused_codec_result_without_second_sampler(monkeypatch) -> None:
    talker = _make_talker()
    talker._fused_codec_sampler_enabled = True
    talker._fused_codec_sampler_prepared = True
    talker._fused_codec_sampler_request_id = "req-fused"
    talker._fused_codec_sampled = torch.tensor([[3]])
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_next_frequencies = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    state = {"step": 0, "min_tokens": 50, "max_tokens": 64}
    talker._request_audio_states["req-fused"] = state
    monkeypatch.setattr(
        talker,
        "_sample_audio_code",
        lambda *_args: pytest.fail("standalone sampler must be bypassed"),
    )
    info = {
        "request_id": "req-fused",
        "audio_state": state,
        "audio_codes": {"accumulated": torch.tensor([1])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
        request_sample_eligible=[True],
    )

    assert output.multimodal_outputs["codes"]["audio"][0].tolist() == [[3]]
    assert talker._fused_codec_sampler_prepared is False
    assert torch.equal(
        talker._request_repetition_frequencies["req-fused"],
        talker._fused_codec_next_frequencies,
    )


def test_talker_batches_codec_transport_at_initial_and_steady_boundaries(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "2")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES", "3")
    talker = _make_talker()
    talker.batched_codec_output = True

    def push(code: int, *, finished: bool = False) -> torch.Tensor:
        return talker._transport_codec_delta(
            "req-batched-output",
            torch.tensor([[code]], dtype=torch.long),
            finished=finished,
            native_duplex=False,
        )

    assert push(1).numel() == 0
    assert push(2).tolist() == [[1, 2]]
    assert push(3).numel() == 0
    assert push(4).numel() == 0
    assert push(5).tolist() == [[3, 4, 5]]
    assert push(6).numel() == 0
    flushed = talker._transport_codec_delta(
        "req-batched-output",
        torch.empty((0, 1), dtype=torch.long),
        finished=True,
        native_duplex=False,
    )
    assert flushed.tolist() == [[6]]


def test_talker_marks_only_publishable_codec_chunks_as_sparse_output(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "2")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES", "3")
    talker = _make_talker()
    talker.batched_codec_output = True
    samples = iter((torch.tensor(3), torch.tensor(4)))
    monkeypatch.setattr(talker, "_sample_audio_code", lambda *_args: next(samples))
    info = {
        "request_id": "req-sparse-output",
        "audio_state": {"step": 0, "min_tokens": 50, "max_tokens": 64},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert first.multimodal_outputs["meta"]["req_id"] == []
    assert first.multimodal_outputs["codes"]["audio"] == []
    assert second.multimodal_outputs["meta"]["req_id"] == ["req-sparse-output"]
    assert second.multimodal_outputs["codes"]["audio"][0].tolist() == [[3, 4]]


def test_talker_deferred_eos_trims_terminal_tail_at_chunk_boundary(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "3")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES", "3")
    talker = _make_talker()
    talker.batched_codec_output = True
    talker.deferred_chunk_eos = True
    samples = iter((torch.tensor(1), torch.tensor(7), torch.tensor(3)))
    monkeypatch.setattr(talker, "_sample_audio_code", lambda *_args: next(samples))
    monkeypatch.setattr(
        talker,
        "_sampled_code_is_eos",
        lambda *_args, **_kwargs: pytest.fail("deferred EOS must avoid per-token scalar reads"),
    )
    info = {
        "request_id": "req-deferred-eos",
        "audio_state": {"step": 0, "min_tokens": 0, "max_tokens": 10},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    outputs = [
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )
        for _ in range(3)
    ]

    assert outputs[0].multimodal_outputs["codes"]["audio"] == []
    assert outputs[1].multimodal_outputs["codes"]["audio"] == []
    assert outputs[2].multimodal_outputs["codes"]["audio"][0].tolist() == [[1]]
    assert outputs[2].multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker._request_transport_codes["req-deferred-eos"] == []


def test_talker_exact_second_step_gate_allows_only_unpublished_steady_work(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "3")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES", "3")
    talker = _make_talker()
    talker.batched_codec_output = True
    talker.deferred_chunk_eos = True
    talker._request_audio_states["req-two-step"] = {
        "step": 1,
        "max_tokens": 10,
        "finished": False,
    }
    talker._request_transport_codes["req-two-step"] = [torch.tensor([2])]
    empty = SimpleNamespace(
        multimodal_outputs={
            "codes": {"audio": []},
            "meta": {
                "req_id": ["req-two-step"],
                "finished": [torch.tensor(False)],
            },
        }
    )

    assert talker.can_run_exact_second_step("req-two-step", empty)

    published = SimpleNamespace(
        multimodal_outputs={
            "codes": {"audio": [torch.tensor([[1, 2, 3]])]},
            "meta": {"req_id": ["req-two-step"], "finished": [False]},
        }
    )
    assert not talker.can_run_exact_second_step("req-two-step", published)

    talker._request_audio_states["req-two-step"]["finished"] = True
    assert not talker.can_run_exact_second_step("req-two-step", empty)


def test_talker_codec_parity_trace_records_complete_sample_and_publish_sequences(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "3")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES", "3")
    talker = _make_talker()
    trace_path = tmp_path / "codec-parity.jsonl"
    talker._codec_parity_trace_path = str(trace_path)
    talker._request_codec_parity = {}
    talker.batched_codec_output = True
    talker.deferred_chunk_eos = True
    samples = iter((torch.tensor(1), torch.tensor(7), torch.tensor(3)))
    monkeypatch.setattr(talker, "_sample_audio_code", lambda *_args: next(samples))
    info = {
        "request_id": "req-parity-secret",
        "audio_state": {"step": 0, "min_tokens": 0, "max_tokens": 10},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    for _ in range(3):
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )
    talker.on_requests_finished(["req-parity-secret"])

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["format"] == "minicpmo45-codec-parity-v2"
    assert payload["sample_count"] == 3
    assert payload["published_count"] == 1
    assert payload["samples"] == [1, 7, 3]
    assert payload["published"] == [1]
    assert payload["steps"][-1]["finished"] is True
    assert payload["final_rng_sha256"] is None
    assert "req-parity-secret" not in trace_path.read_text(encoding="utf-8")


def test_codec_parity_trace_binds_distribution_to_native_sample(tmp_path) -> None:
    talker = _make_talker()
    trace_path = tmp_path / "codec-distribution-parity.jsonl"
    talker._codec_parity_trace_path = str(trace_path)
    talker._request_codec_parity = {}
    talker._request_codec_parity_pending = {}
    talker._codec_temperature = 0.8
    talker._codec_top_k = 5
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._fused_codec_distribution_enabled = True
    talker._fused_codec_sampler_prepared = True
    talker._fused_codec_sampler_request_id = "req-distribution-parity"
    talker._fused_codec_frequencies = torch.zeros(1, 8)
    talker._fused_codec_penalty = torch.tensor([1.05])
    talker.head_code = nn.ModuleList([nn.Linear(4, 8, bias=False)])
    hidden = torch.tensor([[0.5, -0.25, 0.75, 1.0]])
    history = torch.empty(0, dtype=torch.long)
    talker._request_audio_states["req-distribution-parity"] = {
        "step": 0,
        "min_tokens": 2,
        "max_tokens": 64,
    }
    probabilities, candidate_ids = _bounded_codec_distribution(
        hidden,
        talker._fused_codec_frequencies,
        talker.head_code[0].weight,
        talker._fused_codec_penalty,
        temperature=0.8,
        top_k=5,
        top_p=0.85,
        eos_id=7,
        mask_eos=True,
    )
    talker._fused_codec_probabilities = probabilities.clone()
    talker._fused_codec_candidate_ids = candidate_ids.clone()

    sampled = talker._consume_fused_codec_distribution(
        hidden,
        history,
        "req-distribution-parity",
        0,
    )
    talker._record_codec_parity_step(
        "req-distribution-parity",
        sampled,
        sampled.reshape(1, 1),
        step=0,
        min_tokens=2,
        reached_limit=False,
        finished=False,
    )
    talker.on_requests_finished(["req-distribution-parity"])

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["distribution_samples"] == [int(sampled)]
    assert payload["distribution_steps"] == [0]
    assert payload["candidate_ids"] == [candidate_ids.reshape(-1).tolist()]
    assert torch.allclose(
        torch.tensor(payload["probabilities"]),
        probabilities,
    )
    assert payload["candidate_ids_after"] == payload["candidate_ids"]
    assert torch.allclose(
        torch.tensor(payload["probabilities_after"]),
        torch.tensor(payload["probabilities"]),
    )


def test_talker_deferred_eos_flushes_limit_without_boundary_sample(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "4")
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES", "4")
    talker = _make_talker()
    talker.batched_codec_output = True
    talker.deferred_chunk_eos = True
    samples = iter((torch.tensor(1), torch.tensor(2), torch.tensor(6)))
    monkeypatch.setattr(talker, "_sample_audio_code", lambda *_args: next(samples))
    info = {
        "request_id": "req-deferred-limit",
        "audio_state": {"step": 0, "min_tokens": 50, "max_tokens": 3},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    outputs = [
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )
        for _ in range(3)
    ]

    assert outputs[0].multimodal_outputs["codes"]["audio"] == []
    assert outputs[1].multimodal_outputs["codes"]["audio"] == []
    assert outputs[2].multimodal_outputs["codes"]["audio"][0].tolist() == [[1, 2]]
    assert outputs[2].multimodal_outputs["meta"]["finished"][0].item() is True


def _routed(output, index: int):
    return to_payload_element(
        output.multimodal_outputs,
        index,
        index,
        index + 1,
        seq_len=2,
        scheduled_seq_len=2,
    )


@pytest.mark.parametrize(
    ("condition_tokens", "expected"),
    [(3, 64), (100, 1000), (1000, 2048)],
)
def test_audio_token_limit_scales_with_condition_length(
    condition_tokens: int,
    expected: int,
) -> None:
    assert _max_audio_tokens(condition_tokens) == expected


def test_device_native_repetition_penalty_matches_bincount_reference() -> None:
    logits = torch.tensor([[-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, -4.0, 0.25]])
    history = torch.tensor([1, 1, 3, 5, 1, 7, 5, 5, 5, 2, 4, 4, 6, 0, 2, 2, 3])
    recent = history[-16:]
    reference_frequencies = torch.bincount(recent, minlength=logits.shape[-1]).to(logits.dtype)
    expected = _apply_repetition_penalty_from_frequencies(
        logits,
        reference_frequencies,
        penalty=1.05,
    )

    actual = _apply_repetition_penalty(logits, history, penalty=1.05, window_size=16)

    assert torch.equal(actual, expected)


def test_resident_repetition_penalty_skips_scalar_materialization(monkeypatch) -> None:
    logits = torch.tensor([[-2.0, -1.0, 0.5, 1.0]])
    frequencies = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    penalty = torch.tensor([1.05])
    expected = torch.where(
        logits < 0,
        logits * torch.pow(penalty, frequencies),
        logits / torch.pow(penalty, frequencies),
    )

    monkeypatch.setattr(
        torch,
        "as_tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resident penalty must not be rematerialized")
        ),
    )

    actual = _apply_repetition_penalty_from_frequencies(
        logits,
        frequencies,
        penalty=penalty,
    )

    assert torch.equal(actual, expected)


def test_incremental_repetition_frequencies_match_sliding_window() -> None:
    talker = _make_talker()
    logits = torch.zeros(1, 8)
    history = torch.tensor([0, 1, 1, 2, 3, 5, 5, 5, 7, 0, 2, 4, 6, 6, 6, 1])
    frequencies = talker._repetition_frequencies("req", history, logits)
    sampled = torch.tensor(6)

    talker._advance_repetition_frequencies("req", history, sampled, frequencies)

    expected_history = torch.cat([history[-15:], sampled.reshape(1)])
    expected = torch.bincount(expected_history, minlength=8).to(logits.dtype).reshape(1, -1)
    assert torch.equal(talker._request_repetition_frequencies["req"], expected)


def test_fixed_codec_ring_matches_growing_history_for_many_wraps() -> None:
    talker = _make_talker()
    talker._fixed_codec_ring_enabled = True
    logits = torch.zeros(1, 8)
    history = torch.empty(0, dtype=torch.long)
    frequencies = talker._repetition_frequencies("req-ring", history, logits)
    reference: list[int] = []

    for value in [0, 1, 1, 2, 3, 5, 7, 6, 4, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2] * 3:
        sampled = torch.tensor(value)
        talker._advance_repetition_frequencies(
            "req-ring",
            history,
            sampled,
            frequencies,
        )
        reference.append(value)
        reference = reference[-16:]
        frequencies = talker._request_repetition_frequencies["req-ring"]
        history = talker._codec_ring_history("req-ring", history)
        expected = torch.bincount(
            torch.tensor(reference),
            minlength=8,
        ).to(logits.dtype).reshape(1, -1)
        assert torch.equal(frequencies, expected)

    entry = talker._request_codec_rings["req-ring"]
    assert entry["slab"].numel() == 16
    assert entry["length"] == 16
    assert sorted(history.tolist()) == sorted(reference)


def test_fixed_codec_ring_initializes_from_legacy_history() -> None:
    talker = _make_talker()
    talker._fixed_codec_ring_enabled = True
    history = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 3)
    logits = torch.zeros(1, 8)

    frequencies = talker._repetition_frequencies("req-restore", history, logits)
    talker._advance_repetition_frequencies(
        "req-restore",
        history,
        torch.tensor(3),
        frequencies,
    )

    expected_history = torch.cat([history[-15:], torch.tensor([3])])
    expected = torch.bincount(expected_history, minlength=8).to(logits.dtype).reshape(1, -1)
    assert torch.equal(talker._request_repetition_frequencies["req-restore"], expected)


def test_graphable_codec_state_matches_sliding_window_counts() -> None:
    frequencies = torch.zeros(1, 8)
    history = torch.full((16,), -1, dtype=torch.long)
    pending = torch.full((1,), -1, dtype=torch.long)
    vocab_ids = torch.arange(8)
    reference: list[int] = []

    for value in [0, 1, 1, 2, 3, 5, 7, 6, 4, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2] * 3:
        frequencies, history = _graphable_advance_codec_state(
            frequencies,
            history,
            pending,
            vocab_ids,
        )
        expected = torch.bincount(
            torch.tensor(reference[-16:], dtype=torch.long),
            minlength=8,
        ).to(frequencies.dtype).reshape(1, -1)
        assert torch.equal(frequencies, expected)
        assert history[history >= 0].tolist() == reference[-16:]
        pending.fill_(value)
        reference.append(value)

    frequencies, history = _graphable_advance_codec_state(
        frequencies,
        history,
        pending,
        vocab_ids,
    )
    expected = torch.bincount(
        torch.tensor(reference[-16:], dtype=torch.long),
        minlength=8,
    ).to(frequencies.dtype).reshape(1, -1)
    assert torch.equal(frequencies, expected)
    assert history.tolist() == reference[-16:]


def test_bounded_codec_candidates_match_full_warper_distribution() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 128, generator=generator)
    expected_logits = _apply_top_k_top_p(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )
    expected = torch.softmax(expected_logits, dim=-1)

    candidate_logits, candidate_ids = _bounded_top_k_top_p_candidates(
        logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )
    actual = torch.zeros_like(expected).scatter(
        -1,
        candidate_ids,
        torch.softmax(candidate_logits, dim=-1),
    )

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("mask_eos", [False, True])
def test_graphable_codec_distribution_matches_eager_filter(mask_eos: bool) -> None:
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(1, 16, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(128, 16, generator=generator, dtype=torch.bfloat16)
    frequencies = torch.randint(0, 4, (1, 128), generator=generator).float()
    penalty = torch.tensor([1.05])

    logits = torch.nn.functional.linear(hidden, weight).float() / 0.8
    expected_logits = _apply_repetition_penalty_from_frequencies(
        logits,
        frequencies,
        penalty=1.05,
    )
    if mask_eos:
        expected_logits[..., 127] = float("-inf")
    expected_candidates, expected_ids = _bounded_top_k_top_p_candidates(
        expected_logits,
        top_k=25,
        top_p=0.85,
        min_tokens_to_keep=3,
    )
    expected_probabilities = torch.softmax(expected_candidates, dim=-1)

    probabilities, candidate_ids = _bounded_codec_distribution(
        hidden,
        frequencies,
        weight,
        penalty,
        temperature=0.8,
        top_k=25,
        top_p=0.85,
        eos_id=127,
        mask_eos=mask_eos,
    )

    assert torch.equal(candidate_ids, expected_ids)
    assert torch.equal(probabilities, expected_probabilities)

    graphable_probabilities, graphable_ids = _graphable_codec_distribution(
        hidden,
        frequencies,
        weight,
        penalty,
        torch.tensor([mask_eos]),
        temperature=0.8,
        top_k=25,
        top_p=0.85,
        eos_id=127,
    )
    assert torch.equal(graphable_ids, expected_ids)
    assert torch.equal(graphable_probabilities, expected_probabilities)


@pytest.mark.parametrize("mask_eos", [False, True])
def test_inverse_cdf_codec_sample_matches_bounded_distribution(mask_eos: bool) -> None:
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(1, 16, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(128, 16, generator=generator, dtype=torch.bfloat16)
    frequencies = torch.randint(0, 4, (1, 128), generator=generator).float()
    penalty = torch.tensor([1.05])
    uniform = torch.tensor([[0.417]])
    expired = torch.tensor([[7]])
    vocab_ids = torch.arange(128).reshape(1, -1)

    probabilities, candidate_ids = _bounded_codec_distribution(
        hidden,
        frequencies,
        weight,
        penalty,
        temperature=0.8,
        top_k=25,
        top_p=0.85,
        eos_id=127,
        mask_eos=mask_eos,
    )
    expected_position = torch.sum(
        probabilities.cumsum(dim=-1) < uniform,
        dim=-1,
        keepdim=True,
    ).clamp_max_(probabilities.shape[-1] - 1)
    expected_sample = candidate_ids.gather(-1, expected_position)
    expected_frequencies = frequencies + (vocab_ids == expected_sample).float()
    expected_frequencies -= (vocab_ids == expired).float()

    sampled, next_frequencies = _graphable_codec_sample(
        hidden,
        frequencies,
        weight,
        penalty,
        uniform,
        torch.tensor([mask_eos]),
        expired,
        vocab_ids,
        temperature=0.8,
        top_k=25,
        top_p=0.85,
        eos_id=127,
    )

    assert torch.equal(sampled, expected_sample)
    assert torch.equal(next_frequencies, expected_frequencies)


def test_weight_norm_restore_matches_checkpoint_parametrization_in_bfloat16() -> None:
    generator = torch.Generator().manual_seed(42)
    weight_v = torch.randn(8, 16, generator=generator, dtype=torch.bfloat16)
    weight_g = torch.rand(8, 1, generator=generator, dtype=torch.bfloat16)
    linear = nn.utils.parametrizations.weight_norm(
        nn.Linear(16, 8, bias=False, dtype=torch.bfloat16),
        dim=0,
    )
    with torch.no_grad():
        linear.parametrizations.weight.original0.copy_(weight_g)
        linear.parametrizations.weight.original1.copy_(weight_v)

    restored = _restore_weight_norm_weight(weight_g, weight_v)

    assert torch.equal(restored, linear.weight)


def test_talker_emits_request_aligned_codec_deltas_after_compaction(mocker) -> None:
    talker = _make_talker()
    seen: list[tuple[str, list[float], list[int]]] = []

    def sample(hidden, history, request_id, step):
        assert step == 0
        seen.append((request_id, hidden.reshape(-1).tolist(), history.tolist()))
        return torch.tensor(2 if request_id == "req-a" else 3)

    mocker.patch.object(talker, "_sample_audio_code", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.tensor([1])}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
    )

    assert seen == [
        ("req-a", [2.0, 0.0], [1]),
        ("req-b", [3.0, 0.0], []),
    ]
    assert infos[0]["audio_codes"]["accumulated"].tolist() == [1, 2]
    assert infos[1]["audio_codes"]["accumulated"].tolist() == [3]
    assert set(output.multimodal_outputs) == {"codes", "meta"}
    assert "model_outputs" not in output.multimodal_outputs
    assert "sr" not in output.multimodal_outputs
    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[3]]
    assert _routed(output, 0)["meta"]["finished"].item() is False
    assert set(output.multimodal_outputs["meta"]) == {"finished"}
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [0, 0]


def test_direct_stop_sampler_reuses_model_continue_and_stop_decisions(monkeypatch) -> None:
    talker = _make_talker()
    talker.direct_stop_sampler = True
    sampling_metadata = SimpleNamespace(
        max_num_logprobs=None,
        logprob_token_ids={},
    )
    sample_calls = 0

    def sample(*_args) -> torch.Tensor:
        nonlocal sample_calls
        sample_calls += 1
        return torch.tensor(3)

    monkeypatch.setattr(talker, "_sample_audio_code", sample)
    info = {
        "request_id": "req-direct-stop",
        "audio_state": {"step": 0, "min_tokens": 0, "max_tokens": 2},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    first_logits = talker.compute_logits(first.text_hidden_states)
    first_sample = talker.sample(first_logits, sampling_metadata)
    first_constant = first_sample.sampled_token_ids

    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    second_logits = talker.compute_logits(second.text_hidden_states)
    second_sample = talker.sample(second_logits, sampling_metadata)

    third = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    third_logits = talker.compute_logits(third.text_hidden_states)
    third_sample = talker.sample(third_logits, sampling_metadata)

    assert sample_calls == 2
    assert first_logits.argmax(dim=-1).tolist() == [0]
    assert first_sample.sampled_token_ids.tolist() == [[0]]
    assert first_sample.sampled_token_ids is first_constant
    assert second_logits.argmax(dim=-1).tolist() == [1]
    assert second_sample.sampled_token_ids.tolist() == [[1]]
    assert third_logits is second_logits
    assert third_sample.sampled_token_ids is second_sample.sampled_token_ids


def test_pre_minimum_codec_steps_do_not_read_sample_back_to_host(mocker) -> None:
    talker = _make_talker()

    class NoHostReadbackSample:
        def item(self):
            raise AssertionError("pre-minimum codec sample must not be read by the host")

        def reshape(self, *shape):
            return torch.tensor(2).reshape(*shape)

    mocker.patch.object(talker, "_sample_audio_code", return_value=NoHostReadbackSample())
    info = {
        "request_id": "req-no-sync",
        "audio_state": {"step": 0, "min_tokens": 50},
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    assert info["audio_state"]["step"] == 1
    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 0)["meta"]["finished"].item() is False


def test_talker_projects_request_aligned_duplex_metadata(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-a",
            "native_duplex": True,
            "duplex": {"epoch": 3, "turn_id": 7},
            "ids": {"tts": [41]},
            "meta": {
                "native_duplex_segment_text": "first",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-b",
            "native_duplex": True,
            "duplex": {"epoch": 4, "turn_id": 8},
            "ids": {"tts": [42, 99]},
            "meta": {
                "native_duplex_segment_text": "second",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
    ]

    output = talker.make_omni_output(
        torch.ones(2, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    meta = output.multimodal_outputs["meta"]
    assert [value.item() for value in meta["native_duplex"]] == [True, True]
    assert [value.item() for value in meta["duplex_epoch"]] == [3, 4]
    assert [value.item() for value in meta["duplex_turn_id"]] == [7, 8]
    assert "native_duplex_segment_text" not in meta
    assert [bytes(value.tolist()).decode("utf-8") for value in meta["llm_output_text_utf8"]] == [
        "first",
        "second",
    ]
    assert [value.item() for value in meta["turn_end"]] == [False, True]


def test_talker_rejects_native_duplex_without_fence_identity(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    info = {
        "request_id": "req-missing-fence",
        "native_duplex": True,
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    with pytest.raises(RuntimeError, match="requires non-negative integer epoch and turn_id"):
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )


def test_incomplete_prefill_emits_no_code_and_does_not_advance_state(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(2))
    infos = [
        {
            "request_id": "req-prefill",
            "audio_state": {"step": 0},
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-decode",
            "audio_state": {"step": 4},
            "audio_codes": {"accumulated": torch.tensor([1])},
        },
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
        request_sample_eligible=[False, True],
    )

    sample.assert_called_once()
    assert sample.call_args.args[2] == "req-decode"
    assert infos[0]["audio_state"]["step"] == 0
    assert infos[0]["audio_codes"]["accumulated"].numel() == 0
    assert infos[1]["audio_state"]["step"] == 5
    assert _routed(output, 0)["codes"]["audio"].shape == (0, 1)
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[2]]


def test_eos_is_terminal_once_and_never_enters_codec_history(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(7))
    info = {
        "request_id": "req-stop",
        # The real sampler masks EOS before min_tokens. This test injects EOS
        # directly, so make the request eligible for the host-side EOS check.
        "audio_state": {"step": 3, "min_tokens": 0},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    first_logits = talker.compute_logits(first.text_hidden_states)
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    sample.assert_called_once()
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert first.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert first.multimodal_outputs["meta"]["finished"][0].item() is True
    assert second.multimodal_outputs["meta"]["finished"][0].item() is False
    assert first_logits.argmax(dim=-1).tolist() == [1]
    assert talker.compute_logits(second.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_max_token_terminal_drops_unconsumed_codec_delta(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    info = {
        "request_id": "req-limit",
        "audio_state": {"step": 1, "max_tokens": 2},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    # MiniCPMTTS.generate_chunk samples once at the max-token boundary to
    # advance RNG state, but the sampled code is not fed into KV or returned.
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_request_local_state_survives_missing_runner_buffer_update(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_code", return_value=torch.tensor(3))
    first_info = {
        "request_id": "req-local-state",
        "audio_state": {"step": 1, "max_tokens": 3},
        "audio_codes": {"accumulated": torch.tensor([4])},
    }

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[first_info],
        request_token_spans=[(0, 1)],
    )
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-local-state"}],
        request_token_spans=[(0, 1)],
    )

    assert second.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker._request_audio_states["req-local-state"]["step"] == 3


def test_missing_conditioning_fails_clearly() -> None:
    talker = _make_talker()

    with pytest.raises(ValueError, match="tts_token_ids and tts_hidden_states"):
        talker.preprocess(
            torch.tensor([0]),
            None,
            _omni_is_prefill=True,
            request_id="req-invalid",
        )


def test_empty_speech_segment_finishes_without_sampling_codes() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(8, 4)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])
    talker._text_eos_id = 5
    talker._tts_bos_id = 6

    _, embeds, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-empty",
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 4),
    )

    assert torch.equal(embeds, talker.emb_text(torch.tensor([5, 6])))
    assert updates["audio_state"]["finished"] is True

    # Stage 1's sampling min_tokens keeps scheduling decode steps until the stop
    # token becomes eligible, and those steps have no previous code to embed.
    _, decode_embeds, _ = talker.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        request_id="req-empty",
        audio_state=updates["audio_state"],
        audio_codes=updates["audio_codes"],
    )

    assert decode_embeds.shape == (1, 4)


def test_chunked_prefill_tail_aligns_condition_with_prompt_length(mocker) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    condition = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)

    _, embeds, _ = talker.preprocess(
        torch.zeros(9, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=59,
        _omni_prompt_len=68,
        request_id="req-chunked-prefill",
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    assert torch.equal(embeds, condition)
    state = talker._request_audio_states["req-chunked-prefill"]
    assert state["min_tokens"] == 50
    assert state["max_tokens"] == 64


@pytest.mark.parametrize(
    ("meta", "expected_min_tokens"),
    [
        ({"turn_start": True}, 0),
        ({}, 26),
        ({"turn_end": True}, 0),
    ],
)
def test_native_duplex_prefill_uses_official_chunk_limits(
    mocker,
    meta,
    expected_min_tokens,
) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    mocker.patch.object(
        talker,
        "_build_condition_embeddings",
        return_value=torch.ones(3, 2),
    )

    talker.preprocess(
        torch.zeros(3, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-duplex-chunk",
        native_duplex=True,
        meta=meta,
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    state = talker._request_audio_states["req-duplex-chunk"]
    assert state["min_tokens"] == expected_min_tokens
    assert state["max_tokens"] == 26


def test_native_duplex_condition_matches_official_text_plus_audio_bos() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(16, 2)
    talker.projector_semantic = nn.Identity()
    talker._normalize = False
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    with torch.no_grad():
        talker.emb_text.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(16, 2))

    token_ids = torch.tensor([2, 3])
    hidden_states = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    condition = talker._build_condition_embeddings(
        token_ids,
        hidden_states,
        native_duplex=True,
    )

    expected_text = talker.emb_text(token_ids) + hidden_states
    expected = torch.cat(
        [expected_text, talker.emb_text(torch.tensor([talker._tts_bos_id]))],
        dim=0,
    )
    assert torch.equal(condition, expected)
    assert condition.shape[0] == token_ids.shape[0] + 1


def test_request_cleanup_evicts_ar_rng_and_decode_state() -> None:
    talker = _make_talker()
    talker._request_generators["req-done"] = torch.Generator()
    talker._request_audio_states["req-done"] = {"step": 1}
    talker._request_repetition_frequencies["req-done"] = torch.zeros(1, 8)

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_generators
    assert "req-done" not in talker._request_audio_states
    assert "req-done" not in talker._request_repetition_frequencies
