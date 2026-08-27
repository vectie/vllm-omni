# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

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
    _graphable_codec_sample,
    _max_audio_tokens,
    _restore_weight_norm_weight,
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


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._batch_stop_logits = None
    talker._batch_stop_token_ids = None
    talker._stop_logits_constants = None
    talker._stop_token_constants = None
    talker.direct_stop_sampler = False
    talker._request_generators = {}
    talker._request_audio_states = {}
    talker._request_repetition_frequencies = {}
    talker._deferred_cleanup_ids = set()
    talker._codec_vocab_ids = torch.arange(8)
    talker._codec_min_tokens = 50
    talker._codec_seed = 42
    talker._fused_codec_sampler_enabled = False
    talker._fused_codec_sampler_prepared = False
    talker._fused_codec_sampler_request_id = None
    return talker


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
