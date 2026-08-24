# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 Thinker-to-Talker and Talker-to-Code2Wav bridges."""

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.experimental.fullduplex.engine.intermediate import (
    build_duplex_intermediate_buffer,
    set_ref_audio,
    set_tts_handoff,
)
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)
_MINICPMO45_ASYNC_STATE = "_minicpmo45_async_codec_state"
_MINICPMO45_STREAM_RECORD = "_minicpmo45_async_stream_record"
_MINICPMO45_SILENCE_CODE = 4218
_MINICPMO45_MIN_STREAM_BODY_FRAMES = 5
_MINICPMO45_CODEC_CHUNK_FRAMES_ENV = "VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES"
_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES_ENV = "VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES"
_MINICPMO45_CODEC_LEFT_CONTEXT_FRAMES_ENV = "VLLM_OMNI_MINICPMO45_CODEC_LEFT_CONTEXT_FRAMES"


class _MiniCPMO45MetaStruct(MetaStruct):
    """Model-owned metadata for the split Talker-to-Code2Wav bridge."""

    ref_audio_sr: int | None = None
    native_duplex_segment_text: str | None = None
    llm_output_text_utf8: torch.Tensor | None = None
    duplex_turn_id: int | None = None
    duplex_epoch: int | None = None
    segment_end: bool | None = None
    turn_end: bool | None = None
    tts_is_last_chunk: bool | None = None


@dataclass
class _MiniCPMO45StreamRecord:
    internal_id: str
    cache_epoch: int = 0
    chunk_seq: int = 0
    retired_internal_ids: set[str] = field(default_factory=set)
    last_terminal_turn: tuple[int | None, int | None] | None = None


@dataclass
class _MiniCPMO45CodecAccumulator:
    internal_id: str
    pending_parts: list[torch.Tensor] = field(default_factory=list)
    pending_count: int = 0
    pending_text_utf8: list[int] = field(default_factory=list)
    segment_text_recorded: bool = False
    left_context: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    codec_end: int = 0


def _extract_first_audio_ref(multi_modal_data):
    if not isinstance(multi_modal_data, dict):
        return None
    audio_data = multi_modal_data.get("audio")
    if audio_data is None:
        return None
    if isinstance(audio_data, list):
        if not audio_data:
            return None
        audio_data = audio_data[0]

    samples = None
    sample_rate = None
    if isinstance(audio_data, tuple) and len(audio_data) >= 2:
        samples, sample_rate = audio_data[0], audio_data[1]
    elif isinstance(audio_data, dict):
        sample_rate = audio_data.get("sample_rate") or audio_data.get("sampling_rate") or audio_data.get("sr")
        for key in ("audio", "wav", "samples", "array", "waveform"):
            if key in audio_data:
                samples = audio_data[key]
                break
    if samples is None or sample_rate is None:
        return None

    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.ndim > 1:
        if waveform.shape[0] <= 2 and waveform.shape[-1] > waveform.shape[0]:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=-1)
    return waveform.reshape(-1).cpu(), int(sample_rate)


def _extract_native_runtime_ref_audio(data_plane_metadata):
    if not isinstance(data_plane_metadata, dict):
        return None
    runtime_config = data_plane_metadata.get("runtime_config")
    if not isinstance(runtime_config, dict):
        return None
    from vllm_omni.experimental.fullduplex.minicpmo45.input import decode_native_ref_audio_from_config

    waveform = decode_native_ref_audio_from_config({"extra_body": runtime_config})
    if waveform is None:
        return None
    sample_rate = runtime_config.get("ref_audio_sample_rate_hz") or 16000
    return torch.as_tensor(waveform, dtype=torch.float32).reshape(-1).cpu(), int(sample_rate)


def prepare_code2wav_request(
    base_input: dict[str, Any],
    original_prompt: Any,
    _stage0_request: Any,
) -> dict[str, Any]:
    """Attach exact voice conditioning to the explicit Stage-2 prepare event.

    The orchestrator owns when the prepare lifecycle runs. This model hook
    only translates MiniCPM-o's original multimodal prompt into the payload
    Code2Wav already consumes, allowing prompt extraction and request-state
    allocation to overlap Stage 0/1 instead of preparing the default voice and
    throwing that provisional state away when the first real chunk arrives.
    """
    additional_information = base_input.get("additional_information")
    if not isinstance(additional_information, dict):
        additional_information = {}

    multi_modal_data = original_prompt.get("multi_modal_data") if isinstance(original_prompt, Mapping) else None
    ref_audio = _extract_first_audio_ref(multi_modal_data)
    if ref_audio is None:
        ref_audio = _extract_native_runtime_ref_audio(additional_information.get("duplex"))
    if ref_audio is not None:
        waveform, sample_rate = ref_audio
        codes = additional_information.get("codes")
        if not isinstance(codes, dict):
            codes = {}
        meta = additional_information.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        codes["ref"] = waveform
        meta["ref_audio_sr"] = int(sample_rate)
        additional_information["codes"] = codes
        additional_information["meta"] = meta

    base_input["additional_information"] = additional_information
    return base_input


def _coerce_token_id_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _to_transport_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _coerce_int(value):
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return None
        value = flat[0].item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _codec_config(transfer_manager: Any) -> tuple[int, int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_config = getattr(connector, "config", {}) or {}
    config = raw_config.get("extra", raw_config) if isinstance(raw_config, dict) else {}
    config = config if isinstance(config, dict) else {}

    def resolve_int(env_name: str, config_name: str, default: int) -> int:
        env_value = os.environ.get(env_name)
        raw_value = env_value if env_value not in (None, "") else config.get(config_name, default)
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid MiniCPM-o codec chunk config: {env_name}={raw_value!r}") from exc

    chunk_frames = resolve_int(
        _MINICPMO45_CODEC_CHUNK_FRAMES_ENV,
        "codec_chunk_frames",
        25,
    )
    initial_chunk_frames = resolve_int(
        _MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES_ENV,
        "initial_codec_chunk_frames",
        chunk_frames,
    )
    left_context_frames = resolve_int(
        _MINICPMO45_CODEC_LEFT_CONTEXT_FRAMES_ENV,
        "codec_left_context_frames",
        3,
    )
    if chunk_frames <= 0 or initial_chunk_frames <= 0 or left_context_frames < 0:
        raise ValueError(
            "Invalid MiniCPM-o codec chunk config: "
            f"codec_chunk_frames={chunk_frames}, "
            f"initial_codec_chunk_frames={initial_chunk_frames}, "
            f"codec_left_context_frames={left_context_frames}"
        )
    return chunk_frames, initial_chunk_frames, left_context_frames


def _codec_tensor(value: Any, *, to_cpu: bool = True) -> torch.Tensor:
    """Normalize one codec delta without forcing an early device sync.

    Ordinary/full-payload callers need a CPU tensor for transport and keep the
    historical default.  The async MiniCPM-o bridge retains individual Talker
    deltas on the producing device until an entire codec chunk is ready, then
    performs one batched D2H transfer for that chunk.  This is important for
    single-stream NPU serving: copying every one-token delta to CPU inserts a
    host synchronization between every Talker decode step.
    """
    if value is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        tensor = value.detach().to(dtype=torch.long).reshape(-1).contiguous()
        return tensor.to(device="cpu") if to_cpu else tensor
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_codec_tensor(item, to_cpu=to_cpu) for item in value]
        parts = [part for part in parts if part.numel()]
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    if isinstance(value, (int, bool)):
        return torch.tensor([int(value)], dtype=torch.long)
    raise TypeError(f"Unsupported MiniCPM-o codec delta type: {type(value).__name__}")


def _extract_codec_delta(
    pooling_output: Any,
    request_id: str,
    *,
    to_cpu: bool = True,
) -> torch.Tensor:
    if pooling_output is None:
        return torch.empty(0, dtype=torch.long)
    if isinstance(pooling_output, Mapping):
        meta = pooling_output.get("meta")
        routed_id = meta.get("request_id") if isinstance(meta, Mapping) else None
        routed_id = routed_id or pooling_output.get("request_id")
        if routed_id is not None and str(routed_id) != request_id:
            return torch.empty(0, dtype=torch.long)
        codes = pooling_output.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else pooling_output.get("codes.audio")
        return _codec_tensor(audio, to_cpu=to_cpu)
    if isinstance(pooling_output, Sequence) and not isinstance(
        pooling_output,
        (str, bytes, bytearray),
    ):
        delta: list[torch.Tensor] = []
        for item in pooling_output:
            values = (
                _extract_codec_delta(item, request_id, to_cpu=to_cpu)
                if isinstance(item, Mapping)
                else _codec_tensor(item, to_cpu=to_cpu)
            )
            if values.numel():
                delta.append(values)
        return torch.cat(delta) if delta else torch.empty(0, dtype=torch.long)
    return _codec_tensor(pooling_output, to_cpu=to_cpu)


def _append_pending_codec(
    state: _MiniCPMO45CodecAccumulator,
    delta: torch.Tensor,
) -> int:
    """Append a device-resident Talker delta and return the queued length."""
    if delta.numel():
        state.pending_parts.append(delta.reshape(-1))
        state.pending_count += int(delta.numel())
    return state.pending_count


def _take_pending_codec(
    state: _MiniCPMO45CodecAccumulator,
    count: int,
) -> torch.Tensor:
    """Take exactly ``count`` queued codes without a device-to-host copy."""
    available = state.pending_count
    if count < 0 or count > available:
        raise ValueError(f"Invalid MiniCPM-o pending codec take: count={count}, available={available}")
    if count == 0:
        return torch.empty(0, dtype=torch.long)

    taken: list[torch.Tensor] = []
    remaining: list[torch.Tensor] = []
    need = count
    for part in state.pending_parts:
        width = int(part.numel())
        if need <= 0:
            remaining.append(part)
        elif width <= need:
            taken.append(part)
            need -= width
        else:
            taken.append(part[:need])
            remaining.append(part[need:])
            need = 0
    if need:
        raise RuntimeError(f"MiniCPM-o pending codec queue underflowed by {need} codes")
    state.pending_parts = remaining
    state.pending_count = available - count
    return taken[0] if len(taken) == 1 else torch.cat(taken)


def _drop_codec_state(transfer_manager: Any, request_id: str) -> None:
    request_payload = getattr(transfer_manager, "request_payload", None)
    if isinstance(request_payload, dict):
        container = request_payload.get(request_id)
        if isinstance(container, dict):
            container.pop(_MINICPMO45_ASYNC_STATE, None)
            if not container:
                request_payload.pop(request_id, None)
        else:
            request_payload.pop(request_id, None)
    code_accumulators = getattr(transfer_manager, "code_prompt_token_ids", None)
    if hasattr(code_accumulators, "pop"):
        code_accumulators.pop(request_id, None)


def _is_aborted(request: Any) -> bool:
    status_name = getattr(getattr(request, "status", None), "name", "")
    return any(marker in status_name for marker in ("ABORT", "CANCEL", "IGNORED", "ERROR"))


def tts2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Stream request-owned MiniCPM-o codec windows to Code2Wav."""
    external_id = getattr(request, "external_req_id", None)
    internal_id = getattr(request, "request_id", None)
    request_id = str(external_id if external_id is not None else internal_id)
    internal_id = str(internal_id if internal_id is not None else request_id)
    output_meta = multimodal_output.get("meta") if isinstance(multimodal_output, Mapping) else None
    output_meta = output_meta if isinstance(output_meta, Mapping) else {}
    native_duplex = bool(_coerce_int(output_meta.get("native_duplex")))
    duplex_epoch = _coerce_int(output_meta.get("duplex_epoch"))
    duplex_turn_id = _coerce_int(output_meta.get("duplex_turn_id"))
    segment_text_utf8 = output_meta.get("llm_output_text_utf8")
    if not isinstance(segment_text_utf8, torch.Tensor):
        segment_text_utf8 = None
    turn_end = bool(_coerce_int(output_meta.get("turn_end")))
    duplex_turn_key = (duplex_epoch, duplex_turn_id)

    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    container = request_payload.get(request_id)
    if not isinstance(container, dict):
        container = {}
        request_payload[request_id] = container

    record = container.get(_MINICPMO45_STREAM_RECORD)
    if not isinstance(record, _MiniCPMO45StreamRecord):
        record = _MiniCPMO45StreamRecord(internal_id=internal_id)
        container[_MINICPMO45_STREAM_RECORD] = record
    elif internal_id in record.retired_internal_ids:
        return None
    elif record.internal_id != internal_id:
        record.retired_internal_ids.add(record.internal_id)
        record.internal_id = internal_id
        record.cache_epoch += 1
        record.chunk_seq = 0
        record.last_terminal_turn = None
        _drop_codec_state(transfer_manager, request_id)

    if _is_aborted(request):
        record.retired_internal_ids.add(internal_id)
        _drop_codec_state(transfer_manager, request_id)
        return None

    if native_duplex and turn_end and record.last_terminal_turn == duplex_turn_key:
        return None

    state = container.get(_MINICPMO45_ASYNC_STATE)
    if not isinstance(state, _MiniCPMO45CodecAccumulator):
        # Keep one-token Talker deltas device-resident. They are joined and
        # copied to CPU once per publishable codec chunk.
        state = _MiniCPMO45CodecAccumulator(internal_id=internal_id)
        container[_MINICPMO45_ASYNC_STATE] = state

    delta = _extract_codec_delta(multimodal_output, request_id, to_cpu=False)
    pending_count = _append_pending_codec(state, delta)
    pending_text_utf8 = state.pending_text_utf8
    current_text_utf8 = (
        segment_text_utf8.detach().to(device="cpu", dtype=torch.uint8).reshape(-1).tolist()
        if isinstance(segment_text_utf8, torch.Tensor)
        else []
    )
    if native_duplex and current_text_utf8 and not state.segment_text_recorded:
        # Talker repeats the unit text on every codec step. Queue it once for
        # the first Code2Wav payload that can carry audio for this unit.
        pending_text_utf8.extend(current_text_utf8)
        state.segment_text_recorded = True
    request_finished = getattr(request, "is_finished", None)
    finished = bool(is_finished or (callable(request_finished) and request_finished()))
    chunk_frames, initial_chunk_frames, left_context_frames = _codec_config(transfer_manager)
    chunk_seq = record.chunk_seq
    active_chunk_frames = initial_chunk_frames if chunk_seq == 0 else chunk_frames
    flush_pending = finished
    last_chunk = bool(flush_pending and (not native_duplex or turn_end))
    if not flush_pending and pending_count < active_chunk_frames:
        return None

    hold_short_unit = (
        native_duplex
        and flush_pending
        and not last_chunk
        and 0 < pending_count < _MINICPMO45_MIN_STREAM_BODY_FRAMES
    )
    new_token_count = 0 if hold_short_unit else (pending_count if flush_pending else active_chunk_frames)
    # One batched D2H per emitted audio chunk replaces one D2H synchronization
    # per Talker token. Empty control boundaries stay CPU-only.
    new_codes_device = _take_pending_codec(state, new_token_count)
    new_codes = _codec_tensor(new_codes_device) if new_token_count else torch.empty(0, dtype=torch.long)
    codec_start = state.codec_end
    codec_end = codec_start + new_token_count

    if new_token_count:
        if codec_start == 0:
            context = torch.full((left_context_frames,), _MINICPMO45_SILENCE_CODE, dtype=torch.long)
        else:
            context = torch.as_tensor(state.left_context, dtype=torch.long)
        output_codes = torch.cat((context, new_codes))
        history = output_codes
        state.left_context = history[-left_context_frames:].clone() if left_context_frames else history[:0]
    elif last_chunk and codec_start > 0 and torch.as_tensor(state.left_context).numel():
        context = torch.as_tensor(state.left_context, dtype=torch.long)
        output_codes = context
    else:
        context = torch.empty(0, dtype=torch.long)
        output_codes = torch.empty(0, dtype=torch.long)
    state.codec_end = codec_end
    code_flat_numel = int(output_codes.numel())
    if native_duplex and code_flat_numel > 0:
        segment_text_utf8 = torch.tensor(pending_text_utf8, dtype=torch.uint8)
        pending_text_utf8.clear()
    if native_duplex and finished:
        state.segment_text_recorded = False
    if flush_pending and not last_chunk and code_flat_numel == 0:
        # Keep the generic generation connector model-agnostic: a real token
        # makes this control-only TTS boundary schedulable, while the explicit
        # zero length tells Code2Wav to discard the placeholder.
        output_codes = torch.zeros(1, dtype=torch.long)

    if last_chunk and not native_duplex:
        record.retired_internal_ids.add(internal_id)
        _drop_codec_state(transfer_manager, request_id)

    record.chunk_seq = chunk_seq + 1
    ref_audio = None
    ref_audio_sr = None
    if record.cache_epoch == 0 and chunk_seq == 0:
        request_info = getattr(request, "additional_information", None)
        if isinstance(request_info, Mapping):
            codes_info = request_info.get("codes")
            meta_info = request_info.get("meta")
            raw_ref_audio = codes_info.get("ref") if isinstance(codes_info, Mapping) else None
            raw_ref_audio_sr = meta_info.get("ref_audio_sr") if isinstance(meta_info, Mapping) else None
            ref_audio_sr = _coerce_int(raw_ref_audio_sr)
            if raw_ref_audio is not None:
                ref_audio = torch.as_tensor(raw_ref_audio, dtype=torch.float32).reshape(-1).cpu()
    finished_tensor = torch.tensor(last_chunk, dtype=torch.bool)
    payload = OmniPayloadStruct(
        codes=CodesStruct(
            audio=output_codes,
            ref=ref_audio,
        ),
        meta=_MiniCPMO45MetaStruct(
            lifecycle_event="finish" if last_chunk else "chunk",
            lifecycle_generation=record.cache_epoch,
            request_id=request_id,
            chunk_seq=chunk_seq,
            cache_epoch=record.cache_epoch,
            code_flat_numel=code_flat_numel,
            codec_chunk_frames=new_token_count,
            codec_left_context_frames=int(context.numel()),
            left_context_size=int(context.numel()),
            last_chunk=last_chunk,
            stream_finished=finished_tensor,
            finished=finished_tensor,
            is_segment_finished=finished_tensor,
            req_id=[request_id],
            duplex_epoch=duplex_epoch,
            duplex_turn_id=duplex_turn_id,
            llm_output_text_utf8=segment_text_utf8,
            tts_is_last_chunk=flush_pending,
            turn_end=turn_end and last_chunk,
            ref_audio_sr=ref_audio_sr,
        ),
        request_id=request_id,
    )
    if last_chunk and native_duplex:
        record.last_terminal_turn = duplex_turn_key
        record.cache_epoch += 1
        record.chunk_seq = 0
        _drop_codec_state(transfer_manager, request_id)
    return payload


def tts2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: Any,
    request: Any,
) -> OmniPayloadStruct:
    """Build one terminal Code2Wav payload when async chunks are disabled."""
    external_id = getattr(request, "external_req_id", None)
    internal_id = getattr(request, "request_id", None)
    request_id = str(external_id if external_id is not None else internal_id)
    codes = _extract_codec_delta(pooling_output, request_id)
    _, _, left_context_frames = _codec_config(transfer_manager)
    context = (
        torch.full((left_context_frames,), _MINICPMO45_SILENCE_CODE, dtype=torch.long)
        if codes.numel()
        else torch.empty(0, dtype=torch.long)
    )
    output_codes = torch.cat((context, codes))

    request_info = getattr(request, "additional_information", None)
    if not isinstance(request_info, Mapping):
        request_info = {}
    codes_info = request_info.get("codes")
    if not isinstance(codes_info, Mapping):
        codes_info = {}
    meta_info = request_info.get("meta")
    if not isinstance(meta_info, Mapping):
        meta_info = {}
    duplex_info = request_info.get("duplex")
    if not isinstance(duplex_info, Mapping):
        duplex_info = {}

    ref_audio = codes_info.get("ref")
    finished = torch.tensor(True, dtype=torch.bool)
    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=output_codes,
            ref=torch.as_tensor(ref_audio, dtype=torch.float32).reshape(-1) if ref_audio is not None else None,
        ),
        meta=_MiniCPMO45MetaStruct(
            lifecycle_event="finish",
            lifecycle_generation=0,
            request_id=request_id,
            chunk_seq=0,
            cache_epoch=0,
            code_flat_numel=int(output_codes.numel()),
            codec_chunk_frames=int(codes.numel()),
            codec_left_context_frames=int(context.numel()),
            left_context_size=int(context.numel()),
            last_chunk=True,
            stream_finished=finished,
            finished=finished,
            req_id=[request_id],
            ref_audio_sr=_coerce_int(meta_info.get("ref_audio_sr")),
            native_duplex_segment_text=(
                str(meta_info["native_duplex_segment_text"])
                if isinstance(meta_info.get("native_duplex_segment_text"), str)
                else None
            ),
            duplex_turn_id=_coerce_int(duplex_info.get("model_turn_id", duplex_info.get("turn_id"))),
            duplex_epoch=_coerce_int(duplex_info.get("epoch")),
            segment_end=bool(meta_info.get("segment_end", False)),
            turn_end=bool(meta_info.get("turn_end", False)),
            tts_is_last_chunk=True,
        ),
        request_id=request_id,
    )


def tts2code2wav_token_only(
    source_outputs: list[Any],
    _prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build the sync Code2Wav placeholder; codec data arrives by connector."""
    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        codes = multimodal_output.get("codes") if isinstance(multimodal_output, Mapping) else None
        audio = (
            codes.get("audio")
            if isinstance(codes, Mapping)
            else multimodal_output.get("codes.audio")
            if isinstance(multimodal_output, Mapping)
            else None
        )
        codec_codes = _codec_tensor(audio)
        if not codec_codes.numel():
            continue
        prompt_len = int(codec_codes.numel()) + 3
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def _special_token_ids_from_mm_output(mm_output):
    if not isinstance(mm_output, Mapping):
        return {}
    meta = mm_output.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    special_token_ids = mm_output.get("special_token_ids")
    if not isinstance(special_token_ids, dict):
        special_token_ids = {}
    flat_meta = {
        key.removeprefix("meta."): value
        for key, value in mm_output.items()
        if isinstance(key, str) and key.startswith("meta.")
    }
    return {
        key: value
        for key, value in (
            (key, _coerce_int(value))
            for source in (special_token_ids, meta, flat_meta)
            for key, value in source.items()
        )
        if value is not None and value >= 0
    }


def _has_native_duplex_prompt_metadata(mm_output):
    if not isinstance(mm_output, Mapping):
        return False
    return mm_output.get("duplex_prompt_token_ids") is not None or mm_output.get("ids.prompt") is not None


def _require_native_tts_boundary_metadata(special_token_ids):
    if special_token_ids.get("tts_bos_token_id") is None:
        raise ValueError(
            "MiniCPM-o native duplex TTS handoff requires tokenizer-derived "
            "<|tts_bos|> metadata; refusing to infer special token ids."
        )


def _native_tts_boundary_token_ids(special_token_ids):
    # Official duplex conditions the talker on mid-unit <|speak|> tokens AND
    # the <|turn_eos|> token+hidden (its embedding is the trained stop
    # signal); only chunk terminators and framing tokens bound the slice.
    return {
        token_id
        for token_id in (
            special_token_ids.get("tts_eos_token_id"),
            special_token_ids.get("tts_pad_token_id"),
            special_token_ids.get("listen_token_id"),
            special_token_ids.get("chunk_eos_token_id"),
            special_token_ids.get("chunk_tts_eos_token_id"),
            special_token_ids.get("unit_token_id"),
            special_token_ids.get("unit_end_token_id"),
        )
        if token_id is not None
    }


def _decode_native_duplex_token_ids(
    token_ids: list[int],
    streaming_context,
    *,
    request_id: str,
) -> str | None:
    decode_token_ids = getattr(streaming_context, "source_token_decoder", None)
    if not isinstance(decode_token_ids, Callable):
        return None
    decode_ids = [int(token_id) for token_id in token_ids]
    try:
        try:
            return str(decode_token_ids(decode_ids, skip_special_tokens=True))
        except TypeError:
            return str(decode_token_ids(decode_ids))
    except Exception:
        logger.exception("Failed to decode MiniCPM-o duplex token delta for request_id=%s", request_id)
        return ""


def _native_duplex_segment_output_ids(
    output_ids: list[int],
    output_text: str,
    streaming_context,
    *,
    request_id: str,
) -> tuple[list[int], str, bool]:
    """Slice the cumulative thinker output down to the current segment.

    Tracks how many output tokens were already handed to the talker in the
    orchestrator's streaming bridge state. Segment transcript is decoded from
    the same token delta, not from a character cursor over cumulative text.
    """
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return output_ids, output_text, True
    state = bridge_states.setdefault("minicpmo45_tts_handoff", {})
    duplex_state = bridge_states.get("duplex")
    turn_id = duplex_state.get("model_turn_id", duplex_state.get("turn_id")) if isinstance(duplex_state, dict) else None
    if not isinstance(turn_id, int):
        turn_id = None
    if state.get("request_id") != request_id:
        state["request_id"] = request_id
        state["sent_output_len"] = 0
        state["sent_output_ids"] = []
    previous_turn_id = state.get("turn_id")
    sent_len = state.get("sent_output_len", 0)
    prev_output_ids = state.get("sent_output_ids", [])
    if not isinstance(sent_len, int) or sent_len < 0 or sent_len > len(output_ids):
        # Shrunken cumulative output = epoch reset after barge-in.
        sent_len = 0
    elif sent_len and isinstance(prev_output_ids, list):
        prev_prefix = prev_output_ids[:sent_len]
        current_prefix = output_ids[:sent_len]
        if current_prefix != prev_prefix:
            # Some runtimes restart output token ids after a turn boundary while
            # others keep returning cumulative ids. Detect restart from the
            # token prefix instead of clearing the cursor at turn_eos.
            sent_len = 0
    turn_start = sent_len == 0 or previous_turn_id != turn_id
    segment_ids = output_ids[sent_len:]
    decoded_segment_text = _decode_native_duplex_token_ids(
        segment_ids,
        streaming_context,
        request_id=request_id,
    )
    if decoded_segment_text is not None:
        segment_text = decoded_segment_text
    elif sent_len == 0:
        # Non-orchestrator unit tests and legacy callers may not install a
        # decoder. Never slice cumulative text with a stale character cursor.
        segment_text = output_text
    else:
        logger.warning(
            "MiniCPM-o native duplex token delta decoder missing for request_id=%s; "
            "suppressing cumulative transcript fallback",
            request_id,
        )
        segment_text = ""
    state["sent_output_len"] = len(output_ids)
    state["sent_output_ids"] = list(output_ids)
    state["turn_id"] = turn_id
    return segment_ids, segment_text, turn_start


def _native_duplex_data_plane_metadata(streaming_context) -> dict[str, object] | None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    duplex_state = bridge_states.get("duplex")
    if not isinstance(duplex_state, dict):
        return None

    metadata: dict[str, object] = {"data_plane": True}
    session_id = duplex_state.get("session_id")
    if isinstance(session_id, str) and session_id:
        metadata["session_id"] = session_id
    incarnation = duplex_state.get("incarnation")
    if isinstance(incarnation, int):
        metadata["incarnation"] = incarnation
    epoch = duplex_state.get("epoch")
    if isinstance(epoch, int):
        metadata["epoch"] = epoch
    turn_id = duplex_state.get("model_turn_id", duplex_state.get("turn_id"))
    if isinstance(turn_id, int):
        metadata["turn_id"] = turn_id
    session_config = duplex_state.get("session_config")
    if isinstance(session_config, dict):
        metadata["session_config"] = dict(session_config)
    runtime_config = duplex_state.get("runtime_config")
    if isinstance(runtime_config, dict):
        metadata["runtime_config"] = dict(runtime_config)
    return metadata


def _build_tts_scheduler_prompt_token_ids(
    tts_token_ids: torch.Tensor | None,
    llm_output_ids: list[int],
    prompt_token_ids: list[int],
) -> list[int]:
    if tts_token_ids is not None:
        ids = _coerce_token_id_list(tts_token_ids)
        if ids:
            return ids
    if llm_output_ids:
        return llm_output_ids
    if prompt_token_ids:
        return prompt_token_ids[-1:]
    raise ValueError("MiniCPM-o TTS stage requires at least one scheduler prompt token")


def llm2tts(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    _streaming_context=None,
):
    """Build Talker conditioning for ordinary and full-duplex streaming."""
    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    llm_outputs = source_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {}
    for llm_output, p in zip(llm_outputs, prompt):
        if isinstance(p, dict):
            multi_modal_data[llm_output.request_id] = p.get("multi_modal_data", None)
        else:
            multi_modal_data[llm_output.request_id] = getattr(p, "multi_modal_data", None)

    for llm_output in llm_outputs:
        output = llm_output.outputs[0]
        request_mm_output = getattr(llm_output, "multimodal_output", None)
        completion_mm_output = getattr(output, "multimodal_output", None)
        if isinstance(request_mm_output, Mapping):
            mm_output = request_mm_output
        elif isinstance(completion_mm_output, Mapping):
            mm_output = completion_mm_output
        else:
            mm_output = {}
        special_token_ids = _special_token_ids_from_mm_output(mm_output)
        prompt_token_ids = (
            _coerce_token_id_list(mm_output.get("duplex_prompt_token_ids"))
            or _coerce_token_id_list(mm_output.get("ids.prompt"))
            or list(llm_output.prompt_token_ids)
        )
        llm_output_ids = getattr(output, "token_ids", None)
        cumulative_output_ids = getattr(output, "cumulative_token_ids", None)
        if llm_output_ids is None:
            llm_output_ids = cumulative_output_ids or []
        elif cumulative_output_ids is not None and len(cumulative_output_ids) > len(llm_output_ids):
            # DELTA outputs (any streaming client) carry only the newest tokens,
            # while the forwarded hidden states span prompt + whole reply. With
            # the delta alone, tts_bos lands at the end of full_token_ids, the
            # tts_bos..tts_eos slice collapses to nothing and the Talker is
            # conditioned on an empty span, so the reply is silent.
            llm_output_ids = cumulative_output_ids
        # Always copy: CompletionOutput.token_ids can alias the upstream
        # detokenizer's live token list. Forwarding that exact object as the
        # talker prompt makes the stage-1 streaming update extend the list
        # with itself (state and update bind the same object), doubling the
        # thinker's recorded output every segment until the TTS engine input
        # buffer overflows.
        llm_output_ids = list(llm_output_ids)
        thinker_text = getattr(output, "text", "") or ""
        native_turn_start = False
        if _has_native_duplex_prompt_metadata(mm_output):
            # The thinker's resumable duplex request reports cumulative
            # output ids/text, but earlier segments are already folded into
            # the prompt by the scheduler session update, and the forwarded
            # hidden states only cover the current prompt + current segment.
            # Hand stage 1 exactly one segment per handoff so token/hidden
            # alignment holds, the talker prompt grows linearly with new
            # tokens instead of quadratically, and downstream transcripts
            # carry per-unit deltas instead of re-sending the whole reply.
            llm_output_ids, thinker_text, native_turn_start = _native_duplex_segment_output_ids(
                llm_output_ids,
                thinker_text,
                _streaming_context,
                request_id=str(llm_output.request_id),
            )
        prompt_token_ids_len = len(prompt_token_ids)

        latent = mm_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.detach()
        if thinker_hidden_states.ndim == 3 and thinker_hidden_states.shape[0] == 1:
            thinker_hidden_states = thinker_hidden_states.squeeze(0)

        # Build full token sequence and extract TTS region
        full_token_ids = prompt_token_ids + llm_output_ids

        tts_bos_id = special_token_ids.get("tts_bos_token_id")
        is_native_duplex_handoff = _has_native_duplex_prompt_metadata(mm_output)
        if is_native_duplex_handoff:
            _require_native_tts_boundary_metadata(special_token_ids)
        tts_end_ids = _native_tts_boundary_token_ids(special_token_ids)

        # Plain-chat (use_tts_template) fallback: non-duplex requests do not
        # surface special_token_ids, so use MiniCPM-o 4.5's fixed boundaries.
        if tts_bos_id is None and not is_native_duplex_handoff:
            tts_bos_id = 151703
            tts_end_ids = set(tts_end_ids) | {151704, 151645}

        tts_bos_idx = None
        # For native duplex the resumable prompt folds every earlier unit, so
        # a <|tts_bos|> from an already-spoken reply can sit mid-prompt; only
        # a boundary folded as the FINAL prompt token (this unit's decision)
        # or one inside the current segment may start the slice, or stale
        # text would be re-handed to the talker on text-less continuations.
        search_start = max(0, prompt_token_ids_len - 1) if is_native_duplex_handoff else 0
        for idx_t in range(search_start, len(full_token_ids)):
            if full_token_ids[idx_t] == tts_bos_id:
                tts_bos_idx = idx_t + 1
        if tts_bos_idx is None and not is_native_duplex_handoff and llm_output_ids:
            # Audio routing is a model-stage concern, not an OpenAI serving
            # default. Plain chat templates do not include <|tts_bos|>; in
            # that case condition the Talker on the generated assistant span.
            tts_bos_idx = prompt_token_ids_len

        tts_eos_idx = None
        if tts_bos_idx is not None:
            for idx_t in range(tts_bos_idx, len(full_token_ids)):
                if full_token_ids[idx_t] in tts_end_ids:
                    tts_eos_idx = idx_t
                    break

        tts_token_ids_slice = tts_hidden_slice = None
        native_segment_end = False
        if tts_bos_idx is not None and thinker_hidden_states.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else thinker_hidden_states.shape[0]
            if is_native_duplex_handoff and tts_eos_idx is not None:
                boundary_token = full_token_ids[tts_eos_idx]
                native_segment_end = boundary_token in {
                    special_token_ids.get("chunk_eos_token_id"),
                    special_token_ids.get("chunk_tts_eos_token_id"),
                }
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = thinker_hidden_states[tts_bos_idx:end_idx].to(torch.float32).contiguous()
        elif is_native_duplex_handoff:
            # Official MiniCPM-o duplex does not prefill an assistant
            # <|tts_bos|> boundary before generation. A segment delta can
            # start with SEVERAL unit decisions (forced/model listens from
            # chunks that produced no handoff accumulate ahead of the speak),
            # so skip the leading listen run, then the speak decision itself;
            # hidden states for TTS start after it and stop at the next
            # chunk terminator.
            listen_id = special_token_ids.get("listen_token_id")
            speak_id = special_token_ids.get("speak_token_id")
            out_ids = llm_output_ids
            j = 0
            while j < len(out_ids) and out_ids[j] == listen_id:
                j += 1
            if j < len(out_ids) and out_ids[j] == speak_id:
                out_start = j + 1
                out_end = len(out_ids)
                turn_eos_id = special_token_ids.get("turn_eos_token_id")
                for idx_t in range(out_start, len(out_ids)):
                    token_id = out_ids[idx_t]
                    if turn_eos_id is not None and token_id == turn_eos_id:
                        # Native duplex trains the talker on the <|turn_eos|>
                        # embedding itself, but any text sampled after it
                        # belongs to a stale tail and must not enter TTS.
                        out_end = idx_t + 1
                        break
                    if token_id in tts_end_ids:
                        out_end = idx_t
                        native_segment_end = token_id in {
                            special_token_ids.get("chunk_eos_token_id"),
                            special_token_ids.get("chunk_tts_eos_token_id"),
                        }
                        break
                # Map output indices onto hidden rows by END alignment: the
                # leading decision tokens of the delta may ALSO be folded into
                # the resumable prompt (they belong to earlier non-forwarded
                # segments), so prompt_len + delta over-counts them and
                # front-aligned indexing truncates the slice. The hidden
                # tensor's last len(out_ids) rows are the delta's rows.
                hidden_base = int(thinker_hidden_states.shape[0]) - len(out_ids)
                if hidden_base >= 0 and out_end > out_start:
                    tts_token_ids_slice = torch.tensor(out_ids[out_start:out_end], dtype=torch.long)
                    tts_hidden_slice = (
                        thinker_hidden_states[hidden_base + out_start : hidden_base + out_end]
                        .to(torch.float32)
                        .contiguous()
                    )
            elif j < len(out_ids) and out_ids[j] not in tts_end_ids:
                # HF streaming_generate does not require an explicit <|speak|>
                # marker. If a unit starts directly with text, the first token
                # is fed back into the LLM but is not included in
                # total_hidden_in_unit; TTS starts from the following token.
                out_start = j + 1
                out_end = len(out_ids)
                turn_eos_id = special_token_ids.get("turn_eos_token_id")
                for idx_t in range(out_start, len(out_ids)):
                    token_id = out_ids[idx_t]
                    if turn_eos_id is not None and token_id == turn_eos_id:
                        out_end = idx_t + 1
                        break
                    if token_id in tts_end_ids:
                        out_end = idx_t
                        native_segment_end = token_id in {
                            special_token_ids.get("chunk_eos_token_id"),
                            special_token_ids.get("chunk_tts_eos_token_id"),
                        }
                        break
                hidden_base = int(thinker_hidden_states.shape[0]) - len(out_ids)
                if hidden_base >= 0 and out_end > out_start:
                    tts_token_ids_slice = torch.tensor(out_ids[out_start:out_end], dtype=torch.long)
                    tts_hidden_slice = (
                        thinker_hidden_states[hidden_base + out_start : hidden_base + out_end]
                        .to(torch.float32)
                        .contiguous()
                    )
        handoff_ids = _coerce_token_id_list(tts_token_ids_slice) if tts_token_ids_slice is not None else None
        if is_native_duplex_handoff and handoff_ids:
            handoff_text = _decode_native_duplex_token_ids(
                handoff_ids,
                _streaming_context,
                request_id=str(llm_output.request_id),
            )
            if handoff_text is not None:
                # Match the released streaming_generate contract: transcript
                # text comes from total_ids_in_unit, the same slice that
                # conditions the Talker.
                thinker_text = handoff_text
        model_intermediate_buffer = build_duplex_intermediate_buffer(
            request_id=str(llm_output.request_id),
            prompt_token_ids=prompt_token_ids,
            output_token_ids=llm_output_ids,
            output_text=thinker_text,
            stream_output=is_native_duplex_handoff,
            native_duplex=is_native_duplex_handoff,
        )
        if is_native_duplex_handoff:
            turn_eos_id = special_token_ids.get("turn_eos_token_id")
            meta = model_intermediate_buffer.setdefault("meta", {})
            data_plane_metadata = _native_duplex_data_plane_metadata(_streaming_context)
            if data_plane_metadata is not None:
                model_intermediate_buffer["duplex"] = data_plane_metadata
            meta["native_duplex_segment_text"] = thinker_text
            meta.setdefault("override_keys", []).extend(
                [
                    "llm_output_text",
                    ["meta", "native_duplex_segment_text"],
                ]
            )
            if turn_eos_id is not None:
                # The talker detects turn end from <|turn_eos|> inside the
                # handed condition (official conditions on its embedding).
                meta["turn_eos_token_id"] = int(turn_eos_id)
            meta["turn_start"] = native_turn_start
            if native_segment_end:
                meta["segment_end"] = True
        req_mm_data = multi_modal_data.get(llm_output.request_id)
        ref_audio = _extract_first_audio_ref(req_mm_data)
        if ref_audio is None:
            ref_audio = _extract_native_runtime_ref_audio(
                model_intermediate_buffer.get("duplex"),
            )
        if ref_audio is not None:
            ref_waveform, ref_sr = ref_audio
            set_ref_audio(model_intermediate_buffer, _to_transport_list(ref_waveform), ref_sr)
        handoff_hidden = _to_transport_list(tts_hidden_slice) if tts_hidden_slice is not None else None
        native_turn_end_handoff = False
        if is_native_duplex_handoff:
            turn_eos_id = special_token_ids.get("turn_eos_token_id")
            native_turn_end_handoff = turn_eos_id is not None and handoff_ids is not None and turn_eos_id in handoff_ids
            if not handoff_ids:
                continue
        set_tts_handoff(model_intermediate_buffer, handoff_ids, handoff_hidden)
        if native_turn_end_handoff:
            model_intermediate_buffer.setdefault("meta", {})["turn_end"] = True

        if handoff_ids is not None and handoff_hidden is not None:
            condition_suffix_length = 1 if is_native_duplex_handoff else 2
            condition_length = max(len(handoff_ids), len(handoff_hidden)) + condition_suffix_length
            scheduler_prompt_token_ids = [0] * condition_length
            handoff_meta = model_intermediate_buffer.setdefault("meta", {})
            handoff_meta["next_stage_prompt_len"] = condition_length
            # Native duplex resumes one Talker request within a turn, but a new
            # assistant turn must discard the previous turn's prompt and KV.
            if not is_native_duplex_handoff or native_turn_start:
                handoff_meta["replace_streaming_prompt"] = True
        else:
            scheduler_prompt_token_ids = _build_tts_scheduler_prompt_token_ids(
                tts_token_ids_slice,
                llm_output_ids,
                prompt_token_ids,
            )
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=scheduler_prompt_token_ids,
                model_intermediate_buffer=model_intermediate_buffer,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
        if native_turn_end_handoff:
            bridge_states = getattr(_streaming_context, "bridge_states", None)
            duplex_state = bridge_states.get("duplex") if isinstance(bridge_states, dict) else None
            if isinstance(duplex_state, dict):
                current_model_turn_id = duplex_state.get("model_turn_id", duplex_state.get("turn_id", 0))
                if isinstance(current_model_turn_id, int):
                    duplex_state["model_turn_id"] = current_model_turn_id + 1

    return tts_inputs
