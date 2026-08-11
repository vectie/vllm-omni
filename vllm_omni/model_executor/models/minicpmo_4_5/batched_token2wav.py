# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

import importlib
import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_SILENCE_TOKEN = 4218
_NPU_DIT_MLP_GRAPH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH"
_NPU_DIT_MLP_GRAPH_WIDTH_ENV = "VLLM_OMNI_MINICPMO45_NPU_DIT_MLP_GRAPH_WIDTH"
logger = logging.getLogger(__name__)


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def _npu_dit_mlp_graph_enabled(config_value: Any = None) -> bool:
    env_value = os.environ.get(_NPU_DIT_MLP_GRAPH_ENV)
    raw = env_value if env_value not in (None, "") else config_value
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {_NPU_DIT_MLP_GRAPH_ENV}={raw!r}")


def _npu_dit_mlp_graph_width(config_value: Any = None) -> int:
    env_value = os.environ.get(_NPU_DIT_MLP_GRAPH_WIDTH_ENV)
    raw = env_value if env_value not in (None, "") else (50 if config_value is None else config_value)
    try:
        width = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {_NPU_DIT_MLP_GRAPH_WIDTH_ENV}={raw!r}") from exc
    if width <= 0:
        raise ValueError(f"{_NPU_DIT_MLP_GRAPH_WIDTH_ENV} must be positive, got {width}")
    return width


def _ensure_torchair_broadcast_alias() -> None:
    """Repair a TorchAir registration-order incompatibility in vLLM workers.

    vLLM's distributed initialization can register ``npu_define::broadcast``
    before TorchAir imports its experimental converters. In that order,
    TorchAir skips defining its module-local ``op_broadcast`` name and a later
    converter import fails even though the operator itself is available.
    Populate the missing alias without changing the installed torch-npu tree.
    """
    module = importlib.import_module(
        "torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast"
    )
    if not hasattr(module, "op_broadcast"):
        broadcast = torch.ops.npu_define.broadcast
        module.op_broadcast = getattr(broadcast, "default", broadcast)


def _dit_mlp_residual(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> torch.Tensor:
    """Fixed-shape DiT MLP/residual partition used by the NPU graph path."""
    hidden = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    hidden = hidden * (1 + scale) + shift
    hidden = F.linear(hidden, fc1_weight, fc1_bias)
    hidden = F.gelu(hidden, approximate="tanh")
    hidden = F.linear(hidden, fc2_weight, fc2_bias)
    return x + gate * hidden


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(
        self,
        token2wav: Any,
        *,
        npu_dit_mlp_graph: Any = None,
        npu_dit_mlp_graph_width: Any = None,
    ):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        # The upstream streaming path preallocates fixed-size CFM and DiT
        # caches. This adapter never calls that path and supplies dynamically
        # sized request-owned buffers to ``blocks_forward_chunk`` instead.
        decoder = self.flow.decoder
        for module in (decoder, decoder.estimator):
            for buffer_name in ("att_cache_buffer", "cnn_cache_buffer"):
                if buffer_name in module._buffers:
                    setattr(module, buffer_name, None)
        hift_parameter = next(self.hift.parameters(), None)
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "cfm_timeline_base",
            1 - torch.cos(torch.linspace(0, 1, self.n_timesteps + 1, dtype=torch.float32) * 0.5 * torch.pi),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        self._timeline_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
        self._cfm_delta_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
        self._timestep_embedding_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._cfg_workspace: dict[tuple[str, tuple[int, ...], torch.dtype, str, int | None], torch.Tensor] = {}
        self._npu_cfm_graphs: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._npu_cfm_graph_disabled = False
        self._npu_dit_mlp_graph_enabled = _npu_dit_mlp_graph_enabled(npu_dit_mlp_graph)
        self._npu_dit_mlp_graph_width = _npu_dit_mlp_graph_width(npu_dit_mlp_graph_width)
        self._npu_dit_mlp_graph: Any | None = None
        self._npu_dit_mlp_graph_disabled = False
        self._npu_dit_mlp_graph_used = False
        self._warmup_npu_dit_mlp_graph()

    def _warmup_npu_dit_mlp_graph(self) -> None:
        """Compile the fixed-width post-convolution partition during startup."""
        if not self._npu_dit_mlp_graph_enabled:
            return
        estimator = getattr(getattr(self.flow, "decoder", None), "estimator", None)
        blocks = getattr(estimator, "blocks", None)
        if not blocks:
            self._npu_dit_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT MLP graph disabled: estimator blocks are unavailable")
            return
        block = blocks[0]
        weight = getattr(getattr(block, "mlp", None), "fc1", None)
        weight = getattr(weight, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.device.type != "npu":
            self._npu_dit_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT MLP graph disabled: estimator is not on NPU")
            return
        width = self._npu_dit_mlp_graph_width
        hidden_size = int(weight.shape[1])
        x = weight.new_zeros((2, width, hidden_size))
        shift = weight.new_zeros((2, 1, hidden_size))
        scale = weight.new_zeros((2, 1, hidden_size))
        gate = weight.new_zeros((2, 1, hidden_size))
        try:
            graph_fn = self._get_npu_dit_mlp_graph()
            if graph_fn is None:
                return
            with torch.inference_mode():
                graph_fn(
                    x,
                    shift,
                    scale,
                    gate,
                    block.mlp.fc1.weight,
                    block.mlp.fc1.bias,
                    block.mlp.fc2.weight,
                    block.mlp.fc2.bias,
                )
            torch.npu.synchronize()
            logger.info(
                "Compiled MiniCPM-o NPU DiT MLP graph partition for CFG batch=2, width=%d, hidden=%d",
                width,
                hidden_size,
            )
        except Exception:
            self._npu_dit_mlp_graph = None
            self._npu_dit_mlp_graph_disabled = True
            logger.warning("MiniCPM-o NPU DiT MLP graph compilation failed; using eager blocks", exc_info=True)

    def _get_npu_dit_mlp_graph(self):
        if self._npu_dit_mlp_graph_disabled:
            return None
        if self._npu_dit_mlp_graph is None:
            from torch_npu.dynamo import torchair

            _ensure_torchair_broadcast_alias()
            compiler_config = torchair.CompilerConfig()
            self._npu_dit_mlp_graph = torch.compile(
                _dit_mlp_residual,
                backend=torchair.get_npu_backend(compiler_config=compiler_config),
                fullgraph=True,
                dynamic=False,
            )
        return self._npu_dit_mlp_graph

    def _timeline_for(self, value: torch.Tensor) -> torch.Tensor:
        key = (value.device.type, value.device.index, value.dtype)
        timeline = self._timeline_cache.get(key)
        if timeline is None:
            timeline = self.cfm_timeline_base.to(device=value.device, dtype=value.dtype)
            self._timeline_cache[key] = timeline
        return timeline

    def _cfm_deltas_for(self, timeline: torch.Tensor) -> torch.Tensor:
        """Cache the invariant Euler step widths without changing rounding.

        CosyVoice derives every next width from the accumulated time rather
        than taking a direct timeline difference. Reproduce that recurrence
        once per device/dtype, then reuse the resulting scalars for every
        streamed chunk. This removes two tiny eager accelerator operations
        from each non-final CFM step while preserving the original update
        order and values.
        """
        key = (timeline.device.type, timeline.device.index, timeline.dtype)
        cached = self._cfm_delta_cache.get(key)
        if cached is not None:
            return cached
        time = timeline[0]
        dt = timeline[1] - timeline[0]
        deltas: list[torch.Tensor] = []
        for step in range(self.n_timesteps):
            deltas.append(dt)
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time
        cached = torch.stack(deltas).detach()
        self._cfm_delta_cache[key] = cached
        return cached

    def _cfg_pair(self, name: str, value: torch.Tensor, *, zero_unconditional: bool) -> torch.Tensor:
        key = (name, tuple(value.shape), value.dtype, value.device.type, value.device.index)
        pair = self._cfg_workspace.get(key)
        expected_shape = (int(value.shape[0]) * 2, *value.shape[1:])
        if pair is None or tuple(pair.shape) != expected_shape:
            pair = torch.empty(expected_shape, device=value.device, dtype=value.dtype)
            self._cfg_workspace[key] = pair
        batch_size = int(value.shape[0])
        pair[:batch_size].copy_(value)
        if zero_unconditional:
            pair[batch_size:].zero_()
        else:
            pair[batch_size:].copy_(value)
        return pair

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            cached = PromptFeatures(
                speech_tokens=values[0],
                speaker_embedding=values[2],
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        self._prompt_features.pop((prompt_cache_id, prompt_wav), None)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    @staticmethod
    def _estimator_buffers(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn = x.new_empty((depth, batch_size, cnn_channels, cnn_width))
        att = x.new_empty((depth, batch_size, heads, old_att_len + chunk_size, att_width))
        return cnn, att

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time_embedding: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        graph_width = self._npu_dit_mlp_graph_width
        use_mlp_graph = (
            self._npu_dit_mlp_graph_enabled
            and not self._npu_dit_mlp_graph_disabled
            and estimator_input.device.type == "npu"
            and int(estimator_input.shape[0]) == 2
            and int(estimator_input.shape[2]) == graph_width
            and cnn_cache is not None
            and att_cache is not None
        )
        if use_mlp_graph:
            try:
                graph_fn = self._get_npu_dit_mlp_graph()
                if graph_fn is not None:
                    result = self._estimator_blocks_forward_chunk_mlp_graph(
                        estimator,
                        estimator_input,
                        time_embedding,
                        old_cnn,
                        old_att,
                        cnn_out,
                        att_out,
                        graph_fn,
                    )
                    if not self._npu_dit_mlp_graph_used:
                        logger.info(
                            "MiniCPM-o NPU DiT MLP graph replay active for CFG batch=2, width=%d",
                            graph_width,
                        )
                        self._npu_dit_mlp_graph_used = True
                    return result, cnn_out, att_out
            except Exception:
                self._npu_dit_mlp_graph = None
                self._npu_dit_mlp_graph_disabled = True
                logger.warning("MiniCPM-o NPU DiT MLP graph execution failed; using eager blocks", exc_info=True)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
        return result, cnn_out, att_out

    @staticmethod
    def _estimator_blocks_forward_chunk_mlp_graph(
        estimator: nn.Module,
        estimator_input: torch.Tensor,
        time_embedding: torch.Tensor,
        old_cnn: Any,
        old_att: Any,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
        graph_fn: Any,
    ) -> torch.Tensor:
        """Run attention/conv eagerly and replay the fixed-width MLP graph."""
        hidden = estimator.in_proj(estimator_input.transpose(1, 2))
        for block_idx, block in enumerate(estimator.blocks):
            if block.training or block.norm2.weight is not None or block.norm2.bias is not None:
                raise RuntimeError("MiniCPM-o NPU DiT MLP graph requires eval-mode affine-free norm2")
            if float(block.norm2.eps) != 1e-6:
                raise RuntimeError(f"MiniCPM-o NPU DiT MLP graph requires norm2 eps=1e-6, got {block.norm2.eps}")
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                shift_conv,
                scale_conv,
                gate_conv,
            ) = block.adaLN_modulation(time_embedding).chunk(9, dim=-1)

            attention, new_att = block.attn.forward_chunk(
                block.norm1(hidden) * (1 + scale_msa) + shift_msa,
                old_att[block_idx],
                None,
            )
            hidden = hidden + gate_msa * attention

            convolution, new_cnn = block.conv.forward_chunk(
                block.norm3(hidden) * (1 + scale_conv) + shift_conv,
                old_cnn[block_idx],
            )
            hidden = hidden + gate_conv * convolution
            hidden = graph_fn(
                hidden,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                block.mlp.fc1.weight,
                block.mlp.fc1.bias,
                block.mlp.fc2.weight,
                block.mlp.fc2.bias,
            )
            cnn_out[block_idx].copy_(new_cnn)
            att_out[block_idx, :, :, : new_att.shape[2], :].copy_(new_att)

        hidden = estimator.final_layer(hidden, time_embedding)
        return hidden.transpose(1, 2)

    def _estimator_time_embeddings(
        self,
        estimator: nn.Module,
        timeline: torch.Tensor,
        cfg_batch_size: int,
    ) -> torch.Tensor:
        """Cache graph-safe CFM timestep embeddings for inference.

        CosyVoice's ``TimestepEmbedder`` constructs its sinusoidal frequencies
        on CPU and calls ``.to(t)`` for every diffusion step. Besides repeating
        the same host-to-device copy and MLP work for every streamed chunk,
        that transfer is illegal while an Ascend NPU graph is being captured.
        Token2wav weights and the CFM timeline are immutable during serving, so
        compute the step embeddings once and reuse them by shape.

        Lightweight test/dummy estimators that do not expose CosyVoice's
        embedder attributes keep the generic eager behavior.
        """
        embedder = estimator.t_embedder
        frequency_size = getattr(embedder, "frequency_embedding_size", None)
        scale = getattr(embedder, "scale", None)
        mlp = getattr(embedder, "mlp", None)
        if not isinstance(frequency_size, int) or scale is None or not isinstance(mlp, nn.Module):
            return torch.stack(
                [embedder(timeline[step].expand(cfg_batch_size)).unsqueeze(1) for step in range(self.n_timesteps)]
            )

        key = (
            id(embedder),
            cfg_batch_size,
            timeline.device.type,
            timeline.device.index,
            timeline.dtype,
            self.n_timesteps,
        )
        cached = self._timestep_embedding_cache.get(key)
        if cached is not None:
            return cached

        half = frequency_size // 2
        # Match CosyVoice's CPU/default-dtype frequency construction exactly,
        # but perform its one transfer before any graph capture begins.
        frequencies = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half) / half).to(timeline)
        embeddings: list[torch.Tensor] = []
        time = timeline[0].expand(cfg_batch_size)
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            arguments = (time * float(scale))[:, None] * frequencies[None]
            sinusoidal = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
            if frequency_size % 2:
                sinusoidal = torch.cat((sinusoidal, torch.zeros_like(sinusoidal[:, :1])), dim=-1)
            embeddings.append(mlp(sinusoidal).unsqueeze(1))
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
        cached = torch.stack(embeddings).detach()
        self._timestep_embedding_cache[key] = cached
        return cached

    def _decode_cfm_eager(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = self._timeline_for(mu)
        mu_cfg = self._cfg_pair("mu", mu, zero_unconditional=True)
        speakers_cfg = self._cfg_pair("speakers", speakers, zero_unconditional=True)
        cond_cfg = self._cfg_pair("cond", cond, zero_unconditional=True)
        time_embeddings = self._estimator_time_embeddings(estimator, timeline, batch_size * 2)
        deltas = self._cfm_deltas_for(timeline)
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        for step in range(self.n_timesteps):
            old_cnn = cnn_cache[step] if cnn_cache is not None else None
            old_att = att_cache[step] if att_cache is not None else None
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=self._cfg_pair("x", x, zero_unconditional=False),
                mu=mu_cfg,
                time_embedding=time_embeddings[step],
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=old_cnn,
                att_cache=old_att,
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + deltas[step] * velocity
            next_cnn.append(step_cnn)
            next_att.append(step_att)
        return x, torch.stack(next_cnn), torch.stack(next_att)

    @staticmethod
    def _optional_tensor_signature(value: torch.Tensor | None) -> Any:
        if value is None:
            return None
        return tuple(value.shape), value.dtype, value.device.type, value.device.index

    @staticmethod
    def _npu_cfm_graph_cache_limit() -> int:
        raw = os.environ.get("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE", "4")
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "Invalid VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE=%r; using 4",
                raw,
            )
            return 4

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enabled = os.environ.get("VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH", "0").strip().lower()
        if enabled not in {"1", "true", "yes", "on"} or mu.device.type != "npu" or self._npu_cfm_graph_disabled:
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )

        inputs = (mu, speakers, cond, cnn_cache, att_cache)
        key = tuple(self._optional_tensor_signature(value) for value in inputs)
        entry = self._npu_cfm_graphs.get(key)
        if entry is not None:
            self._npu_cfm_graphs.move_to_end(key)
            for static, current in zip(entry["inputs"], inputs, strict=True):
                if static is not None and current is not None:
                    static.copy_(current)
            entry["graph"].replay()
            return tuple(output.clone() for output in entry["outputs"])

        static_inputs = tuple(value.clone() if value is not None else None for value in inputs)
        static_mu, static_speakers, static_cond, static_cnn, static_att = static_inputs
        try:
            with torch.inference_mode():
                self._decode_cfm_eager(
                    static_mu,
                    static_speakers,
                    static_cond,
                    cnn_cache=static_cnn,
                    att_cache=static_att,
                )
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.inference_mode(), torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
                outputs = self._decode_cfm_eager(
                    static_mu,
                    static_speakers,
                    static_cond,
                    cnn_cache=static_cnn,
                    att_cache=static_att,
                )
            graph.replay()
        except Exception:
            self._npu_cfm_graph_disabled = True
            self._npu_cfm_graphs.clear()
            logger.warning("MiniCPM-o NPU CFM graph capture failed; using eager Code2Wav", exc_info=True)
            return self._decode_cfm_eager(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )

        self._npu_cfm_graphs[key] = {
            "graph": graph,
            "inputs": static_inputs,
            "outputs": outputs,
        }
        max_graphs = self._npu_cfm_graph_cache_limit()
        while len(self._npu_cfm_graphs) > max_graphs:
            self._npu_cfm_graphs.popitem(last=False)
        return tuple(output.clone() for output in outputs)

    @staticmethod
    def _split_flow_cache(cache: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": torch.cat(
                        (
                            cache["estimator_att_cache"][:, :, row : row + 1],
                            cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                }
            )
        return result

    @staticmethod
    def _stack_flow_cache(states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tokens, speakers, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        flow_cache = self._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )

        prompt_len = int(features.mels.shape[1])
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        speech, source = self.hift(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states
