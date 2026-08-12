# SPDX-License-Identifier: Apache-2.0
"""Triton-Ascend virtual concat + input projection for MiniCPM-o 4.5."""

from __future__ import annotations

import torch


def virtual_concat_linear(
    x: torch.Tensor,
    mu: torch.Tensor,
    speaker: torch.Tensor,
    cond: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Project four ``[B, C, T]`` inputs without materializing their concat.

    The kernel presents the four tensors as one logical K dimension to a
    single Cube matmul and emits the same ``[B, T, N]`` layout as
    ``F.linear(torch.cat(inputs, dim=1).transpose(1, 2), weight, bias)``.
    """
    from vllm.triton_utils import tl, triton

    if x.device.type != "npu":
        raise ValueError("virtual_concat_linear requires Ascend NPU tensors")
    inputs = (x, mu, speaker, cond)
    if any(value.device != x.device or value.dtype != x.dtype for value in inputs):
        raise ValueError("virtual concat inputs must share device and dtype")
    if any(value.ndim != 3 for value in inputs):
        raise ValueError("virtual concat inputs must have shape [B, C, T]")
    batch, _, width = x.shape
    if any(value.shape[0] != batch or value.shape[2] != width for value in inputs):
        raise ValueError("virtual concat inputs must share batch and width")
    channels = tuple(int(value.shape[1]) for value in inputs)
    total_channels = sum(channels)
    if weight.ndim != 2 or int(weight.shape[1]) != total_channels:
        raise ValueError("projection weight K dimension must equal concatenated channels")
    if weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("projection weight must share input device and dtype")
    output_features = int(weight.shape[0])
    if bias is not None and (
        bias.shape != (output_features,)
        or bias.device != x.device
        or bias.dtype != x.dtype
    ):
        raise ValueError("projection bias must have shape [N] and share device/dtype")

    output = torch.empty((batch, width, output_features), device=x.device, dtype=x.dtype)
    dummy_bias = bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
    rows = int(batch * width)
    # The competition shape has only 100 logical rows. Use two wide Cube
    # programs rather than many small tiles; launch count dominates here.
    block_m, block_n, block_k = 128, 256, 64
    grid = (triton.cdiv(rows, block_m), triton.cdiv(output_features, block_n))
    _virtual_concat_linear_kernel[grid](
        x,
        mu,
        speaker,
        cond,
        weight,
        dummy_bias,
        output,
        rows,
        output_features,
        total_channels,
        width,
        channels[0],
        channels[1],
        channels[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        mu.stride(0),
        mu.stride(1),
        mu.stride(2),
        speaker.stride(0),
        speaker.stride(1),
        speaker.stride(2),
        cond.stride(0),
        cond.stride(1),
        cond.stride(2),
        weight.stride(0),
        weight.stride(1),
        output.stride(1),
        output.stride(2),
        has_bias=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output


try:
    from vllm.triton_utils import tl, triton
except ImportError:
    tl = None
    triton = None


if triton is not None:

    @triton.jit
    def _virtual_concat_linear_kernel(
        x_ptr,
        mu_ptr,
        speaker_ptr,
        cond_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K,
        width,
        x_channels,
        mu_channels,
        speaker_channels,
        stride_xb,
        stride_xc,
        stride_xt,
        stride_mub,
        stride_muc,
        stride_mut,
        stride_sb,
        stride_sc,
        stride_st,
        stride_cb,
        stride_cc,
        stride_ct,
        stride_wn,
        stride_wk,
        stride_ot,
        stride_on,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        batch_indices = rows // width
        time_indices = rows % width
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        mu_start = x_channels
        speaker_start = mu_start + mu_channels
        cond_start = speaker_start + speaker_channels

        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            row_mask = rows[:, None] < M
            k_mask = k[None, :] < K

            x_k = k
            x_values = tl.load(
                x_ptr
                + batch_indices[:, None] * stride_xb
                + x_k[None, :] * stride_xc
                + time_indices[:, None] * stride_xt,
                mask=row_mask & k_mask & (x_k[None, :] < mu_start),
                other=0.0,
            )
            mu_k = k - mu_start
            mu_values = tl.load(
                mu_ptr
                + batch_indices[:, None] * stride_mub
                + mu_k[None, :] * stride_muc
                + time_indices[:, None] * stride_mut,
                mask=row_mask & k_mask & (k[None, :] >= mu_start) & (k[None, :] < speaker_start),
                other=0.0,
            )
            speaker_k = k - speaker_start
            speaker_values = tl.load(
                speaker_ptr
                + batch_indices[:, None] * stride_sb
                + speaker_k[None, :] * stride_sc
                + time_indices[:, None] * stride_st,
                mask=row_mask & k_mask & (k[None, :] >= speaker_start) & (k[None, :] < cond_start),
                other=0.0,
            )
            cond_k = k - cond_start
            cond_values = tl.load(
                cond_ptr
                + batch_indices[:, None] * stride_cb
                + cond_k[None, :] * stride_cc
                + time_indices[:, None] * stride_ct,
                mask=row_mask & k_mask & (k[None, :] >= cond_start),
                other=0.0,
            )
            activations = x_values + mu_values + speaker_values + cond_values
            weights = tl.load(
                weight_ptr + columns[None, :] * stride_wn + k[:, None] * stride_wk,
                mask=(columns[None, :] < N) & (k[:, None] < K),
                other=0.0,
            )
            accumulator += tl.dot(activations, weights)

        if has_bias:
            bias = tl.load(bias_ptr + columns, mask=columns < N, other=0.0)
            accumulator += bias[None, :]
        output_offsets = rows[:, None] * stride_ot + columns[None, :] * stride_on
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            mask=(rows[:, None] < M) & (columns[None, :] < N),
        )

else:
    _virtual_concat_linear_kernel = None
