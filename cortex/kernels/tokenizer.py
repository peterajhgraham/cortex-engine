"""Fused spike tokenizer Triton kernel.

This is the worked reference example for the kernel pattern used throughout
this project. Subsequent kernels (sparse cross-attention, fused RMSNorm) follow
the same structure: PyTorch reference, Triton implementation, dispatcher.

Op being fused
--------------
Original PyTorch (3 separate kernel launches, 3 reads of E indices, 3 writes):
    tokens = neuron_emb[neuron_ids] + time_emb[time_bins] + value_emb[values]

Fused Triton (1 kernel launch, 1 write):
    For each event e and feature dim d:
        tokens[e, d] = neuron_emb[neuron_ids[e], d]
                     + time_emb[time_bins[e], d]
                     + value_emb[values[e], d]

Memory access pattern
---------------------
    - neuron_emb, time_emb, value_emb are gather targets (random access along dim 0)
    - tokens is contiguous output: (E, D)
    - Each program instance handles BLOCK_E events x BLOCK_D features
    - We tile over both axes to maximize L2 reuse of the same embedding rows
      across nearby events with the same indices

Performance notes
-----------------
    - Optimal BLOCK sizes vary with hidden_dim. Autotune over (BLOCK_E, BLOCK_D, num_warps).
    - This kernel is most beneficial at large E and moderate D (the tokenization
      bottleneck regime). At very small E the overhead of kernel launch dominates.

Reference for Triton patterns
-----------------------------
    - Triton tutorial 03 (matmul) for tiling
    - Liger Kernel `cross_entropy.py` for embedding-style indexed loads
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ── PyTorch reference ─────────────────────────────────────────────────────────


def fused_tokenizer_reference(
    neuron_emb: torch.Tensor,  # (N, D)
    time_emb: torch.Tensor,    # (T, D)
    value_emb: torch.Tensor,   # (V, D)
    neuron_ids: torch.Tensor,  # (E,)
    time_bins: torch.Tensor,   # (E,)
    values: torch.Tensor,      # (E,)
) -> torch.Tensor:
    """Reference implementation. The Triton kernel must match this within tolerance."""
    return neuron_emb[neuron_ids] + time_emb[time_bins] + value_emb[values]


# ── Triton kernel ─────────────────────────────────────────────────────────────


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_E": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_E": 128, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_E": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_E": 128, "BLOCK_D": 128}, num_warps=8),
        triton.Config({"BLOCK_E": 256, "BLOCK_D": 64}, num_warps=8),
    ],
    key=["E", "D"],
)
@triton.jit
def _fused_tokenizer_kernel(
    # Embedding tables (all (rows, D))
    neuron_emb_ptr, time_emb_ptr, value_emb_ptr,
    # Index tensors (E,)
    neuron_ids_ptr, time_bins_ptr, values_ptr,
    # Output (E, D)
    out_ptr,
    # Sizes
    E: tl.constexpr,
    D: tl.constexpr,
    # Strides (in elements, not bytes)
    neuron_stride: tl.constexpr,
    time_stride: tl.constexpr,
    value_stride: tl.constexpr,
    out_stride_e: tl.constexpr,
    # Tile shape
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Each program handles BLOCK_E events x BLOCK_D feature dims."""
    pid_e = tl.program_id(0)
    pid_d = tl.program_id(1)

    e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    e_mask = e_offsets < E
    d_mask = d_offsets < D

    # Load index tensors (BLOCK_E,)
    neuron_ids = tl.load(neuron_ids_ptr + e_offsets, mask=e_mask, other=0)
    time_bins = tl.load(time_bins_ptr + e_offsets, mask=e_mask, other=0)
    values = tl.load(values_ptr + e_offsets, mask=e_mask, other=0)

    # Compute gather addresses
    # Each event in the block reads a (BLOCK_D,) row from each embedding table
    neuron_offsets = (neuron_ids[:, None] * neuron_stride) + d_offsets[None, :]
    time_offsets = (time_bins[:, None] * time_stride) + d_offsets[None, :]
    value_offsets = (values[:, None] * value_stride) + d_offsets[None, :]

    mask_2d = e_mask[:, None] & d_mask[None, :]

    n_emb = tl.load(neuron_emb_ptr + neuron_offsets, mask=mask_2d, other=0.0)
    t_emb = tl.load(time_emb_ptr + time_offsets, mask=mask_2d, other=0.0)
    v_emb = tl.load(value_emb_ptr + value_offsets, mask=mask_2d, other=0.0)

    out = n_emb + t_emb + v_emb

    out_offsets = (e_offsets[:, None] * out_stride_e) + d_offsets[None, :]
    tl.store(out_ptr + out_offsets, out, mask=mask_2d)


# ── Public wrapper ────────────────────────────────────────────────────────────


def fused_tokenizer(
    neuron_emb: torch.Tensor,
    time_emb: torch.Tensor,
    value_emb: torch.Tensor,
    neuron_ids: torch.Tensor,
    time_bins: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused spike tokenizer.

    Args:
        neuron_emb, time_emb, value_emb: contiguous (rows, D) tensors
        neuron_ids, time_bins, values: contiguous (E,) int tensors
    Returns:
        tokens: (E, D) tensor in the same dtype as the embedding tables
    """
    if not (neuron_emb.is_contiguous() and time_emb.is_contiguous() and value_emb.is_contiguous()):
        raise ValueError("embedding tables must be contiguous")
    if not (neuron_ids.shape == time_bins.shape == values.shape):
        raise ValueError("index tensors must have identical shapes")

    E = neuron_ids.shape[0]
    D = neuron_emb.shape[1]
    if not (time_emb.shape[1] == D and value_emb.shape[1] == D):
        raise ValueError("all embedding tables must have the same hidden dim")

    out = torch.empty((E, D), dtype=neuron_emb.dtype, device=neuron_emb.device)

    grid = lambda meta: (
        triton.cdiv(E, meta["BLOCK_E"]),
        triton.cdiv(D, meta["BLOCK_D"]),
    )

    _fused_tokenizer_kernel[grid](
        neuron_emb, time_emb, value_emb,
        neuron_ids, time_bins, values,
        out,
        E, D,
        neuron_emb.stride(0), time_emb.stride(0), value_emb.stride(0),
        out.stride(0),
    )
    return out
