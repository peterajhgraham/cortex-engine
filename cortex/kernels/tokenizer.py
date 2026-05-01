"""Fused spike tokenizer Triton kernel.

The canonical kernel pattern for this project: every subsequent kernel follows
the same file structure (reference → kernel → dispatcher → tests → bench).

Operation being fused
---------------------
PyTorch eager launches 3 separate index_select kernels then 2 elementwise-add
kernels, allocating 2 intermediate (E, D) tensors that each round-trip through
HBM:

    tokens = neuron_emb[neuron_ids]       # (E, D)  ← write intermediate 1
           + time_emb[time_bins]          # (E, D)  ← write intermediate 2
           + value_emb[values]            # (E, D)

Memory traffic (eager):
    Reads:  3 × E × D (gathers) + 2 × 2 × E × D (addition reads) = 7 × E × D
    Writes: 2 × E × D (intermediates) + 1 × E × D (final)        = 3 × E × D
    Total:  10 × E × D × sizeof(dtype) bytes

Memory traffic (fused):
    Reads:  3 × E × D (gathers only)
    Writes: 1 × E × D (final output)
    Total:  4 × E × D × sizeof(dtype) bytes  (theoretical minimum)

Tiling strategy
---------------
2D grid: (ceil(E / BLOCK_E), ceil(D / BLOCK_D))

Each program handles a (BLOCK_E, BLOCK_D) tile of the output.  Within the tile:
  - Load BLOCK_E integer indices from each of the 3 index tensors  → 1D vectors
  - Compute 2D gather addresses: idx[e] × row_stride + d_off[d]   → 2D offsets
  - Load BLOCK_E × BLOCK_D floats from each embedding table        → 2D gathers
  - Add in registers, store once

Triton bugs to avoid
--------------------
  - ONLY BLOCK_E and BLOCK_D need `tl.constexpr` (used in tl.arange).
    E, D, and all strides are RUNTIME scalars — marking them constexpr would
    recompile the kernel for every new (E, D) pair, destroying performance.
  - The autotune `key=["E", "D"]` caches the best config per (E, D) pair,
    which is independent of whether E/D are constexpr.

References
----------
  Triton tutorial 03 (matrix multiplication) — tiling pattern
  Liger Kernel cross_entropy.py — indexed gather in Triton
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── PyTorch reference ─────────────────────────────────────────────────────────


def fused_tokenizer_reference(
    neuron_emb: torch.Tensor,  # (N_vocab, D)
    time_emb:   torch.Tensor,  # (T_vocab, D)
    value_emb:  torch.Tensor,  # (V_vocab, D)
    neuron_ids: torch.Tensor,  # (E,) int64
    time_bins:  torch.Tensor,  # (E,) int64
    values:     torch.Tensor,  # (E,) int64
) -> torch.Tensor:             # (E, D)
    """Pure-PyTorch reference.

    The Triton kernel must match this within rtol=1e-3 / atol=1e-2 for
    bfloat16 and within rtol=1e-5 / atol=1e-5 for float32.
    """
    return neuron_emb[neuron_ids] + time_emb[time_bins] + value_emb[values]


# ── Triton kernel ─────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            # Small D — pack more events per block
            triton.Config({"BLOCK_E": 64,  "BLOCK_D": 64},  num_warps=4),
            triton.Config({"BLOCK_E": 128, "BLOCK_D": 64},  num_warps=4),
            triton.Config({"BLOCK_E": 256, "BLOCK_D": 64},  num_warps=8),
            # Large D — larger feature tiles for better vectorisation
            triton.Config({"BLOCK_E": 32,  "BLOCK_D": 128}, num_warps=4),
            triton.Config({"BLOCK_E": 64,  "BLOCK_D": 128}, num_warps=4),
            triton.Config({"BLOCK_E": 128, "BLOCK_D": 128}, num_warps=8),
            triton.Config({"BLOCK_E": 256, "BLOCK_D": 128}, num_warps=8),
            # D=512 production shape
            triton.Config({"BLOCK_E": 16,  "BLOCK_D": 256}, num_warps=4),
            triton.Config({"BLOCK_E": 32,  "BLOCK_D": 256}, num_warps=8),
        ],
        key=["E", "D"],  # re-autotune when input shape changes; does NOT require constexpr
    )
    @triton.jit
    def _fused_tokenizer_kernel(
        # Embedding weight matrices (each row-major, shape (vocab, D))
        neuron_emb_ptr,
        time_emb_ptr,
        value_emb_ptr,
        # Index tensors, shape (E,), dtype int64
        neuron_ids_ptr,
        time_bins_ptr,
        values_ptr,
        # Output tensor, shape (E, D), row-major
        out_ptr,
        # Runtime shape — NOT constexpr; only BLOCK_* need constexpr (used in tl.arange)
        E,              # int: number of spike events
        D,              # int: embedding / hidden dimension
        # Row strides in elements (not bytes); these are runtime values, not constexpr
        neuron_stride,  # int: stride(neuron_emb, 0) == D for contiguous tensors
        time_stride,    # int: stride(time_emb, 0)
        value_stride,   # int: stride(value_emb, 0)
        out_stride_e,   # int: stride(out, 0) == D for contiguous output
        # Tile sizes — must be constexpr because they appear inside tl.arange()
        BLOCK_E: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        """Process one (BLOCK_E, BLOCK_D) tile of the output.

        Memory layout:
            embedding[row, col] is at embedding_ptr + row * stride + col
            output[e, d]        is at out_ptr + e * out_stride_e + d
        """
        pid_e = tl.program_id(0)  # which event tile
        pid_d = tl.program_id(1)  # which feature tile

        # Absolute event and feature indices for this tile
        e_start = pid_e * BLOCK_E
        d_start = pid_d * BLOCK_D

        e_offs = e_start + tl.arange(0, BLOCK_E)  # (BLOCK_E,)
        d_offs = d_start + tl.arange(0, BLOCK_D)  # (BLOCK_D,)

        # Boundary masks — handles E and D that are not multiples of block sizes
        e_mask = e_offs < E  # (BLOCK_E,) bool
        d_mask = d_offs < D  # (BLOCK_D,) bool
        mask_2d = e_mask[:, None] & d_mask[None, :]  # (BLOCK_E, BLOCK_D) bool

        # Load event indices (1D integer gather)
        nid = tl.load(neuron_ids_ptr + e_offs, mask=e_mask, other=0)  # (BLOCK_E,) int64
        tb  = tl.load(time_bins_ptr  + e_offs, mask=e_mask, other=0)  # (BLOCK_E,) int64
        val = tl.load(values_ptr     + e_offs, mask=e_mask, other=0)  # (BLOCK_E,) int64

        # Compute 2D gather addresses for each embedding table:
        #   addr[e, d] = ptr + index[e] * row_stride + d_offs[d]
        n_addr = nid[:, None] * neuron_stride + d_offs[None, :]  # (BLOCK_E, BLOCK_D)
        t_addr = tb[:, None]  * time_stride   + d_offs[None, :]
        v_addr = val[:, None] * value_stride  + d_offs[None, :]

        # Gather float values from each embedding table
        n_emb = tl.load(neuron_emb_ptr + n_addr, mask=mask_2d, other=0.0)
        t_emb = tl.load(time_emb_ptr   + t_addr, mask=mask_2d, other=0.0)
        v_emb = tl.load(value_emb_ptr  + v_addr, mask=mask_2d, other=0.0)

        # Fused addition — happens in registers, no intermediate HBM write
        result = n_emb + t_emb + v_emb

        # Store output
        out_addr = e_offs[:, None] * out_stride_e + d_offs[None, :]
        tl.store(out_ptr + out_addr, result, mask=mask_2d)


# ── Python dispatcher ─────────────────────────────────────────────────────────


def fused_tokenizer(
    neuron_emb: torch.Tensor,
    time_emb:   torch.Tensor,
    value_emb:  torch.Tensor,
    neuron_ids: torch.Tensor,
    time_bins:  torch.Tensor,
    values:     torch.Tensor,
) -> torch.Tensor:
    """Triton-fused spike tokenizer.  Requires CUDA; falls back to reference on other devices.

    Args:
        neuron_emb: (N_vocab, D) float tensor, contiguous
        time_emb:   (T_vocab, D) float tensor, contiguous
        value_emb:  (V_vocab, D) float tensor, contiguous
        neuron_ids: (E,) int64 tensor on the same device
        time_bins:  (E,) int64 tensor on the same device
        values:     (E,) int64 tensor on the same device

    Returns:
        tokens: (E, D) float tensor with the same dtype as the embedding tables

    Raises:
        RuntimeError: if CUDA is not available and triton is not installed
    """
    device = neuron_emb.device
    if device.type != "cuda" or not _TRITON_AVAILABLE:
        # Graceful fallback — matches reference exactly
        return fused_tokenizer_reference(neuron_emb, time_emb, value_emb, neuron_ids, time_bins, values)

    if not (neuron_emb.is_contiguous() and time_emb.is_contiguous() and value_emb.is_contiguous()):
        raise ValueError("embedding tables must be contiguous (call .contiguous() first)")
    if not (neuron_ids.shape == time_bins.shape == values.shape):
        raise ValueError(f"index tensors must have identical shapes, got "
                         f"{neuron_ids.shape}, {time_bins.shape}, {values.shape}")

    D = neuron_emb.shape[1]
    if time_emb.shape[1] != D or value_emb.shape[1] != D:
        raise ValueError("all embedding tables must have the same hidden dim D")

    E = neuron_ids.shape[0]
    if E == 0:
        return torch.empty((0, D), dtype=neuron_emb.dtype, device=device)

    # Ensure indices are contiguous int64 for the kernel
    neuron_ids = neuron_ids.contiguous()
    time_bins  = time_bins.contiguous()
    values     = values.contiguous()

    out = torch.empty((E, D), dtype=neuron_emb.dtype, device=device)

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
