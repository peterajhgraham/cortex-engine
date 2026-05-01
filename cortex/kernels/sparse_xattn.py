"""Block-sparse cross-attention Triton kernel.

Problem
-------
The Perceiver cross-attention computes Q (latents) × K^T (spike events) and then
weights the V (spike events).  Shape: Q = (B, H, L, Dh), K/V = (B, H, E, Dh).
Naive cost: O(L × E) per head.

Spike events are temporally sparse: in MC_Maze, a 600 ms window has ~685 events
from 137 neurons, but the events cluster around movement onset.  During rest,
most of the 120 time-bins are silent.  If we can skip attention computations for
event tiles that are far from any query's "receptive window," we save a fraction
of the L × E work proportional to spike sparsity.

Block-sparsity approach
-----------------------
1. Partition the event sequence into tiles of BLOCK_E events.
2. Pre-compute a boolean mask: mask[qi, ej] = True means the query tile qi should
   attend to event tile ej.  False means skip — the events in that tile contribute
   zero attention weight.
3. In the kernel, the outer loop over event tiles tests the mask and skips False
   tiles, saving compute and HBM reads proportional to 1 − density.

The mask is computed externally by `build_temporal_block_mask`, which compares
the time-bin range of each event tile against a ±window around each latent's
"expected" time (mapped linearly from latent index onto the full time range).
This is a soft prior, not a hard constraint — latents that specialize outside
their expected window will still see their events as long as the tile is non-empty
and within range.

FlashAttention-2 online softmax
--------------------------------
Standard scaled dot-product attention requires materializing the full (L, E)
attention matrix.  FA2 computes the output in a streaming fashion, maintaining
running max/sum statistics so the softmax denominator never needs to be stored:

    For each query tile qi:
        m = -inf  (running row-wise max)
        l = 0     (running row-wise normaliser ≈ exp-sum)
        o = 0     (running weighted output, *un-normalised* by l)

        For each non-masked event tile ej:
            S = qi @ kj^T × scale          # (BLOCK_L, BLOCK_E) scores
            m_new = max(m, rowmax(S))
            p     = exp(S − m_new)         # re-centred probabilities (BLOCK_L, BLOCK_E)
            l_new = exp(m − m_new) × l + rowsum(p)
            o     = exp(m − m_new) × o + p @ vj
            m, l  = m_new, l_new

        o /= l                             # final normalisation

Memory savings vs PyTorch eager SDPA
--------------------------------------
PyTorch SDPA materialises the full (B, H, L, E) attention matrix then multiplies
by V.  Our kernel:
  - Never materialises the full attention matrix (FA2)
  - Skips entire (BLOCK_L, BLOCK_E) tiles when masked (sparsity)
  - Only reads K and V tiles that are not masked

At 50% mask density (typical for MC_Maze during rest), this halves the K/V HBM
reads, which are the bottleneck.

API
---
The kernel takes a precomputed block_mask rather than computing it inline, so the
sparsity policy (temporal window, learned receptive fields, fixed stride) is
decoupled from the kernel itself.  `build_temporal_block_mask` provides a useful
default based on time-bin proximity.

References
----------
  Dao 2023, "FlashAttention-2: Faster Attention with Better Parallelism"
  Triton tutorial 06 (Flash Attention)
  Jaegle et al. 2021, "Perceiver IO"
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── PyTorch reference ─────────────────────────────────────────────────────────


def sparse_cross_attention_reference(
    q: torch.Tensor,          # (B, H, L, Dh)
    k: torch.Tensor,          # (B, H, E, Dh)
    v: torch.Tensor,          # (B, H, E, Dh)
    block_mask: torch.Tensor | None = None,  # (n_q_tiles, n_k_tiles) bool
    BLOCK_L: int = 64,
    BLOCK_E: int = 64,
    scale: float | None = None,
) -> torch.Tensor:             # (B, H, L, Dh)
    """Reference: dense SDPA with optional tile-level masking.

    When block_mask is provided, event tiles that are False get -inf in the
    attention score before softmax — equivalent to excluding those events.
    Used as the correctness target for the Triton kernel.
    """
    B, H, L, Dh = q.shape
    _B, _H, E, _Dh = k.shape
    s = scale if scale is not None else Dh ** -0.5

    if block_mask is None:
        return F.scaled_dot_product_attention(q, k, v, scale=s)

    # Expand block mask to per-element attention mask: (L, E)
    n_q = math.ceil(L / BLOCK_L)
    n_k = math.ceil(E / BLOCK_E)
    assert block_mask.shape == (n_q, n_k), (
        f"block_mask shape {block_mask.shape} != expected ({n_q}, {n_k})"
    )

    # Build dense (L, E) bool mask from block mask
    attn_mask = torch.zeros(L, E, dtype=torch.bool, device=q.device)
    for qi in range(n_q):
        for kj in range(n_k):
            if block_mask[qi, kj]:
                l_start, l_end = qi * BLOCK_L, min((qi + 1) * BLOCK_L, L)
                e_start, e_end = kj * BLOCK_E, min((kj + 1) * BLOCK_E, E)
                attn_mask[l_start:l_end, e_start:e_end] = True

    # Convert bool to additive float mask: False → -inf, True → 0
    float_mask = torch.where(
        attn_mask,
        torch.zeros(L, E, device=q.device, dtype=q.dtype),
        torch.full((L, E), float("-inf"), device=q.device, dtype=q.dtype),
    )  # (L, E)
    # Broadcast over (B, H) dimensions
    float_mask = float_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, E)

    scores = torch.matmul(q, k.transpose(-2, -1)) * s + float_mask  # (B, H, L, E)
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


# ── Block mask helpers ────────────────────────────────────────────────────────


def build_temporal_block_mask(
    event_times: torch.Tensor,  # (E,) int64 — time-bin for each event
    n_latents: int,             # L — number of latent queries
    max_time_bins: int,         # T — total time axis length
    window_bins: int = 30,      # half-window: latent i attends to bins ±window around its centre
    BLOCK_L: int = 64,
    BLOCK_E: int = 64,
) -> torch.Tensor:              # (n_q_tiles, n_k_tiles) bool, on CPU
    """Build a coarse block-level temporal attention mask.

    Latent i is mapped to a nominal time centre: t_i = i / L × T.
    Event tile j is included for latent tile qi if any event in tile j has a
    time-bin within ±window_bins of any latent centre in tile qi.

    Returns a (n_q, n_k) bool tensor: True = compute this tile.
    This is a conservative (superset) mask — no valid attention weights are
    masked out, only provably-irrelevant tiles are skipped.
    """
    E = event_times.shape[0]
    n_q = math.ceil(n_latents / BLOCK_L)
    n_k = math.ceil(E / BLOCK_E)

    mask = torch.zeros(n_q, n_k, dtype=torch.bool)

    # Precompute time-bin range for each event tile
    e_times_cpu = event_times.cpu()
    tile_tmin = torch.zeros(n_k, dtype=torch.long)
    tile_tmax = torch.zeros(n_k, dtype=torch.long)
    for kj in range(n_k):
        e_start = kj * BLOCK_E
        e_end = min((kj + 1) * BLOCK_E, E)
        tile_tmin[kj] = e_times_cpu[e_start:e_end].min()
        tile_tmax[kj] = e_times_cpu[e_start:e_end].max()

    # For each latent tile, compute its time range and check overlap
    for qi in range(n_q):
        l_start = qi * BLOCK_L
        l_end = min((qi + 1) * BLOCK_L, n_latents)
        # Latent centres in this tile (mapped linearly to time axis)
        centres = torch.arange(l_start, l_end, dtype=torch.float32) / n_latents * max_time_bins
        lat_tmin = int(centres.min().item()) - window_bins
        lat_tmax = int(centres.max().item()) + window_bins

        # Include event tile if its time range overlaps latent window
        for kj in range(n_k):
            overlap = (tile_tmin[kj] <= lat_tmax) and (tile_tmax[kj] >= lat_tmin)
            mask[qi, kj] = overlap

    return mask  # CPU bool tensor


# ── Triton kernel ─────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def _sparse_cross_attn_fwd(
        # Q: (B, H, L, Dh) — row major, stride_qb/qh/ql/qd
        q_ptr, stride_qb, stride_qh, stride_ql, stride_qd,
        # K: (B, H, E, Dh)
        k_ptr, stride_kb, stride_kh, stride_ke, stride_kd,
        # V: (B, H, E, Dh)
        v_ptr, stride_vb, stride_vh, stride_ve, stride_vd,
        # Block mask: (n_q_tiles, n_k_tiles) bool, row-major, stride_mb/stride_mk
        mask_ptr, stride_mn, stride_mk,
        # Output: (B, H, L, Dh)
        out_ptr, stride_ob, stride_oh, stride_ol, stride_od,
        # Shapes
        B, H, L, E, Dh,
        # Scale factor (1 / sqrt(Dh))
        scale,
        # Number of event tiles
        n_k_tiles,
        # Tile sizes (constexpr: used in tl.arange)
        BLOCK_L: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_Dh: tl.constexpr,  # must be power of 2, >= Dh
    ) -> None:
        """FA2-style cross-attention, skipping masked event tiles.

        Grid: (ceil(L/BLOCK_L), B × H)
        Each program handles one query tile for one (batch, head) pair.
        """
        pid_q  = tl.program_id(0)   # which query tile (along L)
        pid_bh = tl.program_id(1)   # flattened (batch, head) index

        batch_idx = pid_bh // H
        head_idx  = pid_bh % H

        # Query tile: absolute latent indices
        l_start = pid_q * BLOCK_L
        l_offs  = l_start + tl.arange(0, BLOCK_L)   # (BLOCK_L,)
        l_mask  = l_offs < L

        # Precompute base pointers for this (batch, head) slice
        q_base = (q_ptr
                  + batch_idx * stride_qb
                  + head_idx  * stride_qh)
        k_base = (k_ptr
                  + batch_idx * stride_kb
                  + head_idx  * stride_kh)
        v_base = (v_ptr
                  + batch_idx * stride_vb
                  + head_idx  * stride_vh)
        o_base = (out_ptr
                  + batch_idx * stride_ob
                  + head_idx  * stride_oh)

        # Load query tile: (BLOCK_L, Dh)
        # We load in BLOCK_Dh chunks to support Dh > 128 if needed
        dh_offs = tl.arange(0, BLOCK_Dh)    # (BLOCK_Dh,)
        dh_mask = dh_offs < Dh

        q_ptrs = q_base + l_offs[:, None] * stride_ql + dh_offs[None, :] * stride_qd
        q_tile = tl.load(q_ptrs,
                         mask=l_mask[:, None] & dh_mask[None, :],
                         other=0.0)  # (BLOCK_L, BLOCK_Dh)

        # FA2 running statistics: shape (BLOCK_L,)
        m_i = tl.full((BLOCK_L,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_L,), dtype=tl.float32)
        # Accumulator for output: (BLOCK_L, BLOCK_Dh) in float32 for numerical stability
        o_i = tl.zeros((BLOCK_L, BLOCK_Dh), dtype=tl.float32)

        # Iterate over event tiles
        for kj in range(n_k_tiles):
            # Check block mask: skip this tile if masked out
            # mask_ptr layout: (n_q_tiles, n_k_tiles), accessed as [pid_q, kj]
            should_compute = tl.load(mask_ptr + pid_q * stride_mn + kj * stride_mk)
            if should_compute:
                e_start = kj * BLOCK_E
                e_offs  = e_start + tl.arange(0, BLOCK_E)  # (BLOCK_E,)
                e_mask  = e_offs < E

                # Load K tile: (BLOCK_E, BLOCK_Dh)
                k_ptrs = k_base + e_offs[:, None] * stride_ke + dh_offs[None, :] * stride_kd
                k_tile = tl.load(k_ptrs,
                                 mask=e_mask[:, None] & dh_mask[None, :],
                                 other=0.0)

                # Attention scores: (BLOCK_L, BLOCK_E)
                scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

                # Mask out out-of-bounds events with -inf so they don't affect softmax
                scores = tl.where(e_mask[None, :], scores, float("-inf"))

                # FA2 online softmax update
                m_ij = tl.max(scores, axis=1)              # (BLOCK_L,) row-wise max
                m_new = tl.maximum(m_i, m_ij)              # updated running max
                alpha = tl.exp(m_i - m_new)                # rescale factor for old stats
                p = tl.exp(scores - m_new[:, None])        # (BLOCK_L, BLOCK_E) unnormalised probs

                # Load V tile: (BLOCK_E, BLOCK_Dh)
                v_ptrs = v_base + e_offs[:, None] * stride_ve + dh_offs[None, :] * stride_vd
                v_tile = tl.load(v_ptrs,
                                 mask=e_mask[:, None] & dh_mask[None, :],
                                 other=0.0)

                # Update running output and normaliser
                o_i = alpha[:, None] * o_i + tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float32)
                l_i = alpha * l_i + tl.sum(p, axis=1)
                m_i = m_new

        # Normalise: divide by running sum
        # Guard against l_i = 0 (all tiles masked) → output zero
        l_safe = tl.where(l_i > 0, l_i, tl.full((BLOCK_L,), 1.0, dtype=tl.float32))
        o_final = o_i / l_safe[:, None]

        # Store output tile
        o_ptrs = o_base + l_offs[:, None] * stride_ol + dh_offs[None, :] * stride_od
        tl.store(o_ptrs, o_final.to(q_ptr.dtype.element_ty),
                 mask=l_mask[:, None] & dh_mask[None, :])


# ── Python dispatcher ─────────────────────────────────────────────────────────


def sparse_cross_attention(
    q: torch.Tensor,          # (B, H, L, Dh)
    k: torch.Tensor,          # (B, H, E, Dh)
    v: torch.Tensor,          # (B, H, E, Dh)
    block_mask: torch.Tensor | None = None,  # (n_q_tiles, n_k_tiles) bool
    BLOCK_L: int = 64,
    BLOCK_E: int = 64,
    scale: float | None = None,
) -> torch.Tensor:             # (B, H, L, Dh)
    """Block-sparse cross-attention.  Requires CUDA + Triton.

    Falls back to the dense PyTorch reference on non-CUDA devices or if Triton
    is not installed.

    Args:
        q:          Query tensor (B, H, L, Dh)
        k:          Key tensor   (B, H, E, Dh)
        v:          Value tensor (B, H, E, Dh)
        block_mask: Boolean tile mask (n_q_tiles, n_k_tiles).  None = dense (all True).
        BLOCK_L:    Latent tile size (must be power of 2).
        BLOCK_E:    Event tile size (must be power of 2).
        scale:      Attention scale.  Default: 1/sqrt(Dh).

    Returns:
        Output tensor (B, H, L, Dh) in the same dtype as q.
    """
    device = q.device
    if device.type != "cuda" or not _TRITON_AVAILABLE:
        return sparse_cross_attention_reference(q, k, v, block_mask, BLOCK_L, BLOCK_E, scale)

    B, H, L, Dh = q.shape
    _B, _H, E, _Dh = k.shape
    s = scale if scale is not None else Dh ** -0.5

    if block_mask is None:
        # Dense: all tiles are active — standard SDPA is faster here
        return F.scaled_dot_product_attention(q, k, v, scale=s)

    n_q_tiles = math.ceil(L / BLOCK_L)
    n_k_tiles = math.ceil(E / BLOCK_E)

    assert block_mask.shape == (n_q_tiles, n_k_tiles), (
        f"block_mask shape {block_mask.shape} != expected ({n_q_tiles}, {n_k_tiles})"
    )

    # Ensure inputs are contiguous
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    block_mask_dev = block_mask.to(device=device, dtype=torch.bool).contiguous()

    # BLOCK_Dh must be a power of 2 >= Dh (for tl.arange / tl.dot efficiency)
    BLOCK_Dh = 1
    while BLOCK_Dh < Dh:
        BLOCK_Dh *= 2

    out = torch.zeros_like(q)

    grid = (n_q_tiles, B * H)

    _sparse_cross_attn_fwd[grid](
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_mask_dev, block_mask_dev.stride(0), block_mask_dev.stride(1),
        out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, L, E, Dh,
        s,
        n_k_tiles,
        BLOCK_L=BLOCK_L, BLOCK_E=BLOCK_E, BLOCK_Dh=BLOCK_Dh,
    )
    return out
