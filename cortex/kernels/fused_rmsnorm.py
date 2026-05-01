"""Fused RMSNorm + linear projection Triton kernel.

Problem
-------
In each SelfAttentionBlock and CrossAttentionBlock the pattern:

    norm(x) → linear

appears twice per block:
  1. norm_attn(latents) → qkv_proj  (D → 3D)
  2. norm_mlp(latents)  → mlp[0]   (D → 4D)

PyTorch runs this as two separate kernel launches:
  - LayerNorm kernel: reads x (MxK), writes x_norm (MxK)
  - Linear kernel:    reads x_norm (MxK) and W (KxN), writes output (MxN)

The intermediate x_norm tensor is written to HBM and immediately read back.
At MxK = 8192 x 512 in float32, that is 16 MB of avoidable HBM traffic per
fused pair, or 16 MB x 2 pairs x 7 blocks = 224 MB per forward pass.

Fused kernel
------------
This kernel eliminates x_norm entirely by computing the normalisation inline
during the matmul K reduction.  It uses a *two-pass* strategy:

  Pass 1: iterate over K in BLOCK_K tiles, accumulating x² sum per row
          → compute rms[m] = sqrt(mean_sq[m] + eps) for each row in the tile

  Pass 2: iterate over K x N in (BLOCK_K, BLOCK_N) tiles
          → load x again, apply norm (x / rms x gamma) in registers
          → multiply by W tile and accumulate into the output

Memory traffic (reference, two kernels):
  Reads:  MxK (norm) + MxK (linear) + KxN (weight)  = 2xMxK + KxN
  Writes: MxK (x_norm) + MxN (output)               = MxK + MxN

Memory traffic (fused kernel):
  Reads:  MxK (pass1) + MxK (pass2) + KxN (weight)  = 2xMxK + KxN
  Writes: MxN (output only)                           = MxN

Saving: eliminates 1 write + 1 read of the MxK intermediate = 2xMxK bytes.
For MxK = 8192x512 x 2 bytes (bf16) = 8 MB saved per fused call.

Note on RMSNorm vs LayerNorm
-----------------------------
The model uses `nn.LayerNorm` (which also subtracts the mean).  RMSNorm skips
the mean subtraction, which is both faster and sufficient for transformers (Biao
Zhang & Rico Sennrich 2019; LLaMA, Mistral, and Gemma all use RMSNorm).  This
kernel implements RMSNorm.  If the model is to be fully trained with this kernel,
its LayerNorm layers should be replaced with RMSNorm.  The bench/test files
operate only on the RMSNorm formulation.

For Phase 2, this kernel is the correctness and benchmark reference.  Swapping
the model layers is a separate Phase 2 task.

Grid
----
  (ceil(M / BLOCK_M), ceil(N / BLOCK_N))
  Each program handles one (BLOCK_M, BLOCK_N) output tile.

References
----------
  Zhang & Sennrich 2019, "Root Mean Square Layer Normalization"
  Triton tutorial 05 (layer normalisation)
  Liger Kernel RMSNorm — two-pass pattern
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── PyTorch reference ─────────────────────────────────────────────────────────


def rms_norm_linear_reference(
    x: torch.Tensor,  # (M, K)  input
    gamma: torch.Tensor,  # (K,)    RMSNorm elementwise scale
    w: torch.Tensor,  # (K, N)  linear weight (pre-transposed)
    bias: torch.Tensor | None,  # (N,)    optional bias
    eps: float = 1e-6,
) -> torch.Tensor:  # (M, N)
    """Reference: RMSNorm then linear, as two separate PyTorch ops.

    The Triton kernel must match this within rtol=1e-2 / atol=1e-2 (bfloat16)
    or rtol=1e-4 / atol=1e-4 (float32).  Looser tolerances than the tokenizer
    because floating-point reductions over K accumulate differently between
    PyTorch and Triton.
    """
    # RMSNorm: x / rms(x) x gamma
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = (x_f32 / rms) * gamma.float()
    # Linear: x_norm @ w + bias
    out = x_norm.to(x.dtype) @ w
    if bias is not None:
        out = out + bias
    return out


# ── Triton kernel ─────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.autotune(  # type: ignore[untyped-decorator]  # triton.jit is intentionally untyped
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8),
        ],
        key=["M", "K", "N"],
    )
    @triton.jit  # type: ignore[untyped-decorator]
    def _rms_norm_linear_fwd(  # type: ignore[no-untyped-def]
        x_ptr,  # (M, K)
        gamma_ptr,  # (K,)
        w_ptr,  # (K, N) — weight already in K-major layout (row = input feature)
        bias_ptr,  # (N,) or null
        out_ptr,  # (M, N)
        # Runtime shapes
        M,
        K,
        N,
        # Strides
        stride_xm,
        stride_xk,  # x strides
        stride_wk,
        stride_wn,  # w strides
        stride_om,
        stride_on,  # out strides
        # Scalar params
        eps,
        # Flag: whether bias is present
        HAS_BIAS: tl.constexpr,
        # Tile sizes — constexpr because used in tl.arange
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        """Two-pass fused RMSNorm + linear.

        Pass 1: compute RMS for each row in the BLOCK_M tile (reduces over K).
        Pass 2: apply norm x gamma inline during the matmul K-reduction.

        Both passes read x from HBM.  The normalised intermediate is NEVER
        written to HBM — it lives entirely in registers.
        """
        bm = tl.program_id(0)  # output row tile
        bn = tl.program_id(1)  # output col tile

        m_start = bm * BLOCK_M
        n_start = bn * BLOCK_N

        m_offs = m_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        n_offs = n_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        m_mask = m_offs < M
        n_mask = n_offs < N

        # ── Pass 1: accumulate x² sum per row ────────────────────────────────
        # We iterate over K in BLOCK_K chunks to keep register usage bounded.
        rms_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
            k_mask = k_offs < K

            x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_tile = tl.load(
                x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )  # (BLOCK_M, BLOCK_K)
            x_f32 = x_tile.to(tl.float32)
            rms_sq += tl.sum(x_f32 * x_f32, axis=1)  # (BLOCK_M,) accumulate

        # Compute per-row RMS  (divide by K, not BLOCK_K x n_chunks, to get the mean)
        rms = tl.sqrt(rms_sq / K + eps)  # (BLOCK_M,)

        # ── Pass 2: norm x gamma inline, then tiled matmul ───────────────────
        # Accumulator for output tile (BLOCK_M, BLOCK_N) in float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # Load x tile and apply RMSNorm
            x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(
                tl.float32
            )  # (BLOCK_M, BLOCK_K)

            # Load gamma for this K chunk
            gamma_tile = tl.load(gamma_ptr + k_offs, mask=k_mask, other=0.0).to(
                tl.float32
            )  # (BLOCK_K,)

            # Apply norm: divide by rms, scale by gamma
            x_norm = (x_tile / rms[:, None]) * gamma_tile[None, :]  # (BLOCK_M, BLOCK_K)

            # Load W tile: (BLOCK_K, BLOCK_N)
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(
                tl.float32
            )  # (BLOCK_K, BLOCK_N)

            # Matmul accumulation
            acc = tl.dot(x_norm, w_tile, acc)

        # Add bias if present
        if HAS_BIAS:
            bias_tile = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias_tile[None, :]

        # Store output (cast back to input dtype)
        out_ptrs = out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
        tl.store(out_ptrs, acc.to(x_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


# ── Python dispatcher ─────────────────────────────────────────────────────────


def rms_norm_linear(
    x: torch.Tensor,  # (M, K) or (B, L, K) — will be reshaped
    gamma: torch.Tensor,  # (K,)
    w: torch.Tensor,  # (K, N)
    bias: torch.Tensor | None = None,  # (N,)
    eps: float = 1e-6,
) -> torch.Tensor:  # same leading dims as x, with K → N
    """Fused RMSNorm + linear.  Requires CUDA + Triton; falls back to reference otherwise.

    Accepts arbitrary leading dimensions (e.g., (B, L, K)) by flattening them
    to (M, K) before calling the kernel, then reshaping the output.

    Args:
        x:     Input of shape (..., K).
        gamma: RMSNorm scale, shape (K,).
        w:     Linear weight, shape (K, N).
        bias:  Optional linear bias, shape (N,).
        eps:   Epsilon for numerical stability in the RMS denominator.

    Returns:
        Output of shape (..., N).
    """
    leading_shape = x.shape[:-1]
    K = x.shape[-1]
    M = x.numel() // K
    x_2d = x.reshape(M, K)

    device = x.device
    if device.type != "cuda" or not _TRITON_AVAILABLE:
        out = rms_norm_linear_reference(x_2d, gamma, w, bias, eps)
        return out.reshape(*leading_shape, w.shape[1])

    N = w.shape[1]

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    out = torch.empty((M, N), dtype=x.dtype, device=device)

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _rms_norm_linear_fwd[grid](
        x_2d,
        gamma,
        w,
        bias if bias is not None else x_2d,  # dummy pointer; HAS_BIAS guards the load
        out,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        HAS_BIAS=(bias is not None),
    )
    return out.reshape(*leading_shape, N)
