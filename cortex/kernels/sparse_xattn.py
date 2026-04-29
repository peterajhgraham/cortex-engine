"""Sparse cross-attention Triton kernel.

The most novel kernel in the project. Naive cross-attention from spike tokens
into the latent array does work proportional to L * E (latents x events). When
spike events are temporally sparse, much of that work is producing
near-zero attention scores for events far from the latent's effective receptive
field.

Optimization strategy
---------------------
Two-level tiling:
    - Outer tile: (BLOCK_L, BLOCK_E) along latents x events
    - Inner: standard FlashAttention-2 online softmax within the tile

Sparsity exploitation
---------------------
Each spike event has a time bin. Each latent has a learned but bounded effective
receptive field in time. We pre-compute a coarse mask at the tile level: skip
entire (BLOCK_L, BLOCK_E) tiles where no event falls in any latent's receptive
field. This reduces compute by ~3x in the regime we actually run in.

Implementation TODO list for Claude Code (Phase 2.3):
    [ ] PyTorch reference (correctness target)
    [ ] Coarse temporal mask precomputation kernel
    [ ] Main attention kernel with FA-2 style online softmax
    [ ] Backward pass for training (matched recompute pattern)
    [ ] Correctness test against reference
    [ ] Benchmark sweep (E from 1k to 64k, L=128)
    [ ] Roofline analysis writeup in benchmarks/kernels/sparse_xattn.md

References:
    - Dao 2023, FlashAttention-2
    - Triton tutorial 06 (FlashAttention)
    - Liger Kernel for production patterns
"""

from __future__ import annotations

import torch


def sparse_cross_attention_reference(
    q: torch.Tensor,  # (B, H, L, D)
    k: torch.Tensor,  # (B, H, E, D)
    v: torch.Tensor,  # (B, H, E, D)
    event_times: torch.Tensor,  # (B, E) int — for sparsity mask construction
    latent_receptive_field_ms: int = 200,
) -> torch.Tensor:
    """Reference impl. Used in tests; not optimized."""
    # TODO: build a per-(L, E) mask from event_times and latent receptive fields,
    # then call SDPA with the mask
    raise NotImplementedError
