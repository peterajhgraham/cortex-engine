"""Fused RMSNorm + linear projection Triton kernel.

Standard fusion pattern: RMSNorm followed by a linear layer is a frequent
sequence in transformers. Fusing them eliminates an intermediate write/read
of the activations.

Op being fused
--------------
    y = norm(x) @ W
    where norm(x) = x / sqrt(mean(x^2) + eps) * gamma

Implementation TODO list for Claude Code (Phase 2.4):
    [ ] PyTorch reference
    [ ] Fused forward kernel (norm + matmul)
    [ ] Backward pass (recompute norm to avoid storing intermediates)
    [ ] Correctness test
    [ ] Benchmark vs unfused reference

This is the simplest of the three kernels and should be implemented last, as
practice consolidating the patterns established in the more complex ones.
"""
