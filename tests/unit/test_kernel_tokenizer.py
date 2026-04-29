"""Correctness test for the fused tokenizer Triton kernel.

This is the canonical pattern: every Triton kernel in this project has a test
that compares to a PyTorch reference and fails the build if numerical
equivalence is broken.
"""

from __future__ import annotations

import pytest
import torch

from cortex.kernels.tokenizer import fused_tokenizer, fused_tokenizer_reference


@pytest.mark.gpu
@pytest.mark.parametrize(
    "E,D,N,T,V",
    [
        (128, 64, 256, 512, 8),
        (1024, 128, 256, 512, 8),
        (4096, 256, 512, 1024, 8),
        (16384, 256, 512, 1024, 16),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_tokenizer_matches_reference(
    E: int, D: int, N: int, T: int, V: int, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    torch.manual_seed(0)

    neuron_emb = torch.randn(N, D, device=device, dtype=dtype)
    time_emb = torch.randn(T, D, device=device, dtype=dtype)
    value_emb = torch.randn(V, D, device=device, dtype=dtype)

    neuron_ids = torch.randint(0, N, (E,), device=device, dtype=torch.int64)
    time_bins = torch.randint(0, T, (E,), device=device, dtype=torch.int64)
    values = torch.randint(0, V, (E,), device=device, dtype=torch.int64)

    ref = fused_tokenizer_reference(neuron_emb, time_emb, value_emb, neuron_ids, time_bins, values)
    out = fused_tokenizer(neuron_emb, time_emb, value_emb, neuron_ids, time_bins, values)

    rtol = 1e-3 if dtype == torch.bfloat16 else 1e-5
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
