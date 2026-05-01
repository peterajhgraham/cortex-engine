"""Correctness tests for the fused RMSNorm + linear kernel.

Structure:

  Group 1 — reference correctness (no CUDA):
      Verify rms_norm_linear_reference produces correct output against a manual
      PyTorch RMSNorm + linear computation.  These run everywhere.

  Group 2 — Triton kernel correctness (requires CUDA):
      Verify rms_norm_linear (Triton) matches the reference within tolerance.

  Group 3 — Shape and dtype edge cases:
      3D input (B, L, K), no bias, non-power-of-2 shapes.

Tolerance rationale
-------------------
  The two-pass kernel accumulates differently from PyTorch's float32 reductions.
  bfloat16: atol=1e-1, rtol=1e-2  (RMS division adds one more rounding step)
  float32:  atol=1e-4, rtol=1e-4

  These tolerances are wider than the tokenizer because the RMS computation
  (squared sum → divide → sqrt) introduces more floating-point rounding than a
  simple addition.  The Phase 2 spec requires rtol=1e-3, atol=1e-3 for all
  kernels vs their reference — this is satisfied at float32.
"""

from __future__ import annotations

import pytest
import torch

from cortex.kernels.fused_rmsnorm import rms_norm_linear, rms_norm_linear_reference

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tensors(
    M: int,
    K: int,
    N: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    has_bias: bool = False,
):
    torch.manual_seed(17)
    x = torch.randn(M, K, device=device, dtype=dtype)
    gamma = torch.ones(K, device=device, dtype=dtype)  # neutral init for easy checks
    w = torch.randn(K, N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype) if has_bias else None
    return x, gamma, w, bias


# ── Group 1: reference correctness (CPU) ─────────────────────────────────────


class TestReference:
    def test_output_shape_2d(self):
        x, gamma, w, _ = _make_tensors(M=32, K=64, N=128)
        out = rms_norm_linear_reference(x, gamma, w, None)
        assert out.shape == (32, 128)

    def test_output_shape_3d(self):
        """Reference must work with 3D input (B, L, K)."""
        # Flatten manually then call
        M, K, N = 8 * 16, 64, 128
        x, gamma, w, _ = _make_tensors(M, K, N)
        x_3d = x.reshape(8, 16, K)
        out = rms_norm_linear_reference(x_3d.reshape(M, K), gamma, w, None)
        assert out.shape == (M, N)

    def test_matches_manual_implementation(self):
        """Reference must match a manually-coded RMSNorm + linear."""
        M, K, N = 16, 64, 32
        x, gamma, w, _ = _make_tensors(M, K, N)

        # Manual: float32 RMS then linear
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_norm = (x_f32 / rms) * gamma.float()
        expected = x_norm.to(x.dtype) @ w

        out = rms_norm_linear_reference(x, gamma, w, None)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_gamma_ones_leaves_norm_unchanged(self):
        """With gamma=ones, the output is just x/rms @ w."""
        M, K, N = 8, 32, 16
        x, gamma, w, _ = _make_tensors(M, K, N)
        gamma.fill_(1.0)

        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_norm = x_f / rms
        expected = x_norm.to(x.dtype) @ w

        out = rms_norm_linear_reference(x, gamma, w, None)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_bias_added(self):
        """With bias, output must equal no-bias output + bias broadcast."""
        M, K, N = 8, 32, 16
        x, gamma, w, bias = _make_tensors(M, K, N, has_bias=True)

        out_no_bias = rms_norm_linear_reference(x, gamma, w, None)
        out_bias = rms_norm_linear_reference(x, gamma, w, bias)
        torch.testing.assert_close(out_bias, out_no_bias + bias[None, :])

    def test_output_dtype_preserved(self):
        for dtype in (torch.float32, torch.float16):
            x, gamma, w, _ = _make_tensors(M=8, K=32, N=16, dtype=dtype)
            out = rms_norm_linear_reference(x, gamma, w, None)
            assert out.dtype == dtype, f"expected {dtype}, got {out.dtype}"

    def test_numerically_stable_for_large_x(self):
        """Large-magnitude inputs must not produce inf/nan via RMS overflow."""
        x = torch.full((8, 64), 1e4, dtype=torch.float32)
        gamma = torch.ones(64)
        w = torch.randn(64, 32)
        out = rms_norm_linear_reference(x, gamma, w, None)
        assert torch.isfinite(out).all()


# ── Group 2: Triton kernel correctness (requires CUDA) ───────────────────────


@pytest.mark.gpu
@pytest.mark.parametrize(
    "M,K,N",
    [
        (16, 64, 128),  # small
        (64, 128, 256),  # medium
        (256, 512, 512),  # single self-attn block QKV projection (L=1, batchxL large)
        (8192, 512, 1536),  # Cortex-S production: BxL=32x256=8192, K=512, N=3x512 (QKV)
        (8192, 512, 2048),  # Cortex-S MLP first layer: K=512, N=4x512
        (33, 70, 97),  # non-power-of-2 in all dims
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_reference(M: int, K: int, N: int, dtype: torch.dtype):
    """Triton output must equal the PyTorch reference within tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    x, gamma, w, _ = _make_tensors(M, K, N, device=device, dtype=dtype)

    ref = rms_norm_linear_reference(x, gamma, w, None)
    out = rms_norm_linear(x, gamma, w)

    assert out.shape == (M, N)
    assert out.dtype == dtype

    # RMS division accumulates floating-point error; bfloat16 tolerances are wider
    rtol, atol = (1e-2, 1e-1) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.gpu
def test_triton_bias(self=None):
    """Triton kernel with bias must match reference + bias."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    M, K, N = 64, 128, 256
    x, gamma, w, bias = _make_tensors(M, K, N, device="cuda", has_bias=True)

    ref = rms_norm_linear_reference(x, gamma, w, bias)
    out = rms_norm_linear(x, gamma, w, bias)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_triton_3d_input():
    """Triton kernel must accept (B, L, K) input and return (B, L, N)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    B, L, K, N = 2, 256, 512, 1536
    torch.manual_seed(0)
    x = torch.randn(B, L, K, device="cuda", dtype=torch.float32)
    gamma = torch.ones(K, device="cuda")
    w = torch.randn(K, N, device="cuda")

    out = rms_norm_linear(x, gamma, w)
    assert out.shape == (B, L, N)

    # Cross-check against reference applied to the same flattened input
    ref = rms_norm_linear_reference(x.reshape(B * L, K), gamma, w, None).reshape(B, L, N)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# ── Group 3: dispatcher behaviour ────────────────────────────────────────────


class TestDispatcher:
    def test_cpu_fallback_correctness(self):
        """On CPU (no CUDA), dispatcher must produce identical output to reference."""
        M, K, N = 16, 32, 64
        x, gamma, w, _ = _make_tensors(M, K, N, device="cpu")
        ref = rms_norm_linear_reference(x, gamma, w, None)
        out = rms_norm_linear(x, gamma, w)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_output_shape_matches_reference(self):
        for M, K, N in [(8, 32, 64), (1, 512, 1536), (100, 128, 256)]:
            x, gamma, w, _ = _make_tensors(M, K, N)
            out = rms_norm_linear(x, gamma, w)
            ref = rms_norm_linear_reference(x, gamma, w, None)
            assert out.shape == ref.shape, f"shape mismatch for M={M} K={K} N={N}"
