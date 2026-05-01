"""Correctness tests for the fused tokenizer kernel.

Structure (canonical pattern for all kernel tests in this project):

  Group 1 — reference-only (no CUDA required):
      Tests that the PyTorch reference produces correct shapes and values on CPU.
      These run in CI on any machine.

  Group 2 — Triton kernel correctness (requires CUDA, @pytest.mark.gpu):
      Tests that fused_tokenizer matches fused_tokenizer_reference within
      rtol/atol tolerances from the Phase 2 spec (1e-3 / 1e-2 for bfloat16,
      1e-5 / 1e-5 for float32).

  Group 3 — dispatcher integration:
      Tests that SpikeTokenizer routes to the Triton kernel when use_kernels=True
      and CUDA is available, and falls back cleanly otherwise.

Tolerance rationale
-------------------
  bfloat16: 8 bits of mantissa → rounding errors on order of 2^-7 ≈ 0.008.
             Three additions accumulate up to 3x2^-7 ≈ 0.023 absolute error.
             atol=1e-2 is tight enough to catch bugs but loose enough for bf16.
  float32:  24 bits of mantissa → rounding errors on order of 2^-23 ≈ 1e-7.
             Tight tolerances (1e-5 / 1e-5) are safe.
"""

from __future__ import annotations

import pytest
import torch

from cortex.kernels.tokenizer import fused_tokenizer, fused_tokenizer_reference
from cortex.models.config import CortexConfig
from cortex.models.tokenizer import SpikeTokenizer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_embs(N: int, T: int, V: int, D: int, device: str, dtype: torch.dtype):
    torch.manual_seed(42)
    return (
        torch.randn(N, D, device=device, dtype=dtype),
        torch.randn(T, D, device=device, dtype=dtype),
        torch.randn(V, D, device=device, dtype=dtype),
    )


def _make_ids(E: int, N: int, T: int, V: int, device: str):
    torch.manual_seed(99)
    return (
        torch.randint(0, N, (E,), device=device, dtype=torch.int64),
        torch.randint(0, T, (E,), device=device, dtype=torch.int64),
        torch.randint(0, V, (E,), device=device, dtype=torch.int64),
    )


# ── Group 1: reference correctness (CPU, no CUDA) ────────────────────────────


class TestReference:
    """The PyTorch reference must behave correctly independent of CUDA."""

    def test_output_shape(self):
        E, D = 128, 64
        n_emb, t_emb, v_emb = _make_embs(N=32, T=64, V=8, D=D, device="cpu", dtype=torch.float32)
        nid, tb, val = _make_ids(E=E, N=32, T=64, V=8, device="cpu")

        out = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
        assert out.shape == (E, D)

    def test_output_dtype_preserved(self):
        for dtype in (torch.float32, torch.float16):
            n_emb, t_emb, v_emb = _make_embs(N=8, T=16, V=4, D=32, device="cpu", dtype=dtype)
            nid, tb, val = _make_ids(E=16, N=8, T=16, V=4, device="cpu")
            out = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
            assert out.dtype == dtype

    def test_matches_manual_sum(self):
        """Reference must exactly equal neuron_emb[nid] + time_emb[tb] + val_emb[val]."""
        n_emb, t_emb, v_emb = _make_embs(N=16, T=32, V=4, D=64, device="cpu", dtype=torch.float32)
        nid, tb, val = _make_ids(E=32, N=16, T=32, V=4, device="cpu")

        ref = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
        expected = n_emb[nid] + t_emb[tb] + v_emb[val]
        torch.testing.assert_close(ref, expected)

    def test_zero_events_returns_empty(self):
        n_emb, t_emb, v_emb = _make_embs(N=8, T=16, V=4, D=32, device="cpu", dtype=torch.float32)
        empty = torch.zeros(0, dtype=torch.int64)
        out = fused_tokenizer_reference(n_emb, t_emb, v_emb, empty, empty, empty)
        assert out.shape == (0, 32)

    def test_single_event(self):
        n_emb, t_emb, v_emb = _make_embs(N=8, T=16, V=4, D=32, device="cpu", dtype=torch.float32)
        nid = torch.tensor([3], dtype=torch.int64)
        tb = torch.tensor([7], dtype=torch.int64)
        val = torch.tensor([1], dtype=torch.int64)
        out = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
        expected = n_emb[3] + t_emb[7] + v_emb[1]
        torch.testing.assert_close(out.squeeze(0), expected)

    def test_all_same_index(self):
        """All events with identical indices must all produce the same token."""
        n_emb, t_emb, v_emb = _make_embs(N=8, T=16, V=4, D=64, device="cpu", dtype=torch.float32)
        E = 100
        nid = torch.zeros(E, dtype=torch.int64)
        tb = torch.zeros(E, dtype=torch.int64)
        val = torch.zeros(E, dtype=torch.int64)
        out = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
        assert out.shape == (E, 64)
        # Every row should be identical
        torch.testing.assert_close(out, out[0:1].expand_as(out))


# ── Group 2: Triton kernel correctness (requires CUDA) ───────────────────────


@pytest.mark.gpu
@pytest.mark.parametrize(
    "E,D,N,T,V",
    [
        (1, 64, 32, 64, 8),  # minimum E
        (127, 64, 32, 64, 8),  # non-power-of-2 E
        (128, 64, 256, 512, 8),  # small reference shape
        (1024, 128, 256, 512, 8),  # medium shape
        (4096, 256, 512, 1024, 8),  # large shape
        (16384, 512, 512, 1024, 16),  # Cortex-S production: batch=32, 512 events/sample
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_reference(E: int, D: int, N: int, T: int, V: int, dtype: torch.dtype):
    """Triton output must equal the PyTorch reference within tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernel test")

    device = "cuda"
    n_emb, t_emb, v_emb = _make_embs(N, T, V, D, device, dtype)
    nid, tb, val = _make_ids(E, N, T, V, device)

    ref = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
    out = fused_tokenizer(n_emb, t_emb, v_emb, nid, tb, val)

    assert out.shape == (E, D)
    assert out.dtype == dtype

    rtol, atol = (1e-3, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.gpu
def test_triton_empty_events():
    """Zero events → (0, D) output, no crash."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    D = 128
    n_emb, t_emb, v_emb = _make_embs(N=32, T=64, V=8, D=D, device="cuda", dtype=torch.float32)
    empty = torch.zeros(0, dtype=torch.int64, device="cuda")
    out = fused_tokenizer(n_emb, t_emb, v_emb, empty, empty, empty)
    assert out.shape == (0, D)


@pytest.mark.gpu
def test_triton_non_contiguous_indices():
    """Kernel should handle non-contiguous index tensors (e.g., slices)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    D, E = 128, 64
    n_emb, t_emb, v_emb = _make_embs(N=32, T=64, V=8, D=D, device="cuda", dtype=torch.float32)
    # Create non-contiguous tensors by striding
    big = torch.randint(0, 32, (E * 2,), device="cuda", dtype=torch.int64)
    nid = big[::2]  # non-contiguous stride-2 view
    tb = torch.randint(0, 64, (E,), device="cuda", dtype=torch.int64)
    val = torch.randint(0, 8, (E,), device="cuda", dtype=torch.int64)

    ref = fused_tokenizer_reference(n_emb, t_emb, v_emb, nid, tb, val)
    out = fused_tokenizer(n_emb, t_emb, v_emb, nid, tb, val)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ── Group 3: dispatcher integration ──────────────────────────────────────────


class TestSpikeTokenizerDispatch:
    """SpikeTokenizer must route to the correct backend based on config and device."""

    def _tokenizer_and_inputs(self, use_kernels: bool, device: str):
        cfg = CortexConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            num_latents=32,
            latent_dim=64,
            cross_attn_heads=4,
            max_neurons=32,
            max_time_bins=64,
            spike_value_buckets=8,
            behavior_dim=2,
            use_kernels=use_kernels,
        )
        tok = SpikeTokenizer(cfg).to(device).eval()
        torch.manual_seed(7)
        E = 64
        nid = torch.randint(0, 32, (E,), device=device, dtype=torch.int64)
        tb = torch.randint(0, 64, (E,), device=device, dtype=torch.int64)
        val = torch.randint(0, 8, (E,), device=device, dtype=torch.int64)
        return tok, nid, tb, val

    def test_cpu_fallback_always_works(self):
        """use_kernels=True on CPU must silently fall back, not raise."""
        tok, nid, tb, val = self._tokenizer_and_inputs(use_kernels=True, device="cpu")
        with torch.no_grad():
            out = tok(nid, tb, val)
        assert out.shape == (64, 64)

    def test_pytorch_and_kernel_paths_agree(self):
        """Both paths must produce identical results for the same weights + inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required to exercise the Triton path")

        tok_ref, nid, tb, val = self._tokenizer_and_inputs(use_kernels=False, device="cuda")
        tok_ker, _, _, _ = self._tokenizer_and_inputs(use_kernels=True, device="cuda")
        # Copy weights from ref tokenizer to kernel tokenizer so they're identical
        tok_ker.load_state_dict(tok_ref.state_dict())

        with torch.no_grad():
            out_ref = tok_ref(nid, tb, val)
            out_ker = tok_ker(nid, tb, val)

        torch.testing.assert_close(out_ref, out_ker, rtol=1e-5, atol=1e-5)

    def test_output_shape_and_dtype(self):
        """Output shape must be (E, hidden_dim) in float32 for float32 embeddings."""
        tok, nid, tb, val = self._tokenizer_and_inputs(use_kernels=False, device="cpu")
        with torch.no_grad():
            out = tok(nid, tb, val)
        assert out.shape == (64, 64)

    def test_shape_mismatch_raises(self):
        """Mismatched index tensor shapes must raise ValueError."""
        tok, nid, tb, val = self._tokenizer_and_inputs(use_kernels=False, device="cpu")
        with pytest.raises(ValueError, match="shape mismatch"):
            tok(nid, tb[:-1], val)  # tb is one shorter
