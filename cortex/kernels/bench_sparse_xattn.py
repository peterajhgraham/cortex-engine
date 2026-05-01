"""Benchmark: block-sparse cross-attention kernel vs PyTorch SDPA.

CUDA REQUIRED — Triton does not support Apple Metal (MPS) or CPU.
Use `scripts/profile_inference.py` to profile the PyTorch reference on MPS.

Metrics reported
----------------
  ref_ms     : median wall-clock time for sparse_cross_attention_reference (PyTorch)
  sdpa_ms    : median wall-clock time for F.scaled_dot_product_attention (dense baseline)
  triton_ms  : median wall-clock time for sparse_cross_attention (Triton kernel)
  speedup    : ref_ms / triton_ms  (sparse Triton vs sparse PyTorch reference)
  density    : fraction of tiles that are active in the mask

Bandwidth calculation
---------------------
  Theoretical minimum bytes for fused kernel at mask density d:
      Reads:  Q load (BxHxLxDh) + K load (d x BxHxExDh) + V load (d x BxHxExDh)
      Writes: output (BxHxLxDh)
  We report achieved bandwidth using the DENSE (d=1) byte count so numbers are
  comparable across density levels — it shows how efficiently bandwidth is used
  relative to the dense case.

Input shapes
------------
  The primary shapes are from Cortex-S cross-attention:
    B=1,  H=8, L=256, E=512,  Dh=64  — single sample inference
    B=8,  H=8, L=256, E=512,  Dh=64  — batch=8 inference
    B=32, H=8, L=256, E=512,  Dh=64  — training batch
    B=1,  H=4, L=128, E=256,  Dh=32  — Cortex-XS
    B=32, H=8, L=256, E=2048, Dh=64  — continuous batching (4x events)

  Three mask densities are swept for each shape: dense (1.0), 50%, 25%.

Usage
-----
  Requires CUDA:
      python -m cortex.kernels.bench_sparse_xattn [--output path/to/out.json]

  Or via the Makefile:
      make bench-kernels

Output
------
  benchmarks/kernels/sparse_xattn.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

_REQUIRES_CUDA = """\
bench_sparse_xattn: CUDA required.
This benchmark measures the Triton block-sparse cross-attention kernel which
only runs on CUDA devices.  Triton does not support Apple Metal (MPS) or CPU.

To profile the PyTorch reference path on MPS:
    PYTHONPATH=. .venv/bin/python scripts/profile_inference.py --device auto
"""

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

import torch.nn.functional as F  # noqa: E402

from cortex.kernels.sparse_xattn import (  # noqa: E402
    sparse_cross_attention,
    sparse_cross_attention_reference,
)

# ── Timing helpers ────────────────────────────────────────────────────────────


def _bench(fn, args: tuple, *, warmup: int = 25, iters: int = 100) -> float:
    """Return median milliseconds per call (CUDA-synchronised)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def _bandwidth_gbs(
    B: int,
    H: int,
    L: int,
    E: int,
    Dh: int,
    dtype: torch.dtype,
    time_ms: float,
    density: float = 1.0,
) -> float:
    """Achieved HBM bandwidth in GB/s.

    Uses the DENSE theoretical minimum so numbers are comparable across
    density levels (density affects compute, not the bandwidth denominator here).
    """
    dtype_bytes = 2 if dtype == torch.bfloat16 else 4
    elem = B * H * Dh * dtype_bytes
    # Q read + K read (density) + V read (density) + output write
    bytes_moved = elem * (L + density * E + density * E + L)
    return bytes_moved / (time_ms * 1e-3) / 1e9


def _make_block_mask(n_q: int, n_k: int, density: float) -> torch.Tensor:
    """Build a block mask with the given fraction of True tiles.

    Uses a deterministic pattern (first density x n_k columns per row) rather
    than random, so results are reproducible.
    """
    if density >= 1.0:
        return torch.ones(n_q, n_k, dtype=torch.bool)
    # Evenly space True tiles across each row
    n_true = max(1, round(density * n_k))
    mask = torch.zeros(n_q, n_k, dtype=torch.bool)
    indices = torch.linspace(0, n_k - 1, n_true).long()
    mask[:, indices] = True
    return mask


# ── Benchmark runner ──────────────────────────────────────────────────────────


SHAPES: list[tuple[int, int, int, int, int, str]] = [
    (1, 8, 256, 512, 64, "Cortex-S single sample"),
    (8, 8, 256, 512, 64, "Cortex-S batch=8"),
    (32, 8, 256, 512, 64, "Cortex-S training batch"),
    (1, 4, 128, 256, 32, "Cortex-XS"),
    (32, 8, 256, 2048, 64, "continuous batching, 4x events"),
]

DENSITIES: list[float] = [1.0, 0.5, 0.25]

BLOCK_L = 32
BLOCK_E = 32


def run(
    output: Path = Path("benchmarks/kernels/sparse_xattn.json"),
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 25,
    iters: int = 100,
) -> list[dict]:
    device = "cuda"
    results: list[dict] = []

    print(f"\n{'='*88}")
    print(f"  Block-Sparse Cross-Attention Benchmark  |  device: {torch.cuda.get_device_name()}")
    print(f"  dtype: {dtype}  |  warmup={warmup}  iters={iters}")
    print(f"  BLOCK_L={BLOCK_L}  BLOCK_E={BLOCK_E}")
    print(f"{'='*88}")
    header = (
        f"{'B':>3}  {'H':>2}  {'L':>4}  {'E':>5}  {'Dh':>3}  {'dens':>5}"
        f"  {'ref_ms':>8}  {'sdpa_ms':>8}  {'tri_ms':>8}"
        f"  {'speedup':>8}  {'bw_tri':>8}  label"
    )
    print(header)
    print("-" * len(header))

    for B, H, L, E, Dh, label in SHAPES:
        for density in DENSITIES:
            torch.manual_seed(0)
            q = torch.randn(B, H, L, Dh, device=device, dtype=dtype)
            k = torch.randn(B, H, E, Dh, device=device, dtype=dtype)
            v = torch.randn(B, H, E, Dh, device=device, dtype=dtype)
            scale = Dh**-0.5

            n_q = math.ceil(L / BLOCK_L)
            n_k = math.ceil(E / BLOCK_E)
            mask = _make_block_mask(n_q, n_k, density)

            # Sparse reference (PyTorch + mask expansion)
            t_ref = _bench(
                sparse_cross_attention_reference,
                (q, k, v, mask, BLOCK_L, BLOCK_E, scale),
                warmup=warmup,
                iters=iters,
            )

            # Dense SDPA (no mask, flash-attention backend)
            t_sdpa = _bench(
                F.scaled_dot_product_attention,
                (q, k, v),
                warmup=warmup,
                iters=iters,
            )

            # Triton sparse kernel
            t_triton = _bench(
                sparse_cross_attention,
                (q, k, v, mask, BLOCK_L, BLOCK_E, scale),
                warmup=warmup,
                iters=iters,
            )

            speedup = t_ref / t_triton
            bw_tri = _bandwidth_gbs(B, H, L, E, Dh, dtype, t_triton, density)
            act_dens = mask.float().mean().item()

            row = {
                "B": B,
                "H": H,
                "L": L,
                "E": E,
                "Dh": Dh,
                "label": label,
                "dtype": str(dtype),
                "density": round(act_dens, 3),
                "ref_ms": round(t_ref, 4),
                "sdpa_ms": round(t_sdpa, 4),
                "triton_ms": round(t_triton, 4),
                "speedup": round(speedup, 2),
                "triton_bw_gbs": round(bw_tri, 1),
            }
            results.append(row)
            print(
                f"{B:>3}  {H:>2}  {L:>4}  {E:>5}  {Dh:>3}  {act_dens:>5.2f}"
                f"  {t_ref:>8.3f}  {t_sdpa:>8.3f}  {t_triton:>8.3f}"
                f"  {speedup:>7.2f}x  {bw_tri:>7.1f}  {label}"
            )

    print(f"{'='*88}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": "sparse_cross_attention",
        "device": torch.cuda.get_device_name(),
        "dtype": str(dtype),
        "block_l": BLOCK_L,
        "block_e": BLOCK_E,
        "results": results,
    }
    output.write_text(json.dumps(meta, indent=2))
    print(f"Results written to {output}")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/kernels/sparse_xattn.json", type=Path)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(_REQUIRES_CUDA, file=sys.stderr)
        sys.exit(1)

    if not _TRITON_AVAILABLE:
        print("Triton not installed — cannot benchmark kernel path.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    run(output=args.output, dtype=dtype, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
