"""Benchmark: fused RMSNorm + linear kernel vs PyTorch eager.

CUDA REQUIRED — Triton does not support Apple Metal (MPS) or CPU.
Use `scripts/profile_inference.py` to profile the PyTorch reference on MPS.

Metrics reported
----------------
  ref_ms    : median wall-clock for rms_norm_linear_reference (two separate ops)
  triton_ms : median wall-clock for rms_norm_linear (Triton fused kernel)
  speedup   : ref_ms / triton_ms
  ref_bw    : achieved HBM bandwidth for reference (GB/s)
  triton_bw : achieved HBM bandwidth for Triton kernel (GB/s)

Bandwidth calculation
---------------------
  Reference (two separate ops):
      RMSNorm: reads M×K, writes M×K            = 2 × M×K
      Linear:  reads M×K (x_norm) + K×N (W)     = M×K + K×N
               writes M×N                        = M×N
      Total:   3×M×K + K×N + M×N

  Fused kernel (eliminates x_norm HBM round-trip):
      Pass 1 + Pass 2: reads M×K twice + K×N    = 2×M×K + K×N
      Writes: M×N                                = M×N
      Total:  2×M×K + K×N + M×N

  We report bandwidth using the FUSED theoretical minimum for BOTH so the
  metric shows how efficiently each implementation uses HBM:
      bandwidth = (2×M×K + K×N + M×N) × sizeof(dtype) / time

  The reference will show lower bandwidth because it does MORE work (3×M×K read)
  for the same byte denominator — the fused version genuinely saves HBM traffic.

Input shapes
------------
  Shapes span both the QKV projection and MLP first-layer sizes in Cortex-S:

  M       K     N     Label
  -----   ---   ----  --------------------------------
  256     512   1536  Cortex-S: L=256, QKV projection (K=512 → 3×512)
  256     512   2048  Cortex-S: L=256, MLP first layer (K=512 → 4×512)
  8192    512   1536  Cortex-S: B×L=32×256, QKV
  8192    512   2048  Cortex-S: B×L=32×256, MLP
  8192    256   768   Cortex-XS: D=256, QKV (3×256)
  8192    512   512   Self-attn output projection (K=512 → 512)
  1       512   2048  Single-token decode (streaming)

Usage
-----
  Requires CUDA:
      python -m cortex.kernels.bench_fused_rmsnorm [--output path/to/out.json]

  Or via the Makefile:
      make bench-kernels

Output
------
  benchmarks/kernels/fused_rmsnorm.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_REQUIRES_CUDA = """\
bench_fused_rmsnorm: CUDA required.
This benchmark measures the Triton fused RMSNorm+linear kernel which only runs
on CUDA devices.  Triton does not support Apple Metal (MPS) or CPU.

To profile the PyTorch reference path on MPS:
    PYTHONPATH=. .venv/bin/python scripts/profile_inference.py --device auto
"""

try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from cortex.kernels.fused_rmsnorm import rms_norm_linear, rms_norm_linear_reference


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


def _bandwidth_gbs(M: int, K: int, N: int, dtype: torch.dtype, time_ms: float) -> float:
    """Achieved HBM bandwidth in GB/s using the fused-kernel theoretical minimum.

    Fused minimum: 2×M×K (x read twice) + K×N (weight) + M×N (output)
    """
    dtype_bytes = 2 if dtype == torch.bfloat16 else 4
    bytes_moved = (2 * M * K + K * N + M * N) * dtype_bytes
    return bytes_moved / (time_ms * 1e-3) / 1e9


# ── Benchmark runner ──────────────────────────────────────────────────────────


SHAPES: list[tuple[int, int, int, str]] = [
    (256,   512,  1536, "Cortex-S L=256, QKV projection"),
    (256,   512,  2048, "Cortex-S L=256, MLP first layer"),
    (8192,  512,  1536, "Cortex-S B×L=8192, QKV projection"),
    (8192,  512,  2048, "Cortex-S B×L=8192, MLP first layer"),
    (8192,  256,  768,  "Cortex-XS B×L=8192, QKV (D=256)"),
    (8192,  512,  512,  "Cortex-S output projection"),
    (1,     512,  2048, "single-token decode"),
]


def run(
    output: Path = Path("benchmarks/kernels/fused_rmsnorm.json"),
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 25,
    iters: int = 100,
) -> list[dict]:
    device = "cuda"
    results: list[dict] = []

    print(f"\n{'='*80}")
    print(f"  Fused RMSNorm+Linear Benchmark  |  device: {torch.cuda.get_device_name()}")
    print(f"  dtype: {dtype}  |  warmup={warmup}  iters={iters}")
    print(f"{'='*80}")
    header = (
        f"{'M':>6}  {'K':>4}  {'N':>5}"
        f"  {'ref_ms':>8}  {'tri_ms':>8}  {'speedup':>8}"
        f"  {'ref_bw':>8}  {'tri_bw':>8}  label"
    )
    print(header)
    print("-" * len(header))

    for M, K, N, label in SHAPES:
        torch.manual_seed(0)
        x     = torch.randn(M, K, device=device, dtype=dtype)
        gamma = torch.ones(K, device=device, dtype=dtype)
        w     = torch.randn(K, N, device=device, dtype=dtype)

        t_ref    = _bench(rms_norm_linear_reference, (x, gamma, w, None), warmup=warmup, iters=iters)
        t_triton = _bench(rms_norm_linear,           (x, gamma, w),       warmup=warmup, iters=iters)

        speedup = t_ref / t_triton
        bw_ref  = _bandwidth_gbs(M, K, N, dtype, t_ref)
        bw_tri  = _bandwidth_gbs(M, K, N, dtype, t_triton)

        row = {
            "M": M, "K": K, "N": N, "label": label,
            "dtype":     str(dtype),
            "ref_ms":    round(t_ref,    4),
            "triton_ms": round(t_triton, 4),
            "speedup":   round(speedup,  2),
            "ref_bw_gbs":    round(bw_ref, 1),
            "triton_bw_gbs": round(bw_tri, 1),
        }
        results.append(row)
        print(
            f"{M:>6}  {K:>4}  {N:>5}"
            f"  {t_ref:>8.3f}  {t_triton:>8.3f}  {speedup:>7.2f}x"
            f"  {bw_ref:>7.1f}  {bw_tri:>7.1f}  {label}"
        )

    print(f"{'='*80}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": "fused_rmsnorm_linear",
        "device":    torch.cuda.get_device_name(),
        "dtype":     str(dtype),
        "results":   results,
    }
    output.write_text(json.dumps(meta, indent=2))
    print(f"Results written to {output}")
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/kernels/fused_rmsnorm.json", type=Path)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters",  type=int, default=100)
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
