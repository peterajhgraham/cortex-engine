"""Benchmark: fused tokenizer kernel vs PyTorch eager.

CUDA REQUIRED — this benchmark cannot run on MPS or CPU.
Triton does not support Metal; the Triton path is CUDA-only.
MPS timings for the PyTorch reference path are available via
`scripts/profile_inference.py` which runs on any device.

Metrics reported
----------------
  ref_ms    : median wall-clock time for fused_tokenizer_reference (eager PyTorch)
  triton_ms : median wall-clock time for fused_tokenizer (Triton kernel)
  speedup   : ref_ms / triton_ms
  ref_bw    : achieved HBM bandwidth for reference (GB/s)
  triton_bw : achieved HBM bandwidth for Triton kernel (GB/s)

Bandwidth calculation
---------------------
  Theoretical minimum bytes transferred (fused kernel):
      Reads:  3 x E x D (one gather per embedding table)
      Writes: 1 x E x D (output)
      Total:  4 x E x D x sizeof(dtype)

  PyTorch eager (unfused):
      Reads:  3 x E x D (gathers) + 4 x E x D (2 additions, each reads 2 tensors)
      Writes: 2 x E x D (intermediates) + 1 x E x D (final)
      Total: ~10 x E x D x sizeof(dtype)  [may be partially cached at small E]

  We use the theoretical minimum for BOTH when computing bandwidth, so the
  metric tells us how efficiently each implementation uses HBM bandwidth:
      bandwidth = 4 x E x D x sizeof(dtype) / time

Input shapes
------------
  Shapes span from single-sample inference (small E) to full-batch training:

  E       D     Label
  -----   ---   --------------------------------
  512     512   Single MC_Maze sample (600 ms window, ~685 events, Cortex-S)
  2,048   512   Batch of 4
  8,192   512   Batch of 16
  16,384  512   Batch of 32 (training batch)
  65,536  512   Batch of 128 (continuous batching)
  16,384  256   Cortex-XS (D=256)
  65,536  128   High-throughput small model

Usage
-----
  Requires CUDA:
      python -m cortex.kernels.bench_tokenizer [--output path/to/out.json]

  Or via the Makefile:
      make bench-kernels   (runs this module as part of the full benchmark suite)

Output
------
  benchmarks/kernels/tokenizer.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# ── CUDA guard ────────────────────────────────────────────────────────────────

_REQUIRES_CUDA = """\
bench_tokenizer: CUDA required.
This benchmark measures the Triton fused-tokenizer kernel which only runs on
CUDA devices.  Triton does not support Apple Metal (MPS) or plain CPU.

To profile the PyTorch reference path on MPS:
    PYTHONPATH=. .venv/bin/python scripts/profile_inference.py --device auto
"""

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

from cortex.kernels.tokenizer import fused_tokenizer, fused_tokenizer_reference  # noqa: E402

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
    return times[len(times) // 2]  # median


def _bandwidth_gbs(E: int, D: int, dtype: torch.dtype, time_ms: float) -> float:
    """Achieved HBM bandwidth in GB/s using the theoretical-minimum byte count."""
    dtype_bytes = 2 if dtype == torch.bfloat16 else 4  # bf16 or float32
    bytes_moved = 4 * E * D * dtype_bytes  # 3 reads + 1 write
    return bytes_moved / (time_ms * 1e-3) / 1e9


# ── Benchmark runner ──────────────────────────────────────────────────────────


SHAPES: list[tuple[int, int, str]] = [
    (512, 512, "single sample (MC_Maze 600ms)"),
    (2_048, 512, "batch=4, Cortex-S"),
    (8_192, 512, "batch=16, Cortex-S"),
    (16_384, 512, "batch=32, Cortex-S  [training batch]"),
    (65_536, 512, "batch=128, Cortex-S [continuous batching]"),
    (16_384, 256, "batch=32, Cortex-XS (D=256)"),
    (65_536, 128, "high-throughput small model"),
]

VOCAB = {"N": 512, "T": 1024, "V": 16}


def run(
    output: Path = Path("benchmarks/kernels/tokenizer.json"),
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 25,
    iters: int = 100,
) -> list[dict]:
    device = "cuda"
    results: list[dict] = []

    print(f"\n{'='*72}")
    print(f"  Fused Tokenizer Benchmark  |  device: {torch.cuda.get_device_name()}")
    print(f"  dtype: {dtype}  |  warmup={warmup}  iters={iters}")
    print(f"{'='*72}")
    header = f"{'E':>7}  {'D':>4}  {'ref_ms':>8}  {'tri_ms':>8}  {'speedup':>8}  {'ref_bw':>8}  {'tri_bw':>8}  label"
    print(header)
    print("-" * len(header))

    for E, D, label in SHAPES:
        torch.manual_seed(0)
        n_emb = torch.randn(VOCAB["N"], D, device=device, dtype=dtype)
        t_emb = torch.randn(VOCAB["T"], D, device=device, dtype=dtype)
        v_emb = torch.randn(VOCAB["V"], D, device=device, dtype=dtype)
        nid = torch.randint(0, VOCAB["N"], (E,), device=device)
        tb = torch.randint(0, VOCAB["T"], (E,), device=device)
        val = torch.randint(0, VOCAB["V"], (E,), device=device)

        args = (n_emb, t_emb, v_emb, nid, tb, val)
        t_ref = _bench(fused_tokenizer_reference, args, warmup=warmup, iters=iters)
        t_triton = _bench(fused_tokenizer, args, warmup=warmup, iters=iters)

        speedup = t_ref / t_triton
        bw_ref = _bandwidth_gbs(E, D, dtype, t_ref)
        bw_tri = _bandwidth_gbs(E, D, dtype, t_triton)

        row = {
            "E": E,
            "D": D,
            "label": label,
            "dtype": str(dtype),
            "ref_ms": round(t_ref, 4),
            "triton_ms": round(t_triton, 4),
            "speedup": round(speedup, 2),
            "ref_bw_gbs": round(bw_ref, 1),
            "triton_bw_gbs": round(bw_tri, 1),
        }
        results.append(row)
        print(
            f"{E:>7}  {D:>4}  {t_ref:>8.3f}  {t_triton:>8.3f}"
            f"  {speedup:>7.2f}x  {bw_ref:>7.1f}  {bw_tri:>7.1f}  {label}"
        )

    print(f"{'='*72}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": "fused_tokenizer",
        "device": torch.cuda.get_device_name(),
        "dtype": str(dtype),
        "results": results,
    }
    output.write_text(json.dumps(meta, indent=2))
    print(f"Results written to {output}")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/kernels/tokenizer.json", type=Path)
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
