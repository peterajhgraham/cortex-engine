"""Benchmark the fused tokenizer kernel against PyTorch eager.

Produces a JSON report in benchmarks/kernels/tokenizer.json with timings across
input sizes. Used to populate the hero metrics table in the README.

Usage:
    python -m cortex.kernels.bench_tokenizer --output benchmarks/kernels/tokenizer.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer

from cortex.kernels.tokenizer import fused_tokenizer, fused_tokenizer_reference

app = typer.Typer()


def _bench_one(
    fn, args: tuple, *, warmup: int = 25, iters: int = 100
) -> float:
    """Return median ms per call after warmup."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


@app.command()
def main(output: Path = Path("benchmarks/kernels/tokenizer.json")) -> None:
    if not torch.cuda.is_available():
        typer.echo("CUDA required for benchmark")
        raise typer.Exit(1)

    device = "cuda"
    dtype = torch.bfloat16
    output.parent.mkdir(parents=True, exist_ok=True)

    sizes = [
        (1024, 128),
        (4096, 256),
        (16384, 256),
        (65536, 384),
    ]
    results = []

    for E, D in sizes:
        N, T, V = 512, 1024, 16
        torch.manual_seed(0)
        neuron_emb = torch.randn(N, D, device=device, dtype=dtype)
        time_emb = torch.randn(T, D, device=device, dtype=dtype)
        value_emb = torch.randn(V, D, device=device, dtype=dtype)
        neuron_ids = torch.randint(0, N, (E,), device=device)
        time_bins = torch.randint(0, T, (E,), device=device)
        values = torch.randint(0, V, (E,), device=device)

        args = (neuron_emb, time_emb, value_emb, neuron_ids, time_bins, values)
        t_ref = _bench_one(fused_tokenizer_reference, args)
        t_triton = _bench_one(fused_tokenizer, args)

        results.append(
            {
                "E": E,
                "D": D,
                "ref_ms": round(t_ref, 4),
                "triton_ms": round(t_triton, 4),
                "speedup": round(t_ref / t_triton, 2),
            }
        )
        typer.echo(f"E={E:6d} D={D:4d}  ref={t_ref:.3f}ms  triton={t_triton:.3f}ms  ({t_ref/t_triton:.2f}x)")

    with output.open("w") as f:
        json.dump({"benchmark": "fused_tokenizer", "device": torch.cuda.get_device_name(), "results": results}, f, indent=2)
    typer.echo(f"\nWrote {output}")


if __name__ == "__main__":
    app()
