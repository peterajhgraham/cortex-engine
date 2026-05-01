"""Load test: measure inference latency under concurrent request load.

Two modes
---------
  in-process  (default, no --url)
      Instantiates InferenceWorker + Scheduler directly, submits requests via
      scheduler.submit().  Measures latency through the full continuous-batching
      pipeline without HTTP/TCP overhead.  Works on any device (MPS, CPU, CUDA).

  http  (--url http://host:port)
      Fires requests against a live server via httpx.  Use when measuring
      end-to-end latency including uvicorn + serialization, or when server
      and load generator are on different machines.

Metrics
-------
Latency is measured from just before scheduler.submit() / client.post() to
just after the result is returned.  Reported percentiles: p50, p75, p90,
p95, p99, max.

Throughput = successes / total_wall_clock_time.

Usage
-----
  # In-process (default):
  PYTHONPATH=. .venv/bin/python scripts/load_test.py \\
      --concurrency 16 --requests 200 --events 256

  # Against a live server:
  PYTHONPATH=. .venv/bin/python scripts/load_test.py \\
      --url http://localhost:8080 --concurrency 32 --requests 500

  # Write results:
  PYTHONPATH=. .venv/bin/python scripts/load_test.py \\
      --output benchmarks/serving/results.md
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_payload(request_id: str, n_events: int, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    return {
        "request_id": request_id,
        "session_id": "load_test",
        "events": [
            {
                "neuron_id": rng.randint(0, 511),
                "time_bin":  rng.randint(0, 1023),
                "value":     rng.randint(0, 7),
            }
            for _ in range(n_events)
        ],
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (len(sv) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sv[lo]
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def compute_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    return {
        "count": len(latencies),
        "mean":  round(sum(latencies) / len(latencies), 2),
        "min":   round(min(latencies), 2),
        "p50":   round(percentile(latencies, 50), 2),
        "p75":   round(percentile(latencies, 75), 2),
        "p90":   round(percentile(latencies, 90), 2),
        "p95":   round(percentile(latencies, 95), 2),
        "p99":   round(percentile(latencies, 99), 2),
        "max":   round(max(latencies), 2),
    }


# ── In-process runner (scheduler + worker, no HTTP) ───────────────────────────


async def run_inprocess(
    n_requests: int,
    concurrency: int,
    n_events: int,
    max_batch: int = 32,
    batch_timeout_ms: float = 5.0,
) -> dict:
    """Benchmark the scheduler+worker pipeline directly."""
    import torch
    from cortex.models.config import CORTEX_S
    from cortex.serve.scheduler import Scheduler
    from cortex.serve.worker import InferenceWorker, _detect_device, load_model

    device = _detect_device()
    print(f"Device: {device}")

    print("Loading model and warming up (3 batches)…")
    model = load_model(CORTEX_S, device=device)
    worker = InferenceWorker(
        model=model,
        config=CORTEX_S,
        device=device,
        max_batch_size=max_batch,
    )
    worker.warmup(n_iters=3)

    scheduler = Scheduler(
        worker=worker,
        max_batch_size=max_batch,
        batch_timeout_ms=batch_timeout_ms,
        default_deadline_ms=60.0,
    )
    sched_task = asyncio.create_task(scheduler.run())

    print(f"Running {n_requests} requests, concurrency={concurrency}, events={n_events}…")
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: list[str] = []

    async def one_request(i: int) -> None:
        payload = _make_payload(f"req_{i:05d}", n_events, seed=i)
        async with semaphore:
            t0 = time.perf_counter()
            try:
                result = await scheduler.submit(payload, request_id=f"req_{i:05d}")
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:
                errors.append(str(exc))

    t_wall_0 = time.perf_counter()
    await asyncio.gather(*[one_request(i) for i in range(n_requests)])
    total_s = time.perf_counter() - t_wall_0

    await scheduler.stop()
    sched_task.cancel()
    try:
        await sched_task
    except asyncio.CancelledError:
        pass
    worker.shutdown()

    return {
        "mode": "in_process",
        "device": str(device),
        "n_requests": n_requests,
        "concurrency": concurrency,
        "n_events": n_events,
        "max_batch": max_batch,
        "batch_timeout_ms": batch_timeout_ms,
        "successes": len(latencies),
        "failures": len(errors),
        "throughput_rps": round(len(latencies) / total_s, 1),
        "total_s": round(total_s, 2),
        "latency_ms": compute_stats(latencies),
    }


# ── HTTP runner ───────────────────────────────────────────────────────────────


async def run_http(
    url: str,
    n_requests: int,
    concurrency: int,
    n_events: int,
) -> dict:
    import httpx

    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: list[str] = []

    async with httpx.AsyncClient(base_url=url, timeout=30.0) as client:
        async def one_request(i: int) -> None:
            payload = _make_payload(f"req_{i:05d}", n_events, seed=i)
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/decode", json=payload)
                    elapsed = (time.perf_counter() - t0) * 1000
                    if r.status_code == 200:
                        latencies.append(elapsed)
                    else:
                        errors.append(f"HTTP {r.status_code}")
                except Exception as exc:
                    errors.append(str(exc))

        t0 = time.perf_counter()
        await asyncio.gather(*[one_request(i) for i in range(n_requests)])
        total_s = time.perf_counter() - t0

    return {
        "mode": f"http:{url}",
        "device": "remote",
        "n_requests": n_requests,
        "concurrency": concurrency,
        "n_events": n_events,
        "successes": len(latencies),
        "failures": len(errors),
        "throughput_rps": round(len(latencies) / total_s, 1),
        "total_s": round(total_s, 2),
        "latency_ms": compute_stats(latencies),
    }


# ── Report writer ─────────────────────────────────────────────────────────────


def write_report(r: dict, output: Path) -> None:
    lat = r["latency_ms"]
    report = f"""# Inference Engine Load Test — Phase 3

Measured: {time.strftime('%Y-%m-%d')}
Hardware: {r.get('device', 'unknown')}
Mode: {r['mode']}
Requests: {r['n_requests']} total, {r['concurrency']} concurrent
Events per request: {r['n_events']}
Max batch size: {r.get('max_batch', '—')}
Batch timeout: {r.get('batch_timeout_ms', '—')} ms

---

## Summary

| Metric | Value |
|---|---|
| Successes | {r['successes']} / {r['n_requests']} |
| Failures | {r['failures']} |
| Throughput | **{r['throughput_rps']} req/s** |
| Total test time | {r['total_s']} s |

---

## Latency Distribution (ms)

*Measured from scheduler.submit() to result return (includes queue wait + inference).*

| Percentile | ms |
|---|---|
| p50 | {lat.get('p50', '—')} |
| p75 | {lat.get('p75', '—')} |
| p90 | {lat.get('p90', '—')} |
| p95 | {lat.get('p95', '—')} |
| **p99** | **{lat.get('p99', '—')}** |
| max | {lat.get('max', '—')} |
| mean | {lat.get('mean', '—')} |
| min | {lat.get('min', '—')} |

---

## Notes

- **SLO target:** p99 < 30 ms on CUDA A100.  Numbers above are on {r.get('device', 'unknown')}.
- **Mode:** `{r['mode']}` — latency includes scheduler queue wait + inference only,
  NOT HTTP serialization or TCP.
- **Batch dynamics:** up to {r.get('max_batch', 32)} requests per batch, formed
  within a {r.get('batch_timeout_ms', 5)} ms window.  At concurrency={r['concurrency']}
  on {r.get('device', 'unknown')}, batches typically contain
  {min(r['concurrency'], r.get('max_batch', 32))} requests.
- The SLO target requires CUDA hardware; MPS/CPU numbers above are for
  infrastructure correctness validation, not production benchmarking.

## To Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/load_test.py \\
    --concurrency {r['concurrency']} \\
    --requests {r['n_requests']} \\
    --events {r['n_events']}
```
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"Results written to {output}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",         default=None, help="HTTP server URL; omit for in-process")
    parser.add_argument("--requests",    type=int,   default=200)
    parser.add_argument("--concurrency", type=int,   default=16)
    parser.add_argument("--events",      type=int,   default=256)
    parser.add_argument("--max-batch",   type=int,   default=32)
    parser.add_argument("--batch-timeout-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/serving/results.md"))
    args = parser.parse_args()

    if args.url:
        results = asyncio.run(run_http(
            url=args.url,
            n_requests=args.requests,
            concurrency=args.concurrency,
            n_events=args.events,
        ))
    else:
        results = asyncio.run(run_inprocess(
            n_requests=args.requests,
            concurrency=args.concurrency,
            n_events=args.events,
            max_batch=args.max_batch,
            batch_timeout_ms=args.batch_timeout_ms,
        ))

    lat = results["latency_ms"]
    print("\n=== Latency (ms) ===")
    print(f"  Throughput : {results['throughput_rps']} req/s")
    print(f"  p50  : {lat.get('p50')} ms")
    print(f"  p95  : {lat.get('p95')} ms")
    print(f"  p99  : {lat.get('p99')} ms")
    print(f"  max  : {lat.get('max')} ms")
    print(f"  Failures : {results['failures']} / {results['n_requests']}")

    write_report(results, args.output)


if __name__ == "__main__":
    main()
