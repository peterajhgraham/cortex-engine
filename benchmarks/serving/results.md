# Inference Engine Load Test — Phase 3

Measured: 2026-05-01
Hardware: mps
Mode: in_process
Requests: 200 total, 16 concurrent
Events per request: 256
Max batch size: 32
Batch timeout: 5.0 ms

---

## Summary

| Metric | Value |
|---|---|
| Successes | 200 / 200 |
| Failures | 0 |
| Throughput | **157.3 req/s** |
| Total test time | 1.27 s |

---

## Latency Distribution (ms)

*Measured from scheduler.submit() to result return (includes queue wait + inference).*

| Percentile | ms |
|---|---|
| p50 | 70.18 |
| p75 | 72.05 |
| p90 | 137.97 |
| p95 | 356.15 |
| **p99** | **358.46** |
| max | 359.09 |
| mean | 95.74 |
| min | 67.38 |

---

## Notes

- **SLO target:** p99 < 30 ms on CUDA A100.  Numbers above are on mps.
- **Mode:** `in_process` — latency includes scheduler queue wait + inference only,
  NOT HTTP serialization or TCP.
- **Batch dynamics:** up to 32 requests per batch, formed
  within a 5.0 ms window.  At concurrency=16
  on mps, batches typically contain
  16 requests.
- The SLO target requires CUDA hardware; MPS/CPU numbers above are for
  infrastructure correctness validation, not production benchmarking.

## To Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/load_test.py \
    --concurrency 16 \
    --requests 200 \
    --events 256
```
