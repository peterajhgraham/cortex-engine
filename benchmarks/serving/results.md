# Inference Engine Load Test — Phase 3

Measured: 2026-05-19
Hardware: cuda
Mode: in_process_direct
Requests: 500 total, 8 concurrent
Events per request: 256
Max batch size: 32
Batch timeout: 5.0 ms

---

## Summary

| Metric | Value |
|---|---|
| Successes | 500 / 500 |
| Failures | 0 |
| Throughput | **254.7 req/s** |
| Total test time | 1.96 s |

---

## Latency Distribution (ms)

*Measured from scheduler.submit() to result return (includes queue wait + inference).*

| Percentile | ms |
|---|---|
| p50 | 27.28 |
| p75 | 27.41 |
| p90 | 27.55 |
| p95 | 27.67 |
| **p99** | **261.52** |
| max | 276.17 |
| mean | 31.1 |
| min | 26.85 |

---

## Notes

- **SLO target:** p99 < 30 ms on CUDA A10.  Numbers above are on cuda.
- **Mode:** `in_process_direct` — latency includes scheduler queue wait + inference only,
  NOT HTTP serialization or TCP.
- **Batch dynamics:** up to 32 requests per batch, formed
  within a 5.0 ms window.  At concurrency=8
  on cuda, batches typically contain
  8 requests.
- The SLO target requires CUDA hardware; MPS/CPU numbers above are for
  infrastructure correctness validation, not production benchmarking.

## To Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/load_test.py \
    --concurrency 8 \
    --requests 500 \
    --events 256
```
