# Inference Engine Load Test — Phase 3

Measured: 2026-05-18
Hardware: cpu
Mode: in_process_direct
Requests: 200 total, 16 concurrent
Events per request: 64
Max batch size: 32
Batch timeout: 5.0 ms

---

## Summary

| Metric | Value |
|---|---|
| Successes | 200 / 200 |
| Failures | 0 |
| Throughput | **15.0 req/s** |
| Total test time | 13.38 s |

---

## Latency Distribution (ms)

*Measured from scheduler.submit() to result return (includes queue wait + inference).*

| Percentile | ms |
|---|---|
| p50 | 1064.87 |
| p75 | 1087.35 |
| p90 | 1135.65 |
| p95 | 1144.39 |
| **p99** | **1165.98** |
| max | 1180.85 |
| mean | 1030.13 |
| min | 90.11 |

---

## Notes

- **SLO target:** p99 < 30 ms on CUDA A10.  Numbers above are on cpu.
- **Mode:** `in_process_direct` — latency includes scheduler queue wait + inference only,
  NOT HTTP serialization or TCP.
- **Batch dynamics:** up to 32 requests per batch, formed
  within a 5.0 ms window.  At concurrency=16
  on cpu, batches typically contain
  16 requests.
- The SLO target requires CUDA hardware; MPS/CPU numbers above are for
  infrastructure correctness validation, not production benchmarking.

## To Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/load_test.py \
    --concurrency 16 \
    --requests 200 \
    --events 64
```
