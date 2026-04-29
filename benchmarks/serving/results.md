# Phase 3 Serving Benchmarks

> Populated by `make bench-serving` (k6 load tests) and ad-hoc latency
> probes.

## Hardware

TBD

## Latency Distribution at Various Loads

### Constant load: 100 req/s

| Percentile | Latency |
|---|---|
| p50 | TBD |
| p95 | TBD |
| p99 | TBD |
| p99.9 | TBD |

### Saturating load (find the breaking point)

| Target rate (req/s) | Sustained rate | p99 latency | Errors |
|---|---|---|---|
| 100 | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD |
| 2000 | TBD | TBD | TBD |

The system saturates at TBD req/s. Beyond that, queue depth grows unboundedly.

## Batch Effects

| Batch size | Throughput (req/s) | Avg latency | GPU utilization |
|---|---|---|---|
| 1 | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD |

## Cortex-Engine vs Naive PyTorch Baseline

The naive baseline is a FastAPI server that does sync `model(x)` per request,
no batching, no scheduler.

| Metric | Naive PyTorch | Cortex-Engine | Improvement |
|---|---|---|---|
| Throughput at p99 < 50ms | TBD | TBD | TBD |
| Peak memory at saturation | TBD | TBD | TBD |
| GPU utilization at saturation | TBD | TBD | TBD |

## Reproducibility

```bash
make docker-up
docker compose -f ops/docker/docker-compose.yml --profile loadtest run loadgen
```

Full k6 reports in `benchmarks/serving/k6_*.json`.
