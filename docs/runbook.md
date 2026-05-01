# Runbook

Operational procedures for Cortex-Engine.

---

## Deployment

### Local development

```bash
make dev-install
make serve  # http://localhost:8080
```

### Full stack with observability (CUDA host)

```bash
make docker-build
make docker-up
# API:        http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
# OTel:       localhost:4317 (gRPC)
```

### Full stack on a CPU-only machine (Mac / CI)

```bash
docker compose \
  -f ops/docker/docker-compose.yml \
  -f ops/docker/docker-compose.cpu.yml \
  up
```

This uses `Dockerfile.cpu` (Python 3.11-slim, CPU PyTorch) and removes the
NVIDIA GPU device reservation.

### Production deployment (Kubernetes)

The Helm chart lives in `ops/helm/cortex-engine/`. Validate before applying:

```bash
helm lint ops/helm/cortex-engine
helm install cortex ops/helm/cortex-engine \
  --namespace cortex \
  --create-namespace \
  --set config.otelEndpoint=http://otel-collector.observability:4317 \
  --set config.checkpointPath=/checkpoints/cortex_s.pt
```

Rolling update:

```bash
helm upgrade cortex ops/helm/cortex-engine \
  --set image.tag=<new-sha>
```

---

## Observability Stack

| Service | URL | Credentials |
|---|---|---|
| Inference API | http://localhost:8080 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |
| OTel collector | localhost:4317 (gRPC) | — |

### Grafana dashboards

Three dashboards are auto-provisioned at startup from `ops/dashboards/`:

| Dashboard | UID | What to look at |
|---|---|---|
| Traffic | `cortex-traffic` | Request rate, error rate, queue depth |
| Latency | `cortex-latency` | p50/p95/p99 end-to-end, SLO burn gauge |
| Resources | `cortex-resources` | GPU memory, GPU utilization, KV cache hit rate |

All dashboards auto-refresh every 5 seconds and show a 15-minute rolling window by default.

---

## Common Incidents

### High p99 latency (`HighP99Latency` alert)

**Symptoms:** p99 above 50 ms for 2+ minutes.

**Diagnosis:**

1. Open the **Latency** dashboard. Is the entire latency distribution elevated,
   or only the tail?
2. Check **Resources**: is GPU utilization at 100% and/or memory near the limit?
3. Check **Traffic**: has request rate increased beyond normal?
4. Check `cortex_queue_depth` — is the scheduler queue backing up?
5. Pull an OTel trace of a slow request to identify the slow stage
   (enqueue, `scheduler.dispatch`, or model forward).

**Common causes and fixes:**

| Cause | Fix |
|---|---|
| Traffic spike | Horizontal scale or activate rate limiting |
| GPU memory pressure | Reduce `CORTEX_MAX_BATCH` or `kv_cache_pages` |
| Bad client sending oversized windows | Add per-session rate limiting |
| Recent bad deploy | Roll back (see below) |

### Error rate spike (`HighErrorRate` alert)

**Symptoms:** error rate above 0.1% for 2+ minutes.

**Diagnosis:**

1. Filter structured logs by `error_type` (e.g., `worker_error`, `queue_full`).
2. Pull OTel traces of failing requests to see which stage threw.
3. Cross-reference with recent deployments (`git log --oneline -10`).

**Common causes:**

- `queue_full` (HTTP 429): arrival rate > serving capacity; scale out or increase
  `CORTEX_MAX_BATCH` if GPU has headroom.
- `inference_error`: model exception, often an OOM or bad input shape. Check logs
  for the actual Python traceback.
- `worker_error`: unhandled exception in the worker thread. Restart the service.

### GPU OOM

**Symptoms:** `HighGPUMemory` alert; `CUDA out of memory` in logs.

**Immediate mitigation:**

```bash
# Restart the service (releases all GPU memory)
docker compose restart cortex-engine
# or in Kubernetes:
kubectl rollout restart deployment/cortex -n cortex
```

**Tuning levers:**

```bash
# Reduce max batch (less peak activation memory)
CORTEX_MAX_BATCH=16

# Reduce KV cache pages (currently hardcoded in app; will be config in Phase 5)
# Default: num_pages=128 × page_size=64 × hidden_dim=512 × 4 bytes = 16 MB
```

**Long-term:** inspect via `nvidia-smi dmon` and PyTorch memory snapshot:
`torch.cuda.memory._dump_snapshot("mem.pickle")`.

### Queue saturation (`QueueNearSaturation` alert)

**Symptoms:** `cortex_queue_depth` > 200 for 1+ minute.

**Fix:** Scale horizontally — add replicas behind a load balancer. Cortex-Engine
is stateless between requests; each replica runs an independent scheduler and
worker. The KV cache is per-process (not shared).

---

## Routine Operations

### Load a new model checkpoint

1. Train: `make train-s` or custom config
2. Validate on held-out set: `PYTHONPATH=. .venv/bin/python -m cortex.training.eval --checkpoint <path>`
3. Set `CORTEX_CHECKPOINT=/app/checkpoints/<new>.pt` and trigger a rolling restart

### Run a k6 load test

```bash
make docker-up

# In a separate terminal:
docker compose -f ops/docker/docker-compose.yml \
  --profile loadtest \
  run loadgen
```

Results stream to stdout. Watch the Grafana **Latency** dashboard for live
system behavior under load. Results are also written to `benchmarks/serving/`.

### Capture an inference profile

```bash
PYTHONPATH=. .venv/bin/python scripts/profile_inference.py \
  --device auto --batch-size 32 --events-per-sample 512
# Writes benchmarks/profiling/baseline_report.md
```

### Roll back a bad deploy

```bash
# Docker Compose
git checkout <previous-sha> -- ops/docker/docker-compose.yml
docker compose pull && docker compose up -d

# Kubernetes
helm rollback cortex -n cortex
```

### SLO burn-rate dashboard interpretation

The **Latency** panel "SLO budget burn" shows multiples of the error budget consumed:
- **< 1×**: healthy — burning error budget slower than it replenishes
- **1–5×**: warning — investigate; burn-rate alert fires at 1× for 1 hour
- **> 5×**: critical — page on-call; exhausts monthly budget in < 6 days

Full SLO definitions: [`docs/slo.md`](slo.md).
