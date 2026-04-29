# Runbook

Operational procedures for Cortex-Engine.

## Deployment

### Local development

```bash
make dev-install
make serve  # http://localhost:8080
```

### Full stack with observability

```bash
make docker-build
make docker-up
# API:        http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

### Production deployment

The Helm chart is in `ops/helm/cortex-engine/`. Validate before applying:

```bash
helm lint ops/helm/cortex-engine
helm install cortex ops/helm/cortex-engine --namespace cortex --create-namespace
```

## Common Incidents

### High p99 latency

**Symptoms:** `HighP99Latency` alert; p99 above 50ms for 2+ minutes.

**Diagnosis steps:**

1. Check the Grafana **Latency** dashboard. Is the entire distribution shifting up, or only the tail?
2. Look at the **Resources** dashboard. Is GPU utilization saturated?
3. Check the **Traffic** dashboard. Is request rate higher than usual?
4. Check `cortex_queue_depth`. Is the scheduler queue backing up?

**Common causes:**
- Traffic spike beyond capacity → scale horizontally
- Recent deployment introduced a regression → roll back
- GPU memory pressure causing eviction → check `cortex_kv_cache_pages_used`
- Bad client sending unusually large windows → throttle by `session_id`

### Error rate spike

**Symptoms:** `HighErrorRate` alert.

**Diagnosis steps:**

1. Check structured logs filtered by `error_type`
2. Look at OpenTelemetry traces of failing requests
3. Cross-reference with deployment timeline

**Common causes:**
- Bad request payloads from a misbehaving client
- OOM after a memory leak (check `cortex_gpu_memory_used_bytes` trend)
- Downstream dependency failure

### GPU OOM

**Symptoms:** `HighGPUMemory` alert; CUDA OOM in logs.

**Mitigation:**

1. Restart the inference server (releases all GPU memory)
2. Reduce `max_batch_size` temporarily in serving config
3. Reduce `kv_cache_pages` if the cache is the dominant consumer

**Long-term fix:** investigate via `nvidia-smi`, PyTorch memory snapshots, and the
GPU memory dashboard.

## Routine Operations

### Loading a new model checkpoint

1. Train with `make train-s` or your custom config
2. Validate accuracy on held-out set: `python -m cortex.training.eval --checkpoint <path>`
3. Update `serving.checkpoint_path` config
4. Trigger rolling restart

### Running a load test

```bash
make docker-up
docker compose -f ops/docker/docker-compose.yml --profile loadtest run loadgen
```

Results stream to stdout. Check Grafana dashboards for system behavior under load.

### Capturing a profile

```bash
python -m cortex.benchmarks.profile_inference --output benchmarks/profiling/$(date +%Y%m%d).json
```

Open in PyTorch profiler UI or Chrome trace viewer.
