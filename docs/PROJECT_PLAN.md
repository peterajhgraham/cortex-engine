# Project Plan: Cortex-Engine

This document is the extended companion to `CLAUDE.md`. It enumerates every
deliverable, success criterion, and "definition of done" check.

## Phase 1: Model + Training Pipeline

### Tasks

| ID | Task | Files | Status |
|----|------|-------|--------|
| 1.1 | Spike tokenizer | `cortex/models/tokenizer.py` | done |
| 1.2 | Perceiver encoder | `cortex/models/perceiver.py` | done |
| 1.3 | Decoder heads | `cortex/models/cortex.py` | done |
| 1.4 | NLB data loader | `cortex/data/nlb.py` | done |
| 1.5 | FSDP training loop | `cortex/training/train.py` | done |
| 1.6 | W&B integration | `cortex/training/train.py` | done |
| 1.7 | Sharded checkpoints | `cortex/training/checkpoint.py` | done |
| 1.8 | Wiener filter baseline | `cortex/training/baselines.py` | done |
| 1.9 | GRU baseline | `cortex/training/baselines.py` | done |
| 1.10 | Vanilla transformer baseline | `cortex/training/baselines.py` | done |
| 1.11 | Eval pipeline + R² metric | `cortex/training/eval.py` | done |

### Definition of Done

- [ ] All three Cortex sizes (XS, S, M) train to convergence on MC_Maze
- [ ] Cortex-S beats all three baselines by at least 0.05 R² on hand velocity
- [ ] W&B run links pasted into `benchmarks/training/results.md`
- [ ] `make train-s` produces a checkpoint reproducibly
- [ ] All unit tests pass; integration smoke test passes
- [ ] mypy --strict passes on the cortex package

## Phase 2: Going Down to the Metal

### Tasks

| ID | Task | Files | Status |
|----|------|-------|--------|
| 2.1 | Profile baseline inference | `benchmarks/profiling/baseline.md` | done |
| 2.2 | Fused tokenizer kernel | `cortex/kernels/tokenizer.py` | done |
| 2.3 | Sparse cross-attention kernel | `cortex/kernels/sparse_xattn.py` | done |
| 2.4 | Fused RMSNorm kernel | `cortex/kernels/fused_rmsnorm.py` | done |
| 2.5 | Kernel benchmarks | `benchmarks/kernels/*.md` | partial (CUDA perf numbers pending) |
| 2.6 | INT8 calibration | `cortex/quantization/calibrate.py` | done |
| 2.7 | QuantizedLinear + conversion | `cortex/quantization/quantize.py` | done |
| 2.8 | Quantization benchmark | `benchmarks/quantization/results.md` | done |

### Definition of Done

- [ ] Each kernel demonstrates measurable speedup at production input shapes
- [ ] All kernels pass correctness tests at `rtol=1e-3, atol=1e-3`
- [ ] INT8 model loses less than 1% R² vs bf16
- [ ] Profiling reports show bottleneck shift after optimization
- [ ] Roofline analysis included for at least one kernel

## Phase 3: Inference Engine

### Tasks

| ID | Task | Files | Status |
|----|------|-------|--------|
| 3.1 | Pydantic API schemas | `cortex/serve/schemas.py` | done |
| 3.2 | Continuous batching scheduler | `cortex/serve/scheduler.py` | done |
| 3.3 | Inference worker | `cortex/serve/worker.py` | done |
| 3.4 | Streaming KV cache | `cortex/cache/streaming.py` | done |
| 3.5 | FastAPI app | `cortex/serve/app.py` | done |
| 3.6 | WebSocket streaming endpoint | `cortex/serve/app.py` | done |
| 3.7 | Load test scripts | `ops/k6/*.js` | done |
| 3.8 | Latency benchmarks | `benchmarks/serving/*.md` | done |

### Definition of Done

- [ ] p99 latency under 30ms for Cortex-S streaming inference
- [ ] 5x throughput vs naive PyTorch baseline at saturating load
- [ ] Greater than 80% GPU utilization at peak throughput
- [ ] Load test passes k6 SLO threshold
- [ ] Request lifecycle diagram in README

## Phase 4: Operations and Observability

### Tasks

| ID | Task | Files | Status |
|----|------|-------|--------|
| 4.1 | Prometheus metrics | `cortex/serve/metrics.py` | done |
| 4.2 | Grafana dashboards | `ops/dashboards/*.json` | done |
| 4.3 | OpenTelemetry tracing | `cortex/serve/tracing.py` | done |
| 4.4 | Structured logging | `cortex/utils/logging.py` | done |
| 4.5 | Alertmanager rules | `ops/docker/alerts.yml` | done |
| 4.6 | Multi-stage Dockerfile | `ops/docker/Dockerfile` | done |
| 4.7 | docker-compose stack | `ops/docker/docker-compose.yml` | done |
| 4.8 | Helm chart skeleton | `ops/helm/cortex-engine/` | done |
| 4.9 | Runbook | `docs/runbook.md` | done |
| 4.10 | SLO doc | `docs/slo.md` | done |

### Definition of Done

- [ ] `docker compose up` brings up the full stack with live metrics
- [ ] All three Grafana dashboards (traffic, latency, resources) populated
- [ ] At least one end-to-end OTel trace captured in screenshot
- [ ] Helm manifests pass `helm lint`

## Phase 5: Writeup

### Tasks

| ID | Task | Files | Status |
|----|------|-------|--------|
| 5.1 | README polish | `README.md` | done |
| 5.2 | Long-form writeup | `docs/writeup.md` | done |
| 5.3 | Architecture diagrams | `docs/diagrams/` | partial (ASCII in README; Mermaid deferred) |
| 5.4 | Demo gif/video | `docs/demo.gif` | deferred (requires CUDA host) |

### Definition of Done

- [ ] 30-second README test: someone unfamiliar can extract hero metrics in 30s
- [ ] Writeup covers motivation, architecture, optimization, serving, ops, results
- [ ] All diagrams render correctly in GitHub markdown
- [ ] Demo shows end-to-end inference at target latency
