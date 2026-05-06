# Cortex-Engine

![Python](https://img.shields.io/badge/python-3.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-red?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)
[![CI](https://github.com/peterajhgraham/cortex-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/peterajhgraham/cortex-engine/actions/workflows/ci.yml)

Production inference infrastructure for transformer-based neural decoders. Three custom Triton kernels (fused embedding, block-sparse cross-attention, fused RMSNorm+linear), per-channel INT8 quantization with a calibration pipeline (72% weight memory reduction), a continuous-batching inference server with an earliest-deadline-first scheduler and paged streaming KV cache, and a full Prometheus/Grafana/OpenTelemetry observability stack. The model is a Perceiver-style transformer trained on real motor cortex population data from the [Neural Latents Benchmark](https://neurallatents.github.io/). On MPS the server delivers **10.5× throughput** over naive sequential inference; the p99 < 30ms SLO requires CUDA.

---

## Benchmark Summary

### Decoding accuracy (trial-aligned evaluation — NLB protocol)

One sample per reach trial, window −100 ms to +500 ms around `move_onset_time`, target = velocity at onset.
Full report: [`benchmarks/training/trial_aligned_results.md`](benchmarks/training/trial_aligned_results.md).

| Model | Params | R² (hand velocity) |
|---|---|---|
| Wiener filter (ridge) | 137 × 2 | **0.4822** |
| GRU (2-layer bidir) | ~660 K | pending CUDA |
| Vanilla Transformer | ~5 M | pending CUDA |
| Cortex-S | 24.80 M | pending CUDA |

*Sliding-window R² values (published in [`benchmarks/training/results.md`](benchmarks/training/results.md)) are near zero because ~85% of windows are rest periods — the evaluation distribution is wrong, not the models. Trial-aligned evaluation above is the correct comparison.*

### Inference (MPS, Cortex-S, batch=32, 512 events/sample)

| Section | Time (ms) | % of forward |
|---|---|---|
| **Full forward** | **129.0** | 100% |
| Self-attention (×7) | 100.6 | 78.0% |
| Cross-attention | 13.6 | 10.5% |
| `_pack_events` | 7.8 | 6.1% |
| Behavior head | 3.0 | 2.3% |

Full report: [`benchmarks/profiling/baseline_report.md`](benchmarks/profiling/baseline_report.md).

### Triton kernels (correctness verified; benchmarks require CUDA)

| Kernel | What it fuses | Memory saved |
|---|---|---|
| Fused tokenizer | 3 embedding lookups → 1 kernel | Eliminates 2 × (E, D) intermediates |
| Sparse cross-attention | FA2 online softmax + block sparsity | Skips masked event tiles |
| Fused RMSNorm + linear | Norm + matmul → 1 kernel | 112 MB / forward at Cortex-S scale |

Run with `make bench-kernels` on a CUDA host.

### INT8 quantization (MPS, synthetic calibration)

| Configuration | Memory | Reduction |
|---|---|---|
| float32 | 99.2 MB | — |
| INT8 per-channel | **27.8 MB** | **−72%** |

Max weight error: 0.003. 34/35 linear layers quantized. Full report: [`benchmarks/quantization/results.md`](benchmarks/quantization/results.md).

### Serving latency (MPS, in-process, concurrency=16)

| Metric | Value |
|---|---|
| Throughput | **157.3 req/s** |
| p50 | 70.2 ms |
| p99 | 358.5 ms |
| Failures | 0 / 200 |

p99 < 30ms SLO requires CUDA — a batch-of-32 Cortex-S forward pass on an A100 takes ~2–3 ms. Full report: [`benchmarks/serving/results.md`](benchmarks/serving/results.md).

---

## Architecture

```
Spike Events (neuron_id, time_bin, value)
    │
    ▼
┌─────────────────────────────────────────────┐
│  SpikeTokenizer                             │
│  Fused gather: n_emb[nid] + t_emb[tb] +    │
│  v_emb[val]  → (E, D) tokens               │
│  [Triton kernel: Phase 2.2]                 │
└────────────────────┬────────────────────────┘
                     │ (E, D) flat tokens
                     ▼
┌─────────────────────────────────────────────┐
│  Perceiver Cross-Attention                  │
│  Latent queries (L, D) × Spike keys/values  │
│  [Block-sparse Triton kernel: Phase 2.2]    │
└────────────────────┬────────────────────────┘
                     │ (B, L, D) latents
                     ▼
┌─────────────────────────────────────────────┐
│  Self-Attention Stack (N layers)            │
│  RMSNorm → QKV → SDPA → MLP                │
│  [Fused RMSNorm+linear kernel: Phase 2.2]   │
└────────────────────┬────────────────────────┘
                     │ (B, L, D) latents
                     ▼
┌─────────────────────────────────────────────┐
│  Decoder Heads                              │
│  Behavior: cross-attn → scalar per dim      │
│  Masked spike: reconstruction head (SSL)    │
└─────────────────────────────────────────────┘
```

---

## What's Built

### Phase 1 — Model and Training Pipeline ✓

- **Three model sizes:** Cortex-XS (4.83M), Cortex-S (24.80M), Cortex-M (83.51M). Perceiver cross-attention encoder with a fixed latent array (L=256 for Cortex-S); handles variable neuron counts without hard-coding electrode geometry.
- **FSDP training loop:** Mixed-precision bfloat16, sharded checkpointing via `torch.distributed.checkpoint`, cosine LR schedule with warmup. Single-GPU mode on MPS.
- **Three baselines + MC_Maze loader:** Wiener filter, GRU, vanilla Transformer evaluated under identical protocol. pynwb/DANDI pipeline, 137 heldin units, 5 ms bins, trial-aligned and sliding-window dataset modes.

### Phase 2.1 — Profiling ✓

- **Bottleneck found:** `_pack_events` Python loop forced 32 CPU↔MPS sync stalls per forward pass. Replaced with vectorized `cumsum`; saves 4.2 ms / forward (−3.2%).
- **Bottleneck hierarchy:** Self-attention 78%, cross-attention 10.5%, pack_events 6.1%. Triton targets chosen from this, not intuition.

### Phase 2.2 — Three Triton Kernels ✓

- **Fused tokenizer:** 2D kernel fuses three embedding lookups into one pass; eliminates two intermediate (E, D) tensors. 9-config autotuner keyed on (E, D).
- **Block-sparse cross-attention:** FlashAttention-2 online softmax (running m/l/o). Skips entire event tiles where the block mask is false. External mask API keeps sparsity policy decoupled from the kernel.
- **Fused RMSNorm+linear:** Two-pass kernel — accumulate x² for RMS in pass 1, apply norm × gamma inline during `tl.dot()` in pass 2. x_norm never touches HBM.

### Phase 2.6 — INT8 Quantization ✓

- **Per-channel calibration:** Activation scales use 99th-percentile abs-max across calibration batches. Weight scales: absmax per output neuron.
- **`QuantizedLinear`:** Stores INT8 weights + float32 scales; dequantizes to bf16 before matmul — device-agnostic, no quantized CUDA kernel dependency.

### Phase 3 — Inference Engine ✓

- **Scheduler + worker:** `asyncio.PriorityQueue` ordered by deadline (EDF). Worker runs in a `ThreadPoolExecutor`; on CUDA, `compute_stream` and `copy_stream` overlap H2D transfer with the previous batch's compute.
- **`StreamingKVCache`:** Paged embedding cache `(num_pages, page_size, hidden_dim)`. LRU eviction via `OrderedDict`. Exploits the 91.6% overlap between consecutive 600 ms / 50 ms-stride BCI windows.
- **FastAPI server:** `POST /decode`, `WS /stream`, `GET /metrics` (Prometheus sub-app). Admission control raises HTTP 429 when the queue is full.

### Phase 4 — Operations and Observability ✓

- **Three Grafana dashboards** auto-provisioned at startup: traffic (req/s, error rate, queue depth), latency (p50–p99.9, SLO burn gauge, batch-size heatmap), resources (GPU memory, utilization, KV cache hit rate).
- **docker-compose stack:** cortex-engine + Prometheus + Grafana + OTel Collector + k6 loadgen. CPU override in `docker-compose.cpu.yml`. `docker compose up` brings up the full stack.
- **Helm chart + Alertmanager:** GPU node selector/toleration, ServiceMonitor, autoscaling on `cortex_queue_depth`. Alerts for p99 > 50ms, error rate > 0.1%, queue saturation.

### Phase 5 — Writeup ✓

- Engineering postmortem (`docs/writeup.md`, ~3 500 words): architecture decisions, profiling methodology, kernel design, honest results with hardware caveats, what I'd do differently.

---

## Tech Stack

| Tool | Role |
|---|---|
| **PyTorch 2.2+** | FSDP2, `scaled_dot_product_attention` → FlashAttention dispatch, MPS backend for dev on Apple Silicon. |
| **Triton** | Custom GPU kernels in Python; generates PTX directly. CUDA-only. |
| **einops** | Readable tensor reshapes — `rearrange` instead of chains of `.view()`. |
| **Hydra** | Hierarchical config composition with CLI overrides. |
| **Pydantic v2** | Typed config and I/O schemas; `model_validator` for cross-field constraints. |
| **pynwb / DANDI** | NLB data is NWB-native; no conversion step needed. |
| **W&B** | Experiment tracking with offline mode; `wandb sync` to upload later. |
| **pytest** | Unit + integration tests; `@pytest.mark.gpu` skips CUDA-only paths on non-CUDA hardware. |
| **mypy (strict)** | Catches shape bugs before runtime. `py.typed` marker in package. |
| **ruff + black** | Linting and formatting enforced by pre-commit hooks. |
| **FastAPI** | Async inference server; WebSocket streaming and REST batch endpoints. |
| **structlog** | Structured JSON logging throughout `cortex/`. |

---

## How to Run It

### Prerequisites

```bash
# Python 3.11+, git clone, then:
make dev-install          # creates .venv, installs all deps

# Verify install
PYTHONPATH=. .venv/bin/python -c "import cortex; print('ok')"
```

### Run all non-GPU tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# Expected: 116 passed, ~39 skipped (CUDA-only Triton paths)
```

### Profile Cortex-S inference on MPS

```bash
PYTHONPATH=. .venv/bin/python scripts/profile_inference.py \
    --device auto --batch-size 32 --events-per-sample 512
# Writes benchmarks/profiling/baseline_report.md
```

### Train Cortex-S on MC\_Maze

```bash
# Download data first (requires DANDI account or public access):
PYTHONPATH=. .venv/bin/python -c "
from cortex.data.nlb import download_mc_maze
download_mc_maze('data/mc_maze')
"

# Train (MPS, 2000 steps, ~20 min on M4 Pro):
make train-s
# or directly:
PYTHONPATH=. .venv/bin/python scripts/train_benchmark.py \
    --model cortex_s --steps 2000 --device auto
```

### Run baselines

```bash
PYTHONPATH=. .venv/bin/python scripts/baseline_benchmark.py
# Writes benchmarks/training/baselines_raw.json
```

### Calibrate and quantize Cortex-S (INT8)

```bash
PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py \
    --checkpoint checkpoints/cortex_s.pt \
    --data-dir data/mc_maze \
    --calib-batches 50 \
    --output benchmarks/quantization/results.md
# Writes INT8 model to checkpoints/cortex_s_int8.pt
```

### Run kernel benchmarks (CUDA required)

```bash
# Individual kernels:
python -m cortex.kernels.bench_tokenizer
python -m cortex.kernels.bench_sparse_xattn
python -m cortex.kernels.bench_fused_rmsnorm

# All at once:
make bench-kernels
# Writes benchmarks/kernels/*.json
```

### Start the inference server

```bash
# Start FastAPI server (auto-detects device: CUDA → MPS → CPU):
PYTHONPATH=. .venv/bin/python -m uvicorn cortex.serve.app:app --host 0.0.0.0 --port 8080

# With explicit config:
CORTEX_DEVICE=cuda CORTEX_MAX_BATCH=32 CORTEX_DEADLINE_MS=30 \
    PYTHONPATH=. .venv/bin/python -m uvicorn cortex.serve.app:app --host 0.0.0.0 --port 8080

# Health check:
curl http://localhost:8080/health
# Ready check (503 until warmup completes):
curl http://localhost:8080/ready
```

### Run the load test

```bash
# In-process (measures scheduler + worker latency, no HTTP overhead):
PYTHONPATH=. .venv/bin/python scripts/load_test.py \
    --concurrency 16 --requests 200 --events 256
# Writes benchmarks/serving/results.md

# Against a live server:
PYTHONPATH=. .venv/bin/python scripts/load_test.py \
    --url http://localhost:8080 --concurrency 32 --requests 500
```

### Start the full observability stack (docker compose)

Requires Docker. On a CUDA host:

```bash
make docker-build         # builds cortex-engine:latest from Dockerfile
make docker-up            # docker compose up -d

# Services:
#   http://localhost:8080  — inference API
#   http://localhost:9090  — Prometheus
#   http://localhost:3000  — Grafana  (admin / admin)
#   localhost:4317         — OTel Collector (gRPC)

# Run k6 load test against the live stack:
docker compose -f ops/docker/docker-compose.yml --profile loadtest run loadgen
```

On a CPU-only machine (Mac / CI):

```bash
docker compose \
  -f ops/docker/docker-compose.yml \
  -f ops/docker/docker-compose.cpu.yml \
  up
```

### Grafana dashboards

Three dashboards are auto-provisioned at startup (Grafana → Dashboards → Cortex):

**Traffic** (`cortex-traffic`) — request rate by endpoint, error rate by type, queue depth, Little's Law in-flight estimate.

**Latency** (`cortex-latency`) — p50/p95/p99/p99.9 time series with 30 ms / 50 ms thresholds. SLO burn gauge spikes red when the k6 ramping scenario saturates the server. Batch-size heatmap shows continuous batching coalescing requests into batches of 16–32.

**Resources** (`cortex-resources`) — GPU memory and utilization, KV cache pages and hit rate. Hit rate should stay above 80% during streaming sessions given the 91.6% sliding-window overlap.

---

## Project Layout

```
cortex-engine/
├── cortex/
│   ├── models/
│   ├── kernels/
│   ├── quantization/
│   ├── training/
│   ├── data/
│   ├── serve/
│   └── cache/
├── configs/
├── scripts/
├── tests/
├── benchmarks/
│   ├── training/
│   ├── profiling/
│   ├── kernels/
│   ├── quantization/
│   └── serving/
└── docs/
```

---

## License

MIT
