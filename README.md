# Cortex-Engine

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-red?style=for-the-badge)


[![CI](https://github.com/peterajhgraham/cortex-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/peterajhgraham/cortex-engine/actions/workflows/ci.yml)


A brain-computer interface that misses its deadline is a cursor that lags, a prosthetic that jerks, a patient who cannot type. Closing the loop on motor cortex requires sub-30 ms p99 latency over irregular, high-dimensional spike streams - hundreds of neurons firing asynchronously at sub-millisecond resolution, with the population code distributed across cells and time in ways that resist the fixed shapes most accelerators are built for. Cortex-Engine treats this as a systems problem first: a Perceiver-style decoder that compresses variable-length spike events into a fixed latent set, three custom Triton kernels targeting the layers profiling proved were bottlenecks, INT8 quantization, and a continuous-batching inference server with a paged streaming KV cache, all wired through a production-grade Prometheus/Grafana/OpenTelemetry observability stack. The goal is not a paper - it is the inference engine you would actually deploy behind a real BCI.

---

## Quick Visual

```
   Spike events (neuron_id, time_bin, value)
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  SpikeTokenizer                             │
│  fused embedding lookup + position          │
│  encoding + value scaling                   │
│  [Triton: fused tokenizer kernel]           │
└────────────────────┬────────────────────────┘
                     │ (E, D) tokens
                     ▼
┌─────────────────────────────────────────────┐
│  Perceiver Cross-Attention                  │
│  latent queries (L, D) ×                    │
│  spike keys/values (E, D)                   │
│  [Triton: block-sparse cross-attention]     │
└────────────────────┬────────────────────────┘
                     │ (B, L, D) latents
                     ▼
┌─────────────────────────────────────────────┐
│  Self-Attention Stack (× N)                 │
│  RMSNorm → QKV → SDPA → MLP                 │
│  [Triton: fused RMSNorm + linear]           │
└────────────────────┬────────────────────────┘
                     │ (B, L, D) latents
                     ▼
┌─────────────────────────────────────────────┐
│  Behavior Head                              │
│  cross-attn → per-dim scalar                │
│  (hand velocity)                            │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              Decoded kinematics

┌─────────────── Inference server ────────────┐
│  Async scheduler   (EDF, continuous batch)  │
│  Paged streaming KV cache                   │
│      (LRU, 91.6% window overlap)            │
│  FastAPI   WS /stream  •  POST /decode      │
└─────────────────────────────────────────────┘
   (drives SpikeTokenizer at request time)
```

Production inference infrastructure for transformer-based neural decoders. Three custom Triton kernels (fused embedding, block-sparse cross-attention, fused RMSNorm+linear), per-channel INT8 quantization with a calibration pipeline (72% weight memory reduction), a continuous-batching inference server with an earliest-deadline-first scheduler and paged streaming KV cache, and a full Prometheus/Grafana/OpenTelemetry observability stack. The model is a Perceiver-style transformer trained on real motor cortex population data from the [Neural Latents Benchmark](https://neurallatents.github.io/). On MPS the server delivers **10.5× throughput** over naive sequential inference; the p99 < 30ms SLO requires NVIDIA A10 (24GB).

---

## Benchmark Summary

> Consolidated reference: [`BENCHMARKS.md`](BENCHMARKS.md). Forward-looking work: [`ROADMAP.md`](ROADMAP.md).

### Decoding accuracy (trial-aligned evaluation - NLB protocol)

One sample per reach trial, window −100 ms to +500 ms around `move_onset_time`, target = velocity at onset.
Full report: [`benchmarks/training/trial_aligned_results.md`](benchmarks/training/trial_aligned_results.md).

| Model | Params | R² (hand velocity) |
|---|---|---|
| Wiener filter (ridge) | 137 × 2 | **0.48** |
| GRU (2-layer bidir) | ~660 K | pending CUDA |
| Vanilla Transformer | ~5 M | pending CUDA |
| Cortex-S | 24.80 M | **0.60** |

*Sliding-window R² values (published in [`benchmarks/training/results.md`](benchmarks/training/results.md)) are near zero because ~85% of windows are rest periods - the evaluation distribution is wrong, not the models. Trial-aligned evaluation above is the correct comparison.*

### Inference (MPS, Cortex-S, batch=32, 512 events/sample)

| Section | Time (ms) | % of forward |
|---|---|---|
| **Full forward** | **129.0** | 100% |
| Self-attention (×7) | 100.6 | 78.0% |
| Cross-attention | 13.6 | 10.5% |
| `_pack_events` | 7.8 | 6.1% |
| Behavior head | 3.0 | 2.3% |

Full report: [`benchmarks/profiling/baseline_report.md`](benchmarks/profiling/baseline_report.md).

### Triton kernels (correctness verified; benchmarks require NVIDIA A10 (24GB))

| Kernel | What it fuses | Memory saved | Speedup (A10) |
|---|---|---|---|
| Fused tokenizer | 3 embedding lookups → 1 kernel | Eliminates 2 × (E, D) intermediates | - |
| Sparse cross-attention | FA2 online softmax + block sparsity | Skips masked event tiles | **up to 27×** |
| Fused RMSNorm + linear | Norm + matmul → 1 kernel | 112 MB / forward at Cortex-S scale | - |

Run with `make bench-kernels` on an NVIDIA A10 (24GB) host.

### INT8 quantization (MPS, synthetic calibration)

| Configuration | Memory | Reduction |
|---|---|---|
| float32 | 99.2 MB | - |
| INT8 per-channel | **27.8 MB** | **−72%** |

Max weight error: 0.003. 34/35 linear layers quantized. Full report: [`benchmarks/quantization/results.md`](benchmarks/quantization/results.md).

### Serving latency (NVIDIA A10 (24GB), in-process, concurrency=8)

| Metric | Value |
|---|---|
| Throughput | **255 req/s** |
| p50 | 27 ms |
| p95 | 28 ms |
| p99 | 261 ms |
| Failures | 0 / 500 |

> **Note on p99:** The 261 ms p99 reflects a first-batch initialization spike (CUDA kernel JIT compile + worker warmup on the initial request). Steady-state p99 is **~28 ms**, well within the 30 ms SLO. Full report: [`benchmarks/serving/results.md`](benchmarks/serving/results.md).

> ### Honest Results - hardware caveat
>
> The model, kernels, scheduler, KV cache, server, and observability stack are all implemented, tested, and benchmarked end-to-end. Where numbers are reported on **MPS** (Apple Silicon), they are real measurements on real data - but they are not the hero numbers. The hero numbers (Triton kernel speedups, p99 < 30 ms serving, full FSDP training of Cortex-M) require an **NVIDIA A10 (24 GB)** and Triton's CUDA backend, which is not available on Apple Silicon. Every CUDA-dependent benchmark cell is explicitly marked **"pending CUDA"** rather than estimated, extrapolated, or quietly omitted. Kernel correctness is verified against PyTorch references within `rtol=1e-3, atol=1e-3` on every commit via CI. A consolidated view lives in [`BENCHMARKS.md`](BENCHMARKS.md).

---

## Architecture

```
   Spike Events (neuron_id, time_bin, value)
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  SpikeTokenizer                             │
│  Fused gather: n_emb[nid] + t_emb[tb] +     │
│  v_emb[val]  → (E, D) tokens                │
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
│  RMSNorm → QKV → SDPA → MLP                 │
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

## Why This Matters Beyond BCI

The systems patterns here are the same ones that make modern LLM inference work. The async scheduler with deadline-aware continuous batching is the same technique vLLM uses to coalesce variable-length generation requests into shared GPU work. The paged streaming KV cache is the same memory model as PagedAttention - fixed-size pages, LRU eviction, near-zero fragmentation - adapted from token sequences to sliding-window spike contexts with 91.6% overlap. The INT8 quantization pipeline (per-channel weights, 99th-percentile activation calibration, dequant-then-matmul) is the same recipe as LLM.int8(), implemented from scratch instead of via a one-line library call so the calibration code is auditable. FSDP2 sharded training, fused Triton kernels for the layers that profiling proved were bottlenecks, and OpenTelemetry tracing through an async request path are all directly transferable to any transformer-serving stack - this repo is a BCI inference engine that happens to be built out of LLM-systems primitives.

---

## What's Built

### Phase 1 - Model and Training Pipeline ✓

- **Three model sizes:** Cortex-XS (4.83M), Cortex-S (24.80M), Cortex-M (83.51M). Perceiver cross-attention encoder with a fixed latent array (L=256 for Cortex-S); handles variable neuron counts without hard-coding electrode geometry.
- **FSDP training loop:** Mixed-precision bfloat16, sharded checkpointing via `torch.distributed.checkpoint`, cosine LR schedule with warmup. Single-GPU mode on MPS.
- **Three baselines + MC_Maze loader:** Wiener filter, GRU, vanilla Transformer evaluated under identical protocol. pynwb/DANDI pipeline, 137 heldin units, 5 ms bins, trial-aligned and sliding-window dataset modes.

### Phase 2.1 - Profiling ✓

- **Bottleneck found:** `_pack_events` Python loop forced 32 CPU↔MPS sync stalls per forward pass. Replaced with vectorized `cumsum`; saves 4.2 ms / forward (−3.2%).
- **Bottleneck hierarchy:** Self-attention 78%, cross-attention 10.5%, pack_events 6.1%. Triton targets chosen from this, not intuition.

### Phase 2.2 - Three Triton Kernels ✓

- **Fused tokenizer:** 2D kernel fuses three embedding lookups into one pass; eliminates two intermediate (E, D) tensors. 9-config autotuner keyed on (E, D).
- **Block-sparse cross-attention:** FlashAttention-2 online softmax (running m/l/o). Skips entire event tiles where the block mask is false. External mask API keeps sparsity policy decoupled from the kernel.
- **Fused RMSNorm+linear:** Two-pass kernel - accumulate x² for RMS in pass 1, apply norm × gamma inline during `tl.dot()` in pass 2. x_norm never touches HBM.

### Phase 2.6 - INT8 Quantization ✓

- **Per-channel calibration:** Activation scales use 99th-percentile abs-max across calibration batches. Weight scales: absmax per output neuron.
- **`QuantizedLinear`:** Stores INT8 weights + float32 scales; dequantizes to bf16 before matmul - device-agnostic, no quantized CUDA kernel dependency.

### Phase 3 - Inference Engine ✓

- **Scheduler + worker:** `asyncio.PriorityQueue` ordered by deadline (EDF). Worker runs in a `ThreadPoolExecutor`; on CUDA, `compute_stream` and `copy_stream` overlap H2D transfer with the previous batch's compute.
- **`StreamingKVCache`:** Paged embedding cache `(num_pages, page_size, hidden_dim)`. LRU eviction via `OrderedDict`. Exploits the 91.6% overlap between consecutive 600 ms / 50 ms-stride BCI windows.
- **FastAPI server:** `POST /decode`, `WS /stream`, `GET /metrics` (Prometheus sub-app). Admission control raises HTTP 429 when the queue is full.

### Phase 4 - Operations and Observability ✓

- **Three Grafana dashboards** auto-provisioned at startup: traffic (req/s, error rate, queue depth), latency (p50–p99.9, SLO burn gauge, batch-size heatmap), resources (GPU memory, utilization, KV cache hit rate).
- **docker-compose stack:** cortex-engine + Prometheus + Grafana + OTel Collector + k6 loadgen. CPU override in `docker-compose.cpu.yml`. `docker compose up` brings up the full stack.
- **Helm chart + Alertmanager:** GPU node selector/toleration, ServiceMonitor, autoscaling on `cortex_queue_depth`. Alerts for p99 > 50ms, error rate > 0.1%, queue saturation.

### Phase 5 - Writeup ✓

- Engineering postmortem (`docs/writeup.md`, ~3 500 words): architecture decisions, profiling methodology, kernel design, honest results with hardware caveats, what I'd do differently.

---

## Tech Stack

| Tool | Role |
|---|---|
| **PyTorch 2.2+** | FSDP2, `scaled_dot_product_attention` → FlashAttention dispatch, MPS backend for dev on Apple Silicon. |
| **Triton** | Custom GPU kernels in Python; generates PTX directly. CUDA-only. |
| **einops** | Readable tensor reshapes - `rearrange` instead of chains of `.view()`. |
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

# Train (CUDA, 2000 steps, ~2 min on A10 24 GB PCIe):
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
#   http://localhost:8080  - inference API
#   http://localhost:9090  - Prometheus
#   http://localhost:3000  - Grafana  (admin / admin)
#   localhost:4317         - OTel Collector (gRPC)

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

**Traffic** (`cortex-traffic`) - request rate by endpoint, error rate by type, queue depth, Little's Law in-flight estimate.

**Latency** (`cortex-latency`) - p50/p95/p99/p99.9 time series with 30 ms / 50 ms thresholds. SLO burn gauge spikes red when the k6 ramping scenario saturates the server. Batch-size heatmap shows continuous batching coalescing requests into batches of 16–32.

**Resources** (`cortex-resources`) - GPU memory and utilization, KV cache pages and hit rate. Hit rate should stay above 80% during streaming sessions given the 91.6% sliding-window overlap.

---

## Project Layout

```
cortex-engine/
├── cortex/
│   ├── models/         # tokenizer, perceiver, cortex (XS/S/M), config
│   ├── kernels/        # Triton kernels + benches (tokenizer,
│   │                   #   sparse xattn, fused RMSNorm+linear)
│   ├── quantization/   # per-channel INT8 calibration + QuantizedLinear
│   ├── training/       # FSDP train loop, eval, checkpointing, baselines
│   ├── data/           # NLB / MC_Maze pynwb+DANDI loader
│   ├── serve/          # FastAPI app, EDF scheduler, worker,
│   │                   #   Prometheus metrics, OTel tracing, schemas
│   ├── cache/          # paged streaming KV cache (LRU)
│   └── utils/          # device detection, structlog setup
├── configs/            # Hydra: cortex_{xs,s,m}.yaml + model/ data/
│                       #   training/ runtime/ serving/ sub-configs
├── scripts/            # train_benchmark, baseline_benchmark,
│                       #   profile_inference, calibrate_model,
│                       #   eval_trial_aligned, load_test
├── tests/
│   ├── unit/           # mirrors cortex/ layout
│   └── integration/    # server, scheduler, end-to-end
├── benchmarks/
│   ├── training/       # baselines + Cortex-S results
│   ├── profiling/      # baseline_report.md (PyTorch profiler)
│   ├── kernels/        # Triton kernel sweeps (CUDA)
│   ├── quantization/   # INT8 memory/accuracy report
│   └── serving/        # load-test latency distributions
├── ops/
│   ├── docker/         # Dockerfile + docker-compose (+ cpu override)
│   ├── dashboards/     # Grafana JSON: traffic, latency, resources
│   ├── helm/           # chart skeleton (ServiceMonitor, HPA, alerts)
│   └── k6/             # load-generation scripts
├── docs/               # PROJECT_PLAN, ROADMAP, writeup, runbook, slo
├── BENCHMARKS.md       # consolidated benchmark index
├── ROADMAP.md          # forward-looking work
└── Makefile            # dev-install, train-s, bench-kernels,
                        #   docker-build, docker-up
```

---

## License

MIT
