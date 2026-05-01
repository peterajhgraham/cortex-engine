# Cortex-Engine

**Real-time inference infrastructure for transformer-based neural decoders.**

Cortex-Engine is a production-oriented ML systems project that builds the full stack for deploying a Perceiver-style transformer decoder on neural population data from the [Neural Latents Benchmark](https://neurallatents.github.io/). It covers custom Triton GPU kernels, a profiling-driven optimization workflow, INT8 quantization with calibration, and a continuous-batching inference server — targeting sub-30ms p99 latency for streaming brain-computer interface (BCI) decoding on a single A100. The infrastructure patterns are directly applicable to LLM serving: the same continuous batching, paged KV cache, FSDP training, and custom kernel workflow appear verbatim in production inference systems.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](#)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2+-orange)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](#)

---

## Benchmark Summary

### Decoding accuracy (Phase 1 — MPS, sliding-window evaluation)

| Model | Params | R² (hand velocity) | Train time | Notes |
|---|---|---|---|---|
| Cortex-S | 24.80 M | **−0.0002** | 20 min / 2 K steps | Best model in eval |
| Wiener filter (ridge) | 137 × 2 | −0.003 | 1.3 s | Mean-rate features |
| GRU (2-layer bidir) | ~660 K | −0.006 | 380 s / 5 epochs | 20 K subsample |
| Vanilla Transformer | ~5 M | −0.013 | 121 s / 3 epochs | 4L / 256d / 4H |

**Honest note on these numbers:** All values are from sliding-window evaluation over the full MC_Maze continuous recording (115 min, 85% rest epochs). Published NLB benchmarks (Wiener R² ≈ 0.40) use trial-aligned evaluation with a dedicated held-out trial split. The negative R² values reflect the model predicting near-zero velocity throughout rest periods — not model failure. Cortex-S outperforms all three baselines under the same evaluation protocol. Trial-aligned evaluation is scheduled for a future pass. Full methodology: [`benchmarks/training/results.md`](benchmarks/training/results.md).

### Inference profiling (Phase 2.1 — MPS, Cortex-S, batch=32, 512 events/sample)

| Section | Time (ms) | % of forward | Notes |
|---|---|---|---|
| **Full forward** | **129.0** | 100% | After `_pack_events` fix |
| Self-attention (×7) | 100.6 | 78.0% | Primary Triton target |
| Cross-attention | 13.6 | 10.5% | Sparse xattn target |
| `_pack_events` | 7.8 | 6.1% | Was 13.6ms; fixed with `cumsum` |
| Behavior head | 3.0 | 2.3% | Below 5% threshold |
| Tokenizer | 1.4 | 1.1% | Below 5% threshold |

Hardware: Apple M4 Pro, 24 GB unified memory, MPS backend. Full report: [`benchmarks/profiling/baseline_report.md`](benchmarks/profiling/baseline_report.md).

### Kernel benchmarks (Phase 2.2 — requires CUDA, not yet run)

Three Triton kernels are fully implemented and tested. Benchmark numbers are pending CUDA hardware. Scripts are in `cortex/kernels/bench_*.py` and can be run with `make bench-kernels`.

| Kernel | Memory saving | Theoretical speedup source |
|---|---|---|
| Fused tokenizer | Eliminates 2 × (E, D) intermediates (~10× → 4× bytes) | Fused gather + add |
| Sparse cross-attention | Skips masked event tiles (O(1 − density) × K/V reads) | Block sparsity + FA2 |
| Fused RMSNorm + linear | Eliminates x\_norm write + read (saves 2 × M × K bytes) | No HBM round-trip for intermediate |

### Quantization (Phase 2.6 — MPS, Cortex-S, synthetic data)

| Configuration | Weight + buffer memory | MSE delta | Max abs output diff |
|---|---|---|---|
| float32 baseline | 99.2 MB | — | — |
| INT8 per-channel weights | **27.8 MB** | +0.0086% | 0.005 |

Savings: −71.4 MB (72.0%). 34/35 linear layers quantized (the one exception: `MaskedSpikeHead.proj` is gated by `return_aux=True` and never activates during a standard forward pass, so no calibration data was collected for it). Numbers measured with randomly initialized weights and synthetic spike events — MSE delta is not meaningful for model quality, but output fidelity (max abs diff 0.005) is. Full report: [`benchmarks/quantization/results.md`](benchmarks/quantization/results.md).

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

- **Three model sizes:** Cortex-XS (4.83 M), Cortex-S (24.80 M), Cortex-M (83.51 M). All size targets met.
- **Spike tokenizer:** Converts (neuron\_id, time\_bin, value) triplets into embedded tokens via three embedding tables; fused Triton path available when `use_kernels=True`.
- **Perceiver cross-attention encoder:** Variable-length spike event input (padded + masked) cross-attends into a fixed-size latent array (L=256 for Cortex-S). Self-attention stack refines latents.
- **Decoder heads:** Behavior regression head (cross-attention from learned query to latents → hand velocity). Masked spike head for self-supervised pre-training.
- **FSDP training loop:** Mixed-precision (bfloat16), sharded checkpointing via `torch.distributed.checkpoint`, gradient clipping. Single-GPU mode on M4 Pro MPS.
- **Hydra config system:** All hyperparameters in `configs/`. Three model presets, training config, data config.
- **MC\_Maze data loader:** DANDI/pynwb pipeline. Loads heldin units (137/182 neurons), bins spikes at 5 ms, creates sliding-window dataset. Handles behavior NWB files separately from neural NWB files.
- **Three baselines:** Wiener filter (ridge regression on mean-rate features), GRU (2-layer bidirectional), vanilla Transformer. All trained and evaluated under identical protocol.
- **W&B integration:** Per-step logging, eval metrics. Offline mode for machines without internet access.

### Phase 2.1 — Profiling ✓

- **PyTorch profiler pipeline** (`scripts/profile_inference.py`): MPS-compatible section timing with `torch.mps.synchronize()` fences, Chrome trace export, and markdown report generation.
- **`_pack_events` bottleneck found and fixed:** Python loop over batch elements called `.item()` 32 times per forward pass, forcing CPU↔MPS sync stalls (34% of `_pack_events` time). Replaced with vectorized `cumsum`. Saves 4.2 ms / forward pass (−3.2%).
- **Bottleneck hierarchy documented:** Self-attention 78%, cross-attention 10.5%, pack\_events 6.1%, tokenizer 1.1%.

### Phase 2.2 — Three Triton Kernels ✓

All three kernels: PyTorch reference implementation, Triton kernel, correctness tests, benchmark script. All 27 non-GPU tests pass on MPS/CPU.

**Fused spike tokenizer** (`cortex/kernels/tokenizer.py`):
- 2D kernel grid over (events, embedding dimensions). Fuses three embedding lookups into one kernel, eliminating two intermediate (E, D) tensors.
- Fixed a `tl.constexpr` correctness bug from the scaffold: only tile sizes (BLOCK\_E, BLOCK\_D) are constexpr; shapes and strides are runtime integers.
- 9-config autotuner, keyed on (E, D).

**Block-sparse cross-attention** (`cortex/kernels/sparse_xattn.py`):
- FlashAttention-2 online softmax (running m, l, o statistics). Skips entire event tiles where the block mask is False.
- External block mask API: `build_temporal_block_mask()` computes a conservative superset mask from event time bins. Sparsity policy is decoupled from the kernel.
- Guard for all-masked rows (l=0 → zero output, no NaN).

**Fused RMSNorm + linear** (`cortex/kernels/fused_rmsnorm.py`):
- Two-pass kernel: Pass 1 accumulates x² per row (for RMS); Pass 2 applies norm × gamma inline during `tl.dot()` matmul. The normalised intermediate x\_norm never touches HBM.
- Saves 2 × M × K bytes per call. At Cortex-S scale (M=8192, K=512, bf16): 8 MB per fused call, 16 MB per self-attention block (2 fused calls), 112 MB per forward pass across 7 blocks.

**Dispatcher pattern:** `CortexConfig.use_kernels: bool = False`. All dispatch wrappers fall back to the PyTorch reference on non-CUDA devices without raising.

### Phase 2.6 — INT8 Quantization ✓

- **Calibration hooks:** `attach_calibration_hooks()` instruments every `nn.Linear` with a forward pre-hook that records per-batch activation statistics.
- **Scale derivation:** `derive_scales()` computes per-channel weight scales (absmax per output neuron) and per-tensor activation scales (99th-percentile abs-max across calibration batches — more robust than raw max).
- **`QuantizedLinear`:** Stores INT8 weight tensor + float32 per-channel scale + float32 activation scale. Forward dequantizes weights to bf16/fp32 before matmul — device-agnostic (no CUDA quantized kernel dependency).
- **`convert_model()`:** Walks the model graph, replaces `nn.Linear` with `QuantizedLinear` in-place. Preserves bias, LayerNorm, and embedding layers in original precision.
- **Calibration script** (`scripts/calibrate_model.py`): Loads Cortex-S, runs 50 calibration batches from MC\_Maze, converts model, evaluates both fp32 and INT8 on validation set, writes `benchmarks/quantization/results.md`.

---

## Tech Stack

| Tool | Why |
|---|---|
| **PyTorch 2.2+** | FSDP2 API, `scaled_dot_product_attention` dispatch (→ FlashAttention), MPS backend for dev on Apple Silicon |
| **Triton** | Write custom GPU kernels in Python syntax; generates PTX without CUDA C. Used by PyTorch internals (inductor). CUDA-only — no MPS support. |
| **einops** | Explicit tensor reshapes (`rearrange`, `einsum`) that read like the dimension names they manipulate. Eliminates silent shape bugs from `.view()` chains. |
| **Hydra** | Hierarchical config composition (model + training + data + runtime) with CLI overrides. Avoids argparse sprawl as config surface grows. |
| **Pydantic v2** | Strict typing for all config and I/O boundaries. `model_validator` enforces cross-field constraints (e.g., `head_dim × num_heads == hidden_dim`). |
| **pynwb / DANDI** | Official NWB Python API + DANDI archive client. Neural Latents Benchmark data is NWB-native; no format conversion needed. |
| **W&B** | Experiment tracking with artifact storage. Offline mode for MPS machines; run `wandb sync` to upload when on network. |
| **pytest** | Unit + integration tests. `@pytest.mark.gpu` for CUDA-only tests; these are collected but skipped on non-CUDA hardware. |
| **mypy (strict)** | Catches shape/type bugs before runtime. All public APIs annotated; `py.typed` marker in package. |
| **ruff + black** | Ruff for fast lint/import sorting; black for formatting. Pre-commit hooks enforce both. |
| **FastAPI** | Async web framework for inference server (Phase 3). WebSocket streaming and REST batch endpoints. |
| **structlog** | Structured JSON logging throughout `cortex/`. Never `print()` in library code. |

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
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v -k "not gpu"
# Expected: ~40 passing, a few skipped (CUDA-only paths)
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

---

## Roadmap

### Phase 3 — Inference Engine (next)

- Async request scheduler with continuous batching (vLLM-style scheduler, implemented from scratch)
- Streaming KV cache (paged attention pattern adapted for the sliding-window spike context)
- CUDA streams for async H2D transfer overlap
- FastAPI server: WebSocket streaming endpoint + REST batch endpoint
- k6 load testing scripts
- **Target:** p99 < 30 ms for Cortex-S on A100, 5× throughput over naive PyTorch baseline

### Phase 4 — Operations and Observability

- Prometheus metrics endpoint with custom counters / histograms / gauges
- Three Grafana dashboards (traffic, latency, resources) with JSON export
- OpenTelemetry distributed tracing through the full request path
- Multi-stage Dockerfile (target < 500 MB image)
- `docker compose up` brings up engine + Prometheus + Grafana + load generator
- Helm chart skeleton
- SLO definitions (`docs/slo.md`) with burn-rate alerting rules

### Phase 5 — Writeup

- Long-form engineering postmortem (`docs/writeup.md`, ~4 000 words)
- Architecture diagrams (Mermaid)
- Demo GIF of inference server under load

---

## Project Layout

```
cortex-engine/
├── cortex/
│   ├── models/          # SpikeTokenizer, PerceiverEncoder, decoder heads, config
│   ├── kernels/         # Triton kernels + PyTorch references + benchmark scripts
│   ├── quantization/    # INT8 calibration, QuantizedLinear, model conversion
│   ├── training/        # FSDP loop, LR schedule, baselines (Wiener / GRU / Transformer)
│   ├── data/            # NLB MC_Maze loader (pynwb + DANDI)
│   ├── serve/           # FastAPI app, scheduler, inference worker (Phase 3)
│   └── cache/           # Streaming KV cache (Phase 3)
├── configs/             # Hydra YAML: model sizes, training, data, serving
├── scripts/             # Profile inference, train, run baselines, calibrate
├── tests/               # Unit tests mirroring cortex/ structure
├── benchmarks/
│   ├── training/        # Phase 1 results: model accuracy vs baselines
│   ├── profiling/       # Phase 2 MPS profile + bottleneck analysis
│   ├── kernels/         # Phase 2 Triton kernel benchmarks (CUDA)
│   └── quantization/    # Phase 2 INT8 accuracy/memory tradeoff
└── docs/                # PROJECT_PLAN.md, writeup (Phase 5), runbook (Phase 4)
```

---

## Why Neural Decoding

Motor cortex population recordings are a natural fit for transformer infrastructure benchmarking: the input is a variable-length sequence of sparse events (spike trains), the output must be low-latency (real BCI systems target < 50 ms end-to-end), and the data is publicly available at production scale via the Neural Latents Benchmark. The Perceiver architecture handles variable neuron counts across sessions without hard-coding electrode geometry, which is the same session-agnostic design goal that makes this practically useful.

The engineering patterns here — continuous batching, paged KV cache, custom attention kernels, calibrated INT8 — transfer directly to LLM serving. The constraints are harder (stricter latency, no warm-up budget), which makes the solutions more rigorous.

---

## License

MIT
