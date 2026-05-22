# Cortex-Engine — Consolidated Benchmarks

A single reference for every measured number in this repo. Each row links to the
full report with raw JSON, hardware details, and reproduction commands.

> **Hardware caveat.** All measurements taken on Apple Silicon are labeled
> **MPS**; they validate infrastructure correctness and characterize the
> model on the dev box. Triton kernel speedups, full p99 < 30 ms serving,
> and Cortex-M training require **NVIDIA A10 (24 GB)** and are explicitly
> marked **pending CUDA** when not yet measured. Kernel correctness is
> verified against PyTorch references within `rtol=1e-3, atol=1e-3` on
> every commit.

---

## Headline Table

| Area | Metric | Value | Hardware | Status | Report |
|---|---|---|---|---|---|
| Decoding | R² hand velocity, Cortex-S (trial-aligned) | **0.60** | MPS | measured | [training/trial_aligned](benchmarks/training/trial_aligned_results.md) |
| Decoding | R² hand velocity, Wiener baseline | 0.48 | MPS | measured | [training/trial_aligned](benchmarks/training/trial_aligned_results.md) |
| Decoding | R² GRU / vanilla Transformer baselines | pending CUDA | A10 | pending | [training/trial_aligned](benchmarks/training/trial_aligned_results.md) |
| Profiling | Cortex-S full forward, batch=32 | 129.0 ms | MPS | measured | [profiling/baseline](benchmarks/profiling/baseline_report.md) |
| Profiling | Self-attn share of forward (Triton target #1) | 78.0% | MPS | measured | [profiling/baseline](benchmarks/profiling/baseline_report.md) |
| Profiling | `_pack_events` fix (Python loop → cumsum) | −4.2 ms (−3.2%) | MPS | measured | [profiling/baseline](benchmarks/profiling/baseline_report.md) |
| Kernel | Fused tokenizer (3 embeds → 1 kernel) | correctness ✓, speedup pending CUDA | A10 | pending | [kernels](benchmarks/kernels/results.md) |
| Kernel | Block-sparse cross-attention (FA2 + tile skip) | up to **27×** vs eager | A10 | indicative | [kernels](benchmarks/kernels/results.md) |
| Kernel | Fused RMSNorm + linear | 112 MB/forward saved, speedup pending CUDA | A10 | pending | [kernels](benchmarks/kernels/results.md) |
| Quantization | Weight memory, fp32 → INT8 per-channel | 99.2 MB → **27.8 MB** (−72%) | MPS | measured | [quantization](benchmarks/quantization/results.md) |
| Quantization | Max abs output diff fp32 vs INT8 | 0.003 | MPS | measured | [quantization](benchmarks/quantization/results.md) |
| Quantization | Layers covered | 34 / 35 nn.Linear | MPS | measured | [quantization](benchmarks/quantization/results.md) |
| Serving | Throughput, concurrency=8 | **255 req/s** | A10 (in-proc) | measured | [serving](benchmarks/serving/results.md) |
| Serving | p50 / p95 latency | 27 ms / 28 ms | A10 (in-proc) | measured | [serving](benchmarks/serving/results.md) |
| Serving | p99 latency (incl. first-batch JIT spike) | 261 ms | A10 (in-proc) | measured | [serving](benchmarks/serving/results.md) |
| Serving | p99 steady-state | ~28 ms (SLO < 30 ms ✓) | A10 (in-proc) | measured | [serving](benchmarks/serving/results.md) |
| Serving | Failures | 0 / 500 | A10 (in-proc) | measured | [serving](benchmarks/serving/results.md) |

---

## Decoding accuracy — trial-aligned (NLB protocol)

One sample per reach trial, window −100 ms to +500 ms around `move_onset_time`,
target = hand velocity at onset.

| Model | Params | R² (hand velocity) |
|---|---|---|
| Wiener filter (ridge) | 137 × 2 | 0.48 |
| GRU (2-layer bidir) | ~660 K | pending CUDA |
| Vanilla Transformer | ~5 M | pending CUDA |
| **Cortex-S** | 24.80 M | **0.60** |

Sliding-window R² values are near zero because ~85% of windows are rest
periods — the evaluation distribution is wrong, not the models. Trial-aligned
above is the correct comparison. Sliding-window numbers preserved for the
record in [`benchmarks/training/results.md`](benchmarks/training/results.md).

---

## Profiling — Cortex-S forward, batch=32, 512 events/sample (MPS)

| Section | Time (ms) | % of forward | Triton candidate |
|---|---|---|---|
| **Full forward** | **129.0** | 100% | — |
| Self-attention (×7) | 100.6 | 78.0% | YES (RMSNorm+linear fuse) |
| Cross-attention | 13.6 | 10.5% | YES (block-sparse kernel) |
| `_pack_events` | 7.8 | 6.1% | fixed in Python (cumsum) |
| Behavior head | 3.0 | 2.3% | no (< 5% rule) |
| Tokenizer | 1.4 | 1.1% | YES (fuses 3 embeds anyway) |

Profiling chose the Triton targets — not intuition. Full report:
[`benchmarks/profiling/baseline_report.md`](benchmarks/profiling/baseline_report.md).

---

## Triton kernels

| Kernel | What it fuses | Memory saved | Speedup (A10) |
|---|---|---|---|
| Fused tokenizer | 3 embedding lookups → 1 kernel | eliminates 2 × (E, D) intermediates | pending CUDA |
| Sparse cross-attention | FA2 online softmax + block sparsity | skips masked event tiles | up to **27×** |
| Fused RMSNorm + linear | norm + matmul → 1 kernel, x_norm never touches HBM | 112 MB / forward at Cortex-S scale | pending CUDA |

All three are correctness-verified against PyTorch references in
`tests/unit/test_kernels_*.py` within `rtol=1e-3, atol=1e-3`. Run with
`make bench-kernels` on a CUDA host. Full report:
[`benchmarks/kernels/results.md`](benchmarks/kernels/results.md).

---

## INT8 quantization (MPS, Cortex-S)

| Configuration | Memory | Reduction |
|---|---|---|
| float32 | 99.2 MB | — |
| INT8 per-channel | **27.8 MB** | **−72%** |

- Max abs output diff fp32 vs INT8: **0.003**
- Mean abs output diff: 0.001
- Quantized layers: 34 / 35 `nn.Linear`
- Calibration: 50 batches, 99th-percentile abs-max per activation, symmetric
  INT8 per-output-channel for weights.

Full report: [`benchmarks/quantization/results.md`](benchmarks/quantization/results.md).

---

## Serving (NVIDIA A10, in-process, concurrency=8)

| Metric | Value |
|---|---|
| Throughput | **255 req/s** |
| p50 | 27 ms |
| p95 | 28 ms |
| **p99** | 261 ms (first-batch JIT spike); ~28 ms steady-state |
| Failures | 0 / 500 |
| Batch timeout | 5 ms |
| Max batch | 32 |

Mode is `in_process_direct` — latency covers scheduler queue wait +
inference, not HTTP serialization. Full report:
[`benchmarks/serving/results.md`](benchmarks/serving/results.md).

---

## Reproducing everything

```bash
make dev-install                                     # venv + deps
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q    # 116 passed, 39 CUDA-only skipped

PYTHONPATH=. .venv/bin/python scripts/profile_inference.py        # profiling
PYTHONPATH=. .venv/bin/python scripts/baseline_benchmark.py       # baselines
PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py --synthetic  # INT8
PYTHONPATH=. .venv/bin/python scripts/load_test.py --concurrency 8 --requests 500
make bench-kernels                                                 # CUDA only
```
