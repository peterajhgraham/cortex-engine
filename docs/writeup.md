# Building Cortex-Engine: A Real-Time Inference System for Neural Decoders

> An engineering postmortem of building production-grade inference infrastructure for transformer-based neural decoders, from custom Triton kernels through continuous-batching serving.

---

## TL;DR

I built a complete training and inference stack for transformer neural decoders, optimized down to the kernel level and deployed with full observability. The headline numbers: **TBD throughput improvement**, **TBD p99 latency**, **TBD memory reduction** at **TBD R² regression**, all on real motor cortex data from the Neural Latents Benchmark.

This post walks through the engineering decisions, the things that broke, and what I would do differently.

---

## 1. Why This Project

(TODO Phase 5.2)

Sketch:
- The latency budget for closed-loop BCI is genuinely under 50ms wall-clock
- Existing neural decoder code is research-quality; nobody has built a real serving stack
- The systems patterns from LLM serving (continuous batching, paged KV, custom kernels) transfer cleanly but require non-trivial adaptation
- Wanted to demonstrate end-to-end MLE depth on a domain where the work is novel

---

## 2. Architecture Overview

(TODO Phase 5.2)

Sketch:
- Model: Perceiver-IO style encoder over spike-event tokens
- Three sizes (XS / S / M) for scaling analysis
- Pretrained with masked spike prediction + supervised behavior loss
- Fine-tuned per task

Insert architecture diagram here.

---

## 3. Phase 1: The Training Pipeline

(TODO Phase 5.2)

Cover:
- FSDP2 setup, why fully_shard over ShardingStrategy
- Mixed precision policy: bf16 compute, fp32 master, fp32 reductions
- Custom data sharding for streaming spike data
- W&B sweeps for hyperparameter search
- Baseline numbers: Wiener filter, GRU, vanilla transformer
- Cortex-S vs baselines on R²

The interesting story: variable-length event sequences mean FSDP's standard padding-aware accumulation doesn't apply naively. Document the workaround.

---

## 4. Phase 2: Going Down to the Metal

This is the section that earns the project most of its credibility. Show the work.

### 4.1 Profiling first

Always profile before optimizing. Insert flame graph here. The top three bottlenecks identified:
1. (TODO)
2. (TODO)
3. (TODO)

### 4.2 The fused tokenizer kernel

Walk through the kernel in `cortex/kernels/tokenizer.py`. Cover:
- Why the three separate embedding lookups were a memory bandwidth bottleneck
- The tile-over-(events, dims) pattern
- Autotune configurations and what won
- Speedup curve across input sizes (insert benchmark plot)

### 4.3 Sparse cross-attention

(TODO Phase 5.2)

The most novel kernel. Spike events are temporally sparse, so a naive O(L*E) attention does substantial wasted work. Walk through the FlashAttention-2 inspired tiling adapted for the temporal sparsity pattern.

### 4.4 INT8 quantization

(TODO Phase 5.2)

- Calibration methodology (held-out trial subset, 99th percentile abs max)
- Per-channel weight quantization, per-tensor activation quantization
- SmoothQuant-style activation smoothing
- Final accuracy/memory tradeoff numbers

---

## 5. Phase 3: The Inference Engine

(TODO Phase 5.2)

The interesting design problem here: continuous batching for streaming biological signals is *not* the same as continuous batching for LLM generation. The key differences:

1. **No autoregression.** LLM batching packs requests at heterogeneous decode positions. Spike decoders run a single forward pass per window.
2. **Variable event count, fixed window length.** The heterogeneity is in spike counts per window, not tokens generated. Bucket by event count, not by sequence length.
3. **Hard deadlines.** BCI applications have physiological latency budgets. The scheduler is deadline-aware in a way that LLM schedulers usually aren't.

Cover the scheduler design, the streaming KV cache adaptation of paged attention, and the cudagraph capture for fixed batch sizes.

Insert request lifecycle diagram here.

---

## 6. Phase 4: Operations

(TODO Phase 5.2)

- Why Prometheus/Grafana/OpenTelemetry, why this combination
- The SLO methodology (multi-window, multi-burn-rate)
- Dashboard design: traffic / latency / resources
- The runbook approach
- What "production-ready" actually means at the early-career level

Insert dashboard screenshots.

---

## 7. Results

| Metric | Baseline | Cortex-Engine | Improvement |
|---|---|---|---|
| (Populate from benchmarks/) | | | |

---

## 8. What I'd Do Differently

(TODO Phase 5.2)

Be honest. The mistakes are what make this read as real engineering.

Candidates from the build:
- (Fill in real ones from the build)
- The continuous batching scheduler initially used a single FIFO queue; deadline-aware priority queue was a late addition that should have been there from day one
- Initial Triton kernel autotune configs were too narrow; one production input shape fell off the table and got 30% slower
- (etc.)

---

## 9. What This Transfers To

The infrastructure patterns map directly to LLM serving work:

- Triton kernels for spike tokenization map to attention variants for LLMs
- FSDP on a Perceiver maps to FSDP on Llama
- Streaming KV cache for spike windows maps to paged KV for LLM tokens
- Continuous batching for decoding requests maps to chat completion serving
- INT8 calibration with SmoothQuant maps directly

The Cog-neuro context made this a richer engineering problem than a generic LLM rebuild would have been: variable event counts, hard latency constraints, and streaming context all force the design to actually engage with the underlying systems problems rather than copy a known-good pattern.

---

## Code

Everything is at [github.com/peter/cortex-engine](#).

All benchmarks are reproducible: clone, `make dev-install`, `make bench-kernels`, `make docker-up && make bench-serving`.
