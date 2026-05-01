# Cortex-Engine: A Real-Time Neural Decoding Inference System

> Built from the GPU kernels up. Custom Triton ops. FSDP training. INT8 quantization with calibration. Continuous batching inference engine. Sub-30ms p99 latency on a single A100. Full observability. The kind of infrastructure that LLM serving teams build, applied to an architecture where the systems problems are arguably harder.

---

## What This Is, In One Paragraph

A complete training and inference stack for transformer-based neural decoders, engineered for production deployment. The training pipeline uses FSDP with custom data sharding for streaming spike data and W&B-tracked hyperparameter sweeps. The model is optimized with Triton kernels for the bottleneck ops, INT8 quantization with proper calibration, and KV-cache management adapted for streaming biological signals. The inference engine implements continuous batching, async request handling, and runs as a FastAPI service behind a full Prometheus/Grafana observability stack. The hero benchmark: 5x throughput improvement and 4x memory reduction over a vanilla PyTorch baseline, with measurable wins documented at every layer of the stack.

---

## Why This Is the Right Project

### Technical depth, in the dimensions that matter in 2026

| Dimension | What you'll demonstrate |
|---|---|
| GPU programming | Custom Triton kernels for tokenization and fused attention variants |
| Distributed training | FSDP, mixed precision, gradient accumulation, sharded checkpointing |
| Model optimization | Quantization (INT8 + calibration), kernel fusion, profiling-driven decisions |
| Inference systems | Continuous batching, paged KV cache adapted for streaming, async serving |
| Production operations | Prometheus, Grafana, distributed tracing, SLO-driven load testing |
| Engineering taste | Documented tradeoffs, reproducibility, real benchmarks, honest limitations |

### The transferability story

Every component of this stack maps directly to LLM serving work:

- Triton kernels for spike tokenization → Triton kernels for attention variants
- FSDP on a neural decoder → FSDP on Llama
- Streaming KV cache for spike events → paged KV cache for LLM tokens
- Continuous batching for decoding requests → continuous batching for chat completions
- Quantization with calibration → INT8 / INT4 quantization for model weights

You're not learning narrow skills. You're learning the same skills LLM serving teams use, applied to a domain that makes them more impressive precisely because the patterns are not pre-baked into existing libraries.

### The flair

The project decodes motor cortex activity into behavioral output. The latency target (sub-30ms) is real for BCI applications, not arbitrary. The cog neuro background informs which optimizations are acceptable (e.g., quantization noise tradeoffs against neural noise floors). The Blackrock Neurotech experience makes the deployment scenario credible. None of this is the meat. All of it is the seasoning that makes hiring managers remember the repo three days later.

---

## The Architecture, At a Glance

```
                    ┌─────────────────────────────────────┐
                    │        TRAINING PIPELINE            │
                    │  FSDP + W&B + Hydra + Custom Data   │
                    │           Sharding                  │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │      MODEL  (Cortex-XS / S / M)     │
                    │  Spike Tokenizer + Perceiver-style  │
                    │     Cross-Attn + Decoder Heads      │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │      OPTIMIZATION LAYER             │
                    │ Triton Kernels + INT8 Quant + Fused │
                    │   Ops + Streaming KV Cache          │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │       INFERENCE ENGINE              │
                    │  Continuous Batching + Async I/O +  │
                    │     FastAPI + Request Scheduler     │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │      OBSERVABILITY + OPS            │
                    │  Prometheus + Grafana + OpenTel +   │
                    │      k6 Load Tests + SLO Burn       │
                    └─────────────────────────────────────┘
```

Each layer is its own subdirectory in the repo, with its own README, its own tests, its own benchmarks.

---

## Phase 1: The Model (Weeks 1 to 3)

**Goal:** Train a clean transformer-based neural decoder. Three model sizes (XS, S, M). Real data. Real baselines. FSDP from day one even on a single GPU, because the infrastructure is the point.

### What gets built

**Model architecture:** A Perceiver-style encoder that ingests spike events as `(neuron_id, time, value)` triplets and decodes via cross-attention from learned query vectors. Three sizes to enable scaling analysis later.

```
Cortex-XS:  ~5M params,  4 layers, 128 dim,  2 heads
Cortex-S:   ~25M params, 6 layers, 256 dim,  4 heads
Cortex-M:   ~80M params, 8 layers, 384 dim,  6 heads
```

**Training stack:**
- FSDP wrapper even for single-GPU runs (the muscle is the point)
- Hydra config system with a clear separation of model / optim / data / runtime
- Mixed precision (bf16 weights, fp32 master) with proper loss scaling
- Gradient accumulation for effective batch size scaling
- Sharded checkpointing via `torch.distributed.checkpoint`
- W&B integration: per-step metrics, per-epoch eval, hyperparameter sweeps via Sweeps API

**Data:** Neural Latents Benchmark `MC_Maze` and `MC_RTT` datasets via DANDI. Custom streaming dataloader that handles the trial structure and supports both supervised (behavior decoding) and self-supervised (masked spike prediction) objectives.

**Baselines for comparison:**
1. Wiener filter with ridge regularization (the canonical motor BCI baseline)
2. GRU sequence model
3. Vanilla transformer without the Perceiver tokenization

### Deliverables for Phase 1

- `cortex/models/`: clean implementations, shape-annotated, einops-heavy
- `cortex/train.py`: FSDP training loop with checkpointing
- `configs/`: Hydra configs for all three model sizes and the baselines
- `benchmarks/training.md`: training throughput, memory, time-to-target-loss for each model size
- `benchmarks/accuracy.md`: R² and decoding accuracy for all model sizes vs baselines

### What this signals

You can train a model. You can do it with the infrastructure that real labs use. You can compare it to baselines and report numbers honestly. This is table stakes but most candidates do not actually clear it.

---

## Phase 2: Going Down to the Metal (Weeks 4 to 6)

**Goal:** Profile aggressively. Find the bottlenecks. Write custom Triton kernels for them. Quantize with proper calibration. Document every decision with before/after numbers.

This is the phase that separates "trained a model" from "engineered a system." Every move has a measured impact.

### Profiling pass

Run `nsys profile` and the PyTorch profiler on a representative inference workload. Generate flame graphs. Identify the top three operations by time. Document them in `profiling/baseline_report.md`.

Expected bottlenecks (your mileage may vary):
1. The spike tokenization pass (gathering across irregular event lists)
2. The cross-attention from spike tokens into the latent array
3. Layer norm and activation fusion opportunities

### Triton kernel implementations

Write three custom Triton kernels, with documented performance wins:

**Kernel 1: Fused spike tokenization.** Combines neuron-id embedding lookup, time-position encoding, and value scaling into a single kernel. Avoids three separate memory passes.

**Kernel 2: Sparse cross-attention.** Spike events are sparse in time. A naive attention computes a lot of zeros. Write a variant that respects the sparse temporal structure. (This is the most novel piece. Look at FlashAttention-style tiling for inspiration.)

**Kernel 3: Fused RMSNorm + projection.** Standard fusion pattern. Easy win. Include it for completeness and because it shows you know the canonical optimizations.

For each kernel:
- Reference PyTorch implementation
- Triton implementation
- Correctness test (assert numerical equivalence within tolerance)
- Benchmark across input sizes
- A markdown writeup with the memory access pattern explained and a roofline analysis

### Quantization

INT8 weights with calibration. Use either GPTQ-style quantization (one-shot, post-training) or a simple SmoothQuant variant adapted for the Perceiver attention. Document the choice and why.

Calibration set: a held-out subset of the training trials. Run forward passes, collect activation statistics, choose scale factors per channel.

Critical: report quantization-induced accuracy loss honestly. Acceptable target: <1% R² drop for 4x memory reduction.

### Deliverables for Phase 2

- `cortex/kernels/`: all Triton kernels with tests and benchmarks
- `cortex/quantization/`: calibration code, quantized weight storage format
- `benchmarks/kernels.md`: speedup numbers per kernel, with input-size sweeps
- `benchmarks/quantization.md`: accuracy vs memory tradeoff curves
- `profiling/`: before and after profiling reports

### What this signals

You can write GPU code. You can profile and prove your work made things faster. You understand quantization beyond `bitsandbytes.load_in_8bit=True`. You have the engineering discipline to measure everything.

---

## Phase 3: The Inference Engine (Weeks 6 to 9, overlapping)

**Goal:** Build a real serving system. Continuous batching, async request handling, proper backpressure, streaming output. Not a Flask wrapper. A real engine.

### What "real" means here

The naive serving loop is: request comes in, model runs, response goes out. The real serving loop is: requests arrive asynchronously, get queued, get batched dynamically with other in-flight requests at compatible stages, executed on the GPU in batches that maximize utilization, and streamed back as outputs become available.

This is what vLLM and SGLang do for LLMs. You will build the equivalent for neural decoding.

### Components

**Request scheduler:** async, priority-aware, with a deadline-based policy (since BCI applications have hard latency budgets). Implements continuous batching: at every model iteration, decide whether to add new requests to the running batch and whether to evict completed ones.

**Streaming KV cache:** adapted from paged attention concepts, but for streaming biological signals where the "context" is a sliding window of recent spike events rather than a growing token sequence. Page size tuned per model size.

**Inference worker:** runs the optimized model from Phase 2 inside an asyncio event loop, communicating with the scheduler via queues. Uses CUDA streams for overlapping data transfer with compute.

**FastAPI server:** Pydantic-validated request and response schemas. WebSocket endpoint for streaming inference. REST endpoint for batch inference. Health and readiness endpoints. OpenAPI docs auto-generated.

**Benchmarking harness:** a separate process that hammers the API and measures latency distributions, throughput, and tail behavior. Use locust or k6.

### Deliverables for Phase 3

- `cortex/serve/`: scheduler, worker, FastAPI app
- `cortex/cache/`: paged KV cache implementation
- `benchmarks/serving.md`: latency distributions (p50, p95, p99, p99.9), throughput curves, batch size effects
- `benchmarks/load_tests/`: k6 scripts and result reports
- A request lifecycle diagram in the README

### Hero numbers to target

- **p99 latency under 30ms** for single-request streaming inference on Cortex-S
- **5x throughput improvement** over the naive PyTorch baseline at sustained load
- **>80% GPU utilization** at saturating throughput
- **No accuracy regression** vs the unoptimized model

These are realistic targets. They are also the kind of numbers that make ML Leads stop scrolling.

### What this signals

You can build infrastructure. Not "I deployed a Streamlit app." Real infrastructure that handles concurrency correctly, that respects backpressure, that has been load-tested. Your Palantir background reads as verified the moment they see this.

---

## Phase 4: Operations and Observability (Weeks 8 to 10, overlapping)

**Goal:** Make the system operable. Real metrics. Real dashboards. Real alerting. Defined SLOs and demonstrated SLO burn analysis.

### The stack

- **Prometheus** for metrics scraping. The FastAPI server exposes a `/metrics` endpoint via `prometheus-client` with custom metrics: request rate, latency histograms, batch sizes, GPU utilization, cache hit rates, queue depth.
- **Grafana** for dashboards. Three dashboards minimum: traffic overview, latency analysis, resource utilization. Screenshots in the repo.
- **OpenTelemetry** for distributed tracing. Trace a request from API entry through scheduler queue, through inference worker, back to response. Critical for debugging tail latency.
- **Structured logging** via `structlog` with request IDs propagated across async boundaries.
- **Alertmanager rules** for SLO violations.

### SLOs (defined explicitly in the repo)

- 99% of requests complete in <50ms
- 99.9% availability (excluding planned maintenance)
- Error budget: 30 minutes per month

Document how SLO burn would be detected and what the response would be. This is not theater. It is how production systems are actually run, and showing you think this way is differentiating.

### Deployment

- Docker multi-stage build that produces a minimal runtime image (~500MB target)
- `docker-compose.yml` that brings up the inference engine plus Prometheus plus Grafana plus a load generator
- A Helm chart skeleton (does not need to actually deploy to a real cluster, but the manifests should be real and validated)
- A `Makefile` with one-command targets: `make train`, `make serve`, `make benchmark`, `make load-test`

### Deliverables for Phase 4

- `ops/`: all infrastructure code
- `ops/dashboards/`: Grafana dashboard JSON exports
- `docs/runbook.md`: what to do when things break
- `docs/slo.md`: SLO definitions and burn analysis methodology

### What this signals

You think in terms of production systems, not toy projects. You know what observability actually means. You can be put on-call and be useful. This is the box almost nobody checks at the early-career level.

---

## Phase 5: The Writeup (Week 11)

**Goal:** A blog post that closes the loop and a README that does the work in 90 seconds.

### The blog post

~4000 words. Sections:
1. Motivation (why real-time neural decoding, why this is hard)
2. Architecture overview
3. Training pipeline (Phase 1 highlights)
4. Optimization deep dive (Phase 2: kernel walkthrough with code snippets and benchmarks)
5. Serving infrastructure (Phase 3: the lifecycle diagram, the continuous batching explanation)
6. Operations (Phase 4: dashboards, SLOs, what would happen at 10x scale)
7. Results (the hero benchmark table)
8. What I'd do differently
9. What this transfers to (the LLM serving parallels, made explicit)

Tone: engineering postmortem, not marketing. Show the weird bug you found in week 6. Show the kernel that you had to rewrite three times. The honesty is what makes it credible.

### The README

Opens with: hero figure (the latency distribution), one paragraph stating what was built, a results table, a quickstart command, and links to the blog post and the deeper docs.

A reader should know within 30 seconds whether they want to keep reading. If yes, the rest of the README delivers.

---

## Realistic Timeline

| Phase | Weeks | What's done by end |
|---|---|---|
| Phase 1: Model + training | 1-3 | Three model sizes trained, baselines beaten, benchmarks documented |
| Phase 2: Optimization | 4-6 | Custom kernels, INT8 quantization, profiling reports |
| Phase 3: Inference engine | 6-9 (overlap) | Real serving with continuous batching, hero latency numbers |
| Phase 4: Ops | 8-10 (overlap) | Full observability, SLOs, deployment artifacts |
| Phase 5: Writeup | 11 | Blog post, README polish, demo |

Total: ~11 weeks at 10-15 hrs/week. Aggressive but achievable with tight scoping. If real life intervenes, the natural cut points are after Phase 2 (you have a complete optimized model) or after Phase 3 (you have a complete system, just unobserved).

---

## What an ML Lead Sees in 90 Seconds

The reader lands on the repo. In order:

1. The header image: a Grafana dashboard showing live latency percentiles
2. The headline: "Real-time inference engine for transformer-based neural decoders. Custom Triton kernels. FSDP training. Continuous batching. Sub-30ms p99 latency. 5x throughput over PyTorch baseline."
3. A four-row results table: latency, throughput, memory, accuracy. Each compared to baseline.
4. A "How it's built" section: Triton, FSDP, Hydra, Pydantic, FastAPI, Prometheus, Grafana, k6
5. A request lifecycle diagram
6. A link to the writeup
7. Somewhere down the page, almost as an aside, a sentence that explains the model decodes motor cortex activity

By the time they reach the neuro context, they have already cataloged you as someone who can build serving infrastructure. The cog neuro becomes the reason they remember you, not the reason they take you seriously.

That sequence (technical credibility first, domain flair second) is the entire psychological strategy of the repo.

---

## What Makes This Genuinely Cracked

Most early-career portfolios stop at "I trained a model." Some reach "I deployed it with FastAPI." Almost none touch:

- Custom GPU kernels they wrote, with profiling evidence
- FSDP training they actually configured
- Quantization with proper calibration, not a one-line bitsandbytes call
- A continuous batching scheduler they implemented from first principles
- A real observability stack with defined SLOs

This project does all of them. And it does them on a domain where the work is novel because the patterns are not already memorized into existing libraries. That's the combination that makes the repo memorable.

---

## Failure Modes to Avoid

**Optimizing the wrong thing.** Profile first. Always. Do not write a Triton kernel for an op that takes 2% of inference time.

**Skipping the baselines.** A 5x speedup is meaningless without the 1x baseline measured the same way on the same hardware.

**Cargo-culting LLM patterns.** Paged KV cache for LLMs has specific reasons. Apply the concept thoughtfully, not literally. The streaming nature of spike data changes the page eviction policy.

**Making the README a wall of text.** It should hand the reader the punch in 30 seconds. The depth lives in `docs/` and in the blog post.

**Starting the writeup last.** Take notes throughout. The blog post writes itself if you've kept a running engineering log per phase. It becomes a brutal week-of-effort if you start from zero in week 11.

**Hiding the imperfections.** "Here is what didn't work and why" is the section that makes the project read as real engineering rather than as a polished demo.

---

## Final Note

When this is on your GitHub, your story changes. You are no longer a candidate hoping to break into ML. You are someone who has shipped the kind of system the team is hiring you to build. The interview becomes a conversation about which of your engineering decisions you'd revisit, not a defense of whether you have relevant experience.

That is the asymmetry the project creates. Build it.
