# Claude Code Instructions: Cortex-Engine

You are building a real-time inference engine for transformer-based neural decoders. This file is your operating manual. Read it fully before starting any phase.

## Project Identity

**Name:** cortex-engine
**One-line description:** Production inference infrastructure for transformer-based neural decoders. Custom Triton kernels, FSDP training, continuous batching, sub-30ms p99 latency.
**Hero metrics to target:**
- p99 latency under 30ms for Cortex-S streaming inference on a single A100
- 5x throughput improvement over naive PyTorch baseline at saturating load
- 4x memory reduction via INT8 quantization with <1% accuracy regression
- Greater than 80% GPU utilization at peak throughput

## Core Engineering Principles

1. **Profile before optimizing.** Never write a Triton kernel for an op that takes less than 5% of inference time. Always measure first, document the bottleneck, then optimize.
2. **Every claim has a benchmark.** "5x faster" without a reproducible benchmark is marketing. Every speedup gets a script in `benchmarks/` that anyone can run.
3. **Honest reporting.** When the optimized version is slower at certain input sizes, document that. When quantization drops accuracy, report it. Hiring managers respect honest engineering more than polished demos.
4. **FSDP from day one.** Even on single GPU. The infrastructure is the point.
5. **Type everything.** Strict mypy. Pydantic for I/O boundaries. The codebase should read like production code, not research code.
6. **Tests are not optional.** Every kernel has a correctness test against a PyTorch reference. Every API endpoint has integration tests. Smoke tests for the training loop.

## Phase Plan and Order of Operations

Execute strictly in order. Do not start Phase N+1 until Phase N's deliverables are complete and benchmarks are recorded.

### Phase 1: Model + Training Pipeline (Weeks 1-3)

**What you must produce:**
- Three model size variants: Cortex-XS (~5M), Cortex-S (~25M), Cortex-M (~80M)
- Spike tokenizer that converts (neuron_id, time, value) triplets into embedded tokens
- Perceiver-style cross-attention encoder
- Decoder heads for: behavior regression, masked spike prediction
- FSDP training loop with mixed precision (bf16) and sharded checkpointing
- W&B integration: per-step logging, eval metrics, artifact storage
- Hydra config system (already scaffolded; just fill in)
- NLB MC_Maze dataset loader via DANDI/pynwb
- Three baselines: Wiener filter (ridge), GRU sequence model, vanilla Transformer
- Benchmarks documented in `benchmarks/training/results.md`

**Definition of done for Phase 1:**
- All three model sizes train to convergence on MC_Maze
- Cortex-S beats all three baselines on R² for hand velocity decoding
- W&B run links pasted into the training results doc
- Sharded checkpoints saved and loadable
- A `make train-s` command produces a trained Cortex-S checkpoint reproducibly

### Phase 2: Going Down to the Metal (Weeks 4-6)

**What you must produce:**
- Profiling reports (PyTorch profiler + nsys if available) for inference and training
- Three Triton kernels:
  1. Fused spike tokenizer (embedding lookup + position encoding + value scaling)
  2. Sparse cross-attention (exploits temporal sparsity of spike events)
  3. Fused RMSNorm + linear projection
- Each kernel: PyTorch reference, Triton implementation, correctness test, benchmark sweep
- INT8 quantization with calibration on a held-out trial subset
- Quantization accuracy/memory tradeoff documented

**Definition of done for Phase 2:**
- Each kernel demonstrates measurable speedup at production input shapes
- Numerical equivalence to reference within `rtol=1e-3, atol=1e-3`
- Profiling reports show the bottlenecks moved to expected places
- INT8 model loses less than 1% R² vs fp16

### Phase 3: Inference Engine (Weeks 6-9)

**What you must produce:**
- Async request scheduler with continuous batching
- Streaming KV cache (paged attention pattern adapted for sliding-window spike context)
- Inference worker using CUDA streams for overlap
- FastAPI server with WebSocket streaming endpoint and REST batch endpoint
- Pydantic schemas for all I/O
- k6 load testing scripts
- Latency distribution analysis at multiple concurrent loads

**Definition of done for Phase 3:**
- Hero p99 latency target met (under 30ms for Cortex-S)
- 5x throughput improvement over naive PyTorch baseline documented
- Load test reports in `benchmarks/serving/`
- Request lifecycle diagram in README

### Phase 4: Operations and Observability (Weeks 8-10)

**What you must produce:**
- Prometheus metrics endpoint with custom counters/histograms/gauges
- Three Grafana dashboards (traffic, latency, resources) with JSON exports
- OpenTelemetry tracing through the full request path
- Structured logging via structlog
- Alertmanager rules for SLO violations
- Multi-stage Dockerfile (target under 500MB)
- docker-compose stack: engine + Prometheus + Grafana + load generator
- Helm chart skeleton (manifests must validate, full deployment optional)
- Runbook docs

**Definition of done for Phase 4:**
- `docker compose up` brings up the full stack and shows live metrics in Grafana
- SLO definitions documented in `docs/slo.md`
- Dashboard screenshots in README

### Phase 5: Writeup (Week 11)

**What you must produce:**
- Top-level README.md polished to 90-second readability (hero figure, headline, results table, links)
- Long-form blog post in `docs/writeup.md` (~4000 words, engineering postmortem tone)
- Architecture diagrams (use mermaid)
- Demo gif/video of inference engine in action

## File and Directory Conventions

- **Module organization:** Each top-level concern is its own subpackage under `cortex/`. Imports between subpackages should be unidirectional (training imports models, serve imports models, but models imports nothing from the others).
- **Configs live in `configs/`** as Hydra YAML. Never put magic numbers in code; if it's a hyperparameter, it goes in a config.
- **Tests mirror source structure.** `cortex/models/cortex.py` is tested by `tests/unit/test_models_cortex.py`.
- **Benchmarks live in `benchmarks/`** with one subdirectory per phase concern. Each benchmark produces a markdown report and ideally a JSON dump of raw numbers for later comparison.
- **Documentation is in `docs/`.** Public-facing in README, deeper material in `docs/`.

## Coding Standards

- Python 3.11+
- Strict typing: mypy --strict on the cortex package
- Black formatting, ruff linting, isort imports (configured in pyproject.toml)
- Pre-commit hooks must pass before any commit
- All public functions have docstrings with shapes annotated for tensor inputs/outputs
- Use einops for any tensor reshape that involves more than two dimensions
- Logging: never use print in cortex/. Use structlog.

## Tools and Dependencies (already pinned in pyproject.toml)

**Core:** torch, triton, einops, hydra-core, pydantic, structlog
**Data:** pynwb, dandi, numpy, pandas
**Training:** wandb, accelerate (for utilities, not the primary training driver)
**Serving:** fastapi, uvicorn, prometheus-client, opentelemetry-api
**Testing:** pytest, pytest-asyncio, hypothesis
**Quality:** mypy, ruff, black, pre-commit

## What NOT to Do

- Do not use `accelerate launch` as the primary training entry point. Implement FSDP directly. The point is to demonstrate you understand it.
- Do not use bitsandbytes one-liners for quantization. Implement calibration manually.
- Do not use `print` for logging. Use structlog.
- Do not catch broad exceptions. Catch specific ones with context.
- Do not write a Triton kernel before profiling proves the op is a bottleneck.
- Do not skip the baselines. The 5x speedup claim requires a 1x measurement on the same hardware.
- Do not use Flask, Django, or any sync web framework. FastAPI with async only.
- Do not commit large files. Use git-lfs or W&B artifacts for checkpoints.
- Do not import torch in code that is supposed to be torch-free (e.g., the schedulers' core logic).

## Workflow Per Phase

1. Read the phase section in this file.
2. Read the corresponding deliverables in `docs/PROJECT_PLAN.md` (extended detail).
3. Write the tests first where possible, especially for kernels (correctness vs reference).
4. Implement.
5. Run benchmarks. Record numbers in the relevant `benchmarks/` markdown.
6. Update the README with a one-line summary of what was completed.
7. Commit with a message that references the phase: `[phase-2] Triton fused tokenizer kernel`.

## When You Are Stuck

- For Triton: reference the official Triton tutorials and Liger Kernel for patterns.
- For FSDP: PyTorch FSDP tutorials. The FSDP2 API (torch >=2.5) is preferred.
- For continuous batching: read the vLLM source, specifically the scheduler. Understand the concept, then implement from scratch.
- For NLB data: nlb_tools repo on GitHub plus the Pei et al. 2021 paper.
- For Perceiver IO: Jaegle et al. 2021 paper, then the deepmind reference implementation.

If a benchmark target is not met, document the actual numbers honestly and explain why. Honest reporting beats overclaiming every time.

## Success Criteria for the Entire Project

The repo is "done" when an ML Lead opening it for the first time can:

1. In 30 seconds: understand what was built and see the hero metrics
2. In 2 minutes: read the README and understand the architecture
3. In 10 minutes: read the writeup and understand the engineering decisions
4. In 30 minutes: clone, run `make demo`, and see the inference engine respond to a request

If any of those four checkpoints fails, the corresponding artifact (README, writeup, demo) needs more work.

Begin with Phase 1, item 1: implement the spike tokenizer in `cortex/models/tokenizer.py`. The Hydra config and the test stub are already in place. Make the test pass, then move to the next file.
