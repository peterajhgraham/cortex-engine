# Roadmap

Cortex-Engine is feature-complete across its five planned phases — model,
profiling-driven Triton kernels, INT8 quantization, continuous-batching
inference engine, and full observability stack. What follows is the
forward work that would make the project stronger as a research and
production artifact. Each item is scoped tightly enough to be picked up
as a single PR.

## 1. CUDA kernel benchmark sweeps on A10 (24 GB)

**Status:** correctness verified on every commit; speedup cells in
[`BENCHMARKS.md`](BENCHMARKS.md) marked *pending CUDA*.

The three Triton kernels (fused tokenizer, block-sparse cross-attention,
fused RMSNorm + linear) need a full benchmark sweep against PyTorch
eager on real A10 hardware. Run `make bench-kernels` on an A10 instance,
populate the TBD rows in `benchmarks/kernels/results.md`, generate a
roofline plot per kernel, and lock down the autotune configs at
production input shapes (E ∈ {1024, 4096, 16384, 65536}, D ∈ {128, 256,
384, 512}). The 27× block-sparse cross-attention number is indicative
from a single shape — the sweep will produce the real curve.

## 2. Full NLB MC_Maze trial-aligned evaluation with all baselines

**Status:** Cortex-S reaches R² = 0.60 trial-aligned vs Wiener 0.48 on
MPS; GRU and vanilla Transformer baselines marked *pending CUDA* in the
trial-aligned protocol.

The sliding-window baselines exist but the trial-aligned numbers for
the GRU and vanilla Transformer baselines still need to be produced
under identical evaluation protocol. After that, submit the Cortex-S
predictions to the NLB leaderboard and compare against the
Pei et al. 2021 reference points (Wiener ≈ 0.33–0.40, best 2021 entry
≈ 0.62) under the official per-bin velocity scoring rather than the
single-onset target used today.

## 3. Multi-session and cross-subject generalization

The current pipeline trains on a single MC_Maze recording. The
Perceiver-style architecture is specifically designed to handle
variable neuron counts without hard-coding electrode geometry, which
makes it the natural starting point for cross-session transfer. Add a
session-conditioning embedding, train on the full MC_Maze + MC_RTT +
Area2_Bump bundle from NLB, and report (a) zero-shot transfer to a
held-out session, and (b) few-shot fine-tune curves at 1, 10, 100, 1000
trials of new-session data. This is the experiment that turns the
project from "an inference engine" into "the inference engine for a
foundation model of motor cortex."

## 4. End-to-end true-INT8 matmul on CUDA

**Status:** weight-only INT8 with bf16 dequant-then-matmul today — 72%
weight memory reduction, no runtime activation savings.

Wire `cublasLtMatmul` (or a Triton INT8 GEMM) into `QuantizedLinear` so
the matmul itself runs in INT8 on A10 tensor cores. Re-measure on
A10 — expected wins are activation memory reduction and throughput,
both currently absent because the dequant path materializes bf16 inputs.
Add a third row to the quantization table: `INT8 (true matmul, CUDA)`.

## 5. CI on a self-hosted CUDA runner

Today's GitHub Actions CI runs lint, mypy strict, and the 116-test
non-GPU suite. The 39 CUDA-skipped Triton tests have no continuous
coverage. Adding a self-hosted A10 runner that runs `pytest -m gpu`
plus `make bench-kernels` weekly would catch kernel regressions and
keep the *pending CUDA* cells in [`BENCHMARKS.md`](BENCHMARKS.md) honest
over time.

---

Contributions welcome on any of the above. Open an issue first if the
change touches a benchmark number or a Triton kernel — the project's
honest-reporting rule applies to PRs as well as commits.
