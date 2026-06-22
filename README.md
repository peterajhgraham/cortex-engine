# Cortex-Engine

Real-time inference infrastructure for transformer-based neural decoders, built from the GPU kernels up.

A closed-loop brain-computer interface has a hard latency budget — a decoder that misses its deadline is a cursor that lags. The signal is awkward for accelerators: hundreds of neurons firing asynchronously, a population code spread across cells and time in shapes that don't fit a fixed tensor. Cortex-Engine is a Perceiver-style decoder that compresses variable-length spike events into a fixed latent set, three Triton kernels for the layers profiling flagged as bottlenecks, per-channel INT8 quantization, and a continuous-batching server with a paged streaming KV cache — trained on real motor cortex data from the [Neural Latents Benchmark](https://neurallatents.github.io/).

## Results

Decoding accuracy, trial-aligned NLB protocol (one sample per reach, velocity at `move_onset_time`):

| Model | Params | R² (hand velocity) |
|---|---|---|
| Wiener filter (ridge) | 137 × 2 | 0.48 |
| **Cortex-S** | **24.8 M** | **0.60** |

Serving latency, Cortex-S on an NVIDIA A10, concurrency 8:

| Throughput | p50 | p95 | p99 (steady) | Failures |
|---|---|---|---|---|
| 255 req/s | 27 ms | 28 ms | ~28 ms | 0 / 500 |

INT8 quantization gives a 72% weight-memory reduction (99.2 → 27.8 MB) at 0.003 max weight error. The three Triton kernels are correctness-checked against PyTorch references (`rtol=atol=1e-3`) on every commit; block-sparse cross-attention hits up to 27× on the A10.

Full numbers, profiling breakdowns, and the GRU / vanilla-Transformer baselines are in [`BENCHMARKS.md`](BENCHMARKS.md). The engineering story is in [`docs/writeup.md`](docs/writeup.md).

> Numbers tagged *MPS* (Apple Silicon dev machine) are real measurements; the hero numbers — Triton speedups, p99 < 30 ms, full Cortex-M FSDP — need CUDA and are marked *pending CUDA* rather than estimated.

## Architecture

```
Spike events (neuron_id, time_bin, value)
        │
        ▼
  SpikeTokenizer          fused embedding gather → (E, D) tokens   [Triton]
        │
        ▼
  Cross-attention         L latent queries × E spike tokens        [Triton, block-sparse]
        │
        ▼
  Self-attention × N      RMSNorm → QKV → SDPA → MLP                [Triton, fused RMSNorm+linear]
        │
        ▼
  Decoder heads           behavior (hand velocity) · masked-spike (SSL)
```

The latent bottleneck is what makes the model session-agnostic: different recordings have different electrode counts, but the fixed latent array abstracts over that. The serving stack (EDF scheduler, paged KV cache, FastAPI) wraps this model and is instrumented end-to-end with Prometheus / Grafana / OpenTelemetry.

## Quickstart

```bash
make dev-install                 # .venv + deps + pre-commit
make test-fast                   # CPU test suite
make train-s                     # train Cortex-S (needs MC_Maze data)
make serve                       # inference server on :8080
make docker-up                   # full stack: engine + Prometheus + Grafana
```

`make help` lists every target. Training needs the MC_Maze NWB files:

```python
from cortex.data.nlb import download_mc_maze
download_mc_maze("data/mc_maze")
```

## Layout

```
cortex/
  models/        tokenizer, perceiver, cortex (XS/S/M)
  kernels/       Triton kernels + benchmarks
  quantization/  per-channel INT8 calibration
  training/      FSDP loop, eval, checkpointing, baselines
  data/          NLB / MC_Maze loader (pynwb + DANDI)
  serve/         FastAPI app, EDF scheduler, worker, metrics, tracing
  cache/         paged streaming KV cache
configs/         Hydra configs (model / data / training / serving)
benchmarks/      per-concern results + raw JSON
ops/             Dockerfile, docker-compose, Grafana dashboards, Helm, k6
docs/            writeup, roadmap, runbook, SLOs
```

## License

MIT
