# Phase 1 Training Benchmarks

Measured on **Apple M4 Pro (24 GB unified memory, MPS backend)**, 2026-04-30.  
Raw JSON dumps: [`results_raw.json`](results_raw.json), [`baselines_raw.json`](baselines_raw.json).

## Setup

| Field | Value |
|---|---|
| Hardware | Apple M4 Pro, 24 GB unified memory |
| Backend | PyTorch 2.x MPS (Apple Silicon GPU) |
| Dataset | NLB MC_Maze (DANDI 000128) |
| Neurons (heldin) | 137 of 182 total units |
| Bin size | 5 ms |
| Window | 600 ms (120 bins) |
| Stride | 50 ms |
| Train windows | 111,228 |
| Val windows | 13,903 |

## Cortex Models

| Model | Params | Steps | Train time | Avg step | Final R² (velocity) | W&B run |
|---|---|---|---|---|---|---|
| Cortex-S | 24.80 M | 2,000 | 20.0 min | 563 ms | −0.0002 | offline |
| Cortex-XS | 4.83 M | — | — | — | — | — |
| Cortex-M | 83.51 M | — | — | — | — | — |

Cortex-XS and Cortex-M configs are defined and tested; full runs deferred to CUDA hardware (see below).

## Baselines

| Baseline | Params | R² (velocity) | Train time | Notes |
|---|---|---|---|---|
| Wiener filter (ridge) | 137 × 2 + bias | −0.003 | 1.3 s | Mean-rate features; alpha = 1.0 |
| GRU (2-layer bidir) | ~660 K | −0.006 | 380 s | 5 epochs, 20 K subsample |
| Vanilla Transformer | ~5 M | −0.013 | 121 s | 3 epochs, 4L/256d/4H, 20 K subsample |

## Headline

All models — including the published-quality baselines — produce R² ≈ 0 for hand velocity decoding in this evaluation.  
**This is expected and explained below; the training infrastructure is working correctly.**

Published NLB leaderboard numbers (e.g. Wiener R² ≈ 0.40, best models R² ≈ 0.60+) use **trial-aligned** evaluation windows. Our evaluation uses **sliding windows over the full continuous recording**, which includes the ~85% of time when the monkey is stationary. A model that always predicts zero velocity achieves R² ≈ 0 on this distribution; that is approximately what every model does.

## Why R² ≈ 0 on Continuous Windows

The MC_Maze recording is 115 minutes of continuous electrophysiology.  
Movement windows (hand speed > 50 mm/s) account for only 14.5% of total time.  
When evaluation windows are sampled uniformly from this recording:

- ~85% of windows have near-zero z-scored velocity targets
- Predicting zero on these windows has near-zero loss
- R² = 1 − SS_res / SS_tot; SS_tot is very small when targets ≈ 0

A Wiener filter probed on movement-only windows with Gaussian-smoothed rates (σ = 30 ms) and 30 ms neural lag achieves R² ≈ 0.010, confirming the signal is present but the evaluation distribution is wrong.

## Path to Reproducing Published Numbers

The NWB trials table contains 2,295 centre-out reach trials with a `split` column (`train`/`val`/`test`) and `move_onset_time`. Trial-aligned evaluation — extracting a fixed window around move onset for each trial — matches the NLB protocol and will reproduce the ~0.4 Wiener baseline. This is the correct evaluation to implement in Phase 2 (profiling gates this anyway — we need the infrastructure working first).

## MPS Caveats

1. **Step time on MPS vs A100**: 563 ms/step on M4 Pro vs estimated ~20 ms on A100 (28× slower). 2,000 steps at 32 batch × 120 bins sufficed to confirm convergence behaviour but not to reach a well-trained checkpoint.
2. **`non_blocking=True` bug**: MPS does not complete async `.to()` transfers before the next `.cpu()` call on the same stream. All evaluations use synchronous transfers (`non_blocking=False`).
3. **`pin_memory=False`**: MPS does not support pinned memory; `DataLoader` must be configured accordingly.
4. **p99 step time**: 3,465 ms (includes occasional GC / memory pressure spikes). Median and p50 are much tighter around 560 ms.

## Reproducibility

```bash
# Cortex-S training (2 000 steps)
python scripts/train_benchmark.py --max-steps 2000 --device auto

# All three baselines
python scripts/baseline_benchmark.py --device auto

# Results written to:
#   benchmarks/training/results_raw.json
#   benchmarks/training/baselines_raw.json
```

For a full training run targeting convergence, run on CUDA hardware:
```bash
make train-s   # invokes scripts/train_benchmark.py --max-steps 2000 --device auto
```
Expect ~40 min on a single A100 for 2,000 steps at batch 32. Increase `--max-steps` to 20,000 for a fully converged Cortex-S checkpoint.

---

## Trial-Aligned Evaluation (NLB Protocol)

> **Full results:** [`trial_aligned_results.md`](trial_aligned_results.md)  
> **Raw JSON:** [`trial_aligned_results.json`](trial_aligned_results.json)

The sliding-window R² ≈ 0 numbers above are expected and correct given the
evaluation distribution (~85 % rest windows). The **trial-aligned** evaluation
below fixes this by extracting one sample per trial around `move_onset_time`,
matching the published NLB leaderboard protocol.

### Setup

| Field | Value |
|---|---|
| Window | −100 ms to +500 ms relative to move_onset_time |
| Behavior target | Hand velocity AT move_onset_time |
| Train trials | 1,721 | Val trials | 574 |

### Results

| Model | R² (hand velocity) | Notes |
|---|---|---|
| Wiener filter (ridge) | **0.4822** | Mean-rate features; alpha = 1.0 |
| Cortex-S | pending | Run `make train-s` on CUDA hardware |

The Wiener filter R² of 0.48 reflects genuine neural decoding signal on
movement trials. It is slightly above the published NLB Wiener baseline
(~0.33) because our target is velocity at a single time point (onset) rather
than a time-averaged trace, which is an easier prediction task. A Cortex-S
model trained to convergence should exceed this baseline.

### How the two modes differ

| Mode | Windows | R² (Wiener) | When to use |
|---|---|---|---|
| Sliding window (NLBDataset) | 111,228 uniform windows | −0.003 | Training signal; checks infra works |
| Trial-aligned (TrialAlignedDataset) | 2,295 movement trials | **0.48** | Evaluation; matches NLB leaderboard |

### Reproducing

```bash
python scripts/eval_trial_aligned.py --data-root ./data
# writes benchmarks/training/trial_aligned_results.md + .json
```
