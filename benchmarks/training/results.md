# Phase 1 Training Benchmarks

> Populate after each training run. Numbers in this file feed the headline
> table in the top-level README.

## Setup

| Field | Value |
|---|---|
| Hardware | TBD (e.g., A100-80GB x1) |
| Software | torch X.X / triton Y.Y / cuda Z.Z |
| Dataset | NLB MC_Maze |
| Bin size | 5ms |
| Window | 600ms |
| Stride | 50ms |

## Cortex Models

| Model | Params | Train time | Final R² | Best R² | W&B run |
|---|---|---|---|---|---|
| Cortex-XS | ~5M | TBD | TBD | TBD | [link]() |
| Cortex-S  | ~25M | TBD | TBD | TBD | [link]() |
| Cortex-M  | ~80M | TBD | TBD | TBD | [link]() |

## Baselines

| Baseline | Params | R² | Notes |
|---|---|---|---|
| Wiener filter (ridge) | TBD | TBD | Closed-form fit, alpha tuned via CV |
| GRU (2 layer) | TBD | TBD | |
| Vanilla Transformer | TBD | TBD | Same param budget as Cortex-S |

## Headline

Cortex-S beats the strongest baseline by **TBD R²** while training **TBD**x faster than the vanilla transformer on the same hardware.

## Reproducibility

```bash
make train-s
# then read the W&B run dashboard for full curves
```

Hyperparameter sweeps:
```bash
wandb sweep configs/sweeps/cortex_s.yaml
wandb agent <sweep-id>
```
