# Trial-Aligned Evaluation Results

Measured on **Apple M4 Pro (24 GB unified memory)**, 2026-05-05.

## Setup

| Field | Value |
|---|---|
| NWB file | MC_Maze (DANDI 000128), train+behavior split |
| Evaluation mode | Trial-aligned (NLB protocol) |
| Window | −100 ms to +500 ms relative to move_onset_time |
| Bin size | 5 ms |
| Train trials | 1721 |
| Val trials | 574 |
| Behavior target | Hand velocity AT move_onset_time (+0 ms relative to onset) |

## Results

| Model | R² (hand velocity) | Notes |
|---|---|---|
| Wiener filter (ridge) | 0.4822 | Mean-rate features; alpha = 1.0 |
| Cortex-S | N/A — no checkpoint | Run `make train-s` to produce one |

## Interpretation

These numbers use **trial-aligned evaluation**, which matches the published NLB
leaderboard protocol. Every sample is extracted from a centre-out reach trial.
All targets have substantial hand movement, so R² measures genuine decoding
performance rather than the ability to predict near-zero velocity.

Contrast with the sliding-window results in `results.md` (R² ≈ 0 for all
models) where ~85 % of windows sample rest periods.

Published NLB leaderboard references for MC_Maze:
- Wiener filter: R² ≈ 0.33–0.40 (Pei et al. 2021, NLB '21 paper)
- Best NLB '21 entry: R² ≈ 0.62

The behavior target here is **velocity AT move_onset_time** (a single
time point per trial). This is slightly easier than the published NLB
evaluation (which predicts per-bin velocity traces and averages R² across
all bins), so our Wiener R² is somewhat higher than the ~0.33 reported in
Pei et al. 2021. The evaluation still captures the core decoding challenge:
predict movement onset direction and speed from neural activity.
