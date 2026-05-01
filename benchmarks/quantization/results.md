# INT8 Quantization Results — Cortex-S

Measured: 2026-04-30
Hardware: mps
Data source: synthetic (random)
Calibration batches: 50
Evaluation batches: 100

---

## Memory Savings

| Configuration | Weight + buffer memory | Δ vs float |
|---|---|---|
| float32 baseline | 99.2 MB | — |
| INT8 (per-channel weights) | 27.8 MB | −71.4 MB (−72.0%) |

Quantized layers: 34 / 35 nn.Linear layers.
Skipped (no scales or not an nn.Linear): 1 layers.

Note: this measures parameter + buffer tensor storage only. Runtime activation
memory is not reduced by weight-only quantization. Full memory at inference
(with batch=32 activations) is dominated by activations, not weights.

---

## Accuracy

| Model | MSE (synthetic — not meaningful for model quality) | Forward time (100 batches) |
|---|---|---|
| float32 | 1.111848 | 3.39 s |
| INT8 | 1.112757 | 3.41 s |
| delta | +0.000908 (+0.1%) | +0.03 s |

**Note:** Synthetic random weights + random targets → absolute MSE is meaningless. The delta (fp32 vs INT8) shows output fidelity under quantization noise.

---

## Output Fidelity (max absolute difference, first 10 eval batches)

| Metric | Value |
|---|---|
| Max abs diff (fp32 vs INT8 output) | 0.003202 |
| Mean abs diff (fp32 vs INT8 output) | 0.001474 |

A max abs diff < 0.1 at bf16 scale is expected for per-channel weight quantization
on typical transformer activations.

---

## Per-Layer Scale Statistics (first 20 layers)

| Layer | Calib batches | Act p99 abs-max | Act scale | W scale min | W scale max |
|---|---|---|---|---|---|
| `encoder.cross_attn.q_proj` | 50 | 4.3039 | 3.39e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.cross_attn.kv_proj` | 50 | 5.4905 | 4.32e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.cross_attn.out_proj` | 50 | 0.5078 | 4.00e-03 | 3.44e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.0.qkv_proj` | 50 | 5.1386 | 4.05e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.0.out_proj` | 50 | 1.8876 | 1.49e-02 | 3.42e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.0.mlp.0` | 50 | 4.0732 | 3.21e-02 | 3.41e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.0.mlp.2` | 50 | 2.1326 | 1.68e-02 | 1.73e-04 | 1.74e-04 |
| `encoder.self_attn_blocks.1.qkv_proj` | 50 | 3.6978 | 2.91e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.1.out_proj` | 50 | 1.9853 | 1.56e-02 | 3.44e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.1.mlp.0` | 50 | 3.9205 | 3.09e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.1.mlp.2` | 50 | 2.1297 | 1.68e-02 | 1.73e-04 | 1.74e-04 |
| `encoder.self_attn_blocks.2.qkv_proj` | 50 | 4.4204 | 3.48e-02 | 3.44e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.2.out_proj` | 50 | 1.9701 | 1.55e-02 | 3.44e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.2.mlp.0` | 50 | 3.7760 | 2.97e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.2.mlp.2` | 50 | 2.3636 | 1.86e-02 | 1.73e-04 | 1.74e-04 |
| `encoder.self_attn_blocks.3.qkv_proj` | 50 | 3.8779 | 3.05e-02 | 3.44e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.3.out_proj` | 50 | 2.0751 | 1.63e-02 | 3.44e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.3.mlp.0` | 50 | 3.8522 | 3.03e-02 | 3.43e-04 | 3.48e-04 |
| `encoder.self_attn_blocks.3.mlp.2` | 50 | 2.4397 | 1.92e-02 | 1.73e-04 | 1.74e-04 |
| `encoder.self_attn_blocks.4.qkv_proj` | 50 | 3.7840 | 2.98e-02 | 3.42e-04 | 3.48e-04 |

...and 15 more layers. Full table in `benchmarks/quantization/layer_stats.json`.

---

## Calibration Method

- **Weight quantization:** per output-channel, symmetric INT8. Scale = absmax(W[c, :]) / 127.
- **Activation scale:** per-tensor, 99th-percentile abs-max across 50 calibration batches.
- **Forward pass:** weight dequantization (int8 × scale → fp) then standard matmul.
  No CUDA-specific int8 kernel required. True int8 matmul (cuBLAS) is a Phase 3 target.

## To Reproduce

```bash
# Synthetic mode (no data required):
PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py --synthetic

# With real MC_Maze data and a trained checkpoint:
PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py \
    --checkpoint checkpoints/cortex_s.pt \
    --data-dir data/mc_maze \
    --calib-batches 50
```
