"""INT8 quantization calibration script for Cortex-S.

Workflow:
  1. Load the Cortex-S model (from checkpoint, or random weights if none given).
  2. Attach calibration hooks to every nn.Linear.
  3. Run N calibration batches (real MC_Maze data, or synthetic if --synthetic).
  4. Derive per-layer INT8 scales (per-channel weight, per-tensor activation).
  5. Convert the model (replace nn.Linear with QuantizedLinear).
  6. Evaluate both the float model and the INT8 model on a held-out set,
     measuring MSE loss and output fidelity (max abs difference per sample).
  7. Write benchmarks/quantization/results.md and save the INT8 checkpoint.

Usage:
  # With real MC_Maze data and a trained checkpoint:
  PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py \\
      --checkpoint checkpoints/cortex_s.pt \\
      --data-dir data/mc_maze \\
      --calib-batches 50

  # Synthetic mode (no data or checkpoint required — measures memory/fidelity):
  PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py --synthetic

  # Save INT8 checkpoint:
  PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py \\
      --synthetic --output-checkpoint checkpoints/cortex_s_int8.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortex.models.config import CORTEX_S
from cortex.models.cortex import CortexModel
from cortex.quantization.calibrate import (
    ActivationStats,
    LayerScales,
    attach_calibration_hooks,
    calibration_summary,
    convert_model,
    count_quantized_linears,
    derive_scales,
    model_weight_bytes,
    remove_calibration_hooks,
    save_quantized,
)


# ── Synthetic data generation ─────────────────────────────────────────────────


def make_synthetic_batch(
    batch_size: int = 8,
    events_per_sample: int = 512,
    device: str = "cpu",
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a random batch in the format expected by CortexModel.forward."""
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    cfg = CORTEX_S
    E = batch_size * events_per_sample

    neuron_ids = torch.randint(0, cfg.max_neurons,    (E,), generator=rng)
    time_bins  = torch.randint(0, cfg.max_time_bins,  (E,), generator=rng)
    values     = torch.randint(0, cfg.spike_value_buckets, (E,), generator=rng)
    batch_idx  = torch.repeat_interleave(
        torch.arange(batch_size), torch.full((batch_size,), events_per_sample, dtype=torch.long)
    )
    behavior   = torch.randn(batch_size, cfg.behavior_dim, generator=rng)

    return {
        "neuron_ids":    neuron_ids.to(device),
        "time_bins":     time_bins.to(device),
        "values":        values.to(device),
        "batch_indices": batch_idx.to(device),
        "behavior":      behavior.to(device),
    }


def synthetic_dataloader(
    n_batches: int,
    batch_size: int = 8,
    events_per_sample: int = 512,
    device: str = "cpu",
) -> list[dict[str, torch.Tensor]]:
    return [
        make_synthetic_batch(batch_size, events_per_sample, device, seed=i)
        for i in range(n_batches)
    ]


# ── NLB dataloader (real data) ────────────────────────────────────────────────


def real_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int,
    n_batches: int | None,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    from cortex.data.nlb import NLBDataset, collate_events

    cfg = CORTEX_S
    ds = NLBDataset(
        data_root=data_dir,
        dandiset_id="000128",
        split=split,
        bin_size_ms=5,
        window_ms=600,
        stride_ms=50,
        max_neurons=cfg.max_neurons,
        spike_value_buckets=cfg.spike_value_buckets,
        download=False,  # expect data to be present
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_events, drop_last=True)
    batches = []
    for i, batch in enumerate(dl):
        if n_batches is not None and i >= n_batches:
            break
        batches.append({k: v.to(device) for k, v in batch.items()})
    return batches


# ── Forward pass helper ───────────────────────────────────────────────────────


@torch.no_grad()
def run_batches(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
) -> tuple[float, float]:
    """Run model on batches; return (mean_mse, total_seconds)."""
    model.eval()
    mse_sum = 0.0
    t0 = time.perf_counter()
    for batch in batches:
        out = model(
            batch["neuron_ids"],
            batch["time_bins"],
            batch["values"],
            batch["batch_indices"],
        )
        target = batch["behavior"]
        pred = out["behavior"]
        mse_sum += float(((pred - target) ** 2).mean().item())
    elapsed = time.perf_counter() - t0
    return mse_sum / max(len(batches), 1), elapsed


@torch.no_grad()
def measure_output_fidelity(
    model_fp: nn.Module,
    model_q: nn.Module,
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compare float and INT8 outputs on the same batches.

    Returns max and mean absolute difference across all batch × output elements.
    """
    model_fp.eval()
    model_q.eval()
    max_diffs = []
    mean_diffs = []

    for batch in batches[:10]:  # first 10 batches for speed
        out_fp = model_fp(batch["neuron_ids"], batch["time_bins"], batch["values"],
                          batch["batch_indices"])["behavior"]
        out_q  = model_q( batch["neuron_ids"], batch["time_bins"], batch["values"],
                          batch["batch_indices"])["behavior"]
        diff = (out_fp.float() - out_q.float()).abs()
        max_diffs.append(float(diff.max().item()))
        mean_diffs.append(float(diff.mean().item()))

    return {
        "max_abs_diff":  round(max(max_diffs),  6),
        "mean_abs_diff": round(sum(mean_diffs) / len(mean_diffs), 6),
    }


# ── Report writer ─────────────────────────────────────────────────────────────


def write_report(
    output: Path,
    fp_bytes: int,
    int8_bytes: int,
    n_quantized: int,
    n_total_linear: int,
    fp_mse: float,
    int8_mse: float,
    fidelity: dict[str, float],
    calib_batches: int,
    eval_batches: int,
    fp_time_s: float,
    int8_time_s: float,
    layer_stats: list[dict],
    data_source: str,
    device: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fp_mb   = fp_bytes   / 1e6
    int8_mb = int8_bytes / 1e6
    savings_mb  = fp_mb - int8_mb
    savings_pct = 100.0 * savings_mb / fp_mb

    mse_delta_pct = 100.0 * (int8_mse - fp_mse) / max(fp_mse, 1e-9) if fp_mse > 0 else 0.0

    is_synthetic = "synthetic" in data_source
    mse_label = "synthetic — not meaningful for model quality" if is_synthetic else "real data"
    mse_note = (
        "**Note:** Synthetic random weights + random targets → absolute MSE is meaningless. "
        "The delta (fp32 vs INT8) shows output fidelity under quantization noise."
        if is_synthetic else
        f"MSE delta vs float: {mse_delta_pct:+.2f}%. Phase 2 spec requires < 1% R² regression."
    )

    report = f"""# INT8 Quantization Results — Cortex-S

Measured: {time.strftime('%Y-%m-%d')}
Hardware: {device}
Data source: {data_source}
Calibration batches: {calib_batches}
Evaluation batches: {eval_batches}

---

## Memory Savings

| Configuration | Weight + buffer memory | Δ vs float |
|---|---|---|
| float32 baseline | {fp_mb:.1f} MB | — |
| INT8 (per-channel weights) | {int8_mb:.1f} MB | −{savings_mb:.1f} MB (−{savings_pct:.1f}%) |

Quantized layers: {n_quantized} / {n_total_linear} nn.Linear layers.
Skipped (no scales or not an nn.Linear): {n_total_linear - n_quantized} layers.

Note: this measures parameter + buffer tensor storage only. Runtime activation
memory is not reduced by weight-only quantization. Full memory at inference
(with batch=32 activations) is dominated by activations, not weights.

---

## Accuracy

| Model | MSE ({mse_label}) | Forward time ({eval_batches} batches) |
|---|---|---|
| float32 | {fp_mse:.6f} | {fp_time_s:.2f} s |
| INT8 | {int8_mse:.6f} | {int8_time_s:.2f} s |
| delta | {int8_mse - fp_mse:+.6f} ({mse_delta_pct:+.1f}%) | {int8_time_s - fp_time_s:+.2f} s |

{mse_note}

---

## Output Fidelity (max absolute difference, first 10 eval batches)

| Metric | Value |
|---|---|
| Max abs diff (fp32 vs INT8 output) | {fidelity["max_abs_diff"]:.6f} |
| Mean abs diff (fp32 vs INT8 output) | {fidelity["mean_abs_diff"]:.6f} |

A max abs diff < 0.1 at bf16 scale is expected for per-channel weight quantization
on typical transformer activations.

---

## Per-Layer Scale Statistics (first 20 layers)

| Layer | Calib batches | Act p99 abs-max | Act scale | W scale min | W scale max |
|---|---|---|---|---|---|
"""

    for row in layer_stats[:20]:
        act_p99 = f"{row['act_abs_max_p99']:.4f}"
        act_sc  = f"{row['act_scale']:.2e}" if row["act_scale"] else "—"
        w_min   = f"{row['weight_scale_min']:.2e}" if row["weight_scale_min"] else "—"
        w_max   = f"{row['weight_scale_max']:.2e}" if row["weight_scale_max"] else "—"
        report += f"| `{row['layer'][-40:]}` | {row['calib_batches']} | {act_p99} | {act_sc} | {w_min} | {w_max} |\n"

    if len(layer_stats) > 20:
        report += f"\n...and {len(layer_stats) - 20} more layers. Full table in `benchmarks/quantization/layer_stats.json`.\n"

    report += f"""
---

## Calibration Method

- **Weight quantization:** per output-channel, symmetric INT8. Scale = absmax(W[c, :]) / 127.
- **Activation scale:** per-tensor, 99th-percentile abs-max across {calib_batches} calibration batches.
- **Forward pass:** weight dequantization (int8 × scale → fp) then standard matmul.
  No CUDA-specific int8 kernel required. True int8 matmul (cuBLAS) is a Phase 3 target.

## To Reproduce

```bash
# Synthetic mode (no data required):
PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py --synthetic

# With real MC_Maze data and a trained checkpoint:
PYTHONPATH=. .venv/bin/python scripts/calibrate_model.py \\
    --checkpoint checkpoints/cortex_s.pt \\
    --data-dir data/mc_maze \\
    --calib-batches 50
```
"""

    output.write_text(report)
    print(f"Results written to {output}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",  type=Path, default=None,
                        help="Path to float Cortex-S checkpoint (.pt)")
    parser.add_argument("--data-dir",    type=Path, default=Path("data/mc_maze"),
                        help="Root directory for MC_Maze NWB files")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic random data (no real data or checkpoint required)")
    parser.add_argument("--calib-batches", type=int, default=50)
    parser.add_argument("--eval-batches",  type=int, default=100)
    parser.add_argument("--batch-size",    type=int, default=8)
    parser.add_argument("--events-per-sample", type=int, default=512)
    parser.add_argument("--output",      type=Path,
                        default=Path("benchmarks/quantization/results.md"))
    parser.add_argument("--output-checkpoint", type=Path, default=None,
                        help="Where to save the INT8 model (optional)")
    parser.add_argument("--layer-stats-json", type=Path,
                        default=Path("benchmarks/quantization/layer_stats.json"))
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ── Build float model ─────────────────────────────────────────────────────
    cfg = CORTEX_S
    model_fp = CortexModel(cfg).to(device)
    model_fp.eval()

    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        model_fp.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint supplied — using randomly initialized weights.")

    fp_bytes = model_weight_bytes(model_fp)
    print(f"Float model memory: {fp_bytes / 1e6:.1f} MB")

    # ── Calibration data ──────────────────────────────────────────────────────
    use_real = not args.synthetic and args.data_dir.exists()
    if use_real:
        print(f"Loading calibration data from {args.data_dir} …")
        calib_batches = real_dataloader(
            args.data_dir, "train", args.batch_size, args.calib_batches, device
        )
        eval_batches = real_dataloader(
            args.data_dir, "val", args.batch_size, args.eval_batches, device
        )
        data_source = f"MC_Maze DANDI 000128 (real data)"
    else:
        reason = "--synthetic flag" if args.synthetic else f"{args.data_dir} not found"
        print(f"Using synthetic data ({reason})")
        calib_batches = synthetic_dataloader(
            args.calib_batches, args.batch_size, args.events_per_sample, device
        )
        eval_batches = synthetic_dataloader(
            args.eval_batches, args.batch_size, args.events_per_sample, device
        )
        data_source = "synthetic (random)"

    # ── Run calibration ───────────────────────────────────────────────────────
    print(f"\nRunning {len(calib_batches)} calibration batches …")
    hooks = attach_calibration_hooks(model_fp)

    with torch.no_grad():
        for i, batch in enumerate(calib_batches):
            model_fp(
                batch["neuron_ids"],
                batch["time_bins"],
                batch["values"],
                batch["batch_indices"],
            )
            if (i + 1) % 10 == 0:
                print(f"  calibration {i+1}/{len(calib_batches)}")

    remove_calibration_hooks(hooks)
    print(f"Calibration complete — {sum(h.stats.n_batches for h in hooks.values())} hook activations recorded")

    # ── Derive scales and convert ─────────────────────────────────────────────
    scales = derive_scales(model_fp, hooks)
    print(f"Scales derived for {len(scales)} / {sum(1 for m in model_fp.modules() if isinstance(m, nn.Linear))} linear layers")

    import copy
    model_q = convert_model(copy.deepcopy(model_fp), scales, inplace=True)
    model_q.eval()

    int8_bytes = model_weight_bytes(model_q)
    n_quant, n_total = count_quantized_linears(model_q)
    print(f"INT8 model memory: {int8_bytes / 1e6:.1f} MB  (−{(fp_bytes - int8_bytes) / 1e6:.1f} MB, {100*(fp_bytes-int8_bytes)/fp_bytes:.1f}%)")
    print(f"Quantized {n_quant}/{n_total} linear layers")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating on {len(eval_batches)} batches …")
    fp_mse,   fp_time   = run_batches(model_fp, eval_batches)
    int8_mse, int8_time = run_batches(model_q,  eval_batches)
    fidelity = measure_output_fidelity(model_fp, model_q, eval_batches)

    print(f"Float MSE: {fp_mse:.6f}  ({fp_time:.2f}s)")
    print(f"INT8  MSE: {int8_mse:.6f}  ({int8_time:.2f}s)")
    print(f"Max abs diff (fp vs int8): {fidelity['max_abs_diff']:.6f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if args.output_checkpoint is not None:
        save_quantized(model_q, args.output_checkpoint, meta={
            "model": "cortex_s",
            "quantization": "int8_per_channel_weight",
            "calib_batches": len(calib_batches),
            "data_source": data_source,
        })
        print(f"INT8 checkpoint saved to {args.output_checkpoint}")

    layer_stats = calibration_summary(hooks, scales)
    args.layer_stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.layer_stats_json.write_text(json.dumps(layer_stats, indent=2))

    write_report(
        output=args.output,
        fp_bytes=fp_bytes,
        int8_bytes=int8_bytes,
        n_quantized=n_quant,
        n_total_linear=n_total,
        fp_mse=fp_mse,
        int8_mse=int8_mse,
        fidelity=fidelity,
        calib_batches=len(calib_batches),
        eval_batches=len(eval_batches),
        fp_time_s=fp_time,
        int8_time_s=int8_time,
        layer_stats=layer_stats,
        data_source=data_source,
        device=device,
    )


if __name__ == "__main__":
    main()
