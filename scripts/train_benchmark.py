"""Standalone training benchmark: Cortex-S on MC_Maze, no Hydra.

Records per-step timing, loss curves, and final eval R² to
benchmarks/training/results.md.

Usage:
    python scripts/train_benchmark.py [--max-steps 2000] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cortex.data.nlb import NLBDataset, collate_events
from cortex.models import CortexModel
from cortex.models.config import CORTEX_S
from cortex.training.eval import evaluate, r2_score
from cortex.training.train import (
    build_optimizer,
    build_scheduler,
    compute_loss,
)
from cortex.utils.device import select_device
from cortex.utils.logging import configure_logging, get_logger

configure_logging(level="INFO", json=False)
log = get_logger(__name__)


def build_loaders(
    data_root: str, batch_size: int, val_max_samples: int = 2048
) -> tuple[DataLoader, DataLoader]:
    kwargs = dict(
        data_root=data_root,
        dandiset_id="000128",
        bin_size_ms=5,
        window_ms=600,
        stride_ms=50,
        max_neurons=CORTEX_S.max_neurons,
        spike_value_buckets=CORTEX_S.spike_value_buckets,
        download=False,
    )
    train_ds = NLBDataset(split="train", **kwargs)  # type: ignore[arg-type]
    val_ds = NLBDataset(split="val", **kwargs)  # type: ignore[arg-type]

    # Cap val at val_max_samples to keep eval fast during training
    if len(val_ds) > val_max_samples:
        from torch.utils.data import Subset
        import random
        rng = random.Random(42)
        val_idx = rng.sample(range(len(val_ds)), val_max_samples)
        val_ds = Subset(val_ds, val_idx)  # type: ignore[assignment]

    loader_kw = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_events)
    return (
        DataLoader(train_ds, shuffle=True, **loader_kw),
        DataLoader(val_ds, shuffle=False, **loader_kw),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out", default="benchmarks/training/results_raw.json")
    args = parser.parse_args()

    device = select_device(preference=args.device)
    log.info("benchmark_start", device=str(device), max_steps=args.max_steps, model_params=f"{sum(p.numel() for p in CortexModel(CORTEX_S).parameters())/1e6:.1f}M")

    log.info("loading_data")
    t0 = time.time()
    train_loader, val_loader = build_loaders(args.data_root, args.batch_size)
    log.info("data_loaded", train_windows=len(train_loader.dataset),  # type: ignore[arg-type]
             val_windows=len(val_loader.dataset), elapsed_s=f"{time.time()-t0:.1f}")  # type: ignore[arg-type]

    model = CortexModel(CORTEX_S).to(device)
    optimizer = build_optimizer(model, _cfg(args.lr))
    scheduler = build_scheduler(optimizer, args.warmup_steps, args.max_steps)

    step = 0
    epoch = 0
    step_times: list[float] = []
    loss_log: list[dict] = []
    eval_log: list[dict] = []

    total_t0 = time.time()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            batch = {k: v.to(device, non_blocking=False) for k, v in batch.items()}

            model.train()
            t_step = time.time()
            loss, components = compute_loss(
                model, batch, behavior_weight=1.0, masked_spike_weight=0.1
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step_ms = (time.time() - t_step) * 1000
            step_times.append(step_ms)
            step += 1

            if step % 100 == 0:
                elapsed = time.time() - total_t0
                log.info(
                    "train_step",
                    step=step,
                    loss=f"{float(loss):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step_ms=f"{step_ms:.1f}",
                    elapsed_s=f"{elapsed:.0f}",
                )
                loss_log.append({"step": step, "loss": float(loss), **components})

            if step % 500 == 0 or step == args.max_steps:
                log.info("evaluating", step=step)
                results = evaluate(model, val_loader, device)
                log.info(
                    "eval",
                    step=step,
                    r2_velocity=f"{results.r2_velocity:.4f}",
                    mse_velocity=f"{results.mse_velocity:.4f}",
                )
                eval_log.append({
                    "step": step,
                    "r2_velocity": results.r2_velocity,
                    "mse_velocity": results.mse_velocity,
                })

        epoch += 1

    total_elapsed = time.time() - total_t0
    avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    p99_step_ms = sorted(step_times)[int(0.99 * len(step_times))] if step_times else 0.0

    summary = {
        "model": "Cortex-S",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "device": str(device),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "final_r2_velocity": eval_log[-1]["r2_velocity"] if eval_log else None,
        "final_mse_velocity": eval_log[-1]["mse_velocity"] if eval_log else None,
        "train_windows": len(train_loader.dataset),  # type: ignore[arg-type]
        "val_windows": len(val_loader.dataset),  # type: ignore[arg-type]
        "total_elapsed_s": round(total_elapsed, 1),
        "avg_step_ms": round(avg_step_ms, 1),
        "p99_step_ms": round(p99_step_ms, 1),
        "loss_log": loss_log,
        "eval_log": eval_log,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("benchmark_complete", out=str(out_path), final_r2=eval_log[-1]["r2_velocity"] if eval_log else "N/A", total_s=f"{total_elapsed:.0f}")


class _cfg:
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.weight_decay = 0.01
        self.betas = [0.9, 0.95]


if __name__ == "__main__":
    main()
