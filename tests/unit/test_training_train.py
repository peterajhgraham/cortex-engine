"""Unit tests for training loop helpers and a CPU smoke test of the loop."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cortex.data.nlb import NLBDataset, SessionData, collate_events
from cortex.models import CortexConfig, CortexModel
from cortex.models.config import CORTEX_XS
from cortex.training.train import (
    TrainState,
    build_optimizer,
    build_scheduler,
    compute_loss,
    run_training_loop,
)


# ── LR schedule ────────────────────────────────────────────────────────────────


def test_lr_schedule_warmup_then_cosine() -> None:
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = build_scheduler(opt, warmup_steps=10, max_steps=100)

    lrs = []
    for _ in range(100):
        opt.step()
        lrs.append(sched.get_last_lr()[0])
        sched.step()

    # Step 0 multiplier = 1/10, step 9 multiplier = 1.0 (peak), then cosine decay to 0.
    assert lrs[0] == pytest.approx(0.1, abs=1e-6)
    assert lrs[9] == pytest.approx(1.0, abs=1e-6)
    # Rough monotone decay after warmup.
    assert lrs[-1] == pytest.approx(0.0, abs=1e-3)
    assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(10, 99))


def test_lr_schedule_zero_warmup_clamped() -> None:
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = build_scheduler(opt, warmup_steps=0, max_steps=10)
    assert math.isfinite(sched.get_last_lr()[0])


# ── Optimizer ──────────────────────────────────────────────────────────────────


def test_build_optimizer_uses_config_values() -> None:
    cfg = OmegaConf.create(
        {"lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.999]},
    )
    model = torch.nn.Linear(8, 4)
    opt = build_optimizer(model, cfg)  # type: ignore[arg-type]
    g = opt.param_groups[0]
    assert g["lr"] == pytest.approx(1e-3)
    assert g["weight_decay"] == pytest.approx(0.05)
    assert tuple(g["betas"]) == (0.9, 0.999)


# ── Loss ───────────────────────────────────────────────────────────────────────


def _synthetic_batch(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    e = 64
    return {
        "neuron_ids": torch.randint(0, CORTEX_XS.max_neurons, (e,), generator=g),
        "time_bins": torch.randint(0, CORTEX_XS.max_time_bins, (e,), generator=g),
        "values": torch.randint(0, CORTEX_XS.spike_value_buckets, (e,), generator=g),
        "batch_indices": torch.randint(0, 4, (e,), generator=g).sort().values,
        "behavior": torch.randn(4, CORTEX_XS.behavior_dim, generator=g),
    }


def test_compute_loss_returns_finite_loss_and_components() -> None:
    model = CortexModel(CORTEX_XS)
    batch = _synthetic_batch()
    loss, components = compute_loss(model, batch, behavior_weight=1.0, masked_spike_weight=0.0)
    assert torch.isfinite(loss)
    assert "behavior_loss" in components
    # Without targets, masked loss should not appear regardless of weight.
    loss2, components2 = compute_loss(model, batch, behavior_weight=1.0, masked_spike_weight=0.1)
    assert "masked_spike_loss" not in components2


def test_compute_loss_backprop_updates_params() -> None:
    model = CortexModel(CORTEX_XS)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    batch = _synthetic_batch(seed=7)
    before = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}

    loss, _ = compute_loss(model, batch, behavior_weight=1.0, masked_spike_weight=0.0)
    loss.backward()
    optimizer.step()

    moved = sum(
        not torch.equal(before[k], v.detach()) for k, v in model.named_parameters() if k in before
    )
    # At least the behavior head and tokenizer (both touched in forward) must update.
    assert moved >= 2


# ── Training loop smoke ────────────────────────────────────────────────────────


def _make_loop_dataset(seed: int = 0) -> NLBDataset:
    rng = np.random.default_rng(seed)
    bin_counts = rng.poisson(lam=0.3, size=(400, 32)).astype(np.int32)
    t = np.arange(400, dtype=np.float32) / 399.0
    behavior = np.stack([np.sin(t * 6.28), np.cos(t * 6.28)], axis=1).astype(np.float32)
    session = SessionData(
        session_id="loop-synth",
        bin_counts=bin_counts,
        behavior=behavior,
        neuron_id_offset=0,
    )
    return NLBDataset(
        data_root="/tmp/unused",
        dandiset_id="000128",
        split="train",
        bin_size_ms=5,
        window_ms=100,
        stride_ms=50,
        max_neurons=CORTEX_XS.max_neurons,
        spike_value_buckets=CORTEX_XS.spike_value_buckets,
        download=False,
        sessions=[session],
    )


def test_training_loop_smoke_runs_to_max_steps(tmp_path: Path) -> None:
    """End-to-end: 5 training steps on synthetic data, no eval, no W&B."""
    cfg = OmegaConf.create(
        {
            "training": {
                "grad_accum_steps": 1,
                "max_steps": 5,
                "behavior_loss_weight": 1.0,
                "masked_spike_loss_weight": 0.0,
                "grad_clip": 1.0,
                "eval_every": 10_000,
                "checkpoint_every": 10_000,
            }
        }
    )

    ds = _make_loop_dataset()
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_events)

    model = CortexModel(CORTEX_XS)
    optimizer = build_optimizer(
        model, OmegaConf.create({"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95]})
    )
    scheduler = build_scheduler(optimizer, warmup_steps=2, max_steps=5)
    state = TrainState(step=0, epoch=0, best_val_metric=float("-inf"))

    run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=loader,  # unused: eval_every is huge
        cfg=cfg,  # type: ignore[arg-type]
        state=state,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        wandb_run=None,
    )

    assert state.step == 5
    # A final checkpoint always lands at checkpoint_dir/final.
    assert (tmp_path / "final" / "checkpoint.pt").exists()


def test_training_loop_loss_decreases_on_overfit(tmp_path: Path) -> None:
    """Sanity: a tiny model overfitting a tiny dataset reduces training loss."""
    cfg = OmegaConf.create(
        {
            "training": {
                "grad_accum_steps": 1,
                "max_steps": 80,
                "behavior_loss_weight": 1.0,
                "masked_spike_loss_weight": 0.0,
                "grad_clip": 1.0,
                "eval_every": 10_000,
                "checkpoint_every": 10_000,
            }
        }
    )

    # Reduce the model further so 80 steps on CPU is fast.
    tiny = CortexConfig(
        hidden_dim=64, num_layers=2, num_heads=2, head_dim=32,
        num_latents=16, latent_dim=64, cross_attn_heads=2,
        max_neurons=256, max_time_bins=512, behavior_dim=2,
    )
    model = CortexModel(tiny)
    optimizer = build_optimizer(
        model, OmegaConf.create({"lr": 3e-3, "weight_decay": 0.0, "betas": [0.9, 0.95]})
    )
    scheduler = build_scheduler(optimizer, warmup_steps=5, max_steps=80)

    ds = _make_loop_dataset(seed=9)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_events)

    # Snapshot loss at step 0 vs after training.
    first_batch = next(iter(loader))
    initial_loss, _ = compute_loss(
        model, first_batch, behavior_weight=1.0, masked_spike_weight=0.0
    )

    state = TrainState(step=0, epoch=0, best_val_metric=float("-inf"))
    run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=loader,
        cfg=cfg,  # type: ignore[arg-type]
        state=state,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        wandb_run=None,
    )

    final_loss, _ = compute_loss(
        model, first_batch, behavior_weight=1.0, masked_spike_weight=0.0
    )
    assert final_loss.item() < initial_loss.item()


def test_train_state_serializable() -> None:
    state = TrainState(step=42, epoch=3, best_val_metric=0.5)
    d = asdict(state)
    assert d == {"step": 42, "epoch": 3, "best_val_metric": 0.5}
