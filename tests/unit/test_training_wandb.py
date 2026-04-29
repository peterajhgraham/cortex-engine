"""Tests for the W&B integration in the training loop.

We don't talk to wandb.ai — instead we pass a mock object with the methods
the loop expects (`log`, `log_artifact`, `name`) and verify it's exercised.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cortex.data.nlb import NLBDataset, SessionData, collate_events
from cortex.models import CortexModel
from cortex.models.config import CORTEX_XS
from cortex.training.train import (
    TrainState,
    _log_checkpoint_artifact,
    build_optimizer,
    build_scheduler,
    run_training_loop,
)


def _make_dataset() -> NLBDataset:
    rng = np.random.default_rng(0)
    bin_counts = rng.poisson(lam=0.3, size=(400, 32)).astype(np.int32)
    behavior = np.zeros((400, 2), dtype=np.float32)
    session = SessionData("synth", bin_counts, behavior, neuron_id_offset=0)
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


def test_loop_logs_to_wandb_when_run_provided(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "grad_accum_steps": 1,
                "max_steps": 50,  # >= 50 triggers the per-50-step train log
                "behavior_loss_weight": 1.0,
                "masked_spike_loss_weight": 0.0,
                "grad_clip": 1.0,
                "eval_every": 10_000,
                "checkpoint_every": 10_000,
            }
        }
    )
    model = CortexModel(CORTEX_XS)
    optimizer = build_optimizer(
        model, OmegaConf.create({"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95]})
    )
    scheduler = build_scheduler(optimizer, warmup_steps=2, max_steps=50)
    loader = DataLoader(_make_dataset(), batch_size=4, shuffle=False, collate_fn=collate_events)

    wandb_run = MagicMock()
    wandb_run.name = "test-run"

    run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=loader,
        cfg=cfg,  # type: ignore[arg-type]
        state=TrainState(step=0, epoch=0, best_val_metric=float("-inf")),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        wandb_run=wandb_run,
    )

    # The 50-step train log fires once at step 50.
    log_keys = [call.args[0].keys() for call in wandb_run.log.call_args_list]
    assert any("train/loss" in keys for keys in log_keys)
    assert any("train/lr" in keys for keys in log_keys)


def test_log_checkpoint_artifact_is_noop_when_wandb_off(tmp_path: Path) -> None:
    """A None wandb_run must not raise or attempt any I/O."""
    (tmp_path / "checkpoint.pt").write_bytes(b"\x00")
    _log_checkpoint_artifact(None, tmp_path, name="best")  # no exception


def test_log_checkpoint_artifact_uploads_directory(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.pt").write_bytes(b"\x00")

    artifact = MagicMock()
    wandb_run = MagicMock()
    wandb_run.name = "run-name"

    import sys
    import types

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Artifact = MagicMock(return_value=artifact)  # type: ignore[attr-defined]
    sys.modules["wandb"] = fake_wandb
    try:
        _log_checkpoint_artifact(wandb_run, tmp_path, name="best")
    finally:
        del sys.modules["wandb"]

    fake_wandb.Artifact.assert_called_once_with(name="run-name-best", type="model")  # type: ignore[attr-defined]
    artifact.add_dir.assert_called_once_with(str(tmp_path))
    wandb_run.log_artifact.assert_called_once_with(artifact)
