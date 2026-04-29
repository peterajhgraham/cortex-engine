"""Tests for the single-process checkpoint path.

The DCP (multi-rank) path can't be exercised from a single test process. It's
covered by the resolver functions and structural tests below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cortex.models import CortexModel
from cortex.models.config import CORTEX_XS
from cortex.training.checkpoint import (
    _resolve_dcp_load,
    _resolve_dcp_save,
    load_checkpoint,
    save_checkpoint,
)


def _model_and_optim() -> tuple[CortexModel, torch.optim.Optimizer]:
    model = CortexModel(CORTEX_XS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model, optimizer = _model_and_optim()
    # Take a step so optimizer state is non-empty.
    fake_loss = sum(p.sum() for p in model.parameters() if p.requires_grad)
    fake_loss.backward()
    optimizer.step()

    save_checkpoint(tmp_path, model, optimizer, {"step": 7, "epoch": 1})
    assert (tmp_path / "checkpoint.pt").exists()

    new_model, new_optim = _model_and_optim()
    restored_state = load_checkpoint(tmp_path, new_model, new_optim)

    assert restored_state == {"step": 7, "epoch": 1}
    for p_orig, p_new in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.allclose(p_orig, p_new)


def test_load_checkpoint_without_optimizer(tmp_path: Path) -> None:
    model, optimizer = _model_and_optim()
    save_checkpoint(tmp_path, model, optimizer, {"step": 1})

    new_model, _ = _model_and_optim()
    state = load_checkpoint(tmp_path, new_model, optimizer=None)
    assert state == {"step": 1}


def test_load_checkpoint_missing_path_raises(tmp_path: Path) -> None:
    model, _ = _model_and_optim()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does-not-exist", model)


def test_dcp_resolvers_return_callables() -> None:
    """torch.distributed.checkpoint must expose save+load under one of two names."""
    save_fn = _resolve_dcp_save()
    load_fn = _resolve_dcp_load()
    assert callable(save_fn)
    assert callable(load_fn)
