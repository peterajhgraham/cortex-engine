"""FSDP sharded checkpoint save/load.

Uses torch.distributed.checkpoint for distributed-correct saves that work
across different world sizes (resharding on load).

Implementation notes for Claude Code:
    - Always save from rank 0's view but with sharded format
    - Snapshot also includes optimizer state, LR scheduler state, train state
    - For single-GPU: degrade to standard torch.save/load
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from cortex.utils.logging import get_logger

log = get_logger(__name__)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: dict[str, Any],
) -> None:
    """Save a checkpoint. Sharded if distributed, standard otherwise."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if dist.is_initialized() and dist.get_world_size() > 1:
        # TODO Phase 1.7: implement DCP sharded save
        # from torch.distributed.checkpoint import save
        # state_dict = {"model": model.state_dict(), "optim": optimizer.state_dict(), "state": train_state}
        # save(state_dict, checkpoint_id=str(path))
        raise NotImplementedError("DCP sharded save not yet implemented")
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "state": train_state,
            },
            path / "checkpoint.pt",
        )
        log.info("checkpoint_saved", path=str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint. Returns the train_state dict."""
    path = Path(path)
    if (path / "checkpoint.pt").exists():
        ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        log.info("checkpoint_loaded", path=str(path))
        return ckpt.get("state", {})

    # TODO Phase 1.7: DCP sharded load path
    raise FileNotFoundError(f"No checkpoint at {path}")
