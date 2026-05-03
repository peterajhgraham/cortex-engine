"""FSDP sharded checkpoint save/load.

Two paths, picked at runtime:

1. Distributed (world_size > 1): torch.distributed.checkpoint (DCP). Sharded
   save means each rank only writes its own parameter shards, and load works
   even if the new world size differs from the saved world size (resharding
   on load). The on-disk layout is a directory of `.distcp` shard files.

2. Single-process: plain torch.save / torch.load to a single .pt file. This
   is what runs in unit tests and on a laptop.

Both paths use the same checkpoint_dir layout, so callers don't branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn

from cortex.utils.logging import get_logger

log = get_logger(__name__)


_SINGLE_GPU_FILENAME = "checkpoint.pt"
_DCP_SUBDIR = "shards"
_TRAIN_STATE_FILENAME = "train_state.pt"


def _is_distributed_run() -> bool:
    return bool(dist.is_initialized() and dist.get_world_size() > 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: dict[str, Any],
) -> None:
    """Save a checkpoint. Sharded under torch.distributed, single-file otherwise."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if _is_distributed_run():
        _save_dcp(path, model, optimizer, train_state)
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "state": train_state,
            },
            path / _SINGLE_GPU_FILENAME,
        )
        log.info("checkpoint_saved", path=str(path), mode="single")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint. Auto-detects layout.

    Returns the train_state dict (whatever was passed to save_checkpoint).
    """
    path = Path(path)

    single_file = path / _SINGLE_GPU_FILENAME
    if single_file.exists():
        ckpt = torch.load(single_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        log.info("checkpoint_loaded", path=str(path), mode="single")
        return cast(dict[str, Any], ckpt.get("state", {}))

    if (path / _DCP_SUBDIR).exists():
        return _load_dcp(path, model, optimizer)

    raise FileNotFoundError(f"No checkpoint at {path}")


# ── Distributed Checkpoint (DCP) path ──────────────────────────────────────────


def _save_dcp(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: dict[str, Any],
) -> None:
    """Sharded save via torch.distributed.checkpoint.

    Model and optimizer state dicts go through DCP (one .distcp file per rank
    per shard). The plain Python train_state dict is saved separately by
    rank 0 since DCP only handles tensors.
    """
    save_fn = _resolve_dcp_save()
    state_dict = {
        "model": model.state_dict(),
        "optim": _optimizer_state_dict(model, optimizer),
    }
    save_fn(state_dict, checkpoint_id=str(path / _DCP_SUBDIR))
    if dist.get_rank() == 0:
        torch.save(train_state, path / _TRAIN_STATE_FILENAME)
        log.info("checkpoint_saved", path=str(path), mode="dcp", world_size=dist.get_world_size())


def _load_dcp(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    """Sharded load via torch.distributed.checkpoint."""
    if not dist.is_initialized():
        raise RuntimeError(f"DCP checkpoint at {path} requires torch.distributed to be initialized")

    load_fn = _resolve_dcp_load()
    state_dict: dict[str, Any] = {
        "model": model.state_dict(),
        "optim": _optimizer_state_dict(model, optimizer) if optimizer is not None else {},
    }
    load_fn(state_dict, checkpoint_id=str(path / _DCP_SUBDIR))
    model.load_state_dict(state_dict["model"])
    if optimizer is not None and state_dict["optim"]:
        _restore_optimizer_state(model, optimizer, state_dict["optim"])

    train_state_path = path / _TRAIN_STATE_FILENAME
    train_state: dict[str, Any] = {}
    if train_state_path.exists():
        train_state = torch.load(train_state_path, map_location="cpu", weights_only=False)
    log.info("checkpoint_loaded", path=str(path), mode="dcp")
    return train_state


def _resolve_dcp_save() -> Any:
    """torch>=2.3 ships dcp.save; older releases used dcp.save_state_dict."""
    from torch.distributed import checkpoint as dcp

    return getattr(dcp, "save", None) or getattr(dcp, "save_state_dict", None)


def _resolve_dcp_load() -> Any:
    from torch.distributed import checkpoint as dcp

    return getattr(dcp, "load", None) or getattr(dcp, "load_state_dict", None)


def _optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Return an FSDP-friendly optimizer state dict.

    `torch.distributed.checkpoint.state_dict.get_optimizer_state_dict` produces
    a state dict whose keys are the parameter *names*, which DCP can shard
    correctly across resharded loads. Falls back to the plain optimizer state
    dict on torch versions that don't expose the helper.
    """
    try:
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        result: dict[str, Any] = get_optimizer_state_dict(model, optimizer)
        return result
    except ImportError:
        fallback: dict[str, Any] = optimizer.state_dict()
        return fallback


def _restore_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> None:
    """Inverse of _optimizer_state_dict."""
    try:
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        set_optimizer_state_dict(model, optimizer, optim_state_dict=state)
    except ImportError:
        optimizer.load_state_dict(state)
