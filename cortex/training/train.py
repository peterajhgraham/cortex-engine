"""FSDP training loop for Cortex models.

Implementation notes for Claude Code:
    - Use FSDP2 (`torch.distributed.fsdp.fully_shard`) when on torch >= 2.5.
    - Mixed precision: bf16 compute, fp32 master weights, fp32 reductions.
    - Sharded checkpoints via `torch.distributed.checkpoint`.
    - W&B init must be rank-0 only.
    - Gradient accumulation should respect FSDP's `set_requires_gradient_sync`.

This file is a SCAFFOLD. Fill in the TODOs in order. The skeleton works on a
single GPU without distributed init (FSDP becomes a no-op wrapper).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from cortex.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainState:
    """Mutable training state. Snapshotted into checkpoints."""

    step: int
    epoch: int
    best_val_metric: float


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed if torchrun envvars are present.

    Returns:
        (rank, local_rank, world_size). All zeros for single-GPU.
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


@hydra.main(config_path="../../configs", config_name="cortex_s", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        log.info("training_start", config=OmegaConf.to_container(cfg, resolve=True))

    # TODO Phase 1.1: Build model from cfg.model
    # from cortex.models import CortexModel, CortexConfig
    # model_config = CortexConfig(**cfg.model)
    # model = CortexModel(model_config).to(device)

    # TODO Phase 1.2: Wrap with FSDP2
    # from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    # mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # for layer in model.encoder.self_attn_blocks:
    #     fully_shard(layer, mp_policy=mp_policy)
    # fully_shard(model, mp_policy=mp_policy)

    # TODO Phase 1.3: Build dataloaders from cfg.data
    # from cortex.data.nlb import build_dataloaders
    # train_loader, val_loader = build_dataloaders(cfg.data, world_size=world_size, rank=rank)

    # TODO Phase 1.4: Optimizer + LR schedule
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_steps)

    # TODO Phase 1.5: W&B init (rank 0 only)
    # if is_main_process() and cfg.runtime.wandb_enabled:
    #     import wandb
    #     wandb.init(project=cfg.runtime.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))

    # TODO Phase 1.6: Training loop with grad accumulation
    # state = TrainState(step=0, epoch=0, best_val_metric=float("-inf"))
    # while state.step < cfg.training.max_steps:
    #     for batch in train_loader:
    #         loss = compute_loss(model, batch)
    #         loss.backward()
    #         if (state.step + 1) % cfg.training.grad_accum_steps == 0:
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             scheduler.step()
    #         state.step += 1

    # TODO Phase 1.7: Sharded checkpoint save
    # from torch.distributed.checkpoint import save
    # save(state_dict, checkpoint_id=str(checkpoint_path))

    raise NotImplementedError("Fill in TODOs in cortex/training/train.py")


if __name__ == "__main__":
    main()
