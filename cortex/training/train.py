"""FSDP training loop for Cortex models.

Single entry point for all three model sizes. The same code path runs:
    - single-GPU (no distributed init, no FSDP wrap)
    - single-node multi-GPU via torchrun (FSDP2 wrap, NCCL)
    - multi-node via torchrun (same wrapping logic)

Mixed precision is bf16 compute / fp32 reductions, applied through FSDP's
`MixedPrecisionPolicy` so unsharded single-GPU training stays in fp32 by
default — flip `runtime.mixed_precision` to opt in.

References:
    PyTorch FSDP2 docs: https://docs.pytorch.org/docs/stable/distributed.fsdp.html
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn

from cortex.data.nlb import build_dataloaders
from cortex.models import CortexConfig, CortexModel
from cortex.training.checkpoint import save_checkpoint
from cortex.training.eval import evaluate
from cortex.utils.device import select_device
from cortex.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


# ── Distributed setup ──────────────────────────────────────────────────────────


@dataclass
class TrainState:
    """Mutable training state. Snapshotted into checkpoints."""

    step: int
    epoch: int
    best_val_metric: float


def setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed if torchrun envvars are present.

    Returns (rank, local_rank, world_size). All zeros for single-GPU.
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ── Model construction + FSDP wrap ─────────────────────────────────────────────


def build_model(model_cfg: DictConfig) -> CortexModel:
    """Build a CortexModel from a Hydra/Omega model config."""
    config = CortexConfig(**cast(dict[str, Any], OmegaConf.to_container(model_cfg, resolve=True)))
    return CortexModel(config)


def maybe_wrap_fsdp(
    model: nn.Module,
    *,
    enable_mixed_precision: bool,
) -> nn.Module:
    """Wrap each Perceiver self-attention block + the model root with FSDP2.

    Skipped when not running under torch.distributed; on single-GPU we keep
    the bare model so we don't pay distributed-collective overhead.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return model

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as e:
        raise ImportError(
            "FSDP2 requires torch >= 2.5. Upgrade torch or run on a single GPU."
        ) from e

    mp_policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if enable_mixed_precision
        else None
    )

    # Shard each transformer block first, then the root, so the root's
    # forward sees already-sharded children and can stage all-gathers
    # block-by-block rather than once for the whole model.
    if isinstance(model, CortexModel):
        for block in model.encoder.self_attn_blocks:
            fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    return model


# ── Optimizer + LR schedule ────────────────────────────────────────────────────


def build_optimizer(model: nn.Module, training_cfg: DictConfig) -> torch.optim.Optimizer:
    """AdamW with config-driven lr/wd/betas. Decay applied to all params for now."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
        betas=tuple(training_cfg.betas),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup → cosine decay to 0 over [warmup_steps, max_steps]."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Loss ───────────────────────────────────────────────────────────────────────


def compute_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    behavior_weight: float,
    masked_spike_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward pass + weighted multi-task loss.

    Returns (loss, components_dict). The masked-spike head is included only
    when both the model has it AND the loss weight is positive — keeps the
    supervised-only training path zero-overhead.
    """
    use_aux = masked_spike_weight > 0.0 and getattr(model, "masked_spike_head", None) is not None
    out = model(
        neuron_ids=batch["neuron_ids"],
        time_bins=batch["time_bins"],
        values=batch["values"],
        batch_indices=batch["batch_indices"],
        return_aux=use_aux,
    )
    behavior_loss = nn.functional.mse_loss(out["behavior"], batch["behavior"])
    components = {"behavior_loss": float(behavior_loss.detach().item())}

    total: torch.Tensor = behavior_weight * behavior_loss
    if use_aux and "masked_spike_logits" in out and "masked_spike_targets" in batch:
        spike_loss = nn.functional.cross_entropy(
            out["masked_spike_logits"], batch["masked_spike_targets"]
        )
        total = total + masked_spike_weight * spike_loss
        components["masked_spike_loss"] = float(spike_loss.detach().item())

    return total, components


# ── Main training loop ─────────────────────────────────────────────────────────


@hydra.main(config_path="../../configs", config_name="cortex_s", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size = setup_distributed()
    configure_logging(level=cfg.runtime.log_level, json=cfg.runtime.log_json)
    torch.manual_seed(cfg.seed + rank)

    device = select_device(preference=cfg.runtime.device, local_rank=local_rank)

    if is_main_process():
        log.info(
            "training_start",
            world_size=world_size,
            device=str(device),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Model + FSDP
    model = build_model(cfg.model).to(device)
    model = maybe_wrap_fsdp(model, enable_mixed_precision=cfg.runtime.mixed_precision)
    if cfg.runtime.compile:
        model = cast(nn.Module, torch.compile(model))

    # Data
    train_loader, val_loader = build_dataloaders(cfg.data, world_size=world_size, rank=rank)

    # Optim + schedule
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training.warmup_steps, cfg.training.max_steps)

    # W&B
    wandb_run: Any = None
    if is_main_process() and cfg.runtime.wandb_enabled:
        wandb_run = _init_wandb(cfg)

    state = TrainState(step=0, epoch=0, best_val_metric=float("-inf"))
    checkpoint_dir = Path(cfg.runtime.checkpoint_dir) / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_training_loop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            state=state,
            device=device,
            checkpoint_dir=checkpoint_dir,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


def run_training_loop(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: Any,
    val_loader: Any,
    cfg: DictConfig,
    state: TrainState,
    device: torch.device,
    checkpoint_dir: Path,
    wandb_run: Any,
) -> None:
    """The actual training loop, separated from main() for testability."""
    accum = cfg.training.grad_accum_steps
    max_steps = cfg.training.max_steps
    behavior_w = cfg.training.behavior_loss_weight
    masked_w = cfg.training.masked_spike_loss_weight

    while state.step < max_steps:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(state.epoch)

        for batch in train_loader:
            if state.step >= max_steps:
                break
            batch = _move_batch(batch, device)

            model.train()
            loss, components = compute_loss(
                model,
                batch,
                behavior_weight=behavior_w,
                masked_spike_weight=masked_w,
            )
            (loss / accum).backward()

            if (state.step + 1) % accum == 0:
                if cfg.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            state.step += 1

            if is_main_process() and state.step % 50 == 0:
                log.info(
                    "train_step",
                    step=state.step,
                    epoch=state.epoch,
                    lr=scheduler.get_last_lr()[0],
                    loss=float(loss.detach().item()),
                    **components,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach().item()),
                            "train/lr": scheduler.get_last_lr()[0],
                            **{f"train/{k}": v for k, v in components.items()},
                        },
                        step=state.step,
                    )

            if state.step > 0 and state.step % cfg.training.eval_every == 0:
                results = evaluate(model, val_loader, device)
                if is_main_process():
                    log.info("eval_results", step=state.step, **results.as_dict())
                    if wandb_run is not None:
                        wandb_run.log(
                            {f"val/{k}": v for k, v in results.as_dict().items() if v is not None},
                            step=state.step,
                        )
                if results.r2_velocity > state.best_val_metric:
                    state.best_val_metric = results.r2_velocity
                    if is_main_process():
                        best_path = checkpoint_dir / "best"
                        save_checkpoint(best_path, model, optimizer, asdict(state))
                        _log_checkpoint_artifact(wandb_run, best_path, name="best")

            if (
                is_main_process()
                and state.step > 0
                and state.step % cfg.training.checkpoint_every == 0
            ):
                step_path = checkpoint_dir / f"step-{state.step}"
                save_checkpoint(step_path, model, optimizer, asdict(state))
                _log_checkpoint_artifact(wandb_run, step_path, name=f"step-{state.step}")

        state.epoch += 1

    # Final checkpoint
    if is_main_process():
        final_path = checkpoint_dir / "final"
        save_checkpoint(final_path, model, optimizer, asdict(state))
        _log_checkpoint_artifact(wandb_run, final_path, name="final")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _init_wandb(cfg: DictConfig) -> Any:
    """Lazy-import W&B and init a run on rank 0 only."""
    try:
        import wandb
    except ImportError as e:
        raise ImportError("wandb_enabled=true but wandb is not installed") from e
    return wandb.init(
        project=cfg.runtime.wandb_project,
        entity=cfg.runtime.wandb_entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _log_checkpoint_artifact(wandb_run: Any, ckpt_path: Path, name: str) -> None:
    """Upload a checkpoint directory as a W&B artifact. No-op if W&B is off.

    The artifact is reusable across runs via wandb's content-addressed storage,
    so re-uploading the same content is cheap.
    """
    if wandb_run is None:
        return
    try:
        import wandb
    except ImportError:
        return
    artifact = wandb.Artifact(name=f"{wandb_run.name}-{name}", type="model")
    artifact.add_dir(str(ckpt_path))
    wandb_run.log_artifact(artifact)


if __name__ == "__main__":
    main()
