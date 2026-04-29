"""Evaluation pipeline: R² for behavior decoding, accuracy for masked spike head.

The headline metric for the project is R² on hand velocity decoding (the NLB
standard for MC_Maze).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from cortex.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EvalResults:
    """Aggregate evaluation results."""

    r2_velocity: float
    mse_velocity: float
    masked_spike_accuracy: float | None
    n_samples: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "r2_velocity": self.r2_velocity,
            "mse_velocity": self.mse_velocity,
            "masked_spike_accuracy": self.masked_spike_accuracy,
            "n_samples": self.n_samples,
        }


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination, computed across the full eval set.

    R² = 1 - SS_res / SS_tot

    Args:
        y_true: (N, D) ground truth
        y_pred: (N, D) predictions
    Returns:
        Scalar R² averaged across output dimensions.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    # Per-dimension R², average for the headline number
    per_dim = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
    return float(per_dim.mean().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvalResults:
    """Run the model over the full loader and compute aggregate metrics."""
    model.eval()

    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []
    masked_correct = 0
    masked_total = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(
            neuron_ids=batch["neuron_ids"],
            time_bins=batch["time_bins"],
            values=batch["values"],
            batch_indices=batch["batch_indices"],
            return_aux=True,
        )
        all_pred.append(out["behavior"].float().cpu())
        all_true.append(batch["behavior"].float().cpu())

        if "masked_spike_logits" in out and "masked_spike_targets" in batch:
            preds = out["masked_spike_logits"].argmax(dim=-1)
            masked_correct += int((preds == batch["masked_spike_targets"]).sum().item())
            masked_total += int(preds.numel())

    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)
    r2 = r2_score(y_true, y_pred)
    mse = float(((y_true - y_pred) ** 2).mean().item())

    spike_acc = masked_correct / masked_total if masked_total > 0 else None

    log.info("eval_complete", r2=r2, mse=mse, n=len(y_true))

    return EvalResults(
        r2_velocity=r2,
        mse_velocity=mse,
        masked_spike_accuracy=spike_acc,
        n_samples=len(y_true),
    )
