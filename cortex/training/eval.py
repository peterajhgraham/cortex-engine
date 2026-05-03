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
    """Coefficient of determination, averaged across output dimensions.

    R² = 1 - SS_res / SS_tot. Per-dimension R²s are computed separately and
    then averaged to produce the headline scalar — matches the NLB MC_Maze
    convention of reporting one R² per behavioral channel.

    Args:
        y_true: (N, D) ground truth
        y_pred: (N, D) predictions
    Returns:
        Scalar R² in (-inf, 1.0]. Returns 1.0 only when predictions are exact.
    """
    return float(r2_score_per_dim(y_true, y_pred).mean().item())


def r2_score_per_dim(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-output-dimension R². Useful for reporting x-velocity vs y-velocity.

    Returns a (D,) tensor. Constant-target dimensions get R² clamped to 0.0
    rather than producing -inf.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
        y_pred = y_pred.unsqueeze(-1)

    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    constant = ss_tot < 1e-12
    safe_ratio = ss_res / ss_tot.clamp(min=1e-8)
    per_dim = 1.0 - safe_ratio
    # Constant target → undefined R²; report 0 (= "no better than mean").
    return torch.where(constant, torch.zeros_like(per_dim), per_dim)


def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> EvalResults:
    """Run the model over the full loader and compute aggregate metrics."""
    with torch.no_grad():
        model.eval()

        all_pred: list[torch.Tensor] = []
        all_true: list[torch.Tensor] = []
        masked_correct = 0
        masked_total = 0

        for batch in loader:
            # Keep behavior labels on CPU — avoids an MPS non_blocking race where
            # the tensor hasn't committed before we immediately call .cpu().
            behavior_cpu = batch["behavior"].float()
            spike_keys = ("neuron_ids", "time_bins", "values", "batch_indices")
            batch_dev = {k: batch[k].to(device) for k in spike_keys}
            if "masked_spike_targets" in batch:
                batch_dev["masked_spike_targets"] = batch["masked_spike_targets"].to(device)

            out = model(
                neuron_ids=batch_dev["neuron_ids"],
                time_bins=batch_dev["time_bins"],
                values=batch_dev["values"],
                batch_indices=batch_dev["batch_indices"],
                return_aux=True,
            )
            all_pred.append(out["behavior"].float().cpu())
            all_true.append(behavior_cpu)

            if "masked_spike_logits" in out and "masked_spike_targets" in batch_dev:
                preds = out["masked_spike_logits"].argmax(dim=-1)
                masked_correct += int((preds == batch_dev["masked_spike_targets"]).sum().item())
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


def format_comparison_table(rows: dict[str, float]) -> str:
    """Render a baseline-vs-Cortex R² comparison as a markdown table.

    The training results doc consumes this directly.
    """
    lines = ["| Model | R² (hand velocity) |", "|---|---|"]
    for name, r2 in rows.items():
        lines.append(f"| {name} | {r2:.4f} |")
    return "\n".join(lines)
