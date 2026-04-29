"""INT8 calibration: collect activation statistics over a calibration set.

The calibration pass records min/max (or percentile) statistics per linear
layer's input activations. Scale factors are derived from these statistics.

Workflow:
    1. Load fp16/bf16 model
    2. Wrap each nn.Linear with a CalibrationHook that records input statistics
    3. Run N batches from a held-out calibration split
    4. Derive scales (per-channel weights, per-tensor activations)
    5. Replace nn.Linear with QuantizedLinear
    6. Save quantized weights in custom format
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ActivationStats:
    """Per-tensor activation statistics."""

    min_val: float = float("inf")
    max_val: float = float("-inf")
    abs_max_history: list[float] = field(default_factory=list)

    def update(self, x: torch.Tensor) -> None:
        self.min_val = min(self.min_val, x.min().item())
        self.max_val = max(self.max_val, x.max().item())
        self.abs_max_history.append(x.abs().max().item())

    @property
    def percentile_99(self) -> float:
        """99th-percentile abs max, more robust than raw max for activation scales."""
        if not self.abs_max_history:
            return 0.0
        sorted_vals = sorted(self.abs_max_history)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class CalibrationHook:
    """Forward pre-hook that records input activation statistics."""

    def __init__(self) -> None:
        self.stats = ActivationStats()

    def __call__(self, module: nn.Module, args: tuple) -> None:
        x = args[0]
        self.stats.update(x.detach())


def attach_calibration_hooks(model: nn.Module) -> dict[str, CalibrationHook]:
    """Attach hooks to every nn.Linear; return a name->hook map."""
    hooks: dict[str, CalibrationHook] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = CalibrationHook()
            module.register_forward_pre_hook(hook)
            hooks[name] = hook
    return hooks


# TODO Phase 2.Q1: derive_scales(hooks) -> per-layer scale dict
# TODO Phase 2.Q2: QuantizedLinear with INT8 weights, per-channel scales
# TODO Phase 2.Q3: convert_model(model, scales) -> quantized model
# TODO Phase 2.Q4: save/load custom INT8 weight format
