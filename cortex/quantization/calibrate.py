"""INT8 post-training quantization with calibration for Cortex models.

Approach: per-channel weight quantization + per-tensor activation quantization.
The QuantizedLinear forward pass dequantizes weights to the compute dtype before
the matmul, so no device-specific int8 kernel is required.  This trades compute
efficiency for portability — the primary Phase 2 goal is the memory reduction.
True int8 matmul (via torch.int8 + CUDA cublas) is a Phase 3 optimisation.

Quantization scheme
-------------------
Weights (per output channel):
    scale_w[c] = max(|W[c, :]|) / 127.0
    W_int8[c, :] = round(W[c, :] / scale_w[c]).clamp(-127, 127)
    Reconstruction: W_dq[c, :] = W_int8[c, :] x scale_w[c]
    Error: |W - W_dq| ≤ 0.5 x scale_w[c]   (at most 0.5 LSB per element)

Activations (per tensor, tracked across calibration batches):
    scale_a = percentile_99(|x|) / 127.0
    Stored for reference; used if hardware supports int8 matmul.

Why per-channel weights?
    In transformer linear layers, weight rows (output neurons) span very
    different dynamic ranges.  Per-channel scales prevent the narrow-range
    rows from being drowned out by one large-magnitude row.  Per-tensor weight
    quantization loses ~0.5-2% R² in typical transformer decoders; per-channel
    typically loses <0.1%.

Skipped layers
--------------
    Embedding tables: already accessed sparsely (index → row), so the
        effective compute is a gather, not a matmul. Not quantized.
    LayerNorm: tiny parameter count; numerical stability requires fp32.
    Bias vectors: per-neuron biases are tiny; not quantized.

Memory savings (theoretical, weights only):
    float32 → int8:  x0.25 of linear-layer weight storage
    bfloat16 → int8: x0.50 of linear-layer weight storage
    At Cortex-S scale: ~35 linear layers, ~22 M weight parameters.
    bf16 weight storage: 44 MB → INT8: 22 MB (saves 22 MB / 44.2% reduction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Calibration: collect activation statistics ────────────────────────────────


@dataclass
class ActivationStats:
    """Per-layer activation statistics collected during calibration."""

    abs_max_history: list[float] = field(default_factory=list)

    def update(self, x: torch.Tensor) -> None:
        self.abs_max_history.append(float(x.detach().abs().max().item()))

    @property
    def percentile_99(self) -> float:
        """99th-percentile abs-max across calibration batches.

        More robust than the raw maximum: a single batch with an outlier
        activation spike will not inflate the scale for all batches.
        """
        if not self.abs_max_history:
            return 1.0
        vals = sorted(self.abs_max_history)
        idx = max(0, int(len(vals) * 0.99) - 1)
        return vals[idx]

    @property
    def n_batches(self) -> int:
        return len(self.abs_max_history)


class CalibrationHook:
    """Forward pre-hook that records input activation statistics."""

    def __init__(self) -> None:
        self.stats = ActivationStats()
        self._handle: Any = None

    def __call__(self, module: nn.Module, args: tuple) -> None:
        if args:
            self.stats.update(args[0])

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def attach_calibration_hooks(model: nn.Module) -> dict[str, CalibrationHook]:
    """Attach a forward pre-hook to every nn.Linear; return name → hook map."""
    hooks: dict[str, CalibrationHook] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = CalibrationHook()
            hook._handle = module.register_forward_pre_hook(hook)
            hooks[name] = hook
    return hooks


def remove_calibration_hooks(hooks: dict[str, CalibrationHook]) -> None:
    """Remove all hooks added by attach_calibration_hooks."""
    for hook in hooks.values():
        hook.remove()


# ── Scale derivation ──────────────────────────────────────────────────────────


@dataclass
class LayerScales:
    """Quantization scales for one linear layer."""

    weight_scale: torch.Tensor  # (out_features,) float32 — per output channel
    act_scale: float  # scalar — derived from calibration stats


def derive_scales(
    model: nn.Module,
    hooks: dict[str, CalibrationHook],
) -> dict[str, LayerScales]:
    """Compute per-layer INT8 quantization scales from calibration statistics.

    Args:
        model: The calibrated model (weights must be in their original dtype).
        hooks: Name → CalibrationHook map from attach_calibration_hooks.

    Returns:
        Name → LayerScales dict.  Only layers present in both model and hooks
        are included (skips layers that never received activation data).
    """
    scales: dict[str, LayerScales] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in hooks:
            continue
        hook = hooks[name]
        if hook.stats.n_batches == 0:
            continue

        # Per-channel weight scale: absmax per output neuron / 127
        w = module.weight.detach().float()
        weight_scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0  # (out_features,)

        # Per-tensor activation scale: 99th-percentile absmax / 127
        act_scale = max(hook.stats.percentile_99, 1e-8) / 127.0

        scales[name] = LayerScales(
            weight_scale=weight_scale.cpu(),
            act_scale=act_scale,
        )

    return scales


# ── QuantizedLinear ───────────────────────────────────────────────────────────


class QuantizedLinear(nn.Module):
    """nn.Linear replacement with INT8 per-channel weight quantization.

    Weights are stored as int8 tensors.  On forward, they are dequantized to
    the input dtype before the matmul — no CUDA-specific int8 kernel required.
    This gives the full memory savings of INT8 storage while remaining
    compatible with MPS, CPU, and CUDA.

    Note on compute: dequantization adds one elementwise multiply per forward
    pass (weight_int8 x scale → fp).  At Cortex-S scale this is negligible
    (22 M elements x one operation).  True int8 matmul via cuBLAS is a future
    optimisation that would halve compute cost too.
    """

    in_features: int
    out_features: int

    def __init__(
        self,
        weight_int8: torch.Tensor,  # (out_features, in_features), int8
        weight_scale: torch.Tensor,  # (out_features,), float32
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = weight_int8.shape[1]
        self.out_features = weight_int8.shape[0]

        # Buffers participate in state_dict and device transfers; they are not
        # optimiser parameters.
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scale", weight_scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None  # type: ignore[assignment]

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        scales: LayerScales,
    ) -> QuantizedLinear:
        """Quantize a float nn.Linear using pre-computed scales."""
        w = linear.weight.detach().float()

        # Per-channel quantization: divide each output row by its scale
        w_scaled = w / scales.weight_scale.to(w.device)[:, None]
        w_int8 = w_scaled.round().clamp(-127, 127).to(torch.int8)

        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(w_int8, scales.weight_scale.to(w.device), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: (out, in) int8 x (out, 1) float32 → compute dtype
        w_dq = self.weight_int8.float() * self.weight_scale[:, None]
        return F.linear(x, w_dq.to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantized=INT8"
        )


# ── Model conversion ──────────────────────────────────────────────────────────


def convert_model(
    model: nn.Module,
    scales: dict[str, LayerScales],
    inplace: bool = True,
) -> nn.Module:
    """Replace nn.Linear layers with QuantizedLinear where scales exist.

    Layers without scales (e.g., those not seen during calibration, or layers
    explicitly excluded) are left in their original dtype.

    Args:
        model:   The model to quantize.
        scales:  Per-layer scales from derive_scales().
        inplace: If True, modify model in-place; if False, return a deep copy.

    Returns:
        Quantized model.
    """
    import copy

    if not inplace:
        model = copy.deepcopy(model)

    _replace_linears(model, scales, prefix="")
    return model


def _replace_linears(
    module: nn.Module,
    scales: dict[str, LayerScales],
    prefix: str,
) -> None:
    """Recursively walk module tree, replacing nn.Linear where scales exist."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}".lstrip(".")

        if isinstance(child, nn.Linear) and full_name in scales:
            setattr(module, name, QuantizedLinear.from_linear(child, scales[full_name]))
        else:
            _replace_linears(child, scales, full_name)


# ── Model size utilities ──────────────────────────────────────────────────────


def model_weight_bytes(model: nn.Module) -> int:
    """Count bytes used by all parameter and buffer tensors in the model."""
    total = 0
    for t in model.parameters():
        total += t.numel() * t.element_size()
    for t in model.buffers():
        total += t.numel() * t.element_size()
    return total


def count_quantized_linears(model: nn.Module) -> tuple[int, int]:
    """Return (n_quantized, n_total) linear-type layers."""
    n_quant = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    n_float = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return n_quant, n_quant + n_float


# ── Save / load ───────────────────────────────────────────────────────────────


def save_quantized(model: nn.Module, path: Path, meta: dict | None = None) -> None:
    """Save a quantized model's state dict + metadata.

    Format: torch.save({'state_dict': ..., 'meta': ...})
    The state dict contains int8 weight buffers and float32 scale buffers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_quantized(path: Path, model: nn.Module) -> nn.Module:
    """Load quantized state dict into an already-converted model skeleton.

    The model must already have QuantizedLinear layers in the right positions;
    this function only loads the weights.

    Usage:
        model = CortexModel(config)
        scales = derive_scales(model, hooks)
        model = convert_model(model, scales)
        save_quantized(model, "cortex_s_int8.pt")
        ...
        model_loaded = CortexModel(config)
        model_loaded = convert_model(model_loaded, scales)  # same scales
        load_quantized("cortex_s_int8.pt", model_loaded)
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return model


# ── Calibration summary ───────────────────────────────────────────────────────


def calibration_summary(
    hooks: dict[str, CalibrationHook],
    scales: dict[str, LayerScales],
) -> list[dict]:
    """Return a list of per-layer calibration statistics for logging / reporting."""
    rows = []
    for name, hook in hooks.items():
        s = scales.get(name)
        rows.append(
            {
                "layer": name,
                "calib_batches": hook.stats.n_batches,
                "act_abs_max_p99": round(hook.stats.percentile_99, 6),
                "act_scale": round(s.act_scale, 8) if s else None,
                "weight_scale_min": round(float(s.weight_scale.min()), 8) if s else None,
                "weight_scale_max": round(float(s.weight_scale.max()), 8) if s else None,
            }
        )
    return rows
