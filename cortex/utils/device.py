"""Accelerator selection: CUDA → MPS → CPU.

The training pipeline was designed for CUDA, but Apple Silicon dev machines
have only MPS. We pick the best available backend at runtime so the same
training command works on a Mac laptop and a CUDA cluster.

Order of preference:
    1. CUDA (only if explicitly requested or auto-detected)
    2. MPS  (Apple Silicon GPU via Metal)
    3. CPU  (always available)

Some torch ops still don't have MPS kernels in torch 2.2; the runtime will
fall back to CPU per-op when that happens (MPS_FALLBACK env). We log when
this happens so unexpectedly-slow training doesn't look like a bug.
"""

from __future__ import annotations

import os

import torch

from cortex.utils.logging import get_logger

log = get_logger(__name__)


def select_device(preference: str = "auto", local_rank: int = 0) -> torch.device:
    """Resolve a torch.device given a runtime preference.

    Args:
        preference: "auto" | "cuda" | "mps" | "cpu". "auto" picks the best
            backend present on the host.
        local_rank: GPU index for multi-GPU CUDA. Ignored for MPS / CPU.

    Returns:
        torch.device, with a structured log entry recording what was picked.
    """
    pref = preference.lower()

    if pref in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif pref in ("auto", "mps") and _mps_ready():
        # Tell torch to fall back to CPU for ops that lack an MPS kernel
        # rather than crashing. Only set if the user hasn't already opted out.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        device = torch.device("mps")
    else:
        if pref == "cuda":
            log.warning("cuda_requested_unavailable_falling_back", target="cpu")
        elif pref == "mps":
            log.warning("mps_requested_unavailable_falling_back", target="cpu")
        device = torch.device("cpu")

    log.info("device_selected", device=str(device), preference=pref)
    return device


def _mps_ready() -> bool:
    """torch.backends.mps may be missing on non-mac builds; check defensively."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available()) and bool(backend.is_built())


def pin_memory_supported(device: torch.device) -> bool:
    """pin_memory only works for CUDA; MPS and CPU should leave it off."""
    return bool(device.type == "cuda")
