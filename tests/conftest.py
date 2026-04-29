"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch

from cortex.models.config import CORTEX_XS, CortexConfig


@pytest.fixture
def cortex_xs_config() -> CortexConfig:
    """The smallest model config, suitable for fast CPU tests."""
    return CORTEX_XS


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_event_batch(device: torch.device) -> dict[str, torch.Tensor]:
    """A small batch of synthetic spike events for unit tests."""
    torch.manual_seed(0)
    E = 256
    return {
        "neuron_ids": torch.randint(0, 256, (E,), device=device, dtype=torch.int64),
        "time_bins": torch.randint(0, 512, (E,), device=device, dtype=torch.int64),
        "values": torch.randint(0, 8, (E,), device=device, dtype=torch.int64),
        "batch_indices": torch.randint(0, 4, (E,), device=device, dtype=torch.int64).sort().values,
    }
