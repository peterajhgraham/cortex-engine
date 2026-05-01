"""Tests for device selection precedence."""

from __future__ import annotations

from unittest.mock import patch

import torch

from cortex.utils.device import pin_memory_supported, select_device


def test_select_device_picks_cpu_when_nothing_else_available() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("cortex.utils.device._mps_ready", return_value=False),
    ):
        device = select_device(preference="auto")
    assert device.type == "cpu"


def test_select_device_prefers_cuda_in_auto_when_present() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("cortex.utils.device._mps_ready", return_value=True),
    ):
        device = select_device(preference="auto", local_rank=0)
    assert device.type == "cuda"


def test_select_device_falls_back_to_mps_when_no_cuda() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("cortex.utils.device._mps_ready", return_value=True),
    ):
        device = select_device(preference="auto")
    assert device.type == "mps"


def test_select_device_explicit_mps_works_without_cuda() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("cortex.utils.device._mps_ready", return_value=True),
    ):
        device = select_device(preference="mps")
    assert device.type == "mps"


def test_select_device_explicit_cuda_falls_back_to_cpu_when_unavailable() -> None:
    """Explicit cuda request without cuda must NOT silently use mps — that
    would surprise a user who intended to verify cuda is wired up."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("cortex.utils.device._mps_ready", return_value=True),
    ):
        device = select_device(preference="cuda")
    assert device.type == "cpu"


def test_select_device_explicit_cpu_short_circuits() -> None:
    device = select_device(preference="cpu")
    assert device.type == "cpu"


def test_select_device_sets_mps_fallback_env_var(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("cortex.utils.device._mps_ready", return_value=True),
    ):
        select_device(preference="auto")
    import os

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_pin_memory_supported_only_on_cuda() -> None:
    assert pin_memory_supported(torch.device("cuda")) is True
    assert pin_memory_supported(torch.device("mps")) is False
    assert pin_memory_supported(torch.device("cpu")) is False
