"""Wiener-filter baseline integration test.

The synthetic signal is linearly recoverable from the spike window, so a
correctly fit ridge decoder should achieve high R² on a held-out split.
"""

from __future__ import annotations

import numpy as np
import torch

from cortex.data.nlb import NLBDataset, SessionData
from cortex.training.baselines import WienerFilter, fit_wiener, make_dense_loader


def _linear_session(n_bins: int = 2400, n_units: int = 16, seed: int = 0) -> SessionData:
    rng = np.random.default_rng(seed)
    spikes = rng.poisson(lam=0.4, size=(n_bins, n_units)).astype(np.int32)
    proj = rng.normal(size=(n_units, 2)).astype(np.float32) * 0.1
    behavior = (spikes.astype(np.float32) @ proj).astype(np.float32)
    behavior = behavior + rng.normal(scale=0.01, size=behavior.shape).astype(np.float32)
    return SessionData("synth", spikes, behavior, neuron_id_offset=0)


def _make_split(split: str, sessions: list[SessionData]) -> NLBDataset:
    return NLBDataset(
        data_root="/tmp/unused",
        dandiset_id="000128",
        split=split,
        bin_size_ms=5,
        window_ms=50,
        stride_ms=25,
        max_neurons=64,
        spike_value_buckets=8,
        download=False,
        sessions=sessions,
    )


def test_wiener_filter_recovers_linear_signal() -> None:
    sessions = [_linear_session(seed=10)]
    train_loader = make_dense_loader(_make_split("train", sessions), batch_size=32, shuffle=False)
    val_loader = make_dense_loader(_make_split("val", sessions), batch_size=32, shuffle=False)

    _, r2 = fit_wiener(train_loader, val_loader, behavior_dim=2, alpha=1e-2)
    assert r2 > 0.5, f"Wiener failed to recover linear signal: R²={r2}"


def test_wiener_filter_module_shape_check() -> None:
    m = WienerFilter(n_features=20, behavior_dim=2)
    x = torch.zeros(4, 20)
    assert m(x).shape == (4, 2)
