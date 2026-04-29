"""GRU baseline integration test.

Trains for a handful of epochs on a synthetic linear-recoverable dataset and
verifies the model converges to a finite R² without diverging.
"""

from __future__ import annotations

import numpy as np
import torch

from cortex.data.nlb import NLBDataset, SessionData
from cortex.training.baselines import GRUDecoder, make_dense_loader, train_gru


def _session(seed: int) -> SessionData:
    rng = np.random.default_rng(seed)
    spikes = rng.poisson(lam=0.4, size=(2400, 16)).astype(np.int32)
    proj = rng.normal(size=(16, 2)).astype(np.float32) * 0.1
    behavior = (spikes.astype(np.float32) @ proj).astype(np.float32)
    return SessionData("synth", spikes, behavior, neuron_id_offset=0)


def _split(split: str, sessions: list[SessionData]) -> NLBDataset:
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


def test_gru_baseline_trains_and_reports_finite_r2() -> None:
    sessions = [_session(seed=11)]
    train_loader = make_dense_loader(_split("train", sessions), batch_size=16, shuffle=True)
    val_loader = make_dense_loader(_split("val", sessions), batch_size=16, shuffle=False)

    model, r2 = train_gru(
        train_loader,
        val_loader,
        n_neurons=16,
        behavior_dim=2,
        hidden_dim=32,
        num_layers=1,
        lr=3e-3,
        epochs=4,
    )
    assert isinstance(model, GRUDecoder)
    assert torch.isfinite(torch.tensor(r2))
    assert r2 > -1.0


def test_gru_decoder_forward_shapes() -> None:
    model = GRUDecoder(n_neurons=16, hidden_dim=32, num_layers=1, behavior_dim=2)
    x = torch.zeros(4, 10, 16)
    assert model(x).shape == (4, 2)
