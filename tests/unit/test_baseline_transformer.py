"""Vanilla Transformer baseline integration test.

Same synthetic dataset as the other baselines. We just check the harness runs
end-to-end and produces a finite R² — convergence quality varies on tiny CPU
runs and isn't the point of this test.
"""

from __future__ import annotations

import numpy as np
import torch

from cortex.data.nlb import NLBDataset, SessionData
from cortex.training.baselines import VanillaTransformer, make_dense_loader, train_transformer


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


def test_transformer_baseline_runs_end_to_end() -> None:
    sessions = [_session(seed=12)]
    train_loader = make_dense_loader(_split("train", sessions), batch_size=8, shuffle=True)
    val_loader = make_dense_loader(_split("val", sessions), batch_size=8, shuffle=False)

    model, r2 = train_transformer(
        train_loader,
        val_loader,
        n_neurons=16,
        behavior_dim=2,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        max_time_bins=64,
        lr=3e-3,
        epochs=2,
    )
    assert isinstance(model, VanillaTransformer)
    assert torch.isfinite(torch.tensor(r2))


def test_vanilla_transformer_forward_shapes() -> None:
    model = VanillaTransformer(
        n_neurons=16,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        max_time_bins=64,
        behavior_dim=2,
    )
    x = torch.zeros(2, 10, 16)
    assert model(x).shape == (2, 2)
