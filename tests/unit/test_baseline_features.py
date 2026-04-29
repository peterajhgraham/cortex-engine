"""Shared dense-feature pipeline used by all three baselines.

Verifies the DenseWindowDataset / collate / loader plumbing produces the
same windows as NLBDataset, with cross-session neuron offsets respected.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cortex.data.nlb import NLBDataset, SessionData
from cortex.training.baselines import DenseWindowDataset, collate_dense, make_dense_loader


def _linear_session(n_bins: int = 600, n_units: int = 16, seed: int = 0) -> SessionData:
    rng = np.random.default_rng(seed)
    spikes = rng.poisson(lam=0.4, size=(n_bins, n_units)).astype(np.int32)
    proj = rng.normal(size=(n_units, 2)).astype(np.float32) * 0.1
    behavior = (spikes.astype(np.float32) @ proj).astype(np.float32)
    return SessionData("synth", spikes, behavior, neuron_id_offset=0)


def _make_nlb(seed: int) -> NLBDataset:
    return NLBDataset(
        data_root="/tmp/unused",
        dandiset_id="000128",
        split="train",
        bin_size_ms=5,
        window_ms=50,
        stride_ms=25,
        max_neurons=64,
        spike_value_buckets=8,
        download=False,
        sessions=[_linear_session(seed=seed)],
    )


def test_dense_window_dataset_shapes() -> None:
    nlb = _make_nlb(seed=0)
    dense = DenseWindowDataset(nlb)
    item = dense[0]
    assert item["features"].shape == (10, 16)
    assert item["features"].dtype == torch.float32
    assert item["behavior"].shape == (2,)


def test_dense_window_dataset_places_neurons_at_offset() -> None:
    s1 = _linear_session(seed=1)
    s2_base = _linear_session(seed=2)
    s2 = SessionData(
        session_id=s2_base.session_id,
        bin_counts=s2_base.bin_counts,
        behavior=s2_base.behavior,
        neuron_id_offset=s1.bin_counts.shape[1],
    )
    nlb = NLBDataset(
        data_root="/tmp/unused",
        dandiset_id="000128",
        split="train",
        bin_size_ms=5,
        window_ms=50,
        stride_ms=25,
        max_neurons=64,
        download=False,
        sessions=[s1, s2],
    )
    dense = DenseWindowDataset(nlb)
    n1 = s1.bin_counts.shape[1]

    for idx in range(len(dense)):
        if nlb._windows[idx][0] == 1:  # session-1 window
            item = dense[idx]
            assert torch.all(item["features"][:, :n1] == 0)
            assert item["features"][:, n1:].abs().sum() > 0
            return
    pytest.fail("No session-1 windows in dataset; raise n_bins.")


def test_collate_dense_stacks_correctly() -> None:
    nlb = _make_nlb(seed=0)
    dense = DenseWindowDataset(nlb)
    items = [dense[i] for i in range(3)]
    batch = collate_dense(items)
    assert batch.features.shape == (3, 10, 16)
    assert batch.behavior.shape == (3, 2)


def test_make_dense_loader_yields_dense_batches() -> None:
    loader = make_dense_loader(_make_nlb(seed=0), batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.features.shape[0] == 4
    assert batch.features.ndim == 3
    assert batch.behavior.shape[0] == 4
