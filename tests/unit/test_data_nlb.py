"""Unit tests for the NLB data loader.

The DANDI download path is exercised separately as an integration test (slow,
network-bound). These tests use synthetic SessionData to validate every
non-I/O code path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cortex.data.nlb import (
    NLBDataset,
    SessionData,
    _split_windows,
    bin_spikes,
    bucket_values,
    build_window_index,
    collate_events,
    events_from_bin_matrix,
)


def test_bin_spikes_counts() -> None:
    spikes = np.array([0.001, 0.002, 0.006, 0.013], dtype=np.float64)
    counts = bin_spikes(spikes, n_bins=4, bin_size_s=0.005)
    # bin0 [0,0.005): 2 spikes. bin1 [0.005,0.010): 1. bin2 [0.010,0.015): 1. bin3: 0.
    np.testing.assert_array_equal(counts, [2, 1, 1, 0])


def test_bin_spikes_excludes_out_of_range() -> None:
    spikes = np.array([-0.1, 0.0, 0.05, 1.0], dtype=np.float64)
    counts = bin_spikes(spikes, n_bins=10, bin_size_s=0.01)
    # Only 0.0 and 0.05 fall inside [0, 0.1). 1.0 is past the right edge.
    assert counts.sum() == 2


def test_bin_spikes_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        bin_spikes(np.array([0.0]), n_bins=0, bin_size_s=0.01)
    with pytest.raises(ValueError):
        bin_spikes(np.array([0.0]), n_bins=10, bin_size_s=0.0)


def test_bucket_values_clamps_high_counts() -> None:
    counts = np.array([0, 1, 5, 7, 100])
    bucketed = bucket_values(counts, n_buckets=8)
    np.testing.assert_array_equal(bucketed, [0, 1, 5, 7, 7])
    assert bucketed.dtype == np.int64


def test_events_from_bin_matrix_extracts_only_nonzero() -> None:
    matrix = np.array(
        [
            [0, 1, 0],
            [2, 0, 3],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    neuron_ids, time_bins, values = events_from_bin_matrix(matrix, n_buckets=8)
    # Sort by (time, neuron) for deterministic comparison.
    order = np.lexsort((neuron_ids, time_bins))
    np.testing.assert_array_equal(time_bins[order], [0, 1, 1])
    np.testing.assert_array_equal(neuron_ids[order], [1, 0, 2])
    np.testing.assert_array_equal(values[order], [1, 2, 3])


def test_build_window_index_dense() -> None:
    assert build_window_index(10, window_bins=4, stride_bins=2) == [
        (0, 4),
        (2, 6),
        (4, 8),
        (6, 10),
    ]


def test_build_window_index_drops_partial_tail() -> None:
    # 11 bins, window 4, stride 3: starts at 0, 3, 6 → last full window ends at 10.
    assert build_window_index(11, window_bins=4, stride_bins=3) == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]


def test_build_window_index_empty_when_too_short() -> None:
    assert build_window_index(3, window_bins=4, stride_bins=1) == []


def test_split_windows_deterministic_and_disjoint() -> None:
    windows = [(0, i, i + 1) for i in range(100)]
    train = _split_windows(windows, "train", seed=42, fractions=(0.8, 0.1, 0.1))
    val = _split_windows(windows, "val", seed=42, fractions=(0.8, 0.1, 0.1))
    test = _split_windows(windows, "test", seed=42, fractions=(0.8, 0.1, 0.1))
    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    # Re-running with the same seed produces the same partition.
    assert _split_windows(windows, "train", seed=42, fractions=(0.8, 0.1, 0.1)) == train


def test_split_windows_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        _split_windows([], "wrong-split", seed=0, fractions=(0.8, 0.1, 0.1))
    with pytest.raises(ValueError):
        _split_windows([], "train", seed=0, fractions=(0.5, 0.5, 0.5))


# ── Dataset-level tests via synthetic SessionData ──────────────────────────────


def _synthetic_session(
    n_bins: int,
    n_units: int,
    neuron_id_offset: int = 0,
    seed: int = 0,
) -> SessionData:
    rng = np.random.default_rng(seed)
    bin_counts = rng.poisson(lam=0.3, size=(n_bins, n_units)).astype(np.int32)
    # Fake hand velocity: smoothly varying 2D signal.
    t = np.arange(n_bins, dtype=np.float32) / max(n_bins - 1, 1)
    behavior = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1).astype(np.float32)
    return SessionData(
        session_id=f"synthetic-{seed}",
        bin_counts=bin_counts,
        behavior=behavior,
        neuron_id_offset=neuron_id_offset,
    )


def _make_dataset(split: str, sessions: list[SessionData], **overrides: object) -> NLBDataset:
    kwargs: dict[str, object] = {
        "data_root": "/tmp/unused",
        "dandiset_id": "000128",
        "split": split,
        "bin_size_ms": 5,
        "window_ms": 100,
        "stride_ms": 50,
        "max_neurons": 256,
        "spike_value_buckets": 8,
        "behavior_dim": 2,
        "download": False,
        "sessions": sessions,
    }
    kwargs.update(overrides)
    return NLBDataset(**kwargs)  # type: ignore[arg-type]


def test_dataset_yields_correct_shapes_and_dtypes() -> None:
    session = _synthetic_session(n_bins=400, n_units=32)
    ds = _make_dataset("train", [session])

    assert len(ds) > 0
    item = ds[0]
    assert set(item) == {"neuron_ids", "time_bins", "values", "behavior"}
    assert item["neuron_ids"].dtype == torch.int64
    assert item["time_bins"].dtype == torch.int64
    assert item["values"].dtype == torch.int64
    assert item["behavior"].shape == (2,)
    assert item["behavior"].dtype == torch.float32

    # All event tensors share length E.
    e = item["neuron_ids"].numel()
    assert item["time_bins"].numel() == e
    assert item["values"].numel() == e


def test_dataset_time_bins_are_within_window() -> None:
    session = _synthetic_session(n_bins=400, n_units=32, seed=1)
    ds = _make_dataset("train", [session])
    for item in (ds[i] for i in range(min(5, len(ds)))):
        if item["time_bins"].numel() == 0:
            continue
        assert int(item["time_bins"].max()) < ds.window_bins
        assert int(item["time_bins"].min()) >= 0


def test_dataset_neuron_offset_applied_across_sessions() -> None:
    s1 = _synthetic_session(n_bins=400, n_units=10, neuron_id_offset=0, seed=1)
    s2 = _synthetic_session(n_bins=400, n_units=10, neuron_id_offset=10, seed=2)
    ds = _make_dataset("train", [s1, s2])

    seen_max_per_session = {0: 0, 1: 0}
    for win_idx in range(len(ds)):
        session_idx, _, _ = ds._windows[win_idx]
        item = ds[win_idx]
        if item["neuron_ids"].numel() > 0:
            seen_max_per_session[session_idx] = max(
                seen_max_per_session[session_idx], int(item["neuron_ids"].max())
            )
    # Session 0 ids stay below the offset of session 1; session 1 ids start at 10.
    assert seen_max_per_session[0] < 10
    assert seen_max_per_session[1] >= 10


def test_dataset_rejects_window_not_multiple_of_bin() -> None:
    session = _synthetic_session(n_bins=100, n_units=4)
    with pytest.raises(ValueError, match="multiple of bin_size_ms"):
        _make_dataset("train", [session], window_ms=101)


def test_dataset_rejects_neuron_id_overflow() -> None:
    # max_neurons=8 but offset=0 + n_units=10 → tokenizer would index out of range.
    session = _synthetic_session(n_bins=400, n_units=10, neuron_id_offset=0)
    with pytest.raises(ValueError, match="exceeds max_neurons"):
        ds = _make_dataset("train", [session], max_neurons=8)
        # The check fires lazily on first nonempty window.
        for i in range(len(ds)):
            ds[i]


def test_collate_events_concatenates_with_batch_indices() -> None:
    items = [
        {
            "neuron_ids": torch.tensor([1, 2], dtype=torch.int64),
            "time_bins": torch.tensor([0, 1], dtype=torch.int64),
            "values": torch.tensor([1, 1], dtype=torch.int64),
            "behavior": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
        {
            "neuron_ids": torch.tensor([3], dtype=torch.int64),
            "time_bins": torch.tensor([2], dtype=torch.int64),
            "values": torch.tensor([1], dtype=torch.int64),
            "behavior": torch.tensor([2.0, 3.0], dtype=torch.float32),
        },
    ]
    batch = collate_events(items)
    np.testing.assert_array_equal(batch["batch_indices"].numpy(), [0, 0, 1])
    np.testing.assert_array_equal(batch["neuron_ids"].numpy(), [1, 2, 3])
    assert batch["behavior"].shape == (2, 2)


def test_dataset_end_to_end_with_collation_and_model() -> None:
    """Sanity: dataset output flows through CortexModel without shape errors."""
    from cortex.models import CortexModel
    from cortex.models.config import CORTEX_XS

    session = _synthetic_session(n_bins=400, n_units=64, seed=3)
    ds = _make_dataset(
        "train",
        [session],
        max_neurons=CORTEX_XS.max_neurons,
        spike_value_buckets=CORTEX_XS.spike_value_buckets,
    )
    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_events(items)

    model = CortexModel(CORTEX_XS).eval()
    with torch.no_grad():
        out = model(
            neuron_ids=batch["neuron_ids"],
            time_bins=batch["time_bins"],
            values=batch["values"],
            batch_indices=batch["batch_indices"],
        )
    assert out["behavior"].shape == (len(items), CORTEX_XS.behavior_dim)
    assert torch.isfinite(out["behavior"]).all()
