"""Tests for the eval pipeline: R² metric edge cases and end-to-end loop."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cortex.data.nlb import NLBDataset, SessionData, collate_events
from cortex.models import CortexModel
from cortex.models.config import CORTEX_XS
from cortex.training.eval import (
    EvalResults,
    evaluate,
    format_comparison_table,
    r2_score,
    r2_score_per_dim,
)

# ── r2_score ───────────────────────────────────────────────────────────────────


def test_r2_perfect_prediction_is_one() -> None:
    y = torch.randn(50, 2)
    assert r2_score(y, y) == pytest.approx(1.0, abs=1e-5)


def test_r2_mean_prediction_is_zero() -> None:
    y = torch.randn(200, 2)
    pred = y.mean(dim=0, keepdim=True).expand_as(y)
    assert r2_score(y, pred) == pytest.approx(0.0, abs=1e-5)


def test_r2_worse_than_mean_is_negative() -> None:
    y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    bad = -y
    assert r2_score(y, bad) < 0


def test_r2_per_dim_returns_one_value_per_output() -> None:
    y = torch.randn(30, 3)
    pred = y.clone()
    pred[:, 1] += 0.5  # corrupt channel 1
    per_dim = r2_score_per_dim(y, pred)
    assert per_dim.shape == (3,)
    assert per_dim[0] > per_dim[1]  # corruption shows up only in channel 1
    assert per_dim[2] > per_dim[1]


def test_r2_constant_target_yields_zero_not_nan() -> None:
    y = torch.zeros(20, 2)
    pred = torch.randn(20, 2)
    val = r2_score(y, pred)
    assert math.isfinite(val)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_r2_handles_1d_input() -> None:
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = y.clone()
    assert r2_score_per_dim(y, pred).shape == (1,)


def test_r2_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        r2_score(torch.zeros(4, 2), torch.zeros(4, 3))


# ── evaluate() integration ────────────────────────────────────────────────────


def _synthetic_dataset() -> NLBDataset:
    rng = np.random.default_rng(7)
    bin_counts = rng.poisson(lam=0.3, size=(400, 32)).astype(np.int32)
    behavior = rng.normal(size=(400, 2)).astype(np.float32)
    session = SessionData("synth", bin_counts, behavior, neuron_id_offset=0)
    return NLBDataset(
        data_root="/tmp/unused",
        dandiset_id="000128",
        split="val",
        bin_size_ms=5,
        window_ms=100,
        stride_ms=50,
        max_neurons=CORTEX_XS.max_neurons,
        spike_value_buckets=CORTEX_XS.spike_value_buckets,
        download=False,
        sessions=[session],
    )


def test_evaluate_returns_populated_results() -> None:
    model = CortexModel(CORTEX_XS).eval()
    loader = DataLoader(
        _synthetic_dataset(), batch_size=4, shuffle=False, collate_fn=collate_events
    )

    results = evaluate(model, loader, device=torch.device("cpu"))
    assert isinstance(results, EvalResults)
    assert results.n_samples > 0
    assert math.isfinite(results.r2_velocity)
    assert math.isfinite(results.mse_velocity)
    # Untrained model on uncorrelated targets → R² near zero or negative; just check finiteness.


def test_evaluate_results_as_dict_round_trip() -> None:
    res = EvalResults(r2_velocity=0.7, mse_velocity=0.1, masked_spike_accuracy=None, n_samples=100)
    d = res.as_dict()
    assert d["r2_velocity"] == 0.7
    assert d["masked_spike_accuracy"] is None
    assert d["n_samples"] == 100


# ── Comparison table ──────────────────────────────────────────────────────────


def test_format_comparison_table_renders_markdown() -> None:
    table = format_comparison_table(
        {"Wiener": 0.4, "GRU": 0.55, "Transformer": 0.6, "Cortex-S": 0.7}
    )
    assert "| Model | R² (hand velocity) |" in table
    assert "| Wiener | 0.4000 |" in table
    assert "| Cortex-S | 0.7000 |" in table
    # Markdown must have the separator row right after the header.
    rows = table.splitlines()
    assert rows[1] == "|---|---|"
