"""End-to-end smoke test: model forward pass on synthetic events."""

from __future__ import annotations

import torch

from cortex.models import CortexModel, CortexConfig
from cortex.models.config import CORTEX_XS


def test_cortex_xs_forward_pass() -> None:
    """The smallest model should run forward on CPU with synthetic events."""
    config = CORTEX_XS
    model = CortexModel(config)
    model.eval()

    torch.manual_seed(0)
    E = 128
    batch_size = 4
    neuron_ids = torch.randint(0, config.max_neurons, (E,), dtype=torch.int64)
    time_bins = torch.randint(0, config.max_time_bins, (E,), dtype=torch.int64)
    values = torch.randint(0, config.spike_value_buckets, (E,), dtype=torch.int64)
    batch_indices = torch.randint(0, batch_size, (E,), dtype=torch.int64).sort().values

    with torch.no_grad():
        out = model(neuron_ids, time_bins, values, batch_indices)

    assert "behavior" in out
    behavior_batch = int(batch_indices.max().item()) + 1
    assert out["behavior"].shape == (behavior_batch, config.behavior_dim)
    assert torch.isfinite(out["behavior"]).all()


def test_config_validates_head_dim() -> None:
    """head_dim * num_heads must equal hidden_dim."""
    import pytest

    with pytest.raises(ValueError, match="must equal hidden_dim"):
        CortexConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=2,
            head_dim=32,  # 32 * 2 != 128
            num_latents=32,
            latent_dim=128,
            cross_attn_heads=2,
            max_neurons=256,
            max_time_bins=512,
            behavior_dim=2,
        )


def test_tokenizer_shape_validation() -> None:
    """Tokenizer rejects mismatched index shapes."""
    import pytest

    from cortex.models import SpikeTokenizer

    tok = SpikeTokenizer(CORTEX_XS)
    n = torch.zeros(10, dtype=torch.int64)
    t = torch.zeros(11, dtype=torch.int64)
    v = torch.zeros(10, dtype=torch.int64)
    with pytest.raises(ValueError, match="shape mismatch"):
        tok(n, t, v)
