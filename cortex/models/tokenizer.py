"""Spike tokenizer.

Converts (neuron_id, time_bin, value) triplets into embedded tokens for the
Perceiver encoder.

Design rationale:
    Each spike event becomes a single token. The token embedding is the sum of:
        1. A learned embedding indexed by neuron_id  (per-unit identity)
        2. A learned embedding indexed by quantized time_bin  (temporal position)
        3. A learned embedding indexed by quantized spike value bucket
    This is the same factorisation as POYO.  It scales to multiple sessions
    because neuron embeddings are learned from data, not hard-coded.

Phase 2:
    When config.use_kernels is True AND CUDA is available, forward() dispatches
    to cortex.kernels.tokenizer.fused_tokenizer, which fuses the three embedding
    lookups into one kernel and eliminates two intermediate (E, D) tensors.
    On MPS / CPU it falls back to the pure-PyTorch path transparently.

Shapes (E = total spike events across the batch):
    Input:
        neuron_ids: int64 (E,) in [0, max_neurons)
        time_bins:  int64 (E,) in [0, max_time_bins)
        values:     int64 (E,) bucketed into [0, spike_value_buckets)
    Output:
        tokens:     float (E, hidden_dim)
"""

from __future__ import annotations

import torch
from torch import nn

from cortex.models.config import CortexConfig


class SpikeTokenizer(nn.Module):
    """Embed spike events as tokens for the Perceiver encoder."""

    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.config = config

        self.neuron_emb = nn.Embedding(config.max_neurons, config.hidden_dim)
        self.time_emb = nn.Embedding(config.max_time_bins, config.hidden_dim)
        self.value_emb = nn.Embedding(config.spike_value_buckets, config.hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.neuron_emb, self.time_emb, self.value_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(
        self,
        neuron_ids: torch.Tensor,
        time_bins: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a batch of spike events.

        Args:
            neuron_ids: (E,) int64 tensor of neuron indices
            time_bins:  (E,) int64 tensor of quantized time positions
            values:     (E,) int64 tensor of bucketed spike values

        Returns:
            tokens: (E, hidden_dim) float tensor
        """
        if not (neuron_ids.shape == time_bins.shape == values.shape):
            raise ValueError(
                f"shape mismatch: neuron_ids={neuron_ids.shape}, "
                f"time_bins={time_bins.shape}, values={values.shape}"
            )

        if self.config.use_kernels and torch.cuda.is_available():
            from cortex.kernels.tokenizer import fused_tokenizer

            return fused_tokenizer(
                self.neuron_emb.weight,
                self.time_emb.weight,
                self.value_emb.weight,
                neuron_ids,
                time_bins,
                values,
            )

        return self.neuron_emb(neuron_ids) + self.time_emb(time_bins) + self.value_emb(values)
