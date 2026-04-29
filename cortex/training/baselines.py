"""Baseline neural decoders for comparison.

Three baselines, simplest to most capable:
    1. Wiener filter (ridge-regularized linear decoder) — canonical motor BCI baseline
    2. GRU sequence model — standard RNN decoder
    3. Vanilla Transformer — same task, generic architecture, no Perceiver tokenization

The Cortex models must beat all three on R² for hand velocity decoding to count
as a genuine architectural win.
"""

from __future__ import annotations

import torch
from torch import nn


class WienerFilter(nn.Module):
    """Ridge-regularized linear decoder.

    The canonical motor BCI baseline going back to Wessberg et al. 2000.
    Maps a flattened (neurons * time_lags) feature vector to behavior.

    Trained via closed-form ridge solution, not gradient descent. Use the
    `fit_closed_form` method during training rather than the standard loop.
    """

    def __init__(self, n_features: int, behavior_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, behavior_dim, bias=True)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, n_features)
        Returns:
            (B, behavior_dim)
        """
        return self.linear(x)

    @torch.no_grad()
    def fit_closed_form(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Solve the ridge regression in closed form.

        W = (X^T X + alpha * I)^-1 X^T Y

        Args:
            X: (N, n_features)
            Y: (N, behavior_dim)
        """
        n_features = X.shape[1]
        XtX = X.T @ X + self.alpha * torch.eye(n_features, device=X.device, dtype=X.dtype)
        XtY = X.T @ Y
        W = torch.linalg.solve(XtX, XtY)
        self.linear.weight.copy_(W.T)
        self.linear.bias.zero_()


class GRUDecoder(nn.Module):
    """Bidirectional GRU sequence model.

    Reads windowed spike counts as (B, T, n_neurons) and outputs behavior.
    """

    def __init__(self, n_neurons: int, hidden_dim: int, num_layers: int, behavior_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_neurons,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden_dim, behavior_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, n_neurons) windowed spike counts
        Returns:
            (B, behavior_dim) using the final timestep
        """
        out, _ = self.gru(x)
        return self.head(out[:, -1])  # use final timestep


class VanillaTransformer(nn.Module):
    """Standard transformer encoder over windowed spike counts.

    Treats each time bin as a token. Includes a learned [CLS] token whose final
    representation is decoded into behavior.
    """

    def __init__(
        self,
        n_neurons: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_time_bins: int,
        behavior_dim: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_neurons, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_emb = nn.Embedding(max_time_bins + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, behavior_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, n_neurons)
        Returns:
            (B, behavior_dim)
        """
        B, T, _ = x.shape
        h = self.input_proj(x)  # (B, T, D)
        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, D)

        pos = torch.arange(T + 1, device=x.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_emb(pos)

        h = self.encoder(h)
        return self.head(h[:, 0])  # CLS token output
