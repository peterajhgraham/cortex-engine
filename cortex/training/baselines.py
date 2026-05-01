"""Baseline neural decoders for comparison.

Three baselines, simplest to most capable:
    1. Wiener filter (ridge-regularized linear decoder) — canonical motor BCI baseline
    2. GRU sequence model — standard RNN decoder
    3. Vanilla Transformer — same task, generic architecture, no Perceiver tokenization

The Cortex models must beat all three on R² for hand velocity decoding to count
as a genuine architectural win.

The training entry points at the bottom of this module (`fit_wiener`,
`train_gru`, `train_transformer`) consume an `NLBDataset` directly and emit
R² on a held-out loader, so the same dataset that feeds the Cortex training
loop also feeds these baselines without parallel pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cortex.data.nlb import NLBDataset
from cortex.training.eval import r2_score
from cortex.utils.logging import get_logger

log = get_logger(__name__)


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


# ── Dense feature extraction ──────────────────────────────────────────────────


@dataclass
class DenseBatch:
    """A dense windowed batch suitable for baselines.

    features: (B, T, N) float32 spike-count windows.
    behavior: (B, behavior_dim) float32 targets.
    """

    features: torch.Tensor
    behavior: torch.Tensor


class DenseWindowDataset(Dataset):
    """Adapter: yields (window_bins, n_neurons) dense spike counts per sample.

    Wraps an NLBDataset so the baselines see exactly the same windows as the
    Cortex training loop. The total feature dimension is
    `nlb_dataset.window_bins * total_neurons`, where total_neurons is the
    summed unit count across loaded sessions.
    """

    def __init__(self, nlb_dataset: NLBDataset) -> None:
        self.nlb = nlb_dataset
        self.total_neurons = sum(s.bin_counts.shape[1] for s in nlb_dataset.sessions)

    def __len__(self) -> int:
        return len(self.nlb)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        session_idx, start, end = self.nlb._windows[idx]
        session = self.nlb.sessions[session_idx]
        window = session.bin_counts[start:end]  # (T, N_session)

        features = torch.zeros((self.nlb.window_bins, self.total_neurons), dtype=torch.float32)
        offset = session.neuron_id_offset
        features[:, offset : offset + window.shape[1]] = torch.from_numpy(window.astype(np.float32))
        # Z-score to match NLBDataset normalization so baseline R² is comparable.
        raw = session.behavior[end - 1].astype(np.float32, copy=False)
        behavior = (raw - self.nlb._behavior_mean) / self.nlb._behavior_std
        return {"features": features, "behavior": torch.from_numpy(behavior)}


def collate_dense(batch: list[dict[str, torch.Tensor]]) -> DenseBatch:
    return DenseBatch(
        features=torch.stack([b["features"] for b in batch]),
        behavior=torch.stack([b["behavior"] for b in batch]),
    )


def make_dense_loader(
    nlb_dataset: NLBDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Standard DataLoader wrapping the dense adapter."""
    return DataLoader(
        DenseWindowDataset(nlb_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dense,
    )


# ── Baseline training entry points ─────────────────────────────────────────────


def _stack_loader(loader: Iterable[DenseBatch]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect a full DataLoader into (features, behavior) tensors."""
    feats: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        feats.append(batch.features)
        targets.append(batch.behavior)
    return torch.cat(feats), torch.cat(targets)


@torch.no_grad()
def fit_wiener(
    train_loader: Iterable[DenseBatch],
    val_loader: Iterable[DenseBatch],
    *,
    behavior_dim: int,
    alpha: float = 1.0,
) -> tuple[WienerFilter, float]:
    """Fit a ridge decoder in closed form, return (model, val R²).

    Spike windows are flattened to (B, T*N) — that's the canonical Wiener
    feature representation: one weight per (lag, neuron) pair.
    """
    X_train, Y_train = _stack_loader(train_loader)
    n_samples, T, N = X_train.shape
    n_features = T * N
    X_train_flat = X_train.reshape(n_samples, n_features)

    model = WienerFilter(n_features=n_features, behavior_dim=behavior_dim, alpha=alpha)
    model.fit_closed_form(X_train_flat, Y_train)

    X_val, Y_val = _stack_loader(val_loader)
    Y_pred = model(X_val.reshape(X_val.shape[0], -1))
    r2 = r2_score(Y_val, Y_pred)
    log.info("wiener_baseline_done", r2=r2, n_train=n_samples, n_features=n_features)
    return model, r2


def train_gru(
    train_loader: Iterable[DenseBatch],
    val_loader: Iterable[DenseBatch],
    *,
    n_neurons: int,
    behavior_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 20,
    device: torch.device | None = None,
) -> tuple[GRUDecoder, float]:
    """Standard SGD training of the bidirectional GRU baseline.

    Loops over the loader for `epochs` passes; uses MSE on the final-timestep
    output. Returns (model, val R²) so the calling script can record
    comparison numbers.
    """
    device = device or torch.device("cpu")
    model = GRUDecoder(n_neurons, hidden_dim, num_layers, behavior_dim).to(device)
    return _train_dense_model(model, train_loader, val_loader, lr=lr, epochs=epochs, device=device)


def train_transformer(
    train_loader: Iterable[DenseBatch],
    val_loader: Iterable[DenseBatch],
    *,
    n_neurons: int,
    behavior_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    max_time_bins: int = 1024,
    lr: float = 1e-3,
    epochs: int = 20,
    device: torch.device | None = None,
) -> tuple[VanillaTransformer, float]:
    """Standard SGD training of the vanilla Transformer baseline."""
    device = device or torch.device("cpu")
    model = VanillaTransformer(
        n_neurons=n_neurons,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_time_bins=max_time_bins,
        behavior_dim=behavior_dim,
    ).to(device)
    return _train_dense_model(model, train_loader, val_loader, lr=lr, epochs=epochs, device=device)


def _train_dense_model(
    model: nn.Module,
    train_loader: Iterable[DenseBatch],
    val_loader: Iterable[DenseBatch],
    *,
    lr: float,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, float]:
    """Shared SGD harness for the GRU and Transformer baselines."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch.features.to(device)
            y = batch.behavior.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().item())
            n_batches += 1
        log.info(
            "baseline_epoch",
            model=type(model).__name__,
            epoch=epoch,
            train_loss=running / max(n_batches, 1),
        )

    # Validation R²
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            preds.append(model(batch.features.to(device)).cpu())
            targets.append(batch.behavior)
    r2 = r2_score(torch.cat(targets), torch.cat(preds))
    log.info("baseline_val", model=type(model).__name__, r2=r2)
    return model, r2
