"""Baseline benchmark: Wiener, GRU, VanillaTransformer on MC_Maze.

Practical settings for an M-series Mac (< 30 min total):
  - Wiener: mean-rate features (N neurons) with ridge alpha=1.0; fast + stable
  - GRU: 5 epochs on a 20K subsample; hidden=256, bidirectional
  - Transformer: 3 epochs on a 20K subsample; 4L / 256d / 4h

Usage:
    python scripts/baseline_benchmark.py [--device auto]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cortex.data.nlb import NLBDataset
from cortex.models.config import CORTEX_S
from cortex.training.baselines import (
    DenseWindowDataset,
    GRUDecoder,
    VanillaTransformer,
    WienerFilter,
    collate_dense,
    make_dense_loader,
)
from cortex.training.eval import r2_score
from cortex.utils.device import select_device
from cortex.utils.logging import configure_logging, get_logger

configure_logging(level="INFO", json=False)
log = get_logger(__name__)


def build_datasets(data_root: str) -> tuple[NLBDataset, NLBDataset]:
    kw = dict(
        data_root=data_root, dandiset_id="000128",
        bin_size_ms=5, window_ms=600, stride_ms=50,
        max_neurons=CORTEX_S.max_neurons, download=False,
    )
    return NLBDataset(split="train", **kw), NLBDataset(split="val", **kw)  # type: ignore[arg-type]


def _sub_loader(
    dense_ds: DenseWindowDataset, n: int, batch_size: int, shuffle: bool
) -> DataLoader:
    rng = random.Random(42)
    idx = rng.sample(range(len(dense_ds)), min(n, len(dense_ds)))
    return DataLoader(Subset(dense_ds, idx), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_dense)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-subsample", type=int, default=20_000)
    parser.add_argument("--gru-epochs", type=int, default=5)
    parser.add_argument("--transformer-epochs", type=int, default=3)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out", default="benchmarks/training/baselines_raw.json")
    args = parser.parse_args()

    device = select_device(preference=args.device)

    log.info("loading_data")
    train_ds, val_ds = build_datasets(args.data_root)
    dense_train = DenseWindowDataset(train_ds)
    dense_val = DenseWindowDataset(val_ds)
    total_neurons = dense_train.total_neurons
    log.info("data_loaded", train=len(train_ds), val=len(val_ds), neurons=total_neurons)

    val_loader = DataLoader(dense_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dense)

    results: dict = {
        "device": str(device),
        "train_windows_total": len(train_ds),
        "train_subsample": args.train_subsample,
        "val_windows": len(val_ds),
        "total_neurons": total_neurons,
        "window_bins": train_ds.window_bins,
        "baselines": {},
    }

    # ── Wiener filter — mean-rate features ────────────────────────────────────
    # Use per-neuron mean spike rate over the window (N features) instead of all
    # T×N time-lag features. Mean-rate Wiener is the standard NLB baseline.
    log.info("wiener_start")
    t0 = time.time()

    # Collect mean-rate features from the subsampled train set
    rng_np = np.random.default_rng(42)
    train_idx = rng_np.choice(len(train_ds), min(args.train_subsample, len(train_ds)), replace=False)
    X_train_list, Y_train_list = [], []
    for i in train_idx:
        sess_idx, start, end = train_ds._windows[i]
        sess = train_ds.sessions[sess_idx]
        window = sess.bin_counts[start:end].astype(np.float32)  # (T, N)
        mean_rates = window.mean(axis=0)  # (N,)
        # Place in global neuron space
        feat = np.zeros(total_neurons, dtype=np.float32)
        off = sess.neuron_id_offset
        feat[off:off + mean_rates.shape[0]] = mean_rates
        raw_beh = sess.behavior[end - 1].astype(np.float32)
        beh = (raw_beh - train_ds._behavior_mean) / train_ds._behavior_std
        X_train_list.append(feat)
        Y_train_list.append(beh)

    X_train = torch.from_numpy(np.stack(X_train_list))
    Y_train = torch.from_numpy(np.stack(Y_train_list))

    # Build val features too
    X_val_list, Y_val_list = [], []
    for i in range(len(val_ds)):
        sess_idx, start, end = val_ds._windows[i]
        sess = val_ds.sessions[sess_idx]
        window = sess.bin_counts[start:end].astype(np.float32)
        mean_rates = window.mean(axis=0)
        feat = np.zeros(total_neurons, dtype=np.float32)
        off = sess.neuron_id_offset
        feat[off:off + mean_rates.shape[0]] = mean_rates
        raw_beh = sess.behavior[end - 1].astype(np.float32)
        beh = (raw_beh - val_ds._behavior_mean) / val_ds._behavior_std
        X_val_list.append(feat)
        Y_val_list.append(beh)

    X_val = torch.from_numpy(np.stack(X_val_list))
    Y_val = torch.from_numpy(np.stack(Y_val_list))

    wiener = WienerFilter(n_features=total_neurons, behavior_dim=2, alpha=1.0)
    wiener.fit_closed_form(X_train, Y_train)
    Y_pred_wiener = wiener(X_val)
    wiener_r2 = r2_score(Y_val, Y_pred_wiener)
    wiener_elapsed = time.time() - t0
    log.info("wiener_done", r2=wiener_r2, elapsed_s=f"{wiener_elapsed:.1f}")
    results["baselines"]["Wiener"] = {
        "r2_velocity": wiener_r2,
        "elapsed_s": round(wiener_elapsed, 1),
        "n_features": total_neurons,
    }

    # ── GRU ───────────────────────────────────────────────────────────────────
    log.info("gru_start", epochs=args.gru_epochs, subsample=args.train_subsample)
    t0 = time.time()
    gru_train_loader = _sub_loader(dense_train, args.train_subsample, args.batch_size, shuffle=True)
    gru = GRUDecoder(n_neurons=total_neurons, hidden_dim=256, num_layers=2, behavior_dim=2).to(device)
    opt = torch.optim.AdamW(gru.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for ep in range(args.gru_epochs):
        gru.train()
        running = 0.0; nb = 0
        for batch in gru_train_loader:
            x, y = batch.features.to(device), batch.behavior.to(device)
            pred = gru(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.detach()); nb += 1
        log.info("gru_epoch", epoch=ep, loss=f"{running/nb:.4f}")
    gru.eval()
    gru_preds, gru_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            gru_preds.append(gru(batch.features.to(device)).cpu())
            gru_targets.append(batch.behavior)
    gru_r2 = r2_score(torch.cat(gru_targets), torch.cat(gru_preds))
    gru_elapsed = time.time() - t0
    log.info("gru_done", r2=gru_r2, elapsed_s=f"{gru_elapsed:.1f}")
    results["baselines"]["GRU"] = {
        "r2_velocity": gru_r2,
        "elapsed_s": round(gru_elapsed, 1),
        "epochs": args.gru_epochs,
    }

    # ── Vanilla Transformer ───────────────────────────────────────────────────
    log.info("transformer_start", epochs=args.transformer_epochs, subsample=args.train_subsample)
    t0 = time.time()
    tfm_train_loader = _sub_loader(dense_train, args.train_subsample, args.batch_size, shuffle=True)
    tfm = VanillaTransformer(
        n_neurons=total_neurons, hidden_dim=256, num_layers=4, num_heads=4,
        max_time_bins=train_ds.window_bins + 1, behavior_dim=2,
    ).to(device)
    opt_t = torch.optim.AdamW(tfm.parameters(), lr=1e-3)
    for ep in range(args.transformer_epochs):
        tfm.train()
        running = 0.0; nb = 0
        for batch in tfm_train_loader:
            x, y = batch.features.to(device), batch.behavior.to(device)
            pred = tfm(x)
            loss = loss_fn(pred, y)
            opt_t.zero_grad(set_to_none=True); loss.backward(); opt_t.step()
            running += float(loss.detach()); nb += 1
        log.info("transformer_epoch", epoch=ep, loss=f"{running/nb:.4f}")
    tfm.eval()
    tfm_preds, tfm_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            tfm_preds.append(tfm(batch.features.to(device)).cpu())
            tfm_targets.append(batch.behavior)
    tfm_r2 = r2_score(torch.cat(tfm_targets), torch.cat(tfm_preds))
    tfm_elapsed = time.time() - t0
    log.info("transformer_done", r2=tfm_r2, elapsed_s=f"{tfm_elapsed:.1f}")
    results["baselines"]["Transformer"] = {
        "r2_velocity": tfm_r2,
        "elapsed_s": round(tfm_elapsed, 1),
        "epochs": args.transformer_epochs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info(
        "baselines_complete",
        wiener=f"{wiener_r2:.4f}", gru=f"{gru_r2:.4f}", transformer=f"{tfm_r2:.4f}",
        out=str(out_path),
    )


if __name__ == "__main__":
    main()
