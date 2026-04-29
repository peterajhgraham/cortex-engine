"""Neural Latents Benchmark dataset loader.

The MC_Maze dataset (DANDI 000128) is a primate motor cortex recording during a
center-out reaching task. Each trial is a reach to one of 27 maze targets.

For our decoding task:
    Input  = spike events in a `window_ms` window, binned at `bin_size_ms`.
    Output = hand velocity (2D) at the end of that window.

Data flow:
    DANDI archive -> NWB files -> per-session (units, kinematics) -> binned
    spike matrix + behavior trace -> sliding windows -> per-window event lists.

References:
    Pei et al. 2021, "Neural Latents Benchmark '21"
    https://github.com/neurallatents/nlb_tools
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cortex.utils.logging import get_logger

log = get_logger(__name__)


# ── Pure functions: independently testable, no I/O ─────────────────────────────


def bin_spikes(
    spike_times_s: np.ndarray,
    n_bins: int,
    bin_size_s: float,
    t0_s: float = 0.0,
) -> np.ndarray:
    """Bin a single neuron's spike times into integer counts per bin.

    Args:
        spike_times_s: 1D float array of spike times in seconds.
        n_bins:        Number of bins to produce.
        bin_size_s:    Bin width in seconds.
        t0_s:          Time of the first bin's left edge.

    Returns:
        (n_bins,) int32 array of spike counts per bin.
    """
    if bin_size_s <= 0:
        raise ValueError(f"bin_size_s must be positive, got {bin_size_s}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    edges = t0_s + np.arange(n_bins + 1, dtype=np.float64) * bin_size_s
    counts, _ = np.histogram(spike_times_s, bins=edges)
    return counts.astype(np.int32, copy=False)


def bucket_values(counts: np.ndarray, n_buckets: int) -> np.ndarray:
    """Quantize integer spike counts into bucket indices in [0, n_buckets).

    Buckets are inclusive at both ends: a count of 0 is bucket 0, and any count
    >= n_buckets - 1 is clamped to the top bucket. This matches the POYO scheme
    where most bins have 0–1 spikes and the head can ignore rare large counts.
    """
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive, got {n_buckets}")
    return np.clip(counts, 0, n_buckets - 1).astype(np.int64, copy=False)


def events_from_bin_matrix(
    bin_counts: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a (T, N) bin-count matrix into a sparse event list.

    Only nonzero bins become events. This is the natural representation for the
    Perceiver tokenizer — each spike event is one token.

    Args:
        bin_counts: (T, N) int array of spike counts per (time_bin, neuron).
        n_buckets:  Number of value buckets (passed to bucket_values).

    Returns:
        neuron_ids: (E,) int64
        time_bins:  (E,) int64
        values:     (E,) int64 — bucketed counts, all in [1, n_buckets-1]
    """
    if bin_counts.ndim != 2:
        raise ValueError(f"bin_counts must be (T, N), got shape {bin_counts.shape}")

    time_idx, neuron_idx = np.nonzero(bin_counts)
    raw_values = bin_counts[time_idx, neuron_idx]
    values = bucket_values(raw_values, n_buckets)
    return (
        neuron_idx.astype(np.int64, copy=False),
        time_idx.astype(np.int64, copy=False),
        values,
    )


def build_window_index(
    n_time_bins: int,
    window_bins: int,
    stride_bins: int,
) -> list[tuple[int, int]]:
    """Enumerate (start_bin, end_bin) tuples for sliding windows.

    Drops the trailing partial window. Returns [] if no full window fits.
    """
    if window_bins <= 0 or stride_bins <= 0:
        raise ValueError(f"window_bins ({window_bins}) and stride_bins ({stride_bins}) must be > 0")
    if n_time_bins < window_bins:
        return []
    starts = list(range(0, n_time_bins - window_bins + 1, stride_bins))
    return [(s, s + window_bins) for s in starts]


# ── Per-session record ─────────────────────────────────────────────────────────


@dataclass
class SessionData:
    """In-memory representation of one NWB session, prepared for windowing.

    bin_counts shape: (T, N) int32. T = total time bins in the recording (or
    concatenation of trials), N = number of units in this session.
    behavior shape:  (T, behavior_dim) float32.

    Note: across sessions, N can differ. The neuron_id_offset lets us map
    per-session unit indices into the global [0, max_neurons) range expected
    by the tokenizer's neuron embedding table.
    """

    session_id: str
    bin_counts: np.ndarray
    behavior: np.ndarray
    neuron_id_offset: int


# ── Dataset ────────────────────────────────────────────────────────────────────


class NLBDataset(Dataset):
    """Sliding-window view over one or more NLB sessions.

    The dataset materializes binned spike counts and behavior into RAM on
    construction (NLB sessions are small, on the order of tens of MB each).
    Sliding-window indexing is then O(1) per __getitem__.
    """

    def __init__(
        self,
        data_root: Path,
        dandiset_id: str,
        split: str,
        bin_size_ms: int,
        window_ms: int,
        stride_ms: int,
        max_neurons: int,
        spike_value_buckets: int = 8,
        behavior_dim: int = 2,
        download: bool = True,
        sessions: list[SessionData] | None = None,
        split_seed: int = 0,
        split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        self.data_root = Path(data_root)
        self.dandiset_id = dandiset_id
        self.split = split
        self.bin_size_ms = bin_size_ms
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.max_neurons = max_neurons
        self.spike_value_buckets = spike_value_buckets
        self.behavior_dim = behavior_dim

        if window_ms % bin_size_ms != 0:
            raise ValueError(
                f"window_ms ({window_ms}) must be a multiple of bin_size_ms ({bin_size_ms})"
            )
        if stride_ms % bin_size_ms != 0:
            raise ValueError(
                f"stride_ms ({stride_ms}) must be a multiple of bin_size_ms ({bin_size_ms})"
            )
        self.window_bins = window_ms // bin_size_ms
        self.stride_bins = stride_ms // bin_size_ms

        if sessions is None:
            if download:
                _ensure_dandiset(self.data_root, dandiset_id)
            sessions = _load_sessions_from_directory(
                self.data_root / dandiset_id,
                bin_size_ms=bin_size_ms,
                max_neurons=max_neurons,
            )
        self.sessions = sessions

        all_windows: list[tuple[int, int, int]] = []
        for session_idx, session in enumerate(sessions):
            n_bins = session.bin_counts.shape[0]
            for start, end in build_window_index(n_bins, self.window_bins, self.stride_bins):
                all_windows.append((session_idx, start, end))

        self._windows = _split_windows(all_windows, split, split_seed, split_fractions)

        log.info(
            "nlb_dataset_built",
            split=split,
            n_sessions=len(sessions),
            n_windows=len(self._windows),
            window_bins=self.window_bins,
            stride_bins=self.stride_bins,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        session_idx, start, end = self._windows[idx]
        session = self.sessions[session_idx]

        window_counts = session.bin_counts[start:end]
        neuron_ids, time_bins, values = events_from_bin_matrix(
            window_counts, self.spike_value_buckets
        )
        # Map per-session neuron indices into the global embedding space.
        neuron_ids = neuron_ids + session.neuron_id_offset
        if neuron_ids.size > 0 and int(neuron_ids.max()) >= self.max_neurons:
            raise ValueError(
                f"neuron_id {int(neuron_ids.max())} exceeds max_neurons={self.max_neurons}; "
                f"check max_neurons in config"
            )

        # Behavior target: hand velocity at the final bin of the window. Using
        # the end-of-window matches a causal decoder that emits "current
        # velocity" given the recent past.
        behavior = session.behavior[end - 1].astype(np.float32, copy=False)

        return {
            "neuron_ids": torch.from_numpy(neuron_ids),
            "time_bins": torch.from_numpy(time_bins),
            "values": torch.from_numpy(values),
            "behavior": torch.from_numpy(behavior),
        }


# ── Collation and dataloader factory ───────────────────────────────────────────


def collate_events(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate variable-length event sequences into a flat batch.

    All event tensors are concatenated along dim 0; a parallel batch_indices
    tensor records which batch element each event belongs to. This matches the
    interface expected by CortexModel.forward.
    """
    neuron_ids = torch.cat([b["neuron_ids"] for b in batch])
    time_bins = torch.cat([b["time_bins"] for b in batch])
    values = torch.cat([b["values"] for b in batch])
    batch_indices = torch.cat(
        [torch.full((len(b["neuron_ids"]),), i, dtype=torch.int64) for i, b in enumerate(batch)]
    )
    behavior = torch.stack([b["behavior"] for b in batch])
    return {
        "neuron_ids": neuron_ids,
        "time_bins": time_bins,
        "values": values,
        "batch_indices": batch_indices,
        "behavior": behavior,
    }


def build_dataloaders(
    cfg: Any,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Construct train and val dataloaders. Uses DistributedSampler if world_size > 1."""
    train_ds = NLBDataset(
        data_root=cfg.data_root,
        dandiset_id=cfg.dandiset_id,
        split="train",
        bin_size_ms=cfg.bin_size_ms,
        window_ms=cfg.window_ms,
        stride_ms=cfg.stride_ms,
        max_neurons=cfg.max_neurons,
    )
    val_ds = NLBDataset(
        data_root=cfg.data_root,
        dandiset_id=cfg.dandiset_id,
        split="val",
        bin_size_ms=cfg.bin_size_ms,
        window_ms=cfg.window_ms,
        stride_ms=cfg.stride_ms,
        max_neurons=cfg.max_neurons,
    )

    sampler: torch.utils.data.distributed.DistributedSampler | None = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    batch_size = cfg.get("batch_size", 32) if hasattr(cfg, "get") else getattr(cfg, "batch_size", 32)
    num_workers = getattr(cfg, "num_workers", 0)
    pin_memory = getattr(cfg, "pin_memory", False)
    prefetch_factor = getattr(cfg, "prefetch_factor", None)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_events,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=(sampler is None),
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ── DANDI download + NWB loading (I/O, hard to unit-test) ──────────────────────


def _ensure_dandiset(data_root: Path, dandiset_id: str) -> None:
    """Download a dandiset if its directory is missing or empty.

    Uses the dandi CLI under the hood. This is a no-op once the data is on
    disk, so it's safe to call on every Dataset construction.
    """
    target = data_root / dandiset_id
    if target.exists() and any(target.rglob("*.nwb")):
        return

    data_root.mkdir(parents=True, exist_ok=True)
    log.info("dandi_download_start", dandiset_id=dandiset_id, target=str(target))
    try:
        from dandi.download import download as dandi_download
    except ImportError as e:
        raise ImportError(
            "dandi is required to download NLB data. Install with `pip install dandi`."
        ) from e

    url = f"https://dandiarchive.org/dandiset/{dandiset_id}"
    dandi_download([url], output_dir=str(data_root), get_metadata=True, get_assets=True)
    log.info("dandi_download_complete", dandiset_id=dandiset_id)


def _load_sessions_from_directory(
    nwb_dir: Path,
    bin_size_ms: int,
    max_neurons: int,
) -> list[SessionData]:
    """Load every NWB file under nwb_dir into a SessionData list.

    Each session is binned at bin_size_ms and assigned a contiguous neuron-id
    range starting from the running offset.
    """
    nwb_files = sorted(nwb_dir.rglob("*.nwb"))
    if not nwb_files:
        raise FileNotFoundError(f"No .nwb files under {nwb_dir}")

    try:
        from pynwb import NWBHDF5IO
    except ImportError as e:
        raise ImportError(
            "pynwb is required to load NLB data. Install with `pip install pynwb`."
        ) from e

    sessions: list[SessionData] = []
    neuron_offset = 0
    for nwb_path in nwb_files:
        with NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
            nwb = io.read()
            bin_counts, behavior = _extract_session_arrays(nwb, bin_size_ms)

        n_units = bin_counts.shape[1]
        if neuron_offset + n_units > max_neurons:
            raise ValueError(
                f"Session {nwb_path.name} would overflow max_neurons "
                f"({neuron_offset + n_units} > {max_neurons}). "
                f"Increase max_neurons in the model config."
            )
        sessions.append(
            SessionData(
                session_id=nwb_path.stem,
                bin_counts=bin_counts,
                behavior=behavior,
                neuron_id_offset=neuron_offset,
            )
        )
        neuron_offset += n_units
        log.info(
            "nwb_session_loaded",
            session=nwb_path.name,
            time_bins=bin_counts.shape[0],
            n_units=n_units,
            neuron_offset=neuron_offset,
        )

    return sessions


def _extract_session_arrays(nwb: Any, bin_size_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (binned_spikes, hand_velocity) from an NWB file.

    Conventions used here match the NLB MC_Maze format:
        - Spikes are in nwb.units (a DynamicTable with a 'spike_times' column).
        - Hand kinematics are in nwb.processing['behavior']['Position'] for
          position; we numerically differentiate to get velocity.
    """
    units = nwb.units
    spike_times_per_unit: list[np.ndarray] = [
        np.asarray(units["spike_times"][i], dtype=np.float64) for i in range(len(units))
    ]
    n_units = len(spike_times_per_unit)

    behavior_proc = nwb.processing["behavior"]
    position_container = behavior_proc["Position"]
    pos_ts = next(iter(position_container.spatial_series.values()))
    position = np.asarray(pos_ts.data, dtype=np.float32)
    pos_rate = float(pos_ts.rate)
    pos_t0 = float(pos_ts.starting_time)

    bin_size_s = bin_size_ms / 1000.0
    t_end = pos_t0 + position.shape[0] / pos_rate
    n_bins = int((t_end - pos_t0) / bin_size_s)

    bin_counts = np.zeros((n_bins, n_units), dtype=np.int32)
    for unit_idx, spikes in enumerate(spike_times_per_unit):
        bin_counts[:, unit_idx] = bin_spikes(spikes, n_bins, bin_size_s, t0_s=pos_t0)

    velocity = _resample_velocity_to_bins(position, pos_rate, n_bins, bin_size_s)
    return bin_counts, velocity


def _resample_velocity_to_bins(
    position: np.ndarray,
    pos_rate: float,
    n_bins: int,
    bin_size_s: float,
) -> np.ndarray:
    """Numerically differentiate position and resample to one sample per bin."""
    velocity = np.gradient(position, 1.0 / pos_rate, axis=0).astype(np.float32, copy=False)
    samples_per_bin = max(int(round(bin_size_s * pos_rate)), 1)

    n_complete = (velocity.shape[0] // samples_per_bin) * samples_per_bin
    trimmed = velocity[:n_complete]
    reshaped = trimmed.reshape(-1, samples_per_bin, velocity.shape[1])
    binned = reshaped.mean(axis=1)

    if binned.shape[0] >= n_bins:
        return binned[:n_bins]
    pad = np.zeros((n_bins - binned.shape[0], binned.shape[1]), dtype=np.float32)
    return np.concatenate([binned, pad], axis=0)


# ── Train/val/test split helper ────────────────────────────────────────────────


def _split_windows(
    windows: list[tuple[int, int, int]],
    split: str,
    seed: int,
    fractions: tuple[float, float, float],
) -> list[tuple[int, int, int]]:
    """Deterministically partition a window list into train/val/test."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train'|'val'|'test', got {split!r}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {fractions}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(windows))
    n = len(windows)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])

    if split == "train":
        idx = perm[:n_train]
    elif split == "val":
        idx = perm[n_train : n_train + n_val]
    else:
        idx = perm[n_train + n_val :]
    return [windows[i] for i in idx]
