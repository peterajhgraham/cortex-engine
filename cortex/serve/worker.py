"""Inference worker: owns the model and executes batched forward passes.

Design
------
The worker runs in the same process as the FastAPI app but PyTorch inference
is CPU-bound (even on GPU, .synchronize() blocks Python).  To keep the async
event loop responsive, every forward pass runs inside a ThreadPoolExecutor via
loop.run_in_executor().  The event loop submits batches and awaits results
without blocking other coroutines.

CUDA streams
------------
When CUDA is available we use two streams for H2D overlap:

    compute_stream : runs the forward pass
    copy_stream    : transfers the NEXT batch's tensors to GPU while the
                     current batch is computing

On MPS the Metal command queue is managed by the driver; stream-level overlap
is not exposed.  We fall back to a simple synchronous path.

Payload contract
----------------
Each payload passed into run_batch() is a dict with keys:
    request_id : str
    events     : list[dict]   each dict has neuron_id, time_bin, value

The worker returns a parallel list of result dicts:
    request_id  : str
    behavior    : list[float]  length == config.behavior_dim
    inference_ms: float        time the model forward pass took (batch-level)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from cortex.models.config import CortexConfig
from cortex.serve.metrics import BATCH_SIZE, GPU_MEMORY_USED, GPU_UTILIZATION, INFERENCE_LATENCY
from cortex.utils.logging import get_logger

log = get_logger(__name__)


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    config: CortexConfig,
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Load CortexModel onto device.  Returns float32 model (bfloat16 cast at runtime)."""
    from cortex.models.cortex import CortexModel

    if device is None:
        device = _detect_device()

    model = CortexModel(config).to(device).eval()

    if checkpoint_path:
        try:
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = payload.get("model_state_dict", payload)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log.warning("checkpoint_missing_keys", n=len(missing), keys=missing[:3])
            if unexpected:
                log.warning("checkpoint_unexpected_keys", n=len(unexpected), keys=unexpected[:3])
            log.info("checkpoint_loaded", path=checkpoint_path, device=str(device))
        except FileNotFoundError:
            log.warning("checkpoint_not_found", path=checkpoint_path)
    else:
        log.info("model_random_weights", device=str(device))

    return model


class InferenceWorker:
    """Batched inference worker.

    Thread-safe: a single instance is shared across async handler coroutines.
    All calls to run_batch() serialize through the ThreadPoolExecutor (one
    model, one GPU).
    """

    def __init__(
        self,
        model: nn.Module,
        config: CortexConfig,
        device: torch.device,
        max_batch_size: int = 32,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.max_batch_size = max_batch_size

        # Single-thread executor so forward passes never overlap
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

        # CUDA stream for compute; copy stream for H2D overlap
        if device.type == "cuda":
            self._compute_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=device)
            self._copy_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=device)
        else:
            self._compute_stream = None
            self._copy_stream = None

        self._loaded = False
        log.info(
            "worker_init",
            device=str(device),
            max_batch_size=max_batch_size,
            cuda_streams=self._compute_stream is not None,
        )

    def warmup(self, n_iters: int = 3) -> None:
        """Run dummy batches to warm the model and trigger Triton autotune."""
        dummy = [
            {
                "request_id": f"warmup_{i}",
                "events": [{"neuron_id": 0, "time_bin": 0, "value": 0} for _ in range(64)],
            }
            for i in range(min(n_iters, self.max_batch_size))
        ]
        for _ in range(n_iters):
            self._run_batch_sync(dummy)
        self._loaded = True
        log.info("worker_warmup_complete", n_iters=n_iters)

    @property
    def ready(self) -> bool:
        return self._loaded

    async def run_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Async entry point: submit batch to executor, await result."""
        if not payloads:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_batch_sync, payloads)

    def _run_batch_sync(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Synchronous forward pass.  Called inside the ThreadPoolExecutor."""
        t0 = time.perf_counter()
        BATCH_SIZE.observe(len(payloads))

        # ── Build flat event tensors ──────────────────────────────────────────
        all_nids: list[int] = []
        all_tbs: list[int] = []
        all_vals: list[int] = []
        all_bidx: list[int] = []

        for batch_idx, payload in enumerate(payloads):
            for ev in payload.get("events", []):
                all_nids.append(int(ev["neuron_id"]))
                all_tbs.append(int(ev["time_bin"]))
                all_vals.append(int(ev["value"]))
                all_bidx.append(batch_idx)

        behavior_dim = self.config.behavior_dim

        if not all_nids:
            inference_ms = (time.perf_counter() - t0) * 1000
            return [
                {
                    "request_id": p["request_id"],
                    "behavior": [0.0] * behavior_dim,
                    "inference_ms": inference_ms,
                }
                for p in payloads
            ]

        # ── Tensor construction (with optional H2D on copy stream) ────────────
        copy_ctx = (
            torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext()
        )

        with copy_ctx:
            nids = torch.tensor(all_nids, dtype=torch.int64, device=self.device)
            tbs = torch.tensor(all_tbs, dtype=torch.int64, device=self.device)
            vals = torch.tensor(all_vals, dtype=torch.int64, device=self.device)
            bidx = torch.tensor(all_bidx, dtype=torch.int64, device=self.device)

        if self._copy_stream is not None and self._compute_stream is not None:
            # Wait for H2D before compute
            self._compute_stream.wait_stream(self._copy_stream)

        # ── Forward pass ──────────────────────────────────────────────────────
        compute_ctx = (
            torch.cuda.stream(self._compute_stream)
            if self._compute_stream is not None
            else nullcontext()
        )

        with compute_ctx, torch.inference_mode():
            out = self.model(nids, tbs, vals, bidx)

        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        # ── Record metrics ────────────────────────────────────────────────────
        if self.device.type == "cuda":
            GPU_MEMORY_USED.set(torch.cuda.memory_allocated(self.device))
            with contextlib.suppress(Exception):
                # torch.cuda.utilization() calls NVML; may fail if nvml unavailable
                GPU_UTILIZATION.set(float(torch.cuda.utilization(self.device)))

        inference_ms = (time.perf_counter() - t0) * 1000
        INFERENCE_LATENCY.observe(inference_ms / 1000.0)

        # ── Split results per request ─────────────────────────────────────────
        behavior = out["behavior"].float().cpu().tolist()  # (B, behavior_dim)
        results = []
        for i, payload in enumerate(payloads):
            results.append(
                {
                    "request_id": payload["request_id"],
                    "behavior": behavior[i] if i < len(behavior) else [0.0] * behavior_dim,
                    "inference_ms": inference_ms,
                }
            )

        log.debug(
            "batch_complete",
            batch_size=len(payloads),
            inference_ms=round(inference_ms, 2),
            n_events=len(all_nids),
        )
        return results

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        log.info("worker_shutdown")
