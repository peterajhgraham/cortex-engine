"""Inference worker.

Owns the GPU model and the KV cache. Receives batched payloads from the
scheduler, runs forward passes, and returns results.

Implementation notes for Claude Code:
    - Use CUDA streams to overlap H2D copy with compute on the previous batch.
    - Track in-flight batch state so the scheduler can implement preemption.
    - Capture cudagraphs for fixed batch sizes to eliminate launch overhead.
    - When INT8 quantized, the model is loaded with custom weight format from
      cortex.quantization; forward pass dispatches through the quantized linears.
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from torch import nn

from cortex.utils.logging import get_logger

log = get_logger(__name__)


class InferenceWorker:
    """Runs inference for batches submitted by the scheduler."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch_size: int = 32,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch_size = max_batch_size

        # CUDA stream for H2D overlap
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        # TODO Phase 3.5: KV cache instance
        # from cortex.cache import StreamingKVCache
        # self.kv_cache = StreamingKVCache(...)

        # TODO Phase 3.6: cudagraph capture for fixed batch sizes
        self._graphs: dict[int, Any] = {}

    @torch.inference_mode()
    async def run_batch(self, payloads: list[Any]) -> list[Any]:
        """Run inference on a batch of payloads.

        The scheduler is responsible for ensuring `len(payloads) <= max_batch_size`.
        """
        if not payloads:
            return []

        # TODO Phase 3.7: convert payloads to model inputs
        # TODO Phase 3.8: run model forward (under cudagraph if captured)
        # TODO Phase 3.9: split outputs back into per-request results

        await asyncio.sleep(0)  # cooperate with event loop
        return [None] * len(payloads)
