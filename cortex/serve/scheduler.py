"""Continuous batching scheduler.

Concept (mirrors vLLM/SGLang for LLMs, adapted for streaming spike data):

    Requests arrive asynchronously into an asyncio queue with priorities and
    deadlines. A scheduler loop continuously:

        1. Drains newly arrived requests from the queue (up to a timeout)
        2. Forms a batch up to max_batch_size, respecting per-request deadlines
        3. Submits the batch to the inference worker
        4. Resolves each request's future with its slice of the output

Differences from LLM continuous batching:
    - Spike decoders do not autoregress, so batches do not have heterogeneous
      sequence positions to pack. Instead, the heterogeneity is in event count
      per request (variable spikes per window).
    - Padding strategy: bucket requests by event count to minimize wasted work.
    - Deadline-aware: requests with closest deadlines run first, with admission
      control if the queue projects deadline violation.

Implementation TODO list for Claude Code (Phase 3):
    [ ] Implement Request dataclass with future, deadline, priority
    [ ] Implement bucket-aware batching
    [ ] Implement admission control with deadline projection
    [ ] Wire up to InferenceWorker via asyncio.Queue
    [ ] Add Prometheus metrics at every state transition
    [ ] Integration test: 1000 concurrent requests, verify p99 deadline compliance
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from cortex.serve.metrics import BATCH_SIZE, QUEUE_DEPTH, QUEUE_WAIT
from cortex.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(order=True)
class Request:
    """A request inside the scheduler queue.

    Sortable by deadline so a priority queue gives us deadline-first ordering.
    """

    deadline_at: float
    request_id: str = field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False)
    enqueued_at: float = field(default_factory=time.monotonic, compare=False)


class Scheduler:
    """Continuous batching scheduler.

    Owned by the FastAPI app via lifespan. One scheduler per inference worker.
    """

    def __init__(
        self,
        worker: Any,  # TODO: replace with InferenceWorker type
        max_batch_size: int = 32,
        max_queue_size: int = 256,
        batch_timeout_ms: float = 5.0,
    ) -> None:
        self.worker = worker
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0

        self.queue: asyncio.PriorityQueue[Request] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._stop = asyncio.Event()

    async def submit(self, request_id: str, payload: Any, deadline_ms: float) -> Any:
        """Enqueue a request and await its result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        deadline_at = time.monotonic() + (deadline_ms / 1000.0)

        req = Request(deadline_at=deadline_at, request_id=request_id, payload=payload, future=future)
        await self.queue.put(req)
        QUEUE_DEPTH.set(self.queue.qsize())

        return await future

    async def run(self) -> None:
        """Main scheduler loop. Runs until stop() is called."""
        log.info("scheduler_started", max_batch_size=self.max_batch_size)

        while not self._stop.is_set():
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            batch = [first]
            deadline = time.monotonic() + self.batch_timeout_s

            # Drain additional requests up to batch_size or batch_timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            QUEUE_DEPTH.set(self.queue.qsize())
            BATCH_SIZE.observe(len(batch))

            for req in batch:
                QUEUE_WAIT.observe(time.monotonic() - req.enqueued_at)

            # TODO Phase 3: actually run inference and resolve futures
            # results = await self.worker.run_batch([r.payload for r in batch])
            # for req, result in zip(batch, results):
            #     req.future.set_result(result)

            for req in batch:
                if not req.future.done():
                    req.future.set_exception(NotImplementedError("scheduler not wired to worker yet"))

    async def stop(self) -> None:
        self._stop.set()
