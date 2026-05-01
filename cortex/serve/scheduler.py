"""Continuous batching scheduler.

How it works
------------
Requests arrive via submit() and are placed in an asyncio.PriorityQueue ordered
by deadline (earliest deadline first).  The scheduler loop:

    1. Blocks until the first request arrives (with a 100 ms watchdog timeout).
    2. Drains additional requests for up to batch_timeout_ms, building the
       largest batch that fits within max_batch_size.
    3. Dispatches the batch to InferenceWorker.run_batch() — this awaits
       execution in the worker's ThreadPoolExecutor.
    4. Resolves each request's asyncio.Future with its result slice.

Deadline-aware batching
-----------------------
Requests are sorted by deadline in the priority queue.  When budget allows,
requests nearest their deadline are processed first.  This is soft real-time:
the scheduler never drops requests, but latency SLO compliance improves at low
load because the queue doesn't build up behind slower requests.

Admission control
-----------------
If the queue is full (max_queue_size), submit() raises QueueFullError
immediately rather than blocking indefinitely.  Callers should implement
backpressure (e.g., return HTTP 429) rather than letting the queue grow
without bound.

Differences from LLM batching
------------------------------
LLM continuous batching packs requests at different autoregressive positions,
so every request in a batch has a different KV-cache cursor.  Spike decoders
are non-autoregressive: every request is an independent one-shot forward pass.
Batching here is purely for throughput — we fan out to workers as wide as
their GPU can handle.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace

from cortex.serve.metrics import BATCH_SIZE, ERROR_COUNTER, QUEUE_DEPTH, QUEUE_WAIT
from cortex.utils.logging import get_logger

_tracer = trace.get_tracer("cortex.scheduler")

log = get_logger(__name__)


class QueueFullError(Exception):
    """Raised by submit() when the scheduler queue is at capacity."""


@dataclass(order=True)
class Request:
    """A pending request inside the scheduler's priority queue.

    Sorted by deadline_at so asyncio.PriorityQueue gives deadline-first order.
    Non-sort fields carry the actual payload and resolution future.
    """

    deadline_at: float
    request_id: str = field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False)
    enqueued_at: float = field(default_factory=time.monotonic, compare=False)
    # Captured OTel context so inference spans are children of the caller's span
    trace_ctx: Any = field(default=None, compare=False)


class Scheduler:
    """Continuous batching scheduler.

    Lifecycle:
        scheduler = Scheduler(worker, ...)
        task = asyncio.create_task(scheduler.run())
        ...
        await scheduler.stop()
        task.cancel()
    """

    def __init__(
        self,
        worker: Any,
        max_batch_size: int = 32,
        max_queue_size: int = 256,
        batch_timeout_ms: float = 5.0,
        default_deadline_ms: float = 30.0,
    ) -> None:
        self.worker = worker
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self.default_deadline_ms = default_deadline_ms

        self.queue: asyncio.PriorityQueue[Request] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._stop = asyncio.Event()
        self._n_batches_processed = 0

    async def submit(
        self,
        payload: Any,
        deadline_ms: float | None = None,
        request_id: str | None = None,
    ) -> Any:
        """Enqueue a request and await its result.

        Args:
            payload:      Arbitrary dict passed through to the worker.
            deadline_ms:  Time budget from now in ms.  Default: 30 ms.
            request_id:   Caller-supplied ID for tracing; auto-generated if None.

        Returns:
            Whatever the worker returns for this request.

        Raises:
            QueueFullError: if max_queue_size is reached.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        deadline_budget = deadline_ms if deadline_ms is not None else self.default_deadline_ms
        deadline_at = time.monotonic() + (deadline_budget / 1000.0)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        req = Request(
            deadline_at=deadline_at,
            request_id=request_id,
            payload=payload,
            future=future,
            trace_ctx=otel_context.get_current(),
        )

        # Admission control: fail fast rather than block on a full queue
        if self.queue.full():
            ERROR_COUNTER.labels(error_type="queue_full").inc()
            raise QueueFullError(
                f"scheduler queue at capacity ({self.max_queue_size}); try again"
            )

        await self.queue.put(req)
        QUEUE_DEPTH.set(self.queue.qsize())
        return await future

    async def run(self) -> None:
        """Main scheduler loop.  Run as an asyncio background task."""
        log.info(
            "scheduler_started",
            max_batch_size=self.max_batch_size,
            batch_timeout_ms=self.batch_timeout_s * 1000,
            default_deadline_ms=self.default_deadline_ms,
        )

        while not self._stop.is_set():
            # ── Wait for the first request ────────────────────────────────────
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # ── Drain up to max_batch_size within batch_timeout ───────────────
            batch = [first]
            deadline = time.monotonic() + self.batch_timeout_s

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

            # ── Dispatch to worker ────────────────────────────────────────────
            payloads = [r.payload for r in batch]
            # Attach the first request's trace context so the inference span is
            # a child of the caller's HTTP handler span.
            ctx_token = None
            if batch and batch[0].trace_ctx is not None:
                ctx_token = otel_context.attach(batch[0].trace_ctx)
            try:
                with _tracer.start_as_current_span(
                    "scheduler.dispatch",
                    attributes={"batch_size": len(batch)},
                ):
                    results = await self.worker.run_batch(payloads)
            except Exception as exc:
                log.error("worker_error", error=str(exc), batch_size=len(batch))
                ERROR_COUNTER.labels(error_type="worker_error").inc()
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)
                continue
            finally:
                if ctx_token is not None:
                    otel_context.detach(ctx_token)

            # ── Resolve per-request futures ───────────────────────────────────
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)

            self._n_batches_processed += 1
            log.debug(
                "batch_dispatched",
                batch_size=len(batch),
                n_total=self._n_batches_processed,
            )

        log.info("scheduler_stopped", total_batches=self._n_batches_processed)

    async def stop(self) -> None:
        """Signal the run() loop to exit after the current batch."""
        self._stop.set()

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()
