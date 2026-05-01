"""FastAPI inference server.

Endpoints
---------
    POST /decode      Enqueue a spike-event window, await behavior prediction.
    WS   /stream      Streaming inference for live BCI sessions.
    GET  /health      Liveness probe — always 200 when the process is up.
    GET  /ready       Readiness probe — 200 only after model warmup completes.
    GET  /metrics     Prometheus metrics (mounted as ASGI sub-app).

Request lifecycle
-----------------
    1. /decode handler receives DecodeRequest, notes arrival time
    2. Converts events to worker payload dict
    3. Calls scheduler.submit() — this creates a Future and enqueues a Request
    4. The scheduler background task batches pending requests, calls
       worker.run_batch(), and resolves futures with results
    5. Handler awaits the Future, constructs DecodeResponse, returns

Configuration
-------------
    Controlled via environment variables (prefix CORTEX_):
        CORTEX_CHECKPOINT    path to model .pt file (optional)
        CORTEX_MAX_BATCH     max batch size (default 32)
        CORTEX_DEADLINE_MS   default SLO budget in ms (default 30)
        CORTEX_DEVICE        cuda|mps|cpu|auto (default auto)
        CORTEX_WARMUP_ITERS  dummy batches at startup (default 3)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from prometheus_client import make_asgi_app

from cortex.models.config import CORTEX_S
from cortex.serve.metrics import ERROR_COUNTER, QUEUE_DEPTH, REQUEST_COUNTER, REQUEST_LATENCY
from cortex.serve.scheduler import QueueFullError, Scheduler
from cortex.serve.schemas import DecodeRequest, DecodeResponse, HealthResponse, StreamFrame, StreamResponse
from cortex.serve.worker import InferenceWorker, _detect_device, load_model
from cortex.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _env(key: str, default: str) -> str:
    return os.environ.get(f"CORTEX_{key}", default)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize worker + scheduler at startup; clean up on shutdown."""
    configure_logging(level=_env("LOG_LEVEL", "INFO"))
    log.info("server_startup")

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = _env("DEVICE", "auto")
    if device_str == "auto":
        device = _detect_device()
    else:
        device = torch.device(device_str)

    # ── Model ─────────────────────────────────────────────────────────────────
    checkpoint = _env("CHECKPOINT", "") or None
    config = CORTEX_S

    model = load_model(config, checkpoint_path=checkpoint, device=device)

    # ── Worker ────────────────────────────────────────────────────────────────
    max_batch = int(_env("MAX_BATCH", "32"))
    worker = InferenceWorker(model=model, config=config, device=device, max_batch_size=max_batch)

    warmup_iters = int(_env("WARMUP_ITERS", "3"))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, worker.warmup, warmup_iters)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    deadline_ms = float(_env("DEADLINE_MS", "30"))
    scheduler = Scheduler(
        worker=worker,
        max_batch_size=max_batch,
        batch_timeout_ms=float(_env("BATCH_TIMEOUT_MS", "5")),
        default_deadline_ms=deadline_ms,
    )
    scheduler_task = asyncio.create_task(scheduler.run())

    # Attach to app state so handlers can reach them
    app.state.worker    = worker
    app.state.scheduler = scheduler
    app.state.ready     = True

    log.info(
        "server_ready",
        device=str(device),
        max_batch=max_batch,
        deadline_ms=deadline_ms,
    )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("server_shutdown")
    app.state.ready = False
    await scheduler.stop()
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    worker.shutdown()


app = FastAPI(
    title="Cortex-Engine",
    description="Real-time inference for transformer neural decoders",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness probe — always 200 if the process is alive."""
    scheduler: Scheduler | None = getattr(request.app.state, "scheduler", None)
    depth = scheduler.queue_depth if scheduler is not None else 0
    QUEUE_DEPTH.set(depth)
    return HealthResponse(model_loaded=True, queue_depth=depth)


@app.get("/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    """Readiness probe — 503 until model warmup is complete."""
    is_ready = getattr(request.app.state, "ready", False)
    if not is_ready:
        raise HTTPException(status_code=503, detail="model not ready")
    scheduler: Scheduler = request.app.state.scheduler
    return HealthResponse(model_loaded=True, queue_depth=scheduler.queue_depth)


@app.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest, request: Request) -> DecodeResponse:
    """Submit a spike-event window and return behavioral predictions.

    The handler enqueues the request into the continuous-batching scheduler
    and awaits the result.  Multiple concurrent handlers share the same
    scheduler so their requests are batched together.
    """
    REQUEST_COUNTER.labels(endpoint="decode").inc()
    arrival_ms = time.perf_counter() * 1000

    scheduler: Scheduler = request.app.state.scheduler

    payload = {
        "request_id": req.request_id,
        "events": [e.model_dump() for e in req.events],
    }
    deadline_ms = float(req.deadline_ms) if req.deadline_ms is not None else None

    try:
        with REQUEST_LATENCY.labels(endpoint="decode").time():
            result = await scheduler.submit(
                payload=payload,
                deadline_ms=deadline_ms,
                request_id=req.request_id,
            )
    except QueueFullError as exc:
        ERROR_COUNTER.labels(error_type="queue_full").inc()
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        log.error("decode_error", request_id=req.request_id, error=str(exc))
        raise HTTPException(status_code=500, detail="inference failed")

    total_ms = time.perf_counter() * 1000 - arrival_ms
    inference_ms = result.get("inference_ms", 0.0)
    queue_wait_ms = max(0.0, total_ms - inference_ms)

    return DecodeResponse(
        request_id=req.request_id,
        behavior=result["behavior"],
        latency_ms=round(total_ms, 3),
        queue_wait_ms=round(queue_wait_ms, 3),
        inference_ms=round(inference_ms, 3),
    )


@app.websocket("/stream")
async def stream(ws: WebSocket, request: Request) -> None:
    """Streaming inference for live BCI sessions.

    Protocol:
        Client → server: StreamFrame JSON  (sequence_number, events)
        Server → client: StreamResponse JSON (sequence_number, behavior, latency_ms)

    The server processes each frame through the scheduler (and thus benefits
    from dynamic batching with other concurrent stream sessions).
    """
    await ws.accept()
    scheduler: Scheduler = request.app.state.scheduler

    try:
        while True:
            frame_data = await ws.receive_json()
            frame = StreamFrame.model_validate(frame_data)
            t0 = time.perf_counter()

            payload = {
                "request_id": str(uuid.uuid4()),
                "events": [e.model_dump() for e in frame.events],
            }

            try:
                result = await scheduler.submit(payload=payload)
            except Exception as exc:
                log.error("stream_error", seq=frame.sequence_number, error=str(exc))
                await ws.close(code=1011, reason=str(exc))
                return

            latency_ms = (time.perf_counter() - t0) * 1000
            response = StreamResponse(
                sequence_number=frame.sequence_number,
                behavior=result["behavior"],
                latency_ms=round(latency_ms, 3),
            )
            await ws.send_json(response.model_dump())

    except WebSocketDisconnect:
        log.info("websocket_disconnect")
