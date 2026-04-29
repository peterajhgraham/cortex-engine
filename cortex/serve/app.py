"""FastAPI inference server.

Endpoints:
    POST /decode           Single inference (uses scheduler queue + continuous batching)
    WS   /stream           Streaming inference for live BCI sessions
    GET  /health           Liveness and queue depth
    GET  /ready            Readiness (model loaded + warmup complete)
    GET  /metrics          Prometheus metrics

Implementation notes for Claude Code:
    - Use FastAPI lifespan context to start the scheduler and warm the model.
    - Scheduler runs in a background asyncio task; FastAPI handlers enqueue and
      await per-request futures.
    - Prometheus metrics are exposed via prometheus_client's ASGI app mounted
      at /metrics.
    - OpenTelemetry instrumentation wraps the app for distributed tracing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from prometheus_client import make_asgi_app

from cortex.serve.schemas import (
    DecodeRequest,
    DecodeResponse,
    HealthResponse,
    StreamFrame,
    StreamResponse,
)
from cortex.serve.metrics import (
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    QUEUE_DEPTH,
)
from cortex.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the scheduler and warm the model at server boot."""
    configure_logging()
    log.info("server_startup")

    # TODO Phase 3.1: Load model from checkpoint
    # TODO Phase 3.2: Warm up with N dummy requests for cudagraph capture
    # TODO Phase 3.3: Start scheduler background task
    # app.state.scheduler = Scheduler(...)
    # app.state.scheduler_task = asyncio.create_task(app.state.scheduler.run())

    yield

    log.info("server_shutdown")
    # TODO Phase 3.4: Cancel scheduler, save metrics


app = FastAPI(
    title="Cortex-Engine",
    description="Real-time inference for transformer neural decoders",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(model_loaded=False, queue_depth=0)


@app.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    # TODO: actually check scheduler is running and model is loaded
    return HealthResponse(model_loaded=True, queue_depth=0)


@app.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest) -> DecodeResponse:
    """Submit a single decode request and await the result.

    The handler creates a future, hands it to the scheduler, and awaits.
    The scheduler batches with other in-flight requests.
    """
    REQUEST_COUNTER.labels(endpoint="decode").inc()
    with REQUEST_LATENCY.labels(endpoint="decode").time():
        # TODO Phase 3: actually submit to scheduler and await
        raise HTTPException(status_code=501, detail="Not implemented yet")


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """Streaming inference. Client sends StreamFrame messages, gets StreamResponse back."""
    await ws.accept()
    try:
        while True:
            frame_data = await ws.receive_json()
            frame = StreamFrame.model_validate(frame_data)
            # TODO Phase 3: submit to scheduler via streaming priority queue
            response = StreamResponse(
                sequence_number=frame.sequence_number,
                behavior=[0.0, 0.0],
                latency_ms=0.0,
            )
            await ws.send_json(response.model_dump())
    except WebSocketDisconnect:
        log.info("websocket_disconnect")
