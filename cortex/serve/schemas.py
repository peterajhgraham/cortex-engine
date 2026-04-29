"""Pydantic schemas for the inference API.

All request/response shapes are versioned. Breaking changes go through a new
schema version; the old one is supported for one minor release.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpikeEvent(BaseModel):
    """A single spike event in a request window."""

    neuron_id: int = Field(..., ge=0)
    time_bin: int = Field(..., ge=0)
    value: int = Field(..., ge=0)


class DecodeRequest(BaseModel):
    """Single decode request: spike events -> behavioral output."""

    request_id: str
    session_id: str
    events: list[SpikeEvent]
    deadline_ms: int | None = Field(None, gt=0, description="Override default SLO deadline")


class DecodeResponse(BaseModel):
    """Decoded behavioral output."""

    request_id: str
    behavior: list[float]  # length == config.behavior_dim
    latency_ms: float
    queue_wait_ms: float
    inference_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    queue_depth: int


class StreamFrame(BaseModel):
    """A single frame in a streaming decode session.

    The websocket protocol sends one of these per inference window; the client
    receives a corresponding StreamResponse.
    """

    sequence_number: int
    events: list[SpikeEvent]


class StreamResponse(BaseModel):
    sequence_number: int
    behavior: list[float]
    latency_ms: float
