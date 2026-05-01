"""OpenTelemetry distributed tracing.

Wraps the FastAPI app and propagates context through the scheduler and
inference worker. Critical for debugging tail latency: a request that takes
80ms when p50 is 10ms will show exactly which stage was slow.

Spans:
    request           (FastAPI handler)
      ├─ decode.schedule   (scheduler.submit + queue wait + inference)
      │    └─ scheduler.dispatch  (worker.run_batch for the batch)
      └─ (serialize — implicitly the handler after await returns)
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from cortex.utils.logging import get_logger

log = get_logger(__name__)

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as _OTLPExporter,
    )

    _OTLP_AVAILABLE = True
except ImportError:  # package not installed in dev; traces still created locally
    _OTLP_AVAILABLE = False
    _OTLPExporter = None  # type: ignore[assignment,misc]


def configure_tracing(service_name: str = "cortex-engine", endpoint: str | None = None) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service identifier shown in traces.
        endpoint: OTLP gRPC collector endpoint (e.g., "http://otel-collector:4317").
                  If None, spans are created but not exported — useful for local dev.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint and _OTLP_AVAILABLE:
        exporter = _OTLPExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        log.info("tracing_configured", endpoint=endpoint)
    elif endpoint and not _OTLP_AVAILABLE:
        log.warning(
            "tracing_otlp_unavailable",
            detail="opentelemetry-exporter-otlp-proto-grpc not installed; "
            "install it or run `pip install cortex-engine[otlp]`",
        )
    else:
        log.info("tracing_no_export", detail="spans created but not exported")

    trace.set_tracer_provider(provider)


def instrument_fastapi(app: Any) -> None:
    """Auto-instrument a FastAPI application with OTel middleware."""
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
