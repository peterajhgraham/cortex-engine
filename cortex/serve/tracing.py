"""OpenTelemetry distributed tracing.

Wraps the FastAPI app and propagates context through the scheduler and
inference worker. Critical for debugging tail latency: a request that takes
80ms when p50 is 10ms will show exactly which stage was slow.

Spans:
    request           (FastAPI handler)
      ├─ enqueue      (scheduler.submit)
      ├─ queue_wait   (time in queue)
      ├─ inference    (worker.run_batch slice for this request)
      └─ serialize    (response construction)
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from cortex.utils.logging import get_logger

log = get_logger(__name__)


def configure_tracing(service_name: str = "cortex-engine", endpoint: str | None = None) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service identifier in traces
        endpoint: OTLP collector endpoint (e.g., "http://otel-collector:4317"). If None,
                  spans are created but not exported.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        log.info("tracing_configured", endpoint=endpoint)
    else:
        log.info("tracing_configured_no_export")

    trace.set_tracer_provider(provider)


def instrument_fastapi(app: Any) -> None:
    """Auto-instrument a FastAPI application."""
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
