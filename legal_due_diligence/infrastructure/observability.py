"""
Minimal OpenTelemetry tracer for LexGraph-DD.

Behaviour:
- OTEL_ENDPOINT set  → spans exported via OTLP HTTP to that endpoint
- OTEL_ENDPOINT unset → NoOpTracerProvider; spans are created but never exported
  (zero network overhead, zero stdout noise in dev)

Usage in agent code:
    from infrastructure.observability import get_tracer
    with get_tracer().start_as_current_span("clause_extraction") as span:
        span.set_attribute("key", value)
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

_TRACER_NAME = "lexgraph_dd"


def _configure_provider() -> None:
    from core.config import settings

    if not settings.otel_endpoint:
        return  # leave the default NoOpTracerProvider in place

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
        )
        trace.set_tracer_provider(provider)
        logger.info("[observability] OTel spans → %s", settings.otel_endpoint)
    except Exception as exc:
        logger.warning("[observability] OTel setup failed — spans disabled: %s", exc)


_configure_provider()


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)
