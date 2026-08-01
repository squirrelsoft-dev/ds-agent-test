"""OpenTelemetry setup for tracing and logging.

Configures the global tracer provider and optionally exports traces to an
OTLP collector. Also wires up Python's stdlib ``logging`` to emit structured
logs as OpenTelemetry log records when configured.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

SERVICE_NAME = "pi-agent"


def setup_otel(
    *,
    service_name: str = SERVICE_NAME,
    otlp_endpoint: Optional[str] = None,
    console: bool = False,
) -> trace.Tracer:
    """Configure the global OpenTelemetry tracer provider and return a tracer.

    Args:
        service_name: Name of the service reported to the collector.
        otlp_endpoint: OTLP HTTP endpoint (e.g. ``http://localhost:4318/v1/traces``).
            If None, falls back to the ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
        console: If True, also print spans to stdout (useful for local debugging).

    Returns:
        An OpenTelemetry ``Tracer`` instance.
    """
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Resolve the endpoint from the argument or environment.
    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        url = endpoint.rstrip("/") + "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
        logger.info("Exporting traces to OTLP endpoint: %s", url)
    else:
        logger.info(
            "No OTLP endpoint configured; traces will not be exported. "
            "Set OTEL_EXPORTER_OTLP_ENDPOINT or pass otlp_endpoint."
        )

    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    return trace.get_tracer(service_name)
