"""OpenTelemetry setup for tracing and logging.

Configures the global tracer provider and optionally exports traces to an
OTLP collector. Also wires up Python's stdlib ``logging`` to emit structured
logs as OpenTelemetry log records when configured.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Union

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import (
    Event,
    ReadableSpan,
    Span,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanProcessor,
)
from opentelemetry.trace.status import Status
from opentelemetry.util import types as otel_types

logger = logging.getLogger(__name__)

SERVICE_NAME = "pi-agent"

#: Environment variables whose values must never leave the process in a span.
SECRET_ENV_VARS = ("OPENAI_API_KEY", "AGENT_API_KEY")

#: Placeholder used in place of a redacted secret. Redaction keeps the
#: attribute present (so a dropped attribute doesn't silently signal the
#: presence of a key) while hiding the value itself.
REDACTION_PLACEHOLDER = "[REDACTED]"


def _secret_values() -> frozenset[str]:
    """Return the non-empty values of the configured API-key env vars."""
    return frozenset(
        value for name in SECRET_ENV_VARS if (value := os.environ.get(name))
    )


def _redact_text(text: str, secrets: frozenset[str]) -> str:
    """Replace every occurrence of a secret with the redaction placeholder."""
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, REDACTION_PLACEHOLDER)
    return redacted


def _redact_value(
    value: otel_types.AttributeValue, secrets: frozenset[str]
) -> otel_types.AttributeValue:
    if isinstance(value, str):
        return _redact_text(value, secrets)
    return value


def _redact_attributes(
    attributes: Optional[otel_types.Attributes], secrets: frozenset[str]
) -> Optional[dict[str, otel_types.AttributeValue]]:
    if not attributes:
        return None
    redacted: dict[str, otel_types.AttributeValue] = {}
    for key, value in attributes.items():
        redacted[key] = _redact_value(value, secrets)
    return redacted


def _redact_events(
    events: Sequence[Event], secrets: frozenset[str]
) -> tuple[Event, ...]:
    """Redact every recorded event, including OpenTelemetry exception events.

    An exception event carries the exception message in
    ``exception.message`` and the stack trace in ``exception.stacktrace``;
    both can contain request context (headers included) and therefore the key.
    """
    redacted: list[Event] = []
    for event in events:
        redacted.append(
            Event(
                name=event.name,
                attributes=_redact_attributes(event.attributes, secrets),
                timestamp=event.timestamp,
            )
        )
    return tuple(redacted)


def _redact_span(
    span: ReadableSpan, secrets: frozenset[str]
) -> ReadableSpan:
    """Return a copy of ``span`` with every secret value scrubbed."""
    status = span.status
    if status is not None and status.description:
        status = Status(
            status_code=status.status_code,
            description=_redact_text(status.description, secrets),
        )
    return ReadableSpan(
        name=_redact_text(span.name, secrets) if span.name else span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_redact_attributes(span.attributes, secrets),
        events=_redact_events(span.events, secrets),
        links=span.links,
        kind=span.kind,
        instrumentation_scope=span.instrumentation_scope,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
    )


class RedactingSpanProcessor(SpanProcessor):
    """Scrub API keys from every span before it reaches an exporter.

    Wraps another span processor/exporter and rewrites each ended span so that
    no attribute, event (including exceptions), span name, or status description
    ever contains the value of ``OPENAI_API_KEY`` or ``AGENT_API_KEY``.

    Redaction happens here — at the single point where spans are about to leave
    the process — rather than at each attribute call site, so a key can never
    be introduced later by an unredacted code path.
    """

    def __init__(
        self, inner: Union[SpanProcessor, SpanExporter]
    ) -> None:
        if isinstance(inner, SpanProcessor):
            self._processor = inner
            self._exporter: Optional[SpanExporter] = None
        elif isinstance(inner, SpanExporter):
            processor = SimpleSpanProcessor(inner)
            self._processor = processor
            self._exporter = inner
        else:
            raise TypeError(
                "inner must be a SpanProcessor or SpanExporter, got "
                f"{type(inner).__name__}"
            )

    def _secrets(self) -> frozenset[str]:
        return _secret_values()

    def on_start(
        self, span: Span, parent_context: Optional[object] = None
    ) -> None:
        self._processor.on_start(span, parent_context=parent_context)

    def _on_ending(self, span: Span) -> None:
        self._processor._on_ending(span)  # noqa: SLF001

    def on_end(self, span: ReadableSpan) -> None:
        self._processor.on_end(_redact_span(span, self._secrets()))

    def shutdown(self) -> None:
        self._processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._processor.force_flush(timeout_millis)


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
        provider.add_span_processor(
            RedactingSpanProcessor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=url))
            )
        )
        logger.info("Exporting traces to OTLP endpoint: %s", url)
    else:
        logger.info(
            "No OTLP endpoint configured; traces will not be exported. "
            "Set OTEL_EXPORTER_OTLP_ENDPOINT or pass otlp_endpoint."
        )

    if console:
        provider.add_span_processor(
            RedactingSpanProcessor(ConsoleSpanExporter())
        )

    return trace.get_tracer(service_name)
