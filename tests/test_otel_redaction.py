"""Tests for OpenTelemetry span attribute redaction."""

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace.status import Status, StatusCode

from agent.otel import RedactingSpanProcessor


def _make_tracer(exporter: InMemorySpanExporter) -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        RedactingSpanProcessor(SimpleSpanProcessor(exporter))
    )
    return provider.get_tracer("redact-test")


def _span_text(span) -> str:
    """Serialise a span into a single blob so we can search every part of it."""
    return span.to_json(indent=None)


def test_span_attributes_never_contain_api_key(monkeypatch):
    key = "sk-proj-super-secret-openai-key-12345"
    agent_key = "sk-agent-secret-value-98765"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("AGENT_API_KEY", agent_key)

    exporter = InMemorySpanExporter()
    tracer = _make_tracer(exporter)

    with tracer.start_as_current_span("test.span") as span:
        span.set_attribute("llm.api_key", key)
        span.set_attribute("llm.agent_key", agent_key)
        span.set_attribute("normal.attr", "hello world")

    (span,) = exporter.get_finished_spans()
    blob = _span_text(span)

    assert key not in blob
    assert agent_key not in blob
    assert "[REDACTED]" in blob
    # Non-secret attributes are unchanged.
    assert "normal.attr" in blob
    assert "hello world" in blob


def test_error_path_redacts_exception_message_stack_and_events(monkeypatch):
    key = "sk-error-path-secret-value-555"
    monkeypatch.setenv("OPENAI_API_KEY", key)

    exporter = InMemorySpanExporter()
    tracer = _make_tracer(exporter)

    with tracer.start_as_current_span("test.failing") as span:
        try:
            raise RuntimeError(f"request failed with authorization header {key}")
        except RuntimeError as exc:  # noqa: BLE001
            span.record_exception(exc)
            span.add_event("http.call", {"response": f"token {key} rejected"})
            span.set_status(Status(StatusCode.ERROR, f"failed with key {key}"))
            span.set_attribute("config.api_key", key)

    (span,) = exporter.get_finished_spans()
    blob = _span_text(span)

    assert key not in blob
    # The exception event's message and stacktrace must not contain the key.
    assert "[REDACTED]" in blob
    # The recorded event, status description, and attributes are redacted.
    assert "authorization header [REDACTED]" in blob
    assert "token [REDACTED] rejected" in blob


def test_agent_run_error_path_redacts(monkeypatch, tmp_path):
    """Exercise the real agent loop's error path through the exporter."""
    import os

    from agent.agent import Agent, AgentConfig
    from agent.llm import LLMClient
    from agent.tools import ToolRegistry, tool

    key = "sk-real-agent-error-leak-000"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    os.environ["OPENAI_API_KEY"] = key

    reg = ToolRegistry()

    @tool("boom", "Always raises.", {})
    def boom() -> str:
        raise RuntimeError(f"headers: Authorization {key}")

    reg.register(boom)

    class ScriptedLLM(LLMClient):
        def __init__(self):
            self.model = "fake-model"
            self.calls = 0

        async def chat(self, messages, tools=None, temperature=0.7):
            self.calls += 1
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "boom",
                            "arguments": "{}",
                        },
                    }
                ],
            }

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        RedactingSpanProcessor(SimpleSpanProcessor(exporter))
    )
    tracer = provider.get_tracer("agent-error-test")

    config = AgentConfig(model="fake", skills_dir=tmp_path / "skills")
    agent = Agent(
        config=config,
        registry=reg,
        llm=ScriptedLLM(),
        tracer=tracer,
    )

    asyncio.run(agent.run("make it fail"))

    for span in exporter.get_finished_spans():
        assert key not in _span_text(span)
