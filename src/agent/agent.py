"""The core agent loop.

Ties together the LLM client, tool registry, skill loader, and memory file.
Each agent turn is traced with OpenTelemetry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from opentelemetry import trace

from .llm import ChatMessage, LLMClient
from .memory import Memory, memory_tool
from .skills import render_skill_index
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    model: Optional[str] = None
    system_prompt: str = (
        "You are a helpful AI agent. You can use tools to accomplish tasks. "
        "Use the update_memory tool to record important facts and decisions."
    )
    max_turns: int = 10
    skills_dir: Optional[Path] = None
    memory: Optional[Memory] = None
    extra_tools: List = field(default_factory=list)


class Agent:
    """A tool-using, memory-aware agent with OTel tracing."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        llm: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
        tracer: Optional[trace.Tracer] = None,
    ) -> None:
        self.config = config
        self.llm = llm or LLMClient(model=config.model)
        self.registry = registry or ToolRegistry()
        self.tracer = tracer or trace.get_tracer("pi-agent")

        # Set up memory.
        self.memory = config.memory or Memory.discover()

        # Register built-in tools.
        self.registry.register(memory_tool(self.memory))
        for t in config.extra_tools:
            self.registry.register(t)

        # Load skills.
        self.skills: Dict[str, object] = {}
        if config.skills_dir:
            from .skills import load_skills

            self.skills = load_skills(config.skills_dir)

        self._build_system_prompt()

    def _build_system_prompt(self) -> None:
        parts = [self.config.system_prompt]
        parts.append(self.memory.render_for_system())
        parts.append(render_skill_index(self.skills))  # type: ignore[arg-type]
        self.system_prompt = "\n\n".join(parts)

    async def run(self, user_input: str) -> str:
        """Run the agent on a user input, returning the final text answer."""
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_input),
        ]

        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("agent.max_turns", self.config.max_turns)
            span.set_attribute("agent.model", self.llm.model)
            span.set_attribute("agent.tools", ",".join(self.registry.all().keys()) if hasattr(self.registry.all(), "keys") else ",".join(t.name for t in self.registry.all()))

            for turn in range(self.config.max_turns):
                with self.tracer.start_as_current_span(
                    f"agent.turn.{turn}"
                ) as turn_span:
                    turn_span.set_attribute("agent.turn", turn)

                    result = await self.llm.chat(
                        messages, tools=self.registry.schemas()
                    )
                    assistant_msg = ChatMessage(
                        role="assistant",
                        content=result.get("content"),
                        tool_calls=result.get("tool_calls"),
                    )
                    messages.append(assistant_msg)

                    tool_calls = result.get("tool_calls") or []
                    if not tool_calls:
                        # The model answered without calling tools.
                        turn_span.set_attribute("agent.final", True)
                        return result.get("content") or ""

                    # Execute each requested tool call.
                    for call in tool_calls:
                        fn_name = call["function"]["name"]
                        try:
                            args = json.loads(call["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}

                        with self.tracer.start_as_current_span(
                            f"tool.{fn_name}"
                        ) as tool_span:
                            tool_span.set_attribute("tool.name", fn_name)
                            tool_span.set_attribute("tool.arguments", json.dumps(args))
                            tool_obj = self.registry.get(fn_name)
                            if tool_obj is None:
                                output = f"Error: unknown tool '{fn_name}'."
                                tool_span.set_attribute("tool.error", True)
                            else:
                                try:
                                    output = await tool_obj.invoke(args)
                                except Exception as exc:  # noqa: BLE001
                                    logger.exception("Tool %s failed", fn_name)
                                    output = f"Error invoking tool: {exc}"
                                    tool_span.record_exception(exc)
                                    tool_span.set_attribute("tool.error", True)
                            tool_span.set_attribute("tool.output", output[:2000])

                        messages.append(
                            ChatMessage(
                                role="tool",
                                content=output,
                                tool_call_id=call["id"],
                                name=fn_name,
                            )
                        )

            span.set_attribute("agent.turns_exhausted", True)
            return "I've reached the maximum number of turns without a final answer."
