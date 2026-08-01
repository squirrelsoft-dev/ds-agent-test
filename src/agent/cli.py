"""Command-line entry point for the agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from rich.console import Console
from rich.markdown import Markdown

from .agent import Agent, AgentConfig
from .otel import setup_otel
from .tools import ToolRegistry, tool


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register a couple of example tools."""

    @tool(
        name="echo",
        description="Echo back the provided text.",
        arguments_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo."}
            },
            "required": ["text"],
        },
    )
    def echo(text: str) -> str:
        return text

    @tool(
        name="current_time",
        description="Get the current UTC time as an ISO 8601 string.",
        arguments_schema={"type": "object", "properties": {}},
    )
    def current_time() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    registry.register(echo)
    registry.register(current_time)


async def _run(prompt: str, args: argparse.Namespace) -> None:
    console = Console()

    setup_otel(otlp_endpoint=args.otlp_endpoint, console=args.console_traces)

    registry = ToolRegistry()
    _register_builtin_tools(registry)

    skills_dir = Path(args.skills_dir) if args.skills_dir else Path.cwd() / "skills"

    config = AgentConfig(
        model=args.model,
        max_turns=args.max_turns,
        skills_dir=skills_dir,
        extra_tools=[],  # built-ins already registered via registry
    )
    agent = Agent(config=config, registry=registry)

    console.print(f"[bold]Agent[/bold] (model: {agent.llm.model})")
    console.print(f"[dim]Memory file: {agent.memory.path}[/dim]")
    console.print(f"[dim]Skills loaded: {len(agent.skills)}[/dim]")
    console.print()

    answer = await agent.run(prompt)
    console.print(Markdown(answer))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pi-agent",
        description="AI agent with OTel logging, tools, skills, and agents.md memory.",
    )
    parser.add_argument("prompt", nargs="?", help="The user prompt. If omitted, reads from stdin.")
    parser.add_argument("--model", help="Model to use (overrides AGENT_MODEL/OPENAI_MODEL).")
    parser.add_argument("--max-turns", type=int, default=10, help="Max agent turns.")
    parser.add_argument("--skills-dir", default="skills", help="Directory containing skills.")
    parser.add_argument(
        "--otlp-endpoint",
        help="OTLP HTTP endpoint (e.g. http://localhost:4318). "
        "Defaults to OTEL_EXPORTER_OTLP_ENDPOINT.",
    )
    parser.add_argument(
        "--console-traces",
        action="store_true",
        help="Also print spans to the console for local debugging.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)

    # Load .env if present (does not override existing env vars).
    load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("A prompt is required (argument or stdin).")

    asyncio.run(_run(prompt, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
