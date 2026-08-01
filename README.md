# pi-agent

A Python AI agent with **OpenTelemetry logging**, **tools**, **skill loading**, and
**`agents.md` memory file** support.

## Features

- **Agent loop** — a tool-using agent that calls an LLM (OpenAI or any
  OpenAI-compatible endpoint via the OpenAI SDK), executes tool calls, and
  iterates until it produces a final answer.
- **OpenTelemetry logging** — every agent run, turn, and tool call is traced.
  Spans export to an OTLP collector (e.g. Jaeger, Grafana Tempo, or a local
  OTel collector) or print to the console for local debugging.
- **Tools** — a clean tool abstraction with a decorator, JSON-schema argument
  validation, and a registry. Tools are exposed to the model in OpenAI
  function-calling format.
- **Skill loading** — skills are directories containing a `SKILL.md` file
  (with optional frontmatter and bundled scripts). Loaded skills are injected
  into the system prompt.
- **`agents.md` memory** — a persistent Markdown memory file loaded into the
  system prompt each run. The agent can update it via the `update_memory`
  tool, so facts persist across sessions.

## Project layout

```
pi-agent                 # executable entry point (no install needed)
agents.md                # persistent agent memory
skills/                  # loadable skills (each is a dir with SKILL.md)
src/agent/
  cli.py                 # CLI entry point
  agent.py               # core agent loop + AgentConfig
  tools.py               # Tool, ToolRegistry, @tool decorator
  skills.py              # skill loader
  memory.py              # agents.md read/write
  llm.py                 # LLM client wrapper
  otel.py                # OpenTelemetry setup
tests/                   # pytest suite
```

## Quick start

```bash
# 1. Create a virtualenv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Configure credentials (copy .env.example to .env and fill in)
cp .env.example .env
#   OPENAI_API_KEY=sk-...
#   (optionally AGENT_MODEL, OPENAI_BASE_URL, OTEL_EXPORTER_OTLP_ENDPOINT)

# 3. Run
./pi-agent "What time is it, and remember my favorite color is blue."
```

Or via the installed console script:

```bash
pi-agent "your prompt"
```

## OpenTelemetry

By default the agent prints a log line noting there is no OTLP endpoint. To
export traces:

```bash
# Point at a collector (e.g. a local OTel collector on 4318)
./pi-agent --otlp-endpoint http://localhost:4318 "hello"
# or set the env var
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Print spans to the console for local debugging
./pi-agent --console-traces "hello"
```

Each run produces a trace like:

```
agent.run
├── agent.turn.0
│   └── tool.echo
└── agent.turn.1
```

## Adding tools

Tools are plain functions decorated with `@tool`. Add them to the registry in
`cli.py` (or anywhere):

```python
from agent.tools import tool

@tool(
    name="my_tool",
    description="Does something useful.",
    arguments_schema={
        "type": "object",
        "properties": {"arg": {"type": "string"}},
        "required": ["arg"],
    },
)
def my_tool(arg: str) -> str:
    return f"processed {arg}"
```

If you omit `arguments_schema`, it is inferred from the function's type
annotations.

## Adding skills

Create a directory under `skills/` with a `SKILL.md` file:

```markdown
---
name: my-skill
description: What this skill does.
---

# My Skill

Instructions the agent should follow when this skill is relevant.
```

Skills are auto-discovered at startup and injected into the system prompt.

## Memory (`agents.md`)

`agents.md` is loaded into the system prompt at the start of every run. The
agent can update it using the `update_memory` tool, so things it learns
persist across sessions. You can also edit it by hand.

## Running tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```
