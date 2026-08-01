# agents.md

This file is the agent's persistent memory. It is loaded into the system
prompt at the start of every run, and the agent can update it via the
`update_memory` tool.

## Project notes

- This is a demonstration project for an AI agent built with Python.
- The agent supports OpenTelemetry logging, tools, skill loading, and
  this agents.md memory file.

## Preferences

- Prefer clear, concise responses.
- Use tools whenever they can help accomplish a task.
