"""Tool system for the agent.

A tool is a callable with a name, description, and a JSON schema describing
its arguments. Tools are registered in a registry and exposed to the LLM as
function definitions. When the model requests a tool call, the agent invokes
the tool and feeds the result back to the model.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, create_model

# A tool implementation. It may be sync or async.
ToolFn = Callable[..., Any]


@dataclass
class Tool:
    """A single tool the agent can invoke."""

    name: str
    description: str
    fn: ToolFn
    # JSON schema for the tool's arguments (OpenAI function-calling format).
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, arguments: Dict[str, Any]) -> str:
        """Invoke the tool with parsed arguments and return a string result."""
        result = self.fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
        # Normalize non-string results to JSON so they round-trip cleanly.
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)


class ToolRegistry:
    """Registry of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def tool(
    name: str,
    description: str,
    arguments_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[ToolFn], Tool]:
    """Decorator that turns a function into a :class:`Tool`.

    If ``arguments_schema`` is omitted, it is inferred from the function's
    annotated parameters using pydantic.
    """

    def decorator(fn: ToolFn) -> Tool:
        schema = arguments_schema or _infer_schema(fn)
        return Tool(name=name, description=description, fn=fn, parameters=schema)

    return decorator


def _infer_schema(fn: ToolFn) -> Dict[str, Any]:
    """Build a minimal JSON schema from a function's type annotations."""
    hints = getattr(fn, "__annotations__", {})
    sig = inspect.signature(fn)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        ann = hints.get(param_name, Any)
        properties[param_name] = {
            "type": _json_type(ann),
            "description": f"Argument '{param_name}'.",
        }
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _json_type(ann: Any) -> str:
    """Map a Python type annotation to a JSON schema type string."""
    if ann is str:
        return "string"
    if ann is int or ann is float:
        return "number"
    if ann is bool:
        return "boolean"
    if ann is list or getattr(ann, "__origin__", None) is list:
        return "array"
    if ann is dict or getattr(ann, "__origin__", None) is dict:
        return "object"
    return "string"
