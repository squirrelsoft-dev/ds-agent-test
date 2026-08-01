"""Minimal LLM client wrapper around the OpenAI SDK.

Uses the ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` environment variables so
it works with OpenAI, or any OpenAI-compatible endpoint (e.g. a local
server, LiteLLM proxy, or other providers).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant", or "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


class LLMClient:
    """Async client for chat completions with tool-calling support."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or os.environ.get("AGENT_MODEL") or os.environ.get(
            "OPENAI_MODEL"
        ) or "gpt-4o-mini"
        self.client = AsyncOpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-set"),
        )
        logger.info("LLM client configured with model=%s", self.model)

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send a chat request and return the raw assistant message dict."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("Sending chat request with %d messages", len(messages))
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.model_dump(exclude_none=True)
