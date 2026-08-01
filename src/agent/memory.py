"""agents.md memory file support.

The agent maintains a Markdown memory file (``agents.md``) that persists
facts, decisions, and notes across sessions. The memory is loaded into the
system prompt at the start of each run, and the agent can update it via the
``update_memory`` tool.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_FILE = "agents.md"


@dataclass
class Memory:
    """Read/write access to the agents.md memory file."""

    path: Path

    @classmethod
    def discover(cls, cwd: Optional[Path] = None) -> "Memory":
        """Locate the memory file in the working directory (or create it)."""
        base = cwd or Path.cwd()
        path = base / DEFAULT_MEMORY_FILE
        if not path.exists():
            path.touch()
            logger.info("Created memory file at %s", path)
        return cls(path)

    def read(self) -> str:
        """Return the current memory file contents."""
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def update(self, new_content: str) -> None:
        """Atomically replace the memory file contents."""
        # Write to a temp file in the same dir, then rename for atomicity.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        logger.info("Updated memory file at %s", self.path)

    def render_for_system(self) -> str:
        """Render the memory as a system-prompt section."""
        content = self.read().strip()
        if not content:
            return "## Memory\n\n(No memory recorded yet.)"
        return f"## Memory (from {self.path.name})\n\n{content}"


def memory_tool(memory: Memory):
    """Create the ``update_memory`` tool bound to a Memory instance."""
    from .tools import tool

    @tool(
        name="update_memory",
        description=(
            "Update the persistent agents.md memory file. Use this to record "
            "facts, decisions, and notes you want to remember across sessions. "
            "Pass the full new contents of the file."
        ),
        arguments_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full new contents of the memory file.",
                }
            },
            "required": ["content"],
        },
    )
    def update_memory(content: str) -> str:
        memory.update(content)
        return f"Memory updated. The file now contains:\n\n{content}"

    return update_memory
