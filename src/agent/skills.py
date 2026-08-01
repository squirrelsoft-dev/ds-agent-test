"""Skill loading.

Skills are directories containing a ``SKILL.md`` file that describes the
skill, what it does, and how to use it. A skill may also bundle scripts or
other resources. The loader discovers skills on disk and exposes their
instructions to the agent so it can follow them when relevant.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
    path: Path
    instructions: str = ""
    # Paths to any bundled scripts/resources, relative to the skill dir.
    resources: List[Path] = field(default_factory=list)

    def script_path(self, name: str) -> Optional[Path]:
        """Resolve a bundled script by name, if present."""
        candidate = self.path / name
        return candidate if candidate.exists() else None


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def _parse_frontmatter(text: str) -> Dict[str, str]:
    """Parse simple YAML-ish frontmatter from a SKILL.md file."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def load_skill(skill_dir: Path) -> Optional[Skill]:
    """Load a single skill from a directory containing SKILL.md."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    text = skill_file.read_text(encoding="utf-8")
    meta = _parse_frontmatter(text)

    name = meta.get("name") or skill_dir.name
    description = meta.get("description") or ""

    # Strip frontmatter from the instructions body.
    body = _FRONTMATTER_RE.sub("", text, count=1).strip()

    # Collect bundled resources (anything besides SKILL.md).
    resources = [
        p for p in skill_dir.iterdir()
        if p.is_file() and p.name != "SKILL.md"
    ]

    return Skill(
        name=name,
        description=description,
        path=skill_dir,
        instructions=body,
        resources=resources,
    )


def load_skills(skills_dir: Path) -> Dict[str, Skill]:
    """Discover and load all skills under ``skills_dir``.

    Returns a mapping of skill name -> Skill.
    """
    skills: Dict[str, Skill] = {}
    if not skills_dir.exists():
        logger.warning("Skills directory does not exist: %s", skills_dir)
        return skills

    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill = load_skill(entry)
        if skill:
            skills[skill.name] = skill
            logger.info("Loaded skill '%s' from %s", skill.name, entry)
        else:
            logger.debug("Skipping %s (no SKILL.md)", entry)

    return skills


def render_skill_index(skills: Dict[str, Skill]) -> str:
    """Render a compact index of available skills for the system prompt."""
    if not skills:
        return "No skills available."
    lines = ["Available skills:"]
    for name, skill in skills.items():
        lines.append(f"- {name}: {skill.description}")
    return "\n".join(lines)
