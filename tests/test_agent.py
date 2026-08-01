"""Tests for the pi-agent."""

import asyncio
from pathlib import Path

import pytest

from agent.agent import Agent, AgentConfig
from agent.llm import LLMClient
from agent.memory import Memory
from agent.skills import load_skills, render_skill_index
from agent.tools import ToolRegistry, tool


class FakeLLM(LLMClient):
    """A deterministic fake LLM for tests."""

    def __init__(self, script):
        self.model = "fake-model"
        self.script = list(script)
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=0.7):
        self.calls += 1
        return self.script[self.calls - 1]


def make_registry():
    reg = ToolRegistry()

    @tool(
        "echo",
        "Echo text back.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    def echo(text: str) -> str:
        return text

    reg.register(echo)
    return reg


def test_load_skills(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\n# Demo\n\nDo the thing.\n"
    )
    (skill_dir / "helper.py").write_text("print('hi')\n")

    skills = load_skills(tmp_path / "skills")
    assert "demo" in skills
    assert skills["demo"].description == "A demo skill."
    assert "Do the thing." in skills["demo"].instructions
    assert len(skills["demo"].resources) == 1
    assert "demo" in render_skill_index(skills)


def test_memory_roundtrip(tmp_path):
    mem = Memory(tmp_path / "agents.md")
    mem.update("# New memory\n\n- fact 1")
    assert "# New memory" in mem.read()
    assert "fact 1" in mem.render_for_system()


def test_tool_inference_schema():
    reg = make_registry()
    schema = reg.get("echo").parameters
    assert schema["required"] == ["text"]
    assert schema["properties"]["text"]["type"] == "string"


def test_agent_tool_calling_loop(tmp_path):
    script = [
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                }
            ],
        },
        {"content": "echoed: hi"},
    ]
    reg = make_registry()
    config = AgentConfig(model="fake", skills_dir=tmp_path / "skills")
    agent = Agent(config=config, registry=reg, llm=FakeLLM(script))

    answer = asyncio.run(agent.run("test"))
    assert answer == "echoed: hi"
    assert "echo" in [t.name for t in reg.all()]


def test_agent_unknown_tool_handled(tmp_path):
    script = [
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nope", "arguments": "{}"},
                }
            ],
        },
        {"content": "done"},
    ]
    reg = make_registry()
    config = AgentConfig(model="fake", skills_dir=tmp_path / "skills")
    agent = Agent(config=config, registry=reg, llm=FakeLLM(script))
    answer = asyncio.run(agent.run("test"))
    assert answer == "done"
