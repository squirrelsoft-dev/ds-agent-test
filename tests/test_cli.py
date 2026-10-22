"""Tests for the pi-agent CLI, including the offline --list-skills path."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.cli import main


def _write_skill(skill_dir: Path, name: str, description: str) -> None:
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"
    )


def test_list_skills_default_directory(tmp_path, monkeypatch, capsys):
    """Default skills directory is resolved relative to cwd."""
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir / "b-skill", "b-skill", "Second skill.")
    _write_skill(skills_dir / "a-skill", "a-skill", "First skill.")

    assert main(["--list-skills"]) == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed == [
        {"name": "a-skill", "description": "First skill."},
        {"name": "b-skill", "description": "Second skill."},
    ]
    assert captured.err == ""


def test_list_skills_custom_directory(tmp_path, capsys):
    """--skills-dir overrides the default."""
    custom = tmp_path / "my-skills"
    _write_skill(custom / "custom", "custom", "A custom skill.")

    assert main(["--list-skills", "--skills-dir", str(custom)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == [{"name": "custom", "description": "A custom skill."}]


def test_list_skills_sorted_by_name_not_directory(tmp_path, capsys):
    """Skills are sorted by their declared name, not their directory name."""
    skills_dir = tmp_path / "skills"
    # Directory names are alphabetically opposite of declared names.
    _write_skill(skills_dir / "zzz", "alpha", "Alpha skill.")
    _write_skill(skills_dir / "aaa", "omega", "Omega skill.")

    assert main(["--list-skills", "--skills-dir", str(skills_dir)]) == 0
    names = [obj["name"] for obj in json.loads(capsys.readouterr().out)]
    assert names == ["alpha", "omega"]


def test_list_skills_empty_directory(tmp_path, capsys):
    """An empty skills directory produces []."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    assert main(["--list-skills", "--skills-dir", str(skills_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_list_skills_missing_directory(tmp_path, capsys):
    """A missing skills directory produces []."""
    missing = tmp_path / "no-skills"

    assert main(["--list-skills", "--skills-dir", str(missing)]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_list_skills_exact_keys_no_extra_data(tmp_path, capsys):
    """Each object has exactly 'name' and 'description'."""
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir / "demo", "demo", "Demo skill.")

    assert main(["--list-skills", "--skills-dir", str(skills_dir)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == [{"name": "demo", "description": "Demo skill."}]


def test_list_skills_no_stdin_no_telemetry_no_agent(tmp_path, monkeypatch, capsys):
    """--list-skills exits early without touching stdin, telemetry, or the agent."""
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir / "offline", "offline", "Offline skill.")

    # Any of these being called indicates an accidental online path.
    def exploding_read(*args, **kwargs):
        raise AssertionError("stdin.read() must not be called")

    monkeypatch.setattr("sys.stdin.read", exploding_read)

    with patch("agent.cli.setup_otel") as mock_setup_otel:
        with patch("agent.cli.Agent") as mock_agent:
            assert main(["--list-skills", "--skills-dir", str(skills_dir)]) == 0

    assert json.loads(capsys.readouterr().out) == [
        {"name": "offline", "description": "Offline skill."}
    ]
    mock_setup_otel.assert_not_called()
    mock_agent.assert_not_called()
