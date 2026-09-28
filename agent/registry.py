"""
Genius Dynamic Skill Registry & Auto-Discovery Engine.

Scans the `agent/skills/` directory recursively to discover user-authored scripts,
tools, and modules. Exposes them to:
  1. The Autonomous AgentLoop (injected into system prompt)
  2. Terminal CLI (/skills command & direct auto-invocation)
  3. Direct Python callable interface
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """Metadata describing an auto-discovered skill."""
    name: str
    description: str
    file_path: Path
    module_path: str
    is_package: bool = False
    functions: List[str] = field(default_factory=list)
    cli_usage: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)
    author: str = "User / Genius"
    version: str = "1.0.0"

    def to_agent_prompt(self) -> str:
        """Formats the skill description for injection into LLM system prompt."""
        lines = [f"• Skill '{self.name}': {self.description}"]
        if self.cli_usage:
            lines.append(f"  CLI Usage: {self.cli_usage}")
        if self.functions:
            lines.append(f"  Functions: {', '.join(self.functions[:5])}")
        return "\n".join(lines)


class SkillRegistry:
    """Discovers, validates, and manages all user-defined and built-in skills in agent/skills."""

    def __init__(self, agent_dir: Optional[str | Path] = None) -> None:
        if agent_dir is None:
            # Default to <workspace_root>/agent
            self.agent_dir = Path(os.getcwd()).resolve() / "agent"
        else:
            self.agent_dir = Path(agent_dir).resolve()

        self.skills_dir = self.agent_dir / "skills"
        self.skills: Dict[str, SkillInfo] = {}
        self.refresh()

    def refresh(self) -> int:
        """Scans the skills directory and updates the registry."""
        self.skills.clear()
        if not self.skills_dir.exists():
            try:
                self.skills_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create skills dir {self.skills_dir}: {e}")
                return 0

        # Ensure agent_dir is on sys.path so modules can be imported
        agent_parent = str(self.agent_dir.parent)
        if agent_parent not in sys.path:
            sys.path.insert(0, agent_parent)

        # 1. Scan single-file python scripts (.py) in agent/skills/
        for p in self.skills_dir.glob("*.py"):
            if p.name.startswith("__"):
                continue
            skill = self._parse_file_skill(p)
            if skill:
                self.skills[skill.name] = skill

        # 2. Scan subdirectories in agent/skills/ (packages like agent/skills/imagen/)
        for d in self.skills_dir.iterdir():
            if d.is_dir() and not d.name.startswith((".", "_")):
                init_file = d / "__init__.py"
                main_py = d / f"{d.name}.py"
                target_file = None
                if main_py.exists():
                    target_file = main_py
                elif init_file.exists():
                    target_file = init_file
                else:
                    py_files = list(d.glob("*.py"))
                    if py_files:
                        target_file = py_files[0]

                if target_file:
                    skill = self._parse_file_skill(target_file, package_name=d.name)
                    if skill:
                        self.skills[skill.name] = skill

        return len(self.skills)

    def _parse_file_skill(self, file_path: Path, package_name: Optional[str] = None) -> Optional[SkillInfo]:
        """Extracts skill metadata from a python source file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        skill_name = package_name or file_path.stem

        # Extract docstring from beginning of file
        doc_match = re.search(r'^(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', content, re.DOTALL)
        docstring = doc_match.group(1).strip() if doc_match else f"Skill for {skill_name}"
        first_line_desc = docstring.split("\n")[0].strip()

        # Check for explicit SKILL_INFO dict in code
        cli_usage = f"python {file_path.relative_to(self.agent_dir.parent)}"
        functions: List[str] = []

        # Find top-level defs
        for m in re.finditer(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content, re.MULTILINE):
            fname = m.group(1)
            if not fname.startswith("_"):
                functions.append(fname)

        return SkillInfo(
            name=skill_name,
            description=first_line_desc,
            file_path=file_path,
            module_path=f"agent.skills.{skill_name}",
            is_package=bool(package_name),
            functions=functions,
            cli_usage=cli_usage,
        )

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        """Lookup a skill by name (case-insensitive)."""
        clean = name.strip().lower()
        for k, v in self.skills.items():
            if k.lower() == clean:
                return v
        return None

    def list_all(self) -> List[SkillInfo]:
        """Returns all registered skills."""
        return list(self.skills.values())

    def get_agent_prompt_summary(self) -> str:
        """Returns a formatted summary for the LLM system prompt."""
        if not self.skills:
            return "No custom user skills registered yet in agent/skills/."

        lines = [
            "### Available Specialized User Skills (in agent/skills/):",
            "The user has defined custom scripts and skills that you can directly execute or import in your solutions:",
        ]
        for s in self.skills.values():
            lines.append(s.to_agent_prompt())
        lines.append("You can execute them via `run_command` (e.g., `python -m agent.skills.<name> --help`) or import them in Python scripts.")
        return "\n".join(lines)

    def execute_skill_cli(self, name: str, args: List[str]) -> Tuple[bool, str, str]:
        """Runs the skill as a subprocess CLI command."""
        skill = self.get_skill(name)
        if not skill:
            return False, "", f"Skill '{name}' not found in registry."

        cmd = [sys.executable, str(skill.file_path)] + args
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.agent_dir.parent),
                timeout=120,
            )
            return proc.returncode == 0, proc.stdout, proc.stderr
        except Exception as e:
            return False, "", str(e)


# Global singleton instance
_registry_instance: Optional[SkillRegistry] = None


def get_skill_registry(agent_dir: Optional[str | Path] = None) -> SkillRegistry:
    """Returns or creates the global SkillRegistry instance."""
    global _registry_instance
    if _registry_instance is None or agent_dir is not None:
        _registry_instance = SkillRegistry(agent_dir)
    return _registry_instance
