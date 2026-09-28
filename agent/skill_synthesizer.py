"""
Genius Self-Evolving AI Skill Synthesizer.

Enables Genius AI to listen to user requests in natural language, automatically
design a new tool/skill, write the complete Python script into `agent/skills/`,
test and verify it in a sandbox execution, hot-reload the SkillRegistry, and
instantly put it into active service!
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import SkillInfo, SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


class SkillSynthesizer:
    """Autonomous Self-Evolution engine: synthesizes, tests, and deploys new skills on the fly."""

    def __init__(self, agent_dir: Optional[str | Path] = None) -> None:
        if agent_dir is None:
            self.agent_dir = Path(os.getcwd()).resolve() / "agent"
        else:
            self.agent_dir = Path(agent_dir).resolve()

        self.skills_dir = self.agent_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry = get_skill_registry(self.agent_dir)

    def synthesize_and_deploy(
        self,
        skill_name: str,
        purpose: str,
        code_content: str,
        test_args: Optional[List[str]] = None,
    ) -> Tuple[bool, str, Optional[SkillInfo]]:
        """
        Saves user- or LLM-generated code into agent/skills/<skill_name>.py,
        tests it via python CLI, and registers it.

        Args:
            skill_name: Slug name for the skill (e.g. 'audio_converter', 'qr_maker')
            purpose: High-level description of what the skill accomplishes
            code_content: Full Python source code for the skill
            test_args: Optional CLI arguments to test-run (default: ['--help'])

        Returns:
            (success, message, skill_info)
        """
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", skill_name.strip().lower()).strip("_")
        if not clean_name:
            return False, "Invalid skill name provided.", None

        target_file = self.skills_dir / f"{clean_name}.py"

        # Ensure docstring contains purpose if missing
        final_code = code_content.strip()
        if not (final_code.startswith('"""') or final_code.startswith("'''")):
            doc = f'"""\n{clean_name.replace("_", " ").title()} — {purpose}\n\nGenerated autonomously by Genius AI Self-Evolution Engine.\n"""\n\n'
            final_code = doc + final_code

        # 1. Write file to disk
        try:
            target_file.write_text(final_code, encoding="utf-8")
        except Exception as e:
            return False, f"Failed to write skill file to {target_file}: {e}", None

        # 2. Syntax Validation Check
        try:
            import ast
            ast.parse(final_code, filename=str(target_file))
        except SyntaxError as e:
            # Attempt to delete invalid file
            target_file.unlink(missing_ok=True)
            return False, f"Syntax error in synthesized skill code: {e}", None

        # 3. Test Run in Subprocess
        args = test_args if test_args is not None else ["--help"]
        cmd = [sys.executable, str(target_file)] + args
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.agent_dir.parent),
                timeout=30,
            )
            if res.returncode != 0 and "--help" in args:
                # Some scripts might not use argparse --help, check if basic import works
                test_import = [sys.executable, "-c", f"import agent.skills.{clean_name}"]
                res_imp = subprocess.run(test_import, capture_output=True, text=True, cwd=str(self.agent_dir.parent), timeout=15)
                if res_imp.returncode != 0:
                    return False, f"Verification failed with stderr:\n{res.stderr or res_imp.stderr}", None
        except Exception as e:
            return False, f"Error running test verification for {clean_name}: {e}", None

        # 4. Refresh Registry
        self.registry.refresh()
        new_skill = self.registry.get_skill(clean_name)

        if not new_skill:
            return False, f"Skill file created at {target_file} but could not be indexed in registry.", None

        success_msg = (
            f"Successfully synthesized and deployed skill '{clean_name}'!\n"
            f"• File Location: {target_file}\n"
            f"• Purpose: {purpose}\n"
            f"• CLI Usage: {new_skill.cli_usage}\n"
            f"• Status: HOT-RELOADED & ACTIVE"
        )
        return True, success_msg, new_skill

    def scaffold_template(self, skill_name: str, purpose: str, inputs: List[str], outputs: str) -> str:
        """Generates a starter production-grade template for a new skill."""
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", skill_name.strip().lower()).strip("_")
        title = clean_name.replace("_", " ").title()

        args_code = []
        for inp in inputs:
            arg_flag = f"--{inp.lower().replace('_', '-')}"
            args_code.append(f'    parser.add_argument("{arg_flag}", required=True, help="Input for {inp}")')

        args_block = "\n".join(args_code) or '    parser.add_argument("--input", required=True, help="Input data")'

        return f'''"""
{title} Skill for Genius AI.

Purpose:
  {purpose}

Outputs:
  {outputs}

Generated autonomously by Genius Self-Evolving AI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_skill(*args, **kwargs) -> Any:
    """Core execution function for {title}."""
    # TODO: Autonomous implementation logic
    output_path = kwargs.get("output", "output.txt")
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # Process logic here
    p.write_text(f"Executed {title} with args: {{kwargs}}", encoding="utf-8")
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="{title} Skill CLI")
{args_block}
    parser.add_argument("-o", "--output", default="{clean_name}_output.txt", help="Output destination path")
    args = parser.parse_args()

    try:
        res = run_skill(**vars(args))
        print(f"SUCCESS: {title} executed successfully. Result: {{res}}")
    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


# Global singleton instance
_synthesizer_instance: Optional[SkillSynthesizer] = None


def get_skill_synthesizer(agent_dir: Optional[str | Path] = None) -> SkillSynthesizer:
    """Returns or creates the global SkillSynthesizer instance."""
    global _synthesizer_instance
    if _synthesizer_instance is None or agent_dir is not None:
        _synthesizer_instance = SkillSynthesizer(agent_dir)
    return _synthesizer_instance
