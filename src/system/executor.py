"""System command executor with strict human-in-the-loop safety protocol."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

from .guard import ActionSafetyLevel, SafetyGuard


@dataclass
class ExecutionResult:
    """Result of command evaluation and execution."""
    command: str
    safety_level: ActionSafetyLevel
    executed: bool
    stdout: str
    stderr: str
    exit_code: int
    message: str


class SystemExecutor:
    """Executes safe system actions and blocks sensitive ones pending user approval."""

    def __init__(self, default_cwd: Optional[str] = None) -> None:
        self.default_cwd = default_cwd

    def execute(
        self,
        command: str,
        user_confirmed: bool = False,
    ) -> ExecutionResult:
        """Evaluates command safety and executes if permitted."""
        level, reason = SafetyGuard.evaluate_command(command)

        # If sensitive and not confirmed by user, block execution and present command to user
        if level in (ActionSafetyLevel.SENSITIVE, ActionSafetyLevel.BLOCKED) and not user_confirmed:
            return ExecutionResult(
                command=command,
                safety_level=level,
                executed=False,
                stdout="",
                stderr="",
                exit_code=-1,
                message=(
                    f"⚠️ SENSITIVE ACTION BLOCKED (Safety Guard)\n"
                    f"Reason: {reason}\n"
                    f"Command: {command}\n"
                    f"Action Required: Only the USER can execute this sensitive operation."
                ),
            )

        # Execute safe or user-confirmed action
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                cwd=self.default_cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ExecutionResult(
                command=command,
                safety_level=level,
                executed=True,
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                exit_code=proc.returncode,
                message="Execution completed successfully." if proc.returncode == 0 else "Execution completed with errors.",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                command=command,
                safety_level=level,
                executed=False,
                stdout="",
                stderr="Execution timed out after 30 seconds.",
                exit_code=124,
                message="Command timed out.",
            )
        except Exception as e:
            return ExecutionResult(
                command=command,
                safety_level=level,
                executed=False,
                stdout="",
                stderr=str(e),
                exit_code=1,
                message=f"Execution failed: {e}",
            )
