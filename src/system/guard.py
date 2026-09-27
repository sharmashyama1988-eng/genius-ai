"""Safety Guard for OS and system commands preventing destructive actions."""

from __future__ import annotations

import re
from enum import Enum
from typing import Tuple


class ActionSafetyLevel(str, Enum):
    SAFE = "SAFE"
    SENSITIVE = "SENSITIVE"
    BLOCKED = "BLOCKED"


# Highly dangerous / destructive patterns that should never be executed autonomously
SENSITIVE_PATTERNS = [
    r"\b(rmdir|rd)\b",
    r"\b(remove-item|del|erase|rm)\b",
    r"\b(format|diskpart)\b",
    r"\b(reg\s+delete|reg\s+add|regedit)\b",
    r"\b(taskkill|stop-process)\b",
    r"\b(takeown|icacls)\b",
    r"\b(net\s+user|net\s+localgroup)\b",
    r"c:\\windows",
    r"c:\\program files",
    r"c:\\program files \(x86\)",
    r"\bshutdown\b",
    r"\brestart-computer\b",
    r">\s*c:",
    r">\s*\\\\\.",
]

# Read-only and diagnostic commands that are safe to run
SAFE_PATTERNS = [
    r"\b(dir|ls|get-childitem)\b",
    r"\b(type|cat|get-content)\b",
    r"\b(systeminfo|nvidia-smi|wmic)\b",
    r"\b(tasklist|get-process)\b",
    r"\b(ipconfig|ping)\b",
    r"\b(git\s+status|git\s+log|git\s+diff)\b",
    r"\b(python\s+--version|uv\s+--version)\b",
]


class SafetyGuard:
    """Evaluates OS actions and commands against system safety rules."""

    @staticmethod
    def evaluate_command(command: str) -> Tuple[ActionSafetyLevel, str]:
        """Analyzes command and returns safety level and reason."""
        cmd_lower = command.strip().lower()

        # Check for sensitive / destructive operations
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, cmd_lower):
                return (
                    ActionSafetyLevel.SENSITIVE,
                    f"Command contains sensitive or destructive pattern: '{pattern}'. "
                    "Only the user is permitted to execute this operation.",
                )

        # Check for explicitly safe patterns
        for pattern in SAFE_PATTERNS:
            if re.search(pattern, cmd_lower):
                return (ActionSafetyLevel.SAFE, "Read-only / diagnostic action.")

        # Default fallback for arbitrary write or unknown commands
        if any(kw in cmd_lower for kw in ["set-", "new-item", "copy-item", "move-item", "out-file", ">"]):
            return (
                ActionSafetyLevel.SENSITIVE,
                "Command modifies files or system configuration. User review required.",
            )

        return (ActionSafetyLevel.SAFE, "Non-destructive standard execution.")
