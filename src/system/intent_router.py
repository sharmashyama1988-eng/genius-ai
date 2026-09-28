"""
Genius Autonomous Intent Router & Dispatcher.

Detects user intent from natural language input (English, Hindi, Hinglish)
and automatically selects and executes the appropriate command/subsystem:
  - Agentic Mode (_run_agent) for building apps, games, scripts, fixing bugs, refactoring, testing
  - Git Operations (_handle_git_command) for commit, status, push, pull, branch, PR
  - Protocol Runner (protocol.txt) for running tests, builds, linters, dev servers
  - Codebase Indexer for finding symbols, functions, classes, imports
  - Math Solver for mathematical expressions
  - Perplexity Search for live web research
  - Workspace Manager for file creation, reading, image inspection, directory shifting
  - Conversational Deep Reasoning (_process_turn) for explanations, learning, concepts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IntentType(str, Enum):
    AGENT = "agent"            # Autonomous coding/execution loop
    GIT = "git"                # Git automation (commit, push, status, branch)
    PROTOCOL = "protocol"      # protocol.txt registered command (test, build, lint)
    INDEX = "index"            # Codebase symbol map & search
    MATH = "math"              # Mathematical calculation / AST solver
    SEARCH = "search"          # Perplexity-style live search
    WORKSPACE = "workspace"    # File/directory workspace operations
    CHAT = "chat"              # General deep reasoning / conversation


@dataclass
class RoutedIntent:
    intent_type: IntentType
    command: str               # Equivalent slash command (e.g. "/agent", "/git status")
    argument: str              # Extracted task or payload
    confidence: float          # 0.0 to 1.0
    reason: str                # Why this route was chosen
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentRouter:
    """
    Intelligent Zero-Prefix Intent Router for Genius AI.
    Understands English, Hindi, and Hinglish prompts.
    """

    def __init__(self, registered_protocol_cmds: Optional[List[str]] = None) -> None:
        self.protocol_cmds = [c.lower() for c in (registered_protocol_cmds or [])]

    def update_protocol_cmds(self, cmds: List[str]) -> None:
        self.protocol_cmds = [c.lower() for c in cmds]

    def route(self, query: str) -> RoutedIntent:
        """
        Analyzes the user's input and returns the classified RoutedIntent.
        """
        raw = query.strip()
        q = raw.lower()

        # 0. If user explicitly provided a slash command, keep it verbatim
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1].strip() if len(parts) > 1 else ""
            return RoutedIntent(
                intent_type=self._slash_to_intent(cmd),
                command=cmd,
                argument=arg,
                confidence=1.0,
                reason="Explicit slash command provided by user.",
            )

        # 1. Check for Git Intent
        git_intent = self._check_git_intent(raw, q)
        if git_intent:
            return git_intent

        # 2. Check for Protocol Runner Intent (e.g., "run test", "tests chalao", "build project")
        proto_intent = self._check_protocol_intent(raw, q)
        if proto_intent:
            return proto_intent

        # 3. Check for Codebase Indexer / Symbol Search
        index_intent = self._check_index_intent(raw, q)
        if index_intent:
            return index_intent

        # 4. Check for Math Calculation
        math_intent = self._check_math_intent(raw, q)
        if math_intent:
            return math_intent

        # 5. Check for Perplexity Live Search Intent
        search_intent = self._check_search_intent(raw, q)
        if search_intent:
            return search_intent

        # 6. Check for Workspace / File Ops (Shift workspace, view image, etc.)
        ws_intent = self._check_workspace_intent(raw, q)
        if ws_intent:
            return ws_intent

        # 7. Check for Agentic Task Intent (Autonomous Coding / App / Game / Bug Fix / Refactor)
        agent_intent = self._check_agent_intent(raw, q)
        if agent_intent:
            return agent_intent

        # 8. Default: Deep Reasoning Conversational Turn
        return RoutedIntent(
            intent_type=IntentType.CHAT,
            command="",
            argument=raw,
            confidence=0.9,
            reason="Conversational / Conceptual Deep Reasoning query.",
        )

    # ── Specialized Intent Checkers ───────────────────────────────────────────

    def _slash_to_intent(self, cmd: str) -> IntentType:
        cmd_clean = cmd.lower()
        if cmd_clean == "/agent":
            return IntentType.AGENT
        if cmd_clean == "/git":
            return IntentType.GIT
        if cmd_clean == "/run":
            return IntentType.PROTOCOL
        if cmd_clean == "/index":
            return IntentType.INDEX
        if cmd_clean in ("/calc", "/math"):
            return IntentType.MATH
        if cmd_clean == "/search":
            return IntentType.SEARCH
        if cmd_clean in ("/project", "/files", "/create", "/read", "/view"):
            return IntentType.WORKSPACE
        return IntentType.CHAT

    def _check_git_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """Detects git operations in English and Hinglish."""
        # Git Status
        if re.search(r"\b(?:git\s+status|check\s+git\s+status|git\s+check\s+karo|git\s+status\s+(?:kya\s+hai|dekho|check\s+karo))\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git status",
                argument="status",
                confidence=0.98,
                reason="Detected Git status request.",
            )

        # Git Commit
        commit_match = re.search(
            r"(?:git\s+commit(?:\s+-m)?|commit\s+(?:all\s+)?changes?|changes\s+commit\s+kar\s+do|commit\s+kar\s+do)(?:\s*[:\-]\s*|\s+with\s+message\s+|\s+)(.+)?",
            raw,
            re.IGNORECASE,
        )
        if commit_match:
            msg = commit_match.group(1) or "update"
            msg = msg.strip(" \"'")
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command=f"/git commit {msg}",
                argument=f"commit {msg}",
                confidence=0.96,
                reason="Detected Git commit request.",
            )

        # Git Push
        if re.search(r"\b(?:git\s+push|push\s+to\s+(?:github|remote|origin)|github\s+pe\s+push\s+(?:kar\s+do|karo)|push\s+kar\s+do)\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git push",
                argument="push",
                confidence=0.98,
                reason="Detected Git push request.",
            )

        # Git Pull
        if re.search(r"\b(?:git\s+pull|pull\s+from\s+(?:origin|remote|github)|github\s+se\s+pull\s+(?:kar\s+do|karo)|pull\s+kar\s+do)\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git pull",
                argument="pull",
                confidence=0.98,
                reason="Detected Git pull request.",
            )

        # Git Diff
        if re.search(r"\b(?:git\s+diff|show\s+diff|changes\s+dikhao|diff\s+dekho)\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git diff",
                argument="diff",
                confidence=0.95,
                reason="Detected Git diff request.",
            )

        # Git Branches / Branch create
        branch_create = re.search(r"(?:create\s+branch|nayi\s+branch\s+banao|git\s+branch)\s+([a-zA-Z0-9_\-\/]+)", raw, re.IGNORECASE)
        if branch_create:
            bname = branch_create.group(1).strip()
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command=f"/git branch {bname}",
                argument=f"branch {bname}",
                confidence=0.95,
                reason=f"Detected Git branch creation '{bname}'.",
            )
        if re.search(r"\b(?:list\s+branches|show\s+branches|git\s+branches|saari\s+branches\s+dikhao)\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git branch",
                argument="branch",
                confidence=0.95,
                reason="Detected Git branch list request.",
            )

        # Git PR / Conflicts
        if re.search(r"\b(?:merge\s+conflicts?|conflicts\s+check\s+karo|conflicts\s+dekho)\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git conflicts",
                argument="conflicts",
                confidence=0.95,
                reason="Detected Git conflicts check request.",
            )
        if re.search(r"\b(?:generate\s+pr\s+summary|pr\s+summary\s+banao|pull\s+request\s+summary)\b", q):
            return RoutedIntent(
                intent_type=IntentType.GIT,
                command="/git pr",
                argument="pr",
                confidence=0.95,
                reason="Detected Git PR summary request.",
            )

        return None

    def _check_protocol_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """Detects requests to run registered protocol commands like tests, builds, linting."""
        # 1. Tests (pytest, test runner)
        if re.search(r"\b(?:run\s+(?:all\s+)?tests?|tests?\s+run\s+karo|tests?\s+chalao|test\s+execute\s+karo|chalao\s+test|run\s+pytest|pytest\s+chalao)\b", q):
            return RoutedIntent(
                intent_type=IntentType.PROTOCOL,
                command="/run test",
                argument="test",
                confidence=0.97,
                reason="Detected test execution protocol command.",
            )

        # 2. Build
        if re.search(r"\b(?:build\s+project|run\s+build|project\s+build\s+karo|build\s+karo)\b", q):
            return RoutedIntent(
                intent_type=IntentType.PROTOCOL,
                command="/run build",
                argument="build",
                confidence=0.95,
                reason="Detected build protocol command.",
            )

        # 3. Lint / Format
        if re.search(r"\b(?:run\s+lint(?:er)?|code\s+lint\s+karo|lint\s+chalao|check\s+code\s+quality)\b", q):
            return RoutedIntent(
                intent_type=IntentType.PROTOCOL,
                command="/run lint",
                argument="lint",
                confidence=0.95,
                reason="Detected lint protocol command.",
            )
        if re.search(r"\b(?:format\s+code|code\s+format\s+karo|auto\s+format|run\s+black)\b", q):
            return RoutedIntent(
                intent_type=IntentType.PROTOCOL,
                command="/run format",
                argument="format",
                confidence=0.95,
                reason="Detected code format protocol command.",
            )

        # 4. Check dynamic registered commands in protocol.txt (e.g. dev, install, etc.)
        for cmd_name in self.protocol_cmds:
            pattern = rf"\b(?:run|chalao|execute)\s+{re.escape(cmd_name)}\b"
            if re.search(pattern, q):
                return RoutedIntent(
                    intent_type=IntentType.PROTOCOL,
                    command=f"/run {cmd_name}",
                    argument=cmd_name,
                    confidence=0.94,
                    reason=f"Matched registered protocol command '{cmd_name}'.",
                )

        return None

    def _check_index_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """Detects codebase indexing and symbol lookup requests."""
        # Entire codebase indexing
        if re.search(r"\b(?:index\s+(?:the\s+)?codebase|codebase\s+scan\s+karo|scan\s+codebase|symbol\s+map\s+banao|index\s+project)\b", q):
            return RoutedIntent(
                intent_type=IntentType.INDEX,
                command="/index",
                argument="",
                confidence=0.96,
                reason="Detected codebase indexing request.",
            )

        # Symbol lookup: "where is function login", "kahan define hai UserManager", "find symbol parse_query"
        sym_match = re.search(
            r"(?:where\s+is\s+(?:symbol|function|class|method|variable)?|find\s+symbol|search\s+symbol|kahan\s+define\s+hai\s+(?:function|class|method|symbol)?)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            raw,
            re.IGNORECASE,
        )
        if sym_match:
            sym = sym_match.group(1).strip()
            # Avoid common non-symbol words
            if sym.lower() not in ("the", "this", "my", "python", "code", "file", "project"):
                return RoutedIntent(
                    intent_type=IntentType.INDEX,
                    command=f"/index {sym}",
                    argument=sym,
                    confidence=0.93,
                    reason=f"Detected symbol lookup for '{sym}'.",
                )

        return None

    def _check_math_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """Detects pure mathematical calculation queries."""
        # Explicit math prompt: "calculate 15 * 32", "solve 2x + 4 = 10", "math: (40 + 5) / 2"
        math_match = re.search(
            r"^(?:calculate|compute|math|solve|hisab\s+karo|evaluate)\s*[:\s]+([0-9xXyYzZ\+\-\*\/\^\(\)\.\,\s\=\<\>]+)$",
            raw,
            re.IGNORECASE,
        )
        if math_match:
            expr = math_match.group(1).strip()
            return RoutedIntent(
                intent_type=IntentType.MATH,
                command=f"/calc {expr}",
                argument=expr,
                confidence=0.95,
                reason=f"Detected mathematical calculation for '{expr}'.",
            )

        # Direct expression: e.g. "45 * 12 + 100" or "(5 + 3) / 2"
        if re.match(r"^[\(\s]*\d+[\s\+\-\*\/\^\(\)\.\d\=]+[\)\s]*$", raw) and any(op in raw for op in ["+", "*", "/", "^"]):
            return RoutedIntent(
                intent_type=IntentType.MATH,
                command=f"/calc {raw}",
                argument=raw,
                confidence=0.95,
                reason="Detected standalone arithmetic expression.",
            )

        return None

    def _check_search_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """Detects explicit Perplexity-style live search requests."""
        search_match = re.search(
            r"^(?:search\s+web\s+for|search\s+for|live\s+search|web\s+search|search\s+karo)\s+[:\s]*(.+)$",
            raw,
            re.IGNORECASE,
        )
        if search_match:
            target = search_match.group(1).strip()
            return RoutedIntent(
                intent_type=IntentType.SEARCH,
                command=f"/search {target}",
                argument=target,
                confidence=0.95,
                reason=f"Detected explicit web search for '{target}'.",
            )
        return None

    def _check_workspace_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """Detects workspace shifts, file viewing, image inspections."""
        # Shift workspace directory
        shift_match = re.search(
            r"(?:shift|switch|change|set|go\s+to|open)\s+(?:to\s+)?(?:workspace|project|directory|dir|folder|path)\s+(?:to\s+)?([a-zA-Z]:[\\\/][^\s\"']+|[\\\/][^\s\"']+|\.{1,2}[\\\/][^\s\"']+|[a-zA-Z0-9_\-\.\/\\]+)",
            raw,
            re.IGNORECASE,
        )
        if not shift_match:
            shift_match = re.search(
                r"(?:project|workspace|directory|dir|folder|path)\s+([a-zA-Z]:[\\\/][^\s\"']+|[\\\/][^\s\"']+|\.{1,2}[\\\/][^\s\"']+|[a-zA-Z0-9_\-\.\/\\]+)\s*(?:pe\s+)?(?:shift|switch|chalo|set)",
                raw,
                re.IGNORECASE,
            )
        if shift_match:
            target_dir = shift_match.group(1).strip(" \"'")
            return RoutedIntent(
                intent_type=IntentType.WORKSPACE,
                command=f"/project {target_dir}",
                argument=target_dir,
                confidence=0.95,
                reason=f"Detected workspace folder shift to '{target_dir}'.",
                metadata={"action": "shift", "path": target_dir},
            )

        # Image view
        img_match = re.search(
            r"(?:show|view|open|display|inspect)\s+(?:the\s+)?image\s+([a-zA-Z0-9_\-\.\/\\]+\.(?:png|jpg|jpeg|webp|gif|bmp|svg|ico))",
            raw,
            re.IGNORECASE,
        )
        if not img_match:
            img_match = re.search(
                r"(?:image|photo|picture)\s+([a-zA-Z0-9_\-\.\/\\]+\.(?:png|jpg|jpeg|webp|gif|bmp|svg|ico))\s+(?:dikhao|open|view|show|dekhna)",
                raw,
                re.IGNORECASE,
            )
        if img_match:
            img_path = img_match.group(1).strip(" \"'")
            return RoutedIntent(
                intent_type=IntentType.WORKSPACE,
                command=f"/view {img_path}",
                argument=img_path,
                confidence=0.96,
                reason=f"Detected image view request for '{img_path}'.",
                metadata={"action": "view_image", "path": img_path},
            )

        # Read file
        read_match = re.search(
            r"(?:read|open|show|cat|display)\s+(?:the\s+)?file\s+([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9]+)",
            raw,
            re.IGNORECASE,
        )
        if not read_match:
            read_match = re.search(
                r"file\s+([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9]+)\s+(?:read|open|padho|dikhao|show)",
                raw,
                re.IGNORECASE,
            )
        if read_match:
            fname = read_match.group(1).strip(" \"'")
            return RoutedIntent(
                intent_type=IntentType.WORKSPACE,
                command=f"/read {fname}",
                argument=fname,
                confidence=0.95,
                reason=f"Detected file read request for '{fname}'.",
                metadata={"action": "read_file", "path": fname},
            )

        # List files
        if any(p in q for p in ["list files", "show files", "project files", "directory files", "files in project", "files dikhao", "list all files", "ls", "dir"]):
            if len(q.split()) <= 4:
                return RoutedIntent(
                    intent_type=IntentType.WORKSPACE,
                    command="/files",
                    argument="",
                    confidence=0.94,
                    reason="Detected file list request.",
                    metadata={"action": "list_files"},
                )

        return None

    def _check_agent_intent(self, raw: str, q: str) -> Optional[RoutedIntent]:
        """
        Detects autonomous software engineering tasks (Agentic Mode):
        Creating apps, games, scripts, modules, adding features, fixing errors,
        writing unit tests and executing them, refactoring codebases.
        """
        # Exclusion 1: If user explicitly asks for an explanation or definition
        # e.g., "what is snake game", "explain how snake game works", "python mein game kaise banta hai"
        explanation_markers = [
            r"\b(?:what\s+is|what\s+are|why\s+is|why\s+does|how\s+does|explain|samjhao|batao|kya\s+hota\s+hai|kya\s+hai|kaise\s+kaam\s+karta\s+hai)\b",
            r"\b(?:difference\s+between|meaning\s+of|history\s+of|overview\s+of)\b",
        ]
        is_pure_explanation = any(re.search(pat, q) for pat in explanation_markers)

        # But if they say "explain and build" or "code likho aur explain karo", we still want Agent!
        has_direct_build_command = any(re.search(pat, q) for pat in [
            r"\b(?:banao|bana\s+do|likho|likh\s+do|create|build|implement|develop|generate|make)\b",
            r"\b(?:fix\s+karo|theek\s+karo|debug\s+karo|refactor\s+karo)\b",
        ])

        if is_pure_explanation and not has_direct_build_command:
            return None

        # Pattern A: Action Verb + Target Object
        # English: "create/build/make/implement/write a snake game in Python", "add a feature", "fix bug"
        # Hinglish: "ek snake game banao", "python mein calculator bana do", "file mein unit tests likh do"

        # Strong coding target indicators
        coding_targets = [
            r"game", r"app", r"application", r"script", r"tool", r"bot", r"crawler",
            r"scraper", r"api", r"server", r"backend", r"frontend", r"cli", r"program",
            r"unit\s+tests?", r"tests?", r"function", r"module", r"class", r"endpoint",
            r"component", r"website", r"database", r"feature",
        ]
        target_re = r"(?:" + "|".join(coding_targets) + r")"

        # Strong action verbs (English & Hinglish)
        # Hinglish: "banao", "bana do", "likho", "likh do", "karo", "kar do", "jodo", "implement karo"
        # English: "create", "build", "make", "write", "implement", "develop", "code", "generate"
        agent_patterns = [
            # "snake game banao python mein" or "ek python se game banao"
            rf"(?:ek\s+)?.*?\b{target_re}\b.*?\b(?:banao|bana\s+do|likho|likh\s+do|bana\s+ke\s+do)\b",
            # "banao ek snake game" or "likho python script"
            rf"\b(?:banao|bana\s+do|likho|likh\s+do)\b.*?\b{target_re}\b",
            # "create a snake game" / "build an api" / "implement auth feature"
            rf"\b(?:create|build|make|implement|develop|code|write|generate)\b.*?\b{target_re}\b",
            # "fix the bug/error" or "error fix karo"
            r"\b(?:fix|solve|debug|resolve)\b.*?\b(?:bug|error|issue|exception|crash|traceback|failure)\b",
            r"\b(?:bug|error|issue|exception)\b.*?\b(?:fix|solve|debug|theek|sudharo)\s*(?:karo|kar\s+do)?\b",
            # "refactor code/module" or "code refactor karo"
            r"\b(?:refactor|optimize|clean\s+up)\b.*?\b(?:code|module|function|file|class|system)\b",
            r"\b(?:code|module)\b.*?\b(?:refactor|optimize)\s*(?:karo|kar\s+do)?\b",
            # "write unit tests for X and run" or "tests likho aur run karo"
            r"\b(?:write|add|generate)\b.*?\b(?:tests?|unit\s+tests?)\b",
            r"\b(?:tests?|unit\s+tests?)\b.*?\b(?:likho|likh\s+do|add\s+karo|banao)\b",
            # "python script banao" or "fastapi app banao"
            r"\b(?:python|fastapi|flask|django|react|html|css|javascript|node)\b.*?\b(?:banao|bana\s+do|likho|likh\s+do|build|create)\b",
            # "code likho ... aur file banao" / "run karke dikhao"
            r"\b(?:run\s+karke\s+dikhao|file\s+bana\s+ke|code\s+likh\s+ke|chala\s+ke\s+dekho)\b",
        ]

        for pat in agent_patterns:
            if re.search(pat, q, re.IGNORECASE):
                # Clean up task argument
                task = raw
                return RoutedIntent(
                    intent_type=IntentType.AGENT,
                    command=f"/agent {task}",
                    argument=task,
                    confidence=0.96,
                    reason=f"Matched agentic coding pattern: '{pat}'.",
                )

        return None
