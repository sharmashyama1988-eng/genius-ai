"""
Genius Agent Tools — Claude Code-level file & shell operations.

Available Tools (called by AgentLoop from LLM-generated XML tool calls):
  read_file        — Read full contents of a file
  create_file      — Create (or overwrite) a file with given content
  edit_file        — Apply targeted string replacement inside a file (diff-aware)
  append_to_file   — Append lines to end of file
  delete_file      — Delete a file (with guard confirmation)
  list_files       — Recursive directory tree with sizes
  run_command      — Execute a shell command, capture stdout/stderr
  search_files     — grep-style text search across workspace
  project_summary  — High-level project analysis (langs, entry points, structure)
  read_url         — Fetch plain text from a URL (for docs / APIs)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── Tool Result ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool: str
    success: bool
    output: str
    error: str = ""
    elapsed_ms: float = 0.0

    def to_context_block(self) -> str:
        """Returns a string to inject back into LLM context."""
        status = "OK" if self.success else "ERROR"
        lines = [f"[TOOL:{self.tool} STATUS:{status} ({self.elapsed_ms:.0f}ms)]"]
        if self.output:
            lines.append(self.output[:6000])  # cap context injection
        if self.error:
            lines.append(f"[STDERR]: {self.error[:2000]}")
        lines.append(f"[/TOOL:{self.tool}]")
        return "\n".join(lines)


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file. Use relative or absolute path.",
        "params": {"path": "string — file path to read"},
        "example": "<tool_call>\n<name>read_file</name>\n<path>main.py</path>\n</tool_call>",
    },
    {
        "name": "create_file",
        "description": "Create a new file (or overwrite if exists) with the given content. Creates parent directories automatically.",
        "params": {"path": "string — file path", "content": "string — full file content"},
        "example": "<tool_call>\n<name>create_file</name>\n<path>hello.py</path>\n<content>print('Hello World')</content>\n</tool_call>",
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string inside an existing file. old_str must match exactly (including whitespace).",
        "params": {"path": "string", "old_str": "string — exact text to find", "new_str": "string — replacement text"},
        "example": "<tool_call>\n<name>edit_file</name>\n<path>main.py</path>\n<old_str>x = 1</old_str>\n<new_str>x = 42</new_str>\n</tool_call>",
    },
    {
        "name": "append_to_file",
        "description": "Append content to the end of an existing file.",
        "params": {"path": "string", "content": "string — text to append"},
        "example": "<tool_call>\n<name>append_to_file</name>\n<path>log.txt</path>\n<content>New log entry</content>\n</tool_call>",
    },
    {
        "name": "delete_file",
        "description": "Delete a file. Use only when explicitly needed.",
        "params": {"path": "string — file path to delete"},
        "example": "<tool_call>\n<name>delete_file</name>\n<path>old_file.py</path>\n</tool_call>",
    },
    {
        "name": "list_files",
        "description": "List files in a directory (recursive tree). Defaults to current workspace root.",
        "params": {"path": "string (optional) — directory to list"},
        "example": "<tool_call>\n<name>list_files</name>\n<path>.</path>\n</tool_call>",
    },
    {
        "name": "run_command",
        "description": "Execute a shell command and return stdout + stderr. Use for: running Python scripts, installing packages, checking output, testing code.",
        "params": {"command": "string — shell command to run", "cwd": "string (optional) — working directory"},
        "example": "<tool_call>\n<name>run_command</name>\n<command>python snake.py</command>\n</tool_call>",
    },
    {
        "name": "search_files",
        "description": "Search for a text pattern (regex or plain string) across all files in the workspace. Returns matching file paths and line numbers.",
        "params": {"pattern": "string — search term or regex", "path": "string (optional) — directory to search in"},
        "example": "<tool_call>\n<name>search_files</name>\n<pattern>def main</pattern>\n</tool_call>",
    },
    {
        "name": "project_summary",
        "description": "Analyze the project structure: file tree, languages, entry points, dependencies. Always call this first when starting a new task.",
        "params": {"path": "string (optional) — project root"},
        "example": "<tool_call>\n<name>project_summary</name>\n</tool_call>",
    },
    {
        "name": "task_complete",
        "description": "Signal that the task is fully complete. Provide a summary of what was done.",
        "params": {"summary": "string — what was accomplished"},
        "example": "<tool_call>\n<name>task_complete</name>\n<summary>Created snake_game.py with full game logic. Run with: python snake_game.py</summary>\n</tool_call>",
    },
]


def get_tool_system_prompt() -> str:
    """Returns the system prompt section describing all available tools in XML format."""
    lines = [
        "═══════════════════════════════════════════════════════",
        "  GENIUS AGENTIC MODE — AVAILABLE TOOLS",
        "═══════════════════════════════════════════════════════",
        "You are operating as an autonomous coding agent. Use tools to complete tasks.",
        "IMPORTANT RULES:",
        "  1. Always call project_summary FIRST to understand the workspace.",
        "  2. Use read_file before editing any existing file.",
        "  3. After creating/editing code files, use run_command to verify they work.",
        "  4. If a command returns an error, fix it and run again.",
        "  5. When the task is fully done, call task_complete.",
        "  6. Output ONLY tool_call XML blocks. No markdown, no prose between calls.",
        "",
        "TOOL CALL FORMAT:",
        "  <tool_call>",
        "  <name>TOOL_NAME</name>",
        "  <PARAM_NAME>VALUE</PARAM_NAME>",
        "  </tool_call>",
        "",
        "AVAILABLE TOOLS:",
    ]
    for t in TOOL_SCHEMAS:
        lines.append(f"  [{t['name']}] {t['description']}")
        for pname, pdesc in t["params"].items():
            lines.append(f"    • {pname}: {pdesc}")
    lines.append("═══════════════════════════════════════════════════════")
    return "\n".join(lines)


# ─── Tool Executor ────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes parsed tool calls against the real filesystem and shell."""

    BLOCKED_COMMANDS = {
        "rm -rf /", "rmdir /s /q c:\\", "format c:", "del /f /s /q c:\\",
        "shutdown", "mkfs", "dd if=/dev/zero",
    }

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs).expanduser()
        if not p.is_absolute():
            return (self.workspace_root / p).resolve()
        return p.resolve()

    def _time_it(self, fn, *args, **kwargs) -> Tuple[Any, float]:
        t = time.monotonic()
        result = fn(*args, **kwargs)
        return result, round((time.monotonic() - t) * 1000, 1)

    # ── Individual Tools ─────────────────────────────────────────────────

    def read_file(self, path: str) -> ToolResult:
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("read_file", False, "", f"File not found: {p}", 0)
            if p.stat().st_size > 500_000:
                return ToolResult("read_file", False, "", f"File too large (>{500_000} bytes). Use search_files or list_files.", 0)
            content = p.read_text(encoding="utf-8", errors="replace")
            # Add line numbers for easier editing reference
            numbered = "\n".join(f"{i+1:4d}│ {ln}" for i, ln in enumerate(content.splitlines()))
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("read_file", True, f"FILE: {p}\nLINES: {len(content.splitlines())}\n\n{numbered}", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("read_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def create_file(self, path: str, content: str) -> ToolResult:
        t = time.monotonic()
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            lines = len(content.splitlines())
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("create_file", True, f"Created: {p}\nLines: {lines} | Bytes: {len(content.encode())}", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("create_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("edit_file", False, "", f"File not found: {p}", 0)
            original = p.read_text(encoding="utf-8", errors="replace")
            if old_str not in original:
                # Try with normalized line endings
                normalized = original.replace("\r\n", "\n")
                old_normalized = old_str.replace("\r\n", "\n")
                if old_normalized not in normalized:
                    # Show nearby context to help debug
                    preview_lines = original.splitlines()[:30]
                    preview = "\n".join(preview_lines)
                    return ToolResult("edit_file", False, "",
                                      f"old_str not found in {p}.\nFile preview (first 30 lines):\n{preview}", 0)
                updated = normalized.replace(old_normalized, new_str, 1)
            else:
                updated = original.replace(old_str, new_str, 1)
            p.write_text(updated, encoding="utf-8")
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("edit_file", True, f"Edited: {p}\nReplaced {len(old_str)} chars → {len(new_str)} chars", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("edit_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def append_to_file(self, path: str, content: str) -> ToolResult:
        t = time.monotonic()
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                if not content.startswith("\n"):
                    f.write("\n")
                f.write(content)
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("append_to_file", True, f"Appended {len(content)} chars to {p}", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("append_to_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def delete_file(self, path: str) -> ToolResult:
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("delete_file", False, "", f"Not found: {p}", 0)
            p.unlink()
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("delete_file", True, f"Deleted: {p}", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("delete_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def list_files(self, path: str = ".") -> ToolResult:
        t = time.monotonic()
        try:
            root = self._resolve(path)
            if not root.exists():
                return ToolResult("list_files", False, "", f"Path not found: {root}", 0)

            SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", "dist", "build", ".pytest_cache"}
            lines = [f"WORKSPACE: {root}"]
            total_files = 0

            def _walk(p: Path, prefix: str = "", depth: int = 0) -> None:
                nonlocal total_files
                if depth > 6:
                    return
                try:
                    children = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                except PermissionError:
                    return
                for i, child in enumerate(children):
                    if child.name in SKIP or child.name.startswith("."):
                        continue
                    connector = "└── " if i == len(children) - 1 else "├── "
                    if child.is_dir():
                        lines.append(f"{prefix}{connector}{child.name}/")
                        ext = "    " if i == len(children) - 1 else "│   "
                        _walk(child, prefix + ext, depth + 1)
                    else:
                        size = child.stat().st_size
                        size_str = f"{size}B" if size < 1024 else f"{size//1024}KB"
                        lines.append(f"{prefix}{connector}{child.name}  [{size_str}]")
                        total_files += 1

            _walk(root)
            lines.append(f"\nTotal files: {total_files}")
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("list_files", True, "\n".join(lines), elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("list_files", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def run_command(self, command: str, cwd: Optional[str] = None) -> ToolResult:
        t = time.monotonic()
        # Safety block
        cmd_lower = command.lower().strip()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return ToolResult("run_command", False, "", f"BLOCKED: dangerous command '{command}'", 0)

        try:
            work_dir = self._resolve(cwd) if cwd else self.workspace_root
            # Use PowerShell on Windows, bash elsewhere
            if sys.platform == "win32":
                shell_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
            else:
                shell_cmd = ["bash", "-c", command]

            proc = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(work_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            elapsed = round((time.monotonic() - t) * 1000, 1)
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            success = proc.returncode == 0
            output = f"EXIT CODE: {proc.returncode}\n"
            if out:
                output += f"STDOUT:\n{out[:4000]}"
            if err and not success:
                output += f"\nSTDERR:\n{err[:2000]}"
            return ToolResult("run_command", success, output, err if not success else "", elapsed_ms=elapsed)
        except subprocess.TimeoutExpired:
            return ToolResult("run_command", False, "", "Command timed out after 60s", round((time.monotonic() - t) * 1000, 1))
        except Exception as e:
            return ToolResult("run_command", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def search_files(self, pattern: str, path: str = ".") -> ToolResult:
        t = time.monotonic()
        try:
            root = self._resolve(path)
            results: List[str] = []
            SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules"}
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                regex = re.compile(re.escape(pattern), re.IGNORECASE)

            for fpath in root.rglob("*"):
                if any(s in fpath.parts for s in SKIP):
                    continue
                if fpath.is_file() and fpath.suffix in {".py", ".js", ".ts", ".txt", ".md", ".json", ".yaml", ".toml", ".html", ".css", ".sh", ".bat"}:
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="ignore")
                        for i, line in enumerate(text.splitlines(), 1):
                            if regex.search(line):
                                rel = fpath.relative_to(root)
                                results.append(f"{rel}:{i}:  {line.strip()[:120]}")
                                if len(results) >= 50:
                                    break
                    except Exception:
                        continue
                if len(results) >= 50:
                    break

            elapsed = round((time.monotonic() - t) * 1000, 1)
            if not results:
                return ToolResult("search_files", True, f"No matches for '{pattern}'", elapsed_ms=elapsed)
            return ToolResult("search_files", True, f"Found {len(results)} match(es) for '{pattern}':\n" + "\n".join(results), elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("search_files", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def project_summary(self, path: str = ".") -> ToolResult:
        t = time.monotonic()
        try:
            root = self._resolve(path)
            SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules"}
            ext_counts: Dict[str, int] = {}
            entry_points: List[str] = []
            sig_files: List[str] = []

            SIG_MAP = {
                "package.json": "Node.js/JavaScript",
                "tsconfig.json": "TypeScript",
                "pyproject.toml": "Python (Poetry)",
                "requirements.txt": "Python (pip)",
                "setup.py": "Python (setuptools)",
                "Cargo.toml": "Rust",
                "go.mod": "Go",
                "Makefile": "Make build",
                "docker-compose.yml": "Docker Compose",
                "Dockerfile": "Docker",
            }
            ENTRY_NAMES = {"main.py", "app.py", "run.py", "index.py", "server.py", "start.py", "manage.py", "index.js", "app.js", "main.js", "main.go", "main.rs"}

            total_files = 0
            for fpath in root.rglob("*"):
                if any(s in fpath.parts for s in SKIP):
                    continue
                if fpath.is_file():
                    total_files += 1
                    ext = fpath.suffix.lower()
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    if fpath.name in SIG_MAP:
                        sig_files.append(f"  {fpath.name} → {SIG_MAP[fpath.name]}")
                    if fpath.name in ENTRY_NAMES:
                        entry_points.append(str(fpath.relative_to(root)))

            top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:8]
            lang_map = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".rs": "Rust", ".go": "Go", ".java": "Java", ".cpp": "C++", ".c": "C"}
            langs = [lang_map[ext] for ext, _ in top_exts if ext in lang_map]

            summary_lines = [
                f"PROJECT ROOT: {root}",
                f"Total Files: {total_files}",
                f"Languages: {', '.join(langs) or 'Unknown'}",
                f"File Types: {', '.join(f'{ext}({n})' for ext, n in top_exts)}",
            ]
            if entry_points:
                summary_lines.append(f"Entry Points: {', '.join(entry_points[:5])}")
            if sig_files:
                summary_lines.append("Project Signatures:")
                summary_lines.extend(sig_files)

            # Show top-level tree (non-recursive)
            try:
                top = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name))
                tree = [f"  {'[D]' if p.is_dir() else '[F]'} {p.name}" for p in top[:30] if p.name not in SKIP and not p.name.startswith(".")]
                summary_lines.append("\nTop-Level Structure:")
                summary_lines.extend(tree)
            except Exception:
                pass

            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("project_summary", True, "\n".join(summary_lines), elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("project_summary", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def dispatch(self, tool_name: str, params: Dict[str, str]) -> ToolResult:
        """Dispatches a parsed tool call to the correct method."""
        dispatch_map = {
            "read_file": lambda p: self.read_file(p.get("path", "")),
            "create_file": lambda p: self.create_file(p.get("path", ""), p.get("content", "")),
            "edit_file": lambda p: self.edit_file(p.get("path", ""), p.get("old_str", ""), p.get("new_str", "")),
            "append_to_file": lambda p: self.append_to_file(p.get("path", ""), p.get("content", "")),
            "delete_file": lambda p: self.delete_file(p.get("path", "")),
            "list_files": lambda p: self.list_files(p.get("path", ".")),
            "run_command": lambda p: self.run_command(p.get("command", ""), p.get("cwd")),
            "search_files": lambda p: self.search_files(p.get("pattern", ""), p.get("path", ".")),
            "project_summary": lambda p: self.project_summary(p.get("path", ".")),
        }
        handler = dispatch_map.get(tool_name)
        if handler is None:
            return ToolResult(tool_name, False, "", f"Unknown tool: '{tool_name}'")
        return handler(params)
