"""
Genius Agent Tools — Claude Code-level file, shell & skill operations.

Available Tools (called by AgentLoop from LLM-generated XML tool calls):
  read_file           — Read file contents (with line numbers & optional line ranges)
  read_multiple_files — Read multiple files simultaneously
  create_file         — Create (or overwrite) a file with given content
  edit_file           — Apply targeted string replacement inside a file (diff-aware)
  append_to_file      — Append lines to end of file
  insert_at_line      — Insert content at a specific line number
  copy_file           — Copy a file to a new destination
  move_file           — Move or rename a file
  delete_file         — Delete a file (with guard confirmation)
  list_files          — Recursive directory tree with sizes
  run_command         — Execute a shell command, capture stdout/stderr
  search_files        — grep-style text search across workspace
  project_summary     — High-level project analysis (langs, entry points, structure)
  run_skill           — Run a pre-built or custom skill from agent/skills/
  synthesize_skill    — Self-Evolving AI: dynamically synthesize and deploy a new skill
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
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
            lines.append(self.output[:8000])  # generous context injection
        if self.error:
            lines.append(f"[STDERR]: {self.error[:2000]}")
        lines.append(f"[/TOOL:{self.tool}]")
        return "\n".join(lines)


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read file contents. Supports optional start_line and end_line for reading large files.",
        "params": {
            "path": "string — file path to read",
            "start_line": "integer (optional) — 1-based start line",
            "end_line": "integer (optional) — 1-based end line",
        },
        "example": "<tool_call>\n<name>read_file</name>\n<path>main.py</path>\n</tool_call>",
    },
    {
        "name": "read_multiple_files",
        "description": "Read multiple files at once. Paths separated by commas.",
        "params": {"paths": "string — comma-separated list of file paths"},
        "example": "<tool_call>\n<name>read_multiple_files</name>\n<paths>src/app.py, src/config.py</paths>\n</tool_call>",
    },
    {
        "name": "create_file",
        "description": "Create a new file (or overwrite if exists) with full content. Creates parent directories automatically.",
        "params": {"path": "string — file path", "content": "string — full file content"},
        "example": "<tool_call>\n<name>create_file</name>\n<path>hello.py</path>\n<content>print('Hello World')</content>\n</tool_call>",
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string inside an existing file. old_str must match exactly.",
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
        "name": "insert_at_line",
        "description": "Insert lines at a specific 1-based line number in an existing file.",
        "params": {"path": "string", "line": "integer — 1-based line number", "content": "string — text to insert"},
        "example": "<tool_call>\n<name>insert_at_line</name>\n<path>main.py</path>\n<line>10</line>\n<content>import os</content>\n</tool_call>",
    },
    {
        "name": "copy_file",
        "description": "Copy a file from source path to destination path.",
        "params": {"source": "string", "destination": "string"},
        "example": "<tool_call>\n<name>copy_file</name>\n<source>a.txt</source>\n<destination>b.txt</destination>\n</tool_call>",
    },
    {
        "name": "move_file",
        "description": "Move or rename a file.",
        "params": {"source": "string", "destination": "string"},
        "example": "<tool_call>\n<name>move_file</name>\n<source>old.py</source>\n<destination>new.py</destination>\n</tool_call>",
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
        "description": "Execute a shell command and return stdout + stderr. Use for: running scripts, testing code, installing packages.",
        "params": {"command": "string — shell command to run", "cwd": "string (optional) — working directory"},
        "example": "<tool_call>\n<name>run_command</name>\n<command>python snake.py</command>\n</tool_call>",
    },
    {
        "name": "search_files",
        "description": "Search for a text pattern (regex or plain string) across all workspace files. Returns file paths and matching lines.",
        "params": {"pattern": "string — search term or regex", "path": "string (optional) — directory to search in"},
        "example": "<tool_call>\n<name>search_files</name>\n<pattern>def main</pattern>\n</tool_call>",
    },
    {
        "name": "project_summary",
        "description": "Analyze project structure: file tree, languages, entry points, dependencies. Call this first when starting a new task.",
        "params": {"path": "string (optional) — project root"},
        "example": "<tool_call>\n<name>project_summary</name>\n</tool_call>",
    },
    {
        "name": "run_skill",
        "description": "Execute a registered user or built-in skill from agent/skills/ (e.g. pdf_generator, excel_generator, ppt_generator, imagen).",
        "params": {"skill": "string — name of skill", "args": "string — CLI arguments string"},
        "example": "<tool_call>\n<name>run_skill</name>\n<skill>excel_generator</skill>\n<args>-o data.xlsx -t 'Report'</args>\n</tool_call>",
    },
    {
        "name": "synthesize_skill",
        "description": "Self-Evolving AI: Autonomously design and deploy a new skill in agent/skills/. Saves, tests, and hot-reloads it.",
        "params": {
            "skill_name": "string — slug name for the skill (e.g. 'qr_maker')",
            "purpose": "string — what this skill does",
            "code": "string — complete Python code for the skill",
        },
        "example": "<tool_call>\n<name>synthesize_skill</name>\n<skill_name>my_tool</skill_name>\n<purpose>Process CSV data</purpose>\n<code>...code...</code>\n</tool_call>",
    },
    {
        "name": "task_complete",
        "description": "Signal that the task is fully complete. Provide a summary of what was done.",
        "params": {"summary": "string — what was accomplished"},
        "example": "<tool_call>\n<name>task_complete</name>\n<summary>Completed task successfully.</summary>\n</tool_call>",
    },
]


def get_tool_system_prompt() -> str:
    """Returns the system prompt section describing all available tools in XML format."""
    lines = [
        "═══════════════════════════════════════════════════════",
        "  GENIUS AGENTIC MODE — AVAILABLE TOOLS & POWERS",
        "═══════════════════════════════════════════════════════",
        "You are operating as an autonomous coding agent with full file read/write and execution powers.",
        "IMPORTANT RULES:",
        "  1. Call project_summary FIRST if you need to understand workspace architecture.",
        "  2. Use read_file or read_multiple_files before editing any existing files.",
        "  3. Use create_file to write complete files without size constraints.",
        "  4. Use run_command to execute and verify your code.",
        "  5. If an error occurs, analyze it, fix the file, and retry automatically.",
        "  6. Specialized skills (PDF, Excel, PPT, Imagen) are in agent/skills/ — call them via run_skill.",
        "  7. If user asks for a capability you don't have, use synthesize_skill to write and hot-reload it!",
        "  8. When task is fully done, call task_complete.",
        "  9. Output ONLY tool_call XML blocks.",
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
    """Executes parsed tool calls against the real filesystem, shell, and skill registry."""

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

    # ── File Read Operations ──────────────────────────────────────────────

    def read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> ToolResult:
        """Reads file content with line numbers and optional line range."""
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("read_file", False, "", f"File not found: {p}", 0)
            if p.is_dir():
                return ToolResult("read_file", False, "", f"Path is a directory, not a file: {p}", 0)

            # High file size limit (10MB)
            if p.stat().st_size > 10_000_000 and (start_line is None and end_line is None):
                return ToolResult("read_file", False, "", f"File too large (>{10_000_000} bytes). Specify start_line and end_line.", 0)

            content = p.read_text(encoding="utf-8", errors="replace")
            all_lines = content.splitlines()
            total_lines = len(all_lines)

            s_idx = max(0, (start_line - 1) if start_line is not None else 0)
            e_idx = min(total_lines, end_line if end_line is not None else total_lines)

            sliced = all_lines[s_idx:e_idx]
            numbered = "\n".join(f"{s_idx + i + 1:4d}│ {ln}" for i, ln in enumerate(sliced))

            elapsed = round((time.monotonic() - t) * 1000, 1)
            header = f"FILE: {p}\nTOTAL LINES: {total_lines} | SHOWING: {s_idx + 1} to {e_idx}\n\n"
            return ToolResult("read_file", True, f"{header}{numbered}", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("read_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def read_multiple_files(self, paths_str: str) -> ToolResult:
        """Reads multiple files in one batch call."""
        t = time.monotonic()
        try:
            raw_paths = [p.strip() for p in paths_str.split(",") if p.strip()]
            if not raw_paths:
                return ToolResult("read_multiple_files", False, "", "No paths specified.", 0)

            sections = []
            for path in raw_paths:
                res = self.read_file(path)
                if res.success:
                    sections.append(f"═══ {path} ═══\n{res.output}\n")
                else:
                    sections.append(f"═══ {path} (ERROR) ═══\n{res.error}\n")

            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("read_multiple_files", True, "\n".join(sections), elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("read_multiple_files", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    # ── File Write Operations ─────────────────────────────────────────────

    def create_file(self, path: str, content: str) -> ToolResult:
        """Creates or overwrites a file with full content."""
        t = time.monotonic()
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            lines = len(content.splitlines())
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("create_file", True, f"Successfully written to: {p}\nLines: {lines} | Size: {len(content.encode('utf-8'))} bytes", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("create_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """Targeted exact string replacement."""
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("edit_file", False, "", f"File not found: {p}", 0)
            original = p.read_text(encoding="utf-8", errors="replace")

            if old_str not in original:
                # Try with normalized line endings
                norm_orig = original.replace("\r\n", "\n")
                norm_old = old_str.replace("\r\n", "\n")
                if norm_old not in norm_orig:
                    preview = "\n".join(original.splitlines()[:25])
                    return ToolResult("edit_file", False, "", f"Target string not found in {p}.\nFile Preview:\n{preview}", 0)
                updated = norm_orig.replace(norm_old, new_str, 1)
            else:
                updated = original.replace(old_str, new_str, 1)

            p.write_text(updated, encoding="utf-8")
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("edit_file", True, f"Successfully edited {p}.", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("edit_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def append_to_file(self, path: str, content: str) -> ToolResult:
        """Appends text to the end of a file."""
        t = time.monotonic()
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            updated = existing + content
            p.write_text(updated, encoding="utf-8")
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("append_to_file", True, f"Appended {len(content.splitlines())} lines to {p}.", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("append_to_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def insert_at_line(self, path: str, line: int | str, content: str) -> ToolResult:
        """Inserts text at a specific 1-based line number."""
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("insert_at_line", False, "", f"File not found: {p}", 0)
            target_line = int(line)
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            insert_idx = max(0, min(target_line - 1, len(lines)))
            new_lines = content.splitlines()
            lines[insert_idx:insert_idx] = new_lines
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("insert_at_line", True, f"Inserted {len(new_lines)} lines at line {target_line} of {p}.", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("insert_at_line", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def copy_file(self, source: str, destination: str) -> ToolResult:
        """Copies a file."""
        t = time.monotonic()
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
            if not src.exists():
                return ToolResult("copy_file", False, "", f"Source not found: {src}", 0)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("copy_file", True, f"Copied {src} to {dst}.", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("copy_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def move_file(self, source: str, destination: str) -> ToolResult:
        """Moves or renames a file."""
        t = time.monotonic()
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
            if not src.exists():
                return ToolResult("move_file", False, "", f"Source not found: {src}", 0)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("move_file", True, f"Moved {src} to {dst}.", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("move_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def delete_file(self, path: str) -> ToolResult:
        """Deletes a file."""
        t = time.monotonic()
        try:
            p = self._resolve(path)
            if not p.exists():
                return ToolResult("delete_file", False, "", f"File not found: {p}", 0)
            p.unlink()
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("delete_file", True, f"Deleted: {p}", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("delete_file", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    # ── Directory & Shell Operations ──────────────────────────────────────

    def list_files(self, path: str = ".") -> ToolResult:
        """Returns recursive file tree."""
        t = time.monotonic()
        try:
            root = self._resolve(path)
            if not root.exists():
                return ToolResult("list_files", False, "", f"Path not found: {root}", 0)

            SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", "dist", "build"}
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
        """Executes a shell command."""
        t = time.monotonic()
        cmd_lower = command.lower().strip()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return ToolResult("run_command", False, "", f"BLOCKED: dangerous command '{command}'", 0)

        try:
            work_dir = self._resolve(cwd) if cwd else self.workspace_root
            if sys.platform == "win32":
                shell_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
            else:
                shell_cmd = ["bash", "-c", command]

            proc = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(work_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            elapsed = round((time.monotonic() - t) * 1000, 1)
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            success = proc.returncode == 0
            output = f"EXIT CODE: {proc.returncode}\n"
            if out:
                output += f"STDOUT:\n{out[:5000]}"
            if err and not success:
                output += f"\nSTDERR:\n{err[:2000]}"
            return ToolResult("run_command", success, output, err if not success else "", elapsed_ms=elapsed)
        except subprocess.TimeoutExpired:
            return ToolResult("run_command", False, "", "Command timed out after 120s", round((time.monotonic() - t) * 1000, 1))
        except Exception as e:
            return ToolResult("run_command", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def search_files(self, pattern: str, path: str = ".") -> ToolResult:
        """Grep search across all files."""
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
                        for idx, line in enumerate(text.splitlines(), start=1):
                            if regex.search(line):
                                rel = fpath.relative_to(root)
                                results.append(f"{rel}:{idx}: {line.strip()[:150]}")
                                if len(results) >= 50:
                                    break
                    except Exception:
                        pass
                if len(results) >= 50:
                    break

            elapsed = round((time.monotonic() - t) * 1000, 1)
            out = "\n".join(results) or f"No matches found for pattern '{pattern}'."
            return ToolResult("search_files", True, out, elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("search_files", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def project_summary(self, path: str = ".") -> ToolResult:
        """Summarizes project architecture."""
        t = time.monotonic()
        try:
            root = self._resolve(path)
            SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}
            SIG_MAP = {
                "package.json": "Node.js / JS",
                "tsconfig.json": "TypeScript",
                "pyproject.toml": "Python Modern",
                "requirements.txt": "Python Pip",
                "Cargo.toml": "Rust",
                "go.mod": "Go",
                "protocol.txt": "Genius Protocol Commands",
            }
            ext_counts: Dict[str, int] = {}
            sig_files: List[str] = []
            entry_points: List[str] = []
            ENTRY_NAMES = {"main.py", "app.py", "run.py", "index.py", "run_genius.py", "index.js", "app.js", "main.js"}

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
            lang_map = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".rs": "Rust", ".go": "Go"}
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

            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("project_summary", True, "\n".join(summary_lines), elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("project_summary", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    # ── Specialized Skill & Self-Evolution Tools ──────────────────────────

    def run_skill(self, skill: str, args: str = "") -> ToolResult:
        """Executes a discovered skill from agent/skills/."""
        t = time.monotonic()
        try:
            from agent.registry import get_skill_registry
            reg = get_skill_registry(self.workspace_root / "agent")
            skill_info = reg.get_skill(skill)
            if not skill_info:
                return ToolResult("run_skill", False, "", f"Skill '{skill}' not found in registry. Available: {', '.join(reg.skills.keys())}", 0)

            arg_list = args.split() if args else []
            ok, out, err = reg.execute_skill_cli(skill, arg_list)
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("run_skill", ok, out, err if not ok else "", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("run_skill", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    def synthesize_skill(self, skill_name: str, purpose: str, code: str) -> ToolResult:
        """Self-Evolving AI: writes, validates, tests, and deploys a new skill into agent/skills/."""
        t = time.monotonic()
        try:
            from agent.skill_synthesizer import get_skill_synthesizer
            syn = get_skill_synthesizer(self.workspace_root / "agent")
            ok, msg, skill_info = syn.synthesize_and_deploy(skill_name, purpose, code)
            elapsed = round((time.monotonic() - t) * 1000, 1)
            return ToolResult("synthesize_skill", ok, msg, msg if not ok else "", elapsed_ms=elapsed)
        except Exception as e:
            return ToolResult("synthesize_skill", False, "", str(e), round((time.monotonic() - t) * 1000, 1))

    # ── Tool Dispatcher ───────────────────────────────────────────────────

    def dispatch(self, tool_name: str, params: Dict[str, str]) -> ToolResult:
        """Dispatches a parsed tool call to the correct method."""
        dispatch_map = {
            "read_file": lambda p: self.read_file(
                p.get("path", ""),
                int(p["start_line"]) if "start_line" in p and p["start_line"].isdigit() else None,
                int(p["end_line"]) if "end_line" in p and p["end_line"].isdigit() else None,
            ),
            "read_multiple_files": lambda p: self.read_multiple_files(p.get("paths", "")),
            "create_file": lambda p: self.create_file(p.get("path", ""), p.get("content", "")),
            "edit_file": lambda p: self.edit_file(p.get("path", ""), p.get("old_str", ""), p.get("new_str", "")),
            "append_to_file": lambda p: self.append_to_file(p.get("path", ""), p.get("content", "")),
            "insert_at_line": lambda p: self.insert_at_line(p.get("path", ""), p.get("line", 1), p.get("content", "")),
            "copy_file": lambda p: self.copy_file(p.get("source", ""), p.get("destination", "")),
            "move_file": lambda p: self.move_file(p.get("source", ""), p.get("destination", "")),
            "delete_file": lambda p: self.delete_file(p.get("path", "")),
            "list_files": lambda p: self.list_files(p.get("path", ".")),
            "run_command": lambda p: self.run_command(p.get("command", ""), p.get("cwd")),
            "search_files": lambda p: self.search_files(p.get("pattern", ""), p.get("path", ".")),
            "project_summary": lambda p: self.project_summary(p.get("path", ".")),
            "run_skill": lambda p: self.run_skill(p.get("skill", ""), p.get("args", "")),
            "synthesize_skill": lambda p: self.synthesize_skill(p.get("skill_name", ""), p.get("purpose", ""), p.get("code", "")),
        }
        handler = dispatch_map.get(tool_name)
        if handler is None:
            return ToolResult(tool_name, False, "", f"Unknown tool: '{tool_name}'")
        return handler(params)
