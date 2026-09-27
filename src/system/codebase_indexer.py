"""
Genius Codebase Indexer — Full project understanding & symbol mapping.

Capabilities:
  ✓ Symbol map: classes, functions, variables across all files
  ✓ Import/dependency graph (who imports what)
  ✓ Reference tracking: where is a symbol used?
  ✓ Entry point detection (main, __main__, app.run, etc.)
  ✓ Docstring & signature extraction
  ✓ Multi-language support: Python, JS/TS, Go, Rust, Java
  ✓ Protocol.txt command registry loader
"""

from __future__ import annotations

import ast
import configparser
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Symbol:
    name: str
    kind: str          # "function", "class", "method", "variable", "import"
    file: str
    line: int
    signature: str = ""
    docstring: str = ""
    references: List[str] = field(default_factory=list)  # files that reference this symbol


@dataclass
class FileIndex:
    path: str
    language: str
    symbols: List[Symbol]
    imports: List[str]
    exports: List[str]
    size_bytes: int
    lines: int


@dataclass
class CodebaseIndex:
    root: str
    files: List[FileIndex]
    symbol_map: Dict[str, List[Symbol]]      # symbol_name -> [Symbol(file1), Symbol(file2)]
    import_graph: Dict[str, List[str]]       # file -> [files it imports from]
    entry_points: List[str]                  # detected main entry files
    total_files: int
    total_lines: int

    def find_symbol(self, name: str) -> List[Symbol]:
        """Find all definitions of a symbol by name (case-insensitive partial match)."""
        name_lower = name.lower()
        results = []
        for sym_name, syms in self.symbol_map.items():
            if name_lower in sym_name.lower():
                results.extend(syms)
        return results

    def find_references(self, symbol_name: str) -> List[Tuple[str, int, str]]:
        """Find all usages of a symbol across the codebase. Returns (file, line, snippet)."""
        results = []
        root = Path(self.root)
        for fi in self.files:
            try:
                text = Path(fi.path).read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if re.search(r'\b' + re.escape(symbol_name) + r'\b', line):
                        rel = str(Path(fi.path).relative_to(root))
                        results.append((rel, i, line.strip()[:100]))
            except Exception:
                pass
        return results[:30]  # cap at 30 to avoid context explosion

    def summary_for_agent(self) -> str:
        """Returns a concise codebase summary suitable for LLM context injection."""
        lines = [
            f"CODEBASE INDEX: {self.root}",
            f"Files: {self.total_files} | Lines: {self.total_lines}",
        ]
        if self.entry_points:
            lines.append(f"Entry Points: {', '.join(self.entry_points[:5])}")

        lines.append("\nKey Symbols (top 40):")
        shown = 0
        for name, syms in sorted(self.symbol_map.items()):
            for s in syms:
                if s.kind in ("class", "function") and not s.name.startswith("_"):
                    rel = str(Path(s.file).relative_to(self.root)) if os.path.isabs(s.file) else s.file
                    sig = f" {s.signature[:60]}" if s.signature else ""
                    lines.append(f"  [{s.kind}] {s.name}{sig} — {rel}:{s.line}")
                    shown += 1
                    if shown >= 40:
                        break
            if shown >= 40:
                break

        if self.import_graph:
            lines.append("\nKey Dependencies:")
            for f, deps in list(self.import_graph.items())[:8]:
                rel = str(Path(f).relative_to(self.root)) if os.path.isabs(f) else f
                lines.append(f"  {rel} → {', '.join(deps[:5])}")

        return "\n".join(lines)


# ─── Language Parsers ─────────────────────────────────────────────────────────

class PythonParser:
    """Extract symbols from Python source using AST."""

    @staticmethod
    def parse(file_path: str) -> FileIndex:
        path = Path(file_path)
        symbols: List[Symbol] = []
        imports: List[str] = []
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except Exception:
            return FileIndex(
                path=str(path), language="python", symbols=[], imports=[],
                exports=[], size_bytes=path.stat().st_size if path.exists() else 0, lines=0
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                sig = f"({', '.join(args)})"
                doc = ast.get_docstring(node) or ""
                symbols.append(Symbol(
                    name=node.name, kind="function", file=str(path),
                    line=node.lineno, signature=sig, docstring=doc[:100]
                ))
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                symbols.append(Symbol(
                    name=node.name, kind="class", file=str(path),
                    line=node.lineno, docstring=doc[:100]
                ))
                # Methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in item.args.args]
                        sig = f"({', '.join(args)})"
                        symbols.append(Symbol(
                            name=f"{node.name}.{item.name}", kind="method",
                            file=str(path), line=item.lineno, signature=sig
                        ))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    mod = node.module or ""
                    imports.append(mod)

        lines = source.count("\n") + 1
        size = len(source.encode("utf-8"))
        return FileIndex(
            path=str(path), language="python", symbols=symbols,
            imports=list(set(imports)), exports=[], size_bytes=size, lines=lines
        )


class RegexParser:
    """Lightweight regex-based symbol extractor for JS/TS/Go/Rust/Java."""

    PATTERNS: Dict[str, Dict[str, str]] = {
        ".js": {
            "function": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\(|(\w+)\s*:\s*(?:async\s*)?\()",
            "class": r"class\s+(\w+)",
            "import": r'(?:import|require)\s*(?:\(?\s*[\'"])([^\'\"]+)',
        },
        ".ts": {
            "function": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\(|(\w+)\s*:\s*(?:async\s*)?\()",
            "class": r"class\s+(\w+)",
            "import": r'import\s+.*?from\s+[\'"]([^\'"]+)',
        },
        ".go": {
            "function": r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
            "class": r"^type\s+(\w+)\s+struct",
            "import": r'"([^"]+/[^"]+)"',
        },
        ".rs": {
            "function": r"^(?:pub\s+)?fn\s+(\w+)\s*[\(<]",
            "class": r"^(?:pub\s+)?struct\s+(\w+)",
            "import": r"^use\s+([\w:]+)",
        },
        ".java": {
            "function": r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(",
            "class": r"(?:public\s+)?class\s+(\w+)",
            "import": r"^import\s+([\w.]+);",
        },
    }

    @classmethod
    def parse(cls, file_path: str) -> FileIndex:
        path = Path(file_path)
        ext = path.suffix.lower()
        pats = cls.PATTERNS.get(ext, {})
        symbols: List[Symbol] = []
        imports: List[str] = []

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return FileIndex(path=str(path), language=ext[1:], symbols=[], imports=[], exports=[], size_bytes=0, lines=0)

        lang_map = {".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust", ".java": "java"}
        lang = lang_map.get(ext, ext[1:] if ext else "unknown")

        for i, line in enumerate(source.splitlines(), 1):
            for kind, pat in pats.items():
                m = re.search(pat, line)
                if m:
                    name = next((g for g in m.groups() if g), None)
                    if name and name.strip():
                        if kind == "import":
                            imports.append(name.strip())
                        else:
                            symbols.append(Symbol(name=name.strip(), kind=kind, file=str(path), line=i))

        size = len(source.encode("utf-8"))
        lines = source.count("\n") + 1
        return FileIndex(
            path=str(path), language=lang, symbols=symbols,
            imports=list(set(imports)), exports=[], size_bytes=size, lines=lines
        )


# ─── Codebase Indexer ─────────────────────────────────────────────────────────

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", "dist", "build", ".pytest_cache", ".next"}
PYTHON_EXTS = {".py"}
REGEX_EXTS = {".js", ".ts", ".go", ".rs", ".java"}
ALL_EXTS = PYTHON_EXTS | REGEX_EXTS

ENTRY_NAMES = {
    "main.py", "app.py", "run.py", "index.py", "server.py", "start.py",
    "manage.py", "wsgi.py", "asgi.py", "index.js", "app.js", "main.js",
    "server.js", "main.go", "main.rs",
}


class CodebaseIndexer:
    """Scans and indexes an entire project for symbols, imports, and structure."""

    def __init__(self, workspace_root: str) -> None:
        self.root = Path(workspace_root).resolve()

    def build_index(self, max_files: int = 200) -> CodebaseIndex:
        """Full project scan. Returns a CodebaseIndex."""
        file_indices: List[FileIndex] = []
        symbol_map: Dict[str, List[Symbol]] = {}
        import_graph: Dict[str, List[str]] = {}
        entry_points: List[str] = []
        scanned = 0

        for fpath in self._iter_source_files(max_files):
            ext = fpath.suffix.lower()
            if ext in PYTHON_EXTS:
                fi = PythonParser.parse(str(fpath))
            elif ext in REGEX_EXTS:
                fi = RegexParser.parse(str(fpath))
            else:
                continue

            file_indices.append(fi)
            scanned += 1

            # Build symbol map
            for sym in fi.symbols:
                if sym.name not in symbol_map:
                    symbol_map[sym.name] = []
                symbol_map[sym.name].append(sym)

            # Build import graph
            rel = str(fpath.relative_to(self.root))
            import_graph[rel] = fi.imports[:15]

            # Detect entry points
            if fpath.name in ENTRY_NAMES:
                entry_points.append(rel)

        total_lines = sum(fi.lines for fi in file_indices)

        return CodebaseIndex(
            root=str(self.root),
            files=file_indices,
            symbol_map=symbol_map,
            import_graph=import_graph,
            entry_points=entry_points,
            total_files=scanned,
            total_lines=total_lines,
        )

    def _iter_source_files(self, max_files: int):
        """Yields source files to index, skipping noise."""
        count = 0
        for fpath in sorted(self.root.rglob("*")):
            if count >= max_files:
                break
            if any(skip in fpath.parts for skip in SKIP_DIRS):
                continue
            if fpath.is_file() and fpath.suffix.lower() in ALL_EXTS:
                if fpath.stat().st_size < 500_000:  # skip huge generated files
                    yield fpath
                    count += 1

    def find_symbol_context(self, symbol_name: str) -> str:
        """Quick single-symbol lookup without full index build."""
        results = []
        for fpath in self._iter_source_files(100):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.splitlines(), 1):
                    if re.search(r'\b' + re.escape(symbol_name) + r'\b', line):
                        rel = str(fpath.relative_to(self.root))
                        results.append(f"{rel}:{i}: {line.strip()[:100]}")
                        if len(results) >= 20:
                            break
            except Exception:
                pass
            if len(results) >= 20:
                break
        if not results:
            return f"Symbol '{symbol_name}' not found in codebase."
        return f"References to '{symbol_name}':\n" + "\n".join(results)


# ─── Protocol.txt Registry ────────────────────────────────────────────────────

class ProtocolRegistry:
    """
    Loads and executes commands from protocol.txt in the workspace root.

    protocol.txt format:
    ─────────────────────────────────
    [run]
    command = python main.py

    [test]
    command = pytest tests/ -v

    [build]
    command = python setup.py build

    [install]
    command = pip install -r requirements.txt

    [lint]
    command = python -m flake8 src/

    [format]
    command = python -m black src/

    [dev]
    command = python run_genius.py --research auto
    description = Start Genius AI in development mode
    ─────────────────────────────────
    """

    PROTOCOL_NAMES = ["protocol.txt", "Protocol.txt", ".protocol.txt"]

    def __init__(self, workspace_root: str) -> None:
        self.root = Path(workspace_root).resolve()
        self.commands: Dict[str, Dict[str, str]] = {}
        self._load()

    def _find_protocol_file(self) -> Optional[Path]:
        for name in self.PROTOCOL_NAMES:
            p = self.root / name
            if p.exists():
                return p
        return None

    def _load(self) -> None:
        """Load protocol.txt using configparser INI format."""
        proto = self._find_protocol_file()
        if not proto:
            return
        try:
            config = configparser.ConfigParser()
            config.read(str(proto), encoding="utf-8")
            for section in config.sections():
                self.commands[section.lower()] = dict(config[section])
        except Exception:
            # Fallback: simple key=value parser without sections
            try:
                text = proto.read_text(encoding="utf-8", errors="replace")
                current_section = "default"
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1].lower().strip()
                        if current_section not in self.commands:
                            self.commands[current_section] = {}
                    elif "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        if current_section not in self.commands:
                            self.commands[current_section] = {}
                        self.commands[current_section][k.strip().lower()] = v.strip()
            except Exception:
                pass

    def reload(self) -> None:
        """Reload protocol.txt from disk."""
        self.commands.clear()
        self._load()

    def get_command(self, name: str) -> Optional[str]:
        """Get the shell command for a protocol name (e.g. 'test', 'build')."""
        section = self.commands.get(name.lower(), {})
        return section.get("command") or section.get("cmd")

    def list_all(self) -> str:
        """Returns formatted list of all available protocol commands."""
        if not self.commands:
            proto = self._find_protocol_file()
            if not proto:
                return "No protocol.txt found in workspace. Create one to define project commands."
            return "protocol.txt found but empty or could not be parsed."

        lines = ["Available Protocol Commands:", "─" * 42]
        for name, cfg in sorted(self.commands.items()):
            cmd = cfg.get("command") or cfg.get("cmd", "?")
            desc = cfg.get("description") or cfg.get("desc", "")
            lines.append(f"  /run {name:<16}  {cmd[:45]}")
            if desc:
                lines.append(f"  {'':20}  ({desc})")
        lines.append("─" * 42)
        lines.append("Usage: /run <name>   (e.g. /run test, /run build)")
        return "\n".join(lines)

    @property
    def protocol_path(self) -> Optional[Path]:
        return self._find_protocol_file()

    @staticmethod
    def create_default(workspace_root: str, project_lang: str = "python") -> str:
        """Creates a default protocol.txt for the given project type."""
        root = Path(workspace_root)
        proto_path = root / "protocol.txt"

        if project_lang == "python":
            content = """\
# Genius AI — Protocol Command Registry
# Run commands with: /run <name>
# ─────────────────────────────────────

[run]
command = python main.py
description = Run the main application

[test]
command = python -m pytest tests/ -v
description = Run all tests

[install]
command = pip install -r requirements.txt
description = Install dependencies

[lint]
command = python -m flake8 src/ --max-line-length=120
description = Run linter

[format]
command = python -m black src/ --line-length=120
description = Auto-format code

[clean]
command = find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null
description = Remove compiled Python files

[dev]
command = python run_genius.py --research auto
description = Start Genius AI

[git-status]
command = git status && git log --oneline -5
description = Show git status and recent commits

[git-commit]
command = git add -A && git commit -m "feat: autonomous update by Genius"
description = Stage all and commit
"""
        elif project_lang == "node":
            content = """\
# Genius AI — Protocol Command Registry

[run]
command = node index.js

[dev]
command = npm run dev

[test]
command = npm test

[build]
command = npm run build

[install]
command = npm install

[lint]
command = npm run lint

[format]
command = npx prettier --write .
"""
        else:
            content = """\
# Genius AI — Protocol Command Registry
# Add commands below in [section] format

[run]
command = echo "Set your run command here"

[test]
command = echo "Set your test command here"

[build]
command = echo "Set your build command here"
"""
        proto_path.write_text(content, encoding="utf-8")
        return str(proto_path)
