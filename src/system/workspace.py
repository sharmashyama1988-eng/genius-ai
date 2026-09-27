"""Autonomous Project Workspace, File Operations & Image Intelligence Engine for Genius.

Provides seamless directory shifting, autonomous file creation/reading/editing,
directory hierarchy scanning, and zero-dependency binary image dimension parsing & viewing.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class WorkspaceManager:
    """Manages active project directory, file operations, and image inspection."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".ico", ".tiff"}

    def __init__(self, initial_path: Optional[str] = None) -> None:
        self.workspace_root: Path = Path(initial_path or os.getcwd()).resolve()
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)

    def set_workspace(self, target_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Shifts active working directory to target_path and scans project structure."""
        p = Path(target_path).expanduser()
        if not p.is_absolute():
            p = (self.workspace_root / p).resolve()
        else:
            p = p.resolve()

        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                created = True
            except Exception as e:
                return False, f"Failed to create workspace directory '{p}': {e}", {}
        else:
            created = False

        if not p.is_dir():
            return False, f"Target path '{p}' is a file, not a directory.", {}

        self.workspace_root = p
        # Scan project structure
        summary = self.scan_project()

        status_msg = (
            f"Successfully shifted active workspace to: {self.workspace_root}\n"
            f"• Project Root: {self.workspace_root}\n"
            f"• Total Files: {summary.get('file_count', 0)}\n"
            f"• Total Folders: {summary.get('dir_count', 0)}\n"
            f"• Primary Languages: {', '.join(summary.get('languages', [])) or 'None detected'}\n"
            f"• Project Signatures: {', '.join(summary.get('signatures', [])) or 'General Directory'}"
        )
        if created:
            status_msg = f"[Created New Directory]\n" + status_msg

        return True, status_msg, summary

    def get_workspace(self) -> Path:
        """Returns the active workspace root directory."""
        return self.workspace_root

    def resolve_path(self, rel_or_abs: str) -> Path:
        """Resolves relative or absolute path against current active workspace root."""
        p = Path(rel_or_abs).expanduser()
        if not p.is_absolute():
            return (self.workspace_root / p).resolve()
        return p.resolve()

    def scan_project(self, max_files: int = 500) -> Dict[str, Any]:
        """Scans the active project directory detecting languages, file counts, and signatures."""
        counts_by_ext: Dict[str, int] = {}
        total_files = 0
        total_dirs = 0
        signatures = []

        # Common project indicators
        sig_map = {
            "package.json": "Node.js / JavaScript",
            "tsconfig.json": "TypeScript",
            "pyproject.toml": "Python (Poetry/Modern)",
            "requirements.txt": "Python (Pip)",
            "setup.py": "Python (Setuptools)",
            "Cargo.toml": "Rust (Cargo)",
            "go.mod": "Go Module",
            "pom.xml": "Java (Maven)",
            "build.gradle": "Java/Kotlin (Gradle)",
            "docker-compose.yml": "Docker Compose",
            "Dockerfile": "Docker Container",
            ".git": "Git Repository",
        }

        for root, dirs, files in os.walk(self.workspace_root):
            # Skip hidden / venv / cache folders
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")]
            total_dirs += len(dirs)

            for f in files:
                total_files += 1
                if f in sig_map and sig_map[f] not in signatures:
                    signatures.append(sig_map[f])

                ext = Path(f).suffix.lower()
                if ext:
                    counts_by_ext[ext] = counts_by_ext.get(ext, 0) + 1

                if total_files >= max_files:
                    break
            if total_files >= max_files:
                break

        # Map extensions to language labels
        ext_to_lang = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "React TSX",
            ".jsx": "React JSX",
            ".html": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".rs": "Rust",
            ".go": "Go",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
            ".sql": "SQL",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".md": "Markdown",
            ".sh": "Shell",
            ".ps1": "PowerShell",
        }

        langs = []
        for ext, _ in sorted(counts_by_ext.items(), key=lambda x: x[1], reverse=True):
            if ext in ext_to_lang and ext_to_lang[ext] not in langs:
                langs.append(ext_to_lang[ext])

        return {
            "root": str(self.workspace_root),
            "file_count": total_files,
            "dir_count": total_dirs,
            "languages": langs[:6],
            "signatures": signatures,
            "extensions": counts_by_ext,
        }

    def create_file(self, file_path: str, content: str = "", overwrite: bool = True) -> Tuple[bool, str]:
        """Creates a file with given content inside active workspace."""
        target = self.resolve_path(file_path)
        if target.exists() and not overwrite:
            return False, f"File already exists: {target}. Set overwrite=True to replace."

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            size_bytes = target.stat().st_size
            rel_str = str(target.relative_to(self.workspace_root)) if target.is_relative_to(self.workspace_root) else str(target)
            return True, f"File created successfully: {rel_str} ({size_bytes} bytes, {len(content.splitlines())} lines)"
        except Exception as e:
            return False, f"Failed to create file '{target}': {e}"

    def read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Reads file within active workspace. If image, routes to image metadata inspector."""
        target = self.resolve_path(file_path)

        if not target.exists():
            return False, f"File not found: {target}"

        # If it's an image, inspect it
        ext = target.suffix.lower()
        if ext in self.IMAGE_EXTENSIONS:
            ok, msg, _ = self.inspect_image(str(target), auto_open=True)
            return ok, msg

        # Read as text
        try:
            try:
                text = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = target.read_text(encoding="latin-1")

            lines = text.splitlines()
            total_lines = len(lines)

            if start_line is not None or end_line is not None:
                s = max(1, start_line or 1) - 1
                e = min(total_lines, end_line or total_lines)
                sliced = lines[s:e]
                formatted = [f"{i+1:4d} | {l}" for i, l in enumerate(sliced, start=s)]
                content = "\n".join(formatted)
                return True, f"File: {target.name} (Lines {s+1}-{e} of {total_lines}):\n\n{content}"

            rel_str = str(target.relative_to(self.workspace_root)) if target.is_relative_to(self.workspace_root) else str(target)
            return True, f"File: {rel_str} ({total_lines} lines):\n\n{text}"

        except Exception as e:
            return False, f"Could not read file '{target}': {e}"

    def list_files(self, subpath: str = "", max_depth: int = 2) -> Tuple[bool, str]:
        """Lists files and folders in workspace or subpath with tree representation."""
        target_dir = self.resolve_path(subpath) if subpath else self.workspace_root

        if not target_dir.exists() or not target_dir.is_dir():
            return False, f"Directory does not exist: {target_dir}"

        lines = [f"📂 Workspace: {target_dir}"]

        def _walk(curr: Path, depth: int, prefix: str):
            if depth > max_depth:
                return
            try:
                entries = sorted(list(curr.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                return

            # Filter hidden / venv
            entries = [e for e in entries if not e.name.startswith(".") and e.name not in ("node_modules", "__pycache__", "venv", ".venv")]

            for i, e in enumerate(entries):
                is_last = (i == len(entries) - 1)
                connector = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")

                if e.is_dir():
                    lines.append(f"{prefix}{connector}📁 {e.name}/")
                    _walk(e, depth + 1, next_prefix)
                else:
                    sz = e.stat().st_size
                    sz_str = f"{sz} B" if sz < 1024 else (f"{sz/1024:.1f} KB" if sz < 1048576 else f"{sz/1048576:.1f} MB")
                    icon = "🖼️" if e.suffix.lower() in self.IMAGE_EXTENSIONS else "📄"
                    lines.append(f"{prefix}{connector}{icon} {e.name}  ({sz_str})")

        _walk(target_dir, 1, "")
        return True, "\n".join(lines)

    def edit_file(self, file_path: str, old_content: str, new_content: str) -> Tuple[bool, str]:
        """Replaces exact target content with new content in file."""
        target = self.resolve_path(file_path)
        if not target.exists():
            return False, f"File not found: {target}"

        try:
            text = target.read_text(encoding="utf-8")
            if old_content not in text:
                return False, f"Target text chunk not found in '{target.name}'. Verify content."

            occurrences = text.count(old_content)
            if occurrences > 1:
                return False, f"Target text matches {occurrences} occurrences in '{target.name}'. Must be unique."

            updated = text.replace(old_content, new_content, 1)
            target.write_text(updated, encoding="utf-8")
            return True, f"File '{target.name}' updated successfully."
        except Exception as e:
            return False, f"Failed to edit file '{target}': {e}"

    def inspect_image(self, file_path: str, auto_open: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """Extracts image dimensions and metadata without external libraries, and optionally opens it."""
        target = self.resolve_path(file_path)

        if not target.exists():
            return False, f"Image not found: {target}", {}

        ext = target.suffix.lower()
        if ext not in self.IMAGE_EXTENSIONS:
            return False, f"File '{target.name}' is not recognized as an image format.", {}

        file_size = target.stat().st_size
        sz_str = f"{file_size} B" if file_size < 1024 else (f"{file_size/1024:.1f} KB" if file_size < 1048576 else f"{file_size/1048576:.1f} MB")

        width, height = self._get_image_dimensions(target)
        aspect_ratio = ""
        if width and height:
            aspect_ratio = f"{width / height:.2f}:1"
            # Simplify common ratios
            if abs((width / height) - (16 / 9)) < 0.05:
                aspect_ratio = "16:9"
            elif abs((width / height) - (4 / 3)) < 0.05:
                aspect_ratio = "4:3"
            elif abs((width / height) - 1.0) < 0.01:
                aspect_ratio = "1:1 (Square)"

        meta: Dict[str, Any] = {
            "path": str(target),
            "name": target.name,
            "format": ext.upper().replace(".", ""),
            "size_bytes": file_size,
            "size_str": sz_str,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
        }

        opened_str = ""
        if auto_open:
            opened = self.open_image_viewer(str(target))
            if opened:
                opened_str = " (Opened in system image viewer)"

        dim_str = f"{width} × {height} px" if width and height else "Unknown dimensions"
        resp = (
            f"🖼️ Image Analysis: {target.name}{opened_str}\n"
            f"• Full Path: {target}\n"
            f"• Format: {meta['format']}\n"
            f"• Dimensions: {dim_str}\n"
            f"• Aspect Ratio: {aspect_ratio or 'N/A'}\n"
            f"• File Size: {sz_str}\n"
            f"• Status: Ready for vision / project embedding"
        )
        return True, resp, meta

    def open_image_viewer(self, file_path: str) -> bool:
        """Launches native system viewer to display the image."""
        target = self.resolve_path(file_path)
        if not target.exists():
            return False

        try:
            if sys.platform == "win32":
                os.startfile(str(target))  # Native Windows viewer
                return True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
                return True
            else:
                subprocess.Popen(["xdg-open", str(target)])
                return True
        except Exception:
            return False

    def _get_image_dimensions(self, path: Path) -> Tuple[Optional[int], Optional[int]]:
        """Parses image header bytes for PNG, JPEG, GIF, BMP, and WebP using pure standard library."""
        try:
            with open(path, "rb") as f:
                head = f.read(64)

                # PNG
                if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                    w, h = struct.unpack(">II", head[16:24])
                    return w, h

                # GIF
                if head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
                    w, h = struct.unpack("<HH", head[6:10])
                    return w, h

                # BMP
                if head.startswith(b"BM") and len(head) >= 26:
                    w, h = struct.unpack("<ii", head[18:26])
                    return w, abs(h)

                # WebP
                if head.startswith(b"RIFF") and b"WEBP" in head[8:16]:
                    f.seek(12)
                    chunk_type = f.read(4)
                    if chunk_type == b"VP8 ":
                        f.seek(26)
                        data = f.read(4)
                        if len(data) >= 4:
                            w = struct.unpack("<H", data[0:2])[0] & 0x3FFF
                            h = struct.unpack("<H", data[2:4])[0] & 0x3FFF
                            return w, h
                    elif chunk_type == b"VP8L":
                        f.seek(21)
                        b_data = f.read(4)
                        if len(b_data) >= 4:
                            bits = struct.unpack("<I", b_data)[0]
                            w = (bits & 0x3FFF) + 1
                            h = ((bits >> 14) & 0x3FFF) + 1
                            return w, h
                    elif chunk_type == b"VP8X":
                        f.seek(24)
                        d = f.read(6)
                        if len(d) >= 6:
                            w = struct.unpack("<I", d[0:3] + b"\x00")[0] + 1
                            h = struct.unpack("<I", d[3:6] + b"\x00")[0] + 1
                            return w, h

                # JPEG (Scan for SOF marker)
                if head.startswith(b"\xff\xd8"):
                    f.seek(2)
                    while True:
                        marker_data = f.read(2)
                        if len(marker_data) < 2:
                            break
                        if marker_data[0] != 0xFF:
                            break
                        marker = marker_data[1]
                        # SOF markers: 0xC0 to 0xC3, 0xC9 to 0xCB
                        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC9, 0xCA, 0xCB):
                            length_data = f.read(2)
                            sof_data = f.read(5)
                            if len(sof_data) >= 5:
                                h, w = struct.unpack(">HH", sof_data[1:5])
                                return w, h
                            break
                        elif marker in (0xD9, 0xDA):  # EOI or SOS
                            break
                        else:
                            # Skip marker segment
                            length_data = f.read(2)
                            if len(length_data) < 2:
                                break
                            seg_len = struct.unpack(">H", length_data)[0]
                            f.seek(seg_len - 2, os.SEEK_CUR)

        except Exception:
            pass

        return None, None

    def handle_natural_language_intent(self, query: str) -> Optional[Tuple[str, str]]:
        """Detects and executes autonomous project, file, or image commands from natural language."""
        q = query.strip()
        q_lower = q.lower()

        # 1. Project / Workspace Shift
        # Matches: "shift to project F:\..." or "project F:\... pe shift ho jao" or "set workspace to C:\..."
        shift_match = re.search(
            r"(?:shift|switch|change|set|go\s+to|open)\s+(?:to\s+)?(?:workspace|project|directory|dir|folder|path)\s+(?:to\s+)?([a-zA-Z]:[\\\/][^\s\"']+|[\\\/][^\s\"']+|\.{1,2}[\\\/][^\s\"']+|[a-zA-Z0-9_\-\.\/\\]+)",
            q,
            re.IGNORECASE,
        )
        if not shift_match:
            shift_match = re.search(
                r"(?:project|workspace|directory|dir|folder|path)\s+([a-zA-Z]:[\\\/][^\s\"']+|[\\\/][^\s\"']+|\.{1,2}[\\\/][^\s\"']+|[a-zA-Z0-9_\-\.\/\\]+)\s*(?:pe\s+)?(?:shift|switch|chalo|set)",
                q,
                re.IGNORECASE,
            )

        if shift_match:
            target_dir = shift_match.group(1).strip(" \"'")
            ok, msg, _ = self.set_workspace(target_dir)
            return ("workspace_shift", msg)

        # 2. File Creation
        # Matches: "create file main.py with content print('hello')" or "make a file named test.txt"
        create_match = re.search(
            r"(?:create|make|write|generate)\s+(?:a\s+)?file\s+(?:named\s+|called\s+)?([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9]+)(?:\s+(?:with|content|mein)\s*[:\s]*([\s\S]*))?",
            q,
            re.IGNORECASE,
        )
        if create_match:
            fname = create_match.group(1).strip(" \"'")
            content = create_match.group(2) or ""
            # Strip enclosing code backticks if passed
            content = re.sub(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$", r"\1", content.strip())
            ok, msg = self.create_file(fname, content)
            return ("file_create", msg)

        # 3. Image Viewing / Inspection
        # Matches: "show image logo.png" or "view image photo.jpg" or "image test.png dikhao"
        img_match = re.search(
            r"(?:show|view|open|display|inspect)\s+(?:the\s+)?image\s+([a-zA-Z0-9_\-\.\/\\]+\.(?:png|jpg|jpeg|webp|gif|bmp|svg|ico))",
            q,
            re.IGNORECASE,
        )
        if not img_match:
            img_match = re.search(
                r"(?:image|photo|picture)\s+([a-zA-Z0-9_\-\.\/\\]+\.(?:png|jpg|jpeg|webp|gif|bmp|svg|ico))\s+(?:dikhao|open|view|show|dekhna)",
                q,
                re.IGNORECASE,
            )
        if img_match:
            img_path = img_match.group(1).strip(" \"'")
            ok, msg, _ = self.inspect_image(img_path, auto_open=True)
            return ("image_inspect", msg)

        # 4. File Reading
        # Matches: "read file app.py" or "show file config.json" or "open file data.csv"
        read_match = re.search(
            r"(?:read|open|show|cat|display)\s+(?:the\s+)?file\s+([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9]+)",
            q,
            re.IGNORECASE,
        )
        if not read_match:
            read_match = re.search(
                r"file\s+([a-zA-Z0-9_\-\.\/\\]+\.[a-zA-Z0-9]+)\s+(?:read|open|padho|dikhao|show)",
                q,
                re.IGNORECASE,
            )
        if read_match:
            fname = read_match.group(1).strip(" \"'")
            ok, msg = self.read_file(fname)
            return ("file_read", msg)

        # 5. List Files in Project
        if any(p in q_lower for p in ["list files", "show files", "project files", "directory files", "files in project", "files dikhao", "list all files"]):
            ok, msg = self.list_files()
            return ("file_list", msg)

        return None
