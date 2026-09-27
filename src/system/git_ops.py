"""
Genius Git Operations — Claude Code-level git automation.

Capabilities:
  ✓ git status, diff, log, branch listing
  ✓ Stage files (add), commit with auto-generated message
  ✓ Branch creation, switching, merging
  ✓ Pull Request summary generation
  ✓ Merge conflict detection and resolution hints
  ✓ Push to remote
  ✓ gitignore management
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GitStatus:
    branch: str
    is_clean: bool
    staged: List[str]
    unstaged: List[str]
    untracked: List[str]
    ahead: int
    behind: int
    has_conflicts: bool

    def summary(self) -> str:
        lines = [f"Branch: {self.branch}"]
        if self.is_clean:
            lines.append("Working tree clean.")
        if self.staged:
            lines.append(f"Staged ({len(self.staged)}): " + ", ".join(self.staged[:8]))
        if self.unstaged:
            lines.append(f"Modified ({len(self.unstaged)}): " + ", ".join(self.unstaged[:8]))
        if self.untracked:
            lines.append(f"Untracked ({len(self.untracked)}): " + ", ".join(self.untracked[:5]))
        if self.ahead:
            lines.append(f"Ahead of remote by {self.ahead} commit(s).")
        if self.behind:
            lines.append(f"Behind remote by {self.behind} commit(s).")
        if self.has_conflicts:
            lines.append("⚠️  MERGE CONFLICTS DETECTED!")
        return "\n".join(lines)


class GitOps:
    """Git automation for Genius agentic mode — all operations via subprocess git CLI."""

    def __init__(self, workspace_root: str) -> None:
        self.root = Path(workspace_root).resolve()

    def _git(self, *args: str, check: bool = False) -> Tuple[bool, str, str]:
        """Run git command, return (success, stdout, stderr)."""
        cmd = ["git"] + list(args)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.root),
                timeout=30,
            )
            return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()
        except FileNotFoundError:
            return False, "", "git not found in PATH. Install git first."
        except subprocess.TimeoutExpired:
            return False, "", "git command timed out."
        except Exception as e:
            return False, "", str(e)

    def is_git_repo(self) -> bool:
        ok, _, _ = self._git("rev-parse", "--git-dir")
        return ok

    def init(self) -> Tuple[bool, str]:
        ok, out, err = self._git("init")
        return ok, out if ok else err

    def status(self) -> GitStatus:
        """Returns structured git status."""
        _, branch_out, _ = self._git("branch", "--show-current")
        branch = branch_out.strip() or "HEAD (detached)"

        _, status_out, _ = self._git("status", "--porcelain=v1")
        staged, unstaged, untracked = [], [], []
        has_conflicts = False

        for line in status_out.splitlines():
            if len(line) < 3:
                continue
            xy, path = line[:2], line[3:].strip()
            if xy in ("UU", "AA", "DD", "AU", "UA"):
                has_conflicts = True
            if xy[0] in "MADRC" and xy[0] != " ":
                staged.append(path)
            if xy[1] in "MD?":
                if xy[1] == "?":
                    untracked.append(path)
                else:
                    unstaged.append(path)

        _, ahead_behind, _ = self._git("status", "--short", "--branch")
        ahead = behind = 0
        m = re.search(r"ahead (\d+)", ahead_behind)
        if m:
            ahead = int(m.group(1))
        m = re.search(r"behind (\d+)", ahead_behind)
        if m:
            behind = int(m.group(1))

        is_clean = not staged and not unstaged and not untracked

        return GitStatus(
            branch=branch,
            is_clean=is_clean,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
            ahead=ahead,
            behind=behind,
            has_conflicts=has_conflicts,
        )

    def diff(self, staged: bool = False, file: Optional[str] = None) -> str:
        """Returns git diff output."""
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.append(file)
        _, out, _ = self._git(*args)
        return out[:8000] if out else "(no diff)"

    def log(self, n: int = 10, oneline: bool = True) -> str:
        """Returns recent commit log."""
        fmt = "--oneline" if oneline else "--format=%H|%an|%ad|%s"
        _, out, _ = self._git("log", f"-{n}", fmt, "--date=short")
        return out if out else "(no commits yet)"

    def add(self, *paths: str) -> Tuple[bool, str]:
        """Stage files. Pass '.' to stage all."""
        if not paths:
            paths = (".",)
        ok, out, err = self._git("add", *paths)
        return ok, out if ok else err

    def commit(self, message: str) -> Tuple[bool, str]:
        """Create a commit with the given message."""
        if not message.strip():
            return False, "Commit message cannot be empty."
        ok, out, err = self._git("commit", "-m", message)
        return ok, out if ok else err

    def auto_commit(self, description: str = "") -> Tuple[bool, str]:
        """
        Stage all changes and create a descriptive commit.
        Auto-generates a structured commit message from git diff.
        """
        # Stage everything
        ok, _ = self.add(".")
        if not ok:
            return False, "Failed to stage files."

        # Check if there's anything to commit
        st = self.status()
        if st.is_clean and not st.staged:
            return True, "Nothing to commit — working tree clean."

        # Generate commit message from staged diff
        _, diff_out, _ = self._git("diff", "--cached", "--stat")
        stats = diff_out[:400] if diff_out else "various changes"

        if description:
            msg = f"{description}\n\nChanges:\n{stats}"
        else:
            msg = f"feat: autonomous changes by Genius Agent\n\nChanges:\n{stats}"

        return self.commit(msg)

    def create_branch(self, name: str, from_branch: Optional[str] = None) -> Tuple[bool, str]:
        """Create and switch to a new branch."""
        if from_branch:
            ok, out, err = self._git("checkout", "-b", name, from_branch)
        else:
            ok, out, err = self._git("checkout", "-b", name)
        return ok, out if ok else err

    def switch_branch(self, name: str) -> Tuple[bool, str]:
        """Switch to an existing branch."""
        ok, out, err = self._git("checkout", name)
        return ok, out if ok else err

    def list_branches(self) -> str:
        """List all local + remote branches."""
        _, local, _ = self._git("branch")
        _, remote, _ = self._git("branch", "-r")
        result = "Local branches:\n" + local if local else "No local branches."
        if remote:
            result += "\n\nRemote branches:\n" + remote
        return result

    def merge(self, branch: str) -> Tuple[bool, str]:
        """Merge a branch into current branch."""
        ok, out, err = self._git("merge", branch, "--no-edit")
        return ok, (out + "\n" + err).strip()

    def push(self, remote: str = "origin", branch: Optional[str] = None, set_upstream: bool = False) -> Tuple[bool, str]:
        """Push to remote."""
        args = ["push"]
        if set_upstream:
            args += ["-u"]
        args.append(remote)
        if branch:
            args.append(branch)
        ok, out, err = self._git(*args)
        return ok, (out + "\n" + err).strip()

    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> Tuple[bool, str]:
        """Pull from remote."""
        args = ["pull", remote]
        if branch:
            args.append(branch)
        ok, out, err = self._git(*args)
        return ok, (out + "\n" + err).strip()

    def get_conflicts(self) -> Dict[str, str]:
        """
        Returns dict of {filename: conflict_section} for all files with merge conflicts.
        """
        st = self.status()
        conflicts: Dict[str, str] = {}
        if not st.has_conflicts:
            return conflicts

        _, conflict_files, _ = self._git("diff", "--name-only", "--diff-filter=U")
        for fname in conflict_files.splitlines():
            fpath = self.root / fname.strip()
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    # Extract conflict markers
                    if "<<<<<<<" in content:
                        conflicts[fname.strip()] = content[:3000]
                except Exception:
                    pass
        return conflicts

    def resolve_conflict(self, file: str, keep: str = "ours") -> Tuple[bool, str]:
        """
        Resolve a merge conflict by keeping 'ours', 'theirs', or already-edited file.
        keep: 'ours', 'theirs', or 'manual' (file already edited)
        """
        if keep == "ours":
            ok, out, err = self._git("checkout", "--ours", file)
        elif keep == "theirs":
            ok, out, err = self._git("checkout", "--theirs", file)
        else:
            ok, out, err = True, "Manual resolution assumed.", ""
        if ok:
            ok2, _, err2 = self._git("add", file)
            return ok2, f"Resolved {file} (keep={keep}). Staged." if ok2 else err2
        return False, err

    def generate_pr_summary(self, base_branch: str = "main") -> str:
        """
        Generates a Pull Request description from changes vs base_branch.
        Shows: what changed, files modified, commit log.
        """
        _, diff_stat, _ = self._git("diff", f"{base_branch}...HEAD", "--stat")
        _, commits, _ = self._git("log", f"{base_branch}...HEAD", "--oneline")
        _, changed_files, _ = self._git("diff", f"{base_branch}...HEAD", "--name-only")

        lines = [
            "## Pull Request Summary — Generated by Genius AI",
            "",
            "### Commits",
            commits or "(none)",
            "",
            "### Files Changed",
            changed_files or "(none)",
            "",
            "### Diff Statistics",
            diff_stat or "(no diff)",
        ]
        return "\n".join(lines)

    def add_to_gitignore(self, patterns: List[str]) -> Tuple[bool, str]:
        """Add patterns to .gitignore, creating it if needed."""
        gitignore = self.root / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        added = []
        for p in patterns:
            if p not in existing:
                existing += f"\n{p}"
                added.append(p)
        if added:
            gitignore.write_text(existing.strip() + "\n", encoding="utf-8")
            return True, f"Added to .gitignore: {', '.join(added)}"
        return True, "All patterns already in .gitignore."

    def full_status_report(self) -> str:
        """Full formatted status report for display in terminal."""
        if not self.is_git_repo():
            return "Not a git repository. Run: git init"
        st = self.status()
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "  Git Status Report",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            st.summary(),
            "",
            "Recent Commits:",
            self.log(n=5),
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        return "\n".join(lines)
