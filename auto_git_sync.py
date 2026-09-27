"""
Autonomous Git Sync Daemon for Genius.
Periodically scans workspace for modifications, stages files, generates semantic commit messages,
and atomically pushes to GitHub.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Runs a shell command and returns returncode, stdout, stderr."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=45,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def get_git_status(repo_dir: Path) -> list[str]:
    """Returns list of modified/untracked files from git status."""
    code, stdout, _ = run_cmd(["git", "status", "--porcelain"], repo_dir)
    if code != 0 or not stdout:
        return []
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines


def generate_commit_message(changed_lines: list[str]) -> str:
    """Generates a concise, semantic commit message based on modified file paths."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    files = [line.split()[-1] for line in changed_lines]
    categories = set()

    for f in files:
        f_lower = f.lower()
        if "reasoning" in f_lower or "schema" in f_lower or "grounding" in f_lower:
            categories.add("reasoning")
        elif "memory" in f_lower or "episodic" in f_lower:
            categories.add("memory")
        elif "model" in f_lower or "provider" in f_lower:
            categories.add("model")
        elif "retrieval" in f_lower or "search" in f_lower:
            categories.add("retrieval")
        elif "tests" in f_lower:
            categories.add("tests")
        elif "readme" in f_lower or ".md" in f_lower:
            categories.add("docs")
        else:
            categories.add("core")

    scope = "+".join(sorted(categories)) if categories else "workspace"
    file_count = len(files)
    return f"feat({scope}): auto-sync {file_count} file(s) [{now_str}]"


def sync_cycle(repo_dir: Path, branch: str = "main", remote: str = "origin") -> bool:
    """Executes a single git add, commit, and push cycle if changes exist."""
    changes = get_git_status(repo_dir)
    if not changes:
        return False

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📦 Detected {len(changes)} modified file(s). Staging...", flush=True)

    # Stage changes
    c_add, out_add, err_add = run_cmd(["git", "add", "."], repo_dir)
    if c_add != 0:
        print(f"[!] git add failed: {err_add}", flush=True)
        return False

    # Check if anything is actually staged
    c_diff, out_diff, _ = run_cmd(["git", "diff", "--cached", "--quiet"], repo_dir)
    if c_diff == 0:
        return False  # No staged changes

    # Commit
    commit_msg = generate_commit_message(changes)
    c_commit, out_commit, err_commit = run_cmd(["git", "commit", "-m", commit_msg], repo_dir)
    if c_commit != 0:
        print(f"[!] git commit failed: {err_commit}", flush=True)
        return False

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Committed: '{commit_msg}'", flush=True)

    # Push
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚀 Pushing to {remote}/{branch}...", flush=True)
    c_push, out_push, err_push = run_cmd(["git", "push", remote, branch], repo_dir)
    if c_push == 0:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🎉 Successfully pushed to GitHub!", flush=True)
        return True
    else:
        # If upstream branch not set, try with -u
        if "no upstream branch" in err_push or "has no upstream" in err_push:
            c_u, _, _ = run_cmd(["git", "push", "-u", remote, branch], repo_dir)
            if c_u == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🎉 Pushed and set upstream to {remote}/{branch}!", flush=True)
                return True
        print(f"[!] git push failed: {err_push or out_push}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Autonomous Git Sync Daemon for Genius")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    parser.add_argument("--branch", type=str, default="main", help="Target git branch (default: main)")
    parser.add_argument("--remote", type=str, default="origin", help="Target git remote (default: origin)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent

    print(f"⚡ Genius Auto-Git-Sync initialized on {repo_dir} (Target: {args.remote}/{args.branch})")

    if args.once:
        synced = sync_cycle(repo_dir, branch=args.branch, remote=args.remote)
        sys.exit(0 if synced else 1)

    while True:
        try:
            sync_cycle(repo_dir, branch=args.branch, remote=args.remote)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[dim]Auto-Git-Sync stopped by user.[/dim]")
            break
        except Exception as e:
            print(f"[!] Error in sync loop: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
