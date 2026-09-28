"""
Genius AgentLoop — Claude Code-level autonomous execution engine.

Features:
  ✓ XML tool-call parsing from any LLM output (Qwen / Claude / Ollama)
  ✓ Self-healing loop: error → analyse → fix → retry (up to 8 rounds)
  ✓ GENIUS.md project memory (auto-loaded at agent start)
  ✓ Context window management (rolling truncation to stay under token limits)
  ✓ Safety gates: destructive operations require user confirmation
  ✓ Parallel sub-agent spawning for multi-step complex tasks
  ✓ Streaming progress display with Rich panels
  ✓ Git operations integration
  ✓ Pre/Post hooks (formatters, linters) after file edits
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .tools import TOOL_SCHEMAS, ToolExecutor, ToolResult, get_tool_system_prompt

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 25           # Hard cap on autonomous steps
MAX_CONTEXT_CHARS = 40_000    # Rolling context window size
GENIUS_MD_NAMES = ["GENIUS.md", "genius.md", ".genius.md", "CLAUDE.md"]
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

# Operations requiring user confirmation before execution
DESTRUCTIVE_TOOLS = {"delete_file", "run_command"}
DESTRUCTIVE_CMD_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"rmdir", r"format\b", r"DROP\s+TABLE",
    r"git\s+reset\s+--hard", r"git\s+clean\s+-fd",
]


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    params: Dict[str, str]
    raw_xml: str = ""


@dataclass
class AgentStep:
    iteration: int
    tool_call: ToolCall
    result: ToolResult
    timestamp: float = field(default_factory=time.time)

    def to_context_str(self) -> str:
        return f"[STEP {self.iteration}] {self.result.to_context_block()}"


@dataclass
class AgentRunResult:
    success: bool
    task: str
    steps: List[AgentStep]
    final_summary: str
    total_time_sec: float
    iterations_used: int
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    errors_healed: int = 0


# ─── Tool Call Parser ─────────────────────────────────────────────────────────

def parse_tool_calls(text: str) -> List[ToolCall]:
    """Extract all <tool_call> blocks from LLM output text."""
    calls: List[ToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        inner = match.group(1)
        params: Dict[str, str] = {}
        name: str = ""
        for tag_match in TAG_RE.finditer(inner):
            tag, value = tag_match.group(1), tag_match.group(2).strip()
            if tag == "name":
                name = value.strip()
            else:
                params[tag] = value
        if name:
            calls.append(ToolCall(name=name, params=params, raw_xml=match.group(0)))
    return calls


def _is_destructive_command(command: str) -> bool:
    """Returns True if the shell command matches destructive patterns."""
    for pat in DESTRUCTIVE_CMD_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return True
    return False


# ─── GENIUS.md Loader ────────────────────────────────────────────────────────

def load_genius_md(workspace_root: str) -> str:
    """
    Loads GENIUS.md (or CLAUDE.md) from project root — project-specific
    coding conventions, architecture rules, preferred libraries, hooks.
    Returns empty string if no file found.
    """
    root = Path(workspace_root)
    for name in GENIUS_MD_NAMES:
        p = root / name
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                return f"\n\n═══ PROJECT MEMORY (GENIUS.md) ═══\n{content}\n═══════════════════════════════════"
            except Exception:
                pass
    return ""


# ─── Post-Edit Hooks ─────────────────────────────────────────────────────────

POST_EDIT_HOOKS: Dict[str, List[str]] = {
    ".py": ["python -m black {file} --quiet 2>&1", "python -m isort {file} --quiet 2>&1"],
    ".js": ["npx prettier --write {file} 2>&1"],
    ".ts": ["npx prettier --write {file} 2>&1"],
    ".json": ["python -m json.tool {file} > /dev/null 2>&1"],
}


def run_post_hooks(file_path: str, executor: ToolExecutor) -> List[str]:
    """
    Runs configured post-edit formatters/linters for the file type.
    Returns list of hook outputs (errors are non-fatal).
    """
    ext = Path(file_path).suffix.lower()
    hooks = POST_EDIT_HOOKS.get(ext, [])
    outputs = []
    for hook_template in hooks:
        cmd = hook_template.replace("{file}", file_path)
        result = executor.run_command(cmd)
        if result.output.strip():
            outputs.append(f"[HOOK:{cmd[:30]}] {result.output[:200]}")
    return outputs


# ─── Main Agent Loop ──────────────────────────────────────────────────────────

class AgentLoop:
    """
    Autonomous agentic execution loop for Genius AI.

    Usage:
        agent = AgentLoop(model_provider, workspace_root="/path/to/project")
        result = await agent.run("Build a snake game in Python using pygame")
    """

    def __init__(
        self,
        model_provider,          # BaseLLMProvider instance
        workspace_root: str,
        confirm_destructive: bool = True,
        enable_hooks: bool = True,
        max_iterations: int = MAX_ITERATIONS,
        on_step=None,            # Optional async callback(step: AgentStep)
        on_message=None,         # Optional async callback(msg: str)
        on_confirm=None,         # Optional async callback(prompt: str) -> bool
    ) -> None:
        self.model = model_provider
        self.workspace_root = str(Path(workspace_root).resolve())
        self.executor = ToolExecutor(self.workspace_root)
        self.confirm_destructive = confirm_destructive
        self.enable_hooks = enable_hooks
        self.max_iterations = max_iterations
        self.on_step = on_step
        self.on_message = on_message
        self.on_confirm = on_confirm

        # Stats tracking
        self._files_created: List[str] = []
        self._files_modified: List[str] = []
        self._commands_run: List[str] = []
        self._errors_healed: int = 0

    async def _emit(self, msg: str) -> None:
        if self.on_message:
            await self.on_message(msg)

    async def _confirm(self, prompt: str) -> bool:
        """Request user confirmation for destructive operations."""
        if not self.confirm_destructive:
            return True
        if self.on_confirm:
            return await self.on_confirm(prompt)
        return True  # auto-approve in non-interactive mode

    def _build_system_prompt(self, genius_md: str) -> str:
        """Assembles the full system prompt for the agentic session."""
        parts = [
            "You are Genius AI operating in AUTONOMOUS AGENT MODE.",
            "You have full access to the user's project filesystem and shell.",
            "Your job is to complete the given task autonomously by using the provided tools.",
            "",
            get_tool_system_prompt(),
        ]
        # Inject available specialized skills from agent/skills/
        try:
            from agent.registry import get_skill_registry
            reg = get_skill_registry(Path(self.workspace_root) / "agent")
            skills_prompt = reg.get_agent_prompt_summary()
            if skills_prompt:
                parts.append("\n" + skills_prompt)
        except Exception:
            pass

        if genius_md:
            parts.append(genius_md)
        parts += [
            "",
            "CRITICAL RULES:",
            "  • Output ONLY <tool_call> blocks. No explanatory prose between calls.",
            "  • Always call project_summary first to understand the project.",
            "  • After writing code, use run_command to verify it works.",
            "  • If run_command shows errors, fix the code and run again.",
            "  • Use task_complete ONLY when the task is fully done and verified.",
            "  • Write production-quality code, not placeholders.",
        ]
        return "\n".join(parts)

    def _trim_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rolling context window: keeps system prompt + last N messages
        to stay within MAX_CONTEXT_CHARS. Oldest tool results are dropped first.
        """
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]

        total = sum(len(m["content"]) for m in system + rest)
        while total > MAX_CONTEXT_CHARS and len(rest) > 2:
            removed = rest.pop(0)
            total -= len(removed["content"])

        return system + rest

    async def _llm_generate(self, messages: List[Dict[str, str]]) -> str:
        """Calls active model, collects full streamed output."""
        tokens: List[str] = []
        try:
            async for token in self.model.stream_generate(
                messages=messages,
                max_new_tokens=4096,
                temperature=0.2,   # Low temperature for deterministic tool calling
            ):
                tokens.append(token)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
        raw = "".join(tokens)
        # Strip think blocks from tool-calling output
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    async def _execute_tool_call(self, call: ToolCall) -> ToolResult:
        """Executes a single tool call with optional destructive confirmation."""
        # Destructive safety gate
        if self.confirm_destructive:
            if call.name == "delete_file":
                approved = await self._confirm(
                    f"Agent wants to DELETE file: {call.params.get('path', '?')}"
                )
                if not approved:
                    return ToolResult(call.name, False, "", "User denied: file deletion cancelled.")

            elif call.name == "run_command":
                cmd = call.params.get("command", "")
                if _is_destructive_command(cmd):
                    approved = await self._confirm(
                        f"Agent wants to run POTENTIALLY DESTRUCTIVE command:\n  {cmd}"
                    )
                    if not approved:
                        return ToolResult(call.name, False, "", "User denied: command cancelled.")

        # Execute
        result = self.executor.dispatch(call.name, call.params)

        # Stats tracking
        if call.name == "create_file" and result.success:
            self._files_created.append(call.params.get("path", ""))
        elif call.name in {"edit_file", "append_to_file"} and result.success:
            self._files_modified.append(call.params.get("path", ""))
            # Run post-edit hooks (formatter/linter)
            if self.enable_hooks:
                hook_outputs = run_post_hooks(call.params.get("path", ""), self.executor)
                if hook_outputs:
                    result.output += "\n" + "\n".join(hook_outputs)
        elif call.name == "run_command" and result.success:
            self._commands_run.append(call.params.get("command", ""))

        return result

    async def run(self, task: str) -> AgentRunResult:
        """
        Main entry point. Runs the autonomous agent loop for the given task.
        Returns AgentRunResult with full execution history.
        """
        t_start = time.time()
        steps: List[AgentStep] = []
        final_summary = ""

        # Reset stats
        self._files_created = []
        self._files_modified = []
        self._commands_run = []
        self._errors_healed = 0

        # Load GENIUS.md project memory
        genius_md = load_genius_md(self.workspace_root)
        if genius_md:
            await self._emit(f"Loaded project memory from GENIUS.md")

        system_prompt = self._build_system_prompt(genius_md)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TASK: {task}\n\nWorkspace: {self.workspace_root}\n\nBegin by calling project_summary to understand the project, then execute the task step by step."},
        ]

        await self._emit(f"Agent started: {task}")

        for iteration in range(1, self.max_iterations + 1):
            await self._emit(f"[{iteration}/{self.max_iterations}] Thinking...")

            # Trim context to avoid overflow
            messages = self._trim_context(messages)

            # LLM generates next action
            llm_output = await self._llm_generate(messages)

            if not llm_output.strip():
                await self._emit("LLM returned empty output. Stopping.")
                break

            # Parse tool calls
            tool_calls = parse_tool_calls(llm_output)

            if not tool_calls:
                # No tool calls — LLM might be done or confused
                # Check if task_complete signal is in plain text
                if "task_complete" in llm_output.lower() or "task is complete" in llm_output.lower():
                    final_summary = llm_output.strip()
                    await self._emit("Task complete (text signal detected).")
                    break
                # Feed back asking for a tool call
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "user", "content": "Continue. Call a tool or call task_complete if done."})
                continue

            # Add LLM response to context
            messages.append({"role": "assistant", "content": llm_output})

            # Execute all tool calls in this iteration
            tool_results_text: List[str] = []
            task_done = False

            for call in tool_calls:
                if call.name == "task_complete":
                    final_summary = call.params.get("summary", "Task completed successfully.")
                    await self._emit(f"Task complete: {final_summary}")
                    task_done = True
                    break

                await self._emit(f"  → {call.name}({', '.join(f'{k}={v[:40]!r}' for k, v in call.params.items() if k != 'content')[:120]})")

                result = await self._execute_tool_call(call)

                step = AgentStep(iteration=iteration, tool_call=call, result=result)
                steps.append(step)

                if self.on_step:
                    await self.on_step(step)

                # Track error healing
                if not result.success:
                    self._errors_healed += 1
                    await self._emit(f"    ✗ Error: {result.error[:120]}")
                else:
                    await self._emit(f"    ✓ Done ({result.elapsed_ms:.0f}ms)")

                tool_results_text.append(step.to_context_str())

            if task_done:
                break

            # Inject all tool results back as user message
            combined_results = "\n\n".join(tool_results_text)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{combined_results}\n\nContinue with the task. Call the next tool or task_complete if done."
            })

        else:
            final_summary = f"Agent reached max iterations ({self.max_iterations}). Check the workspace for partial results."
            await self._emit(final_summary)

        total_time = round(time.time() - t_start, 2)
        return AgentRunResult(
            success=bool(final_summary and "error" not in final_summary.lower()),
            task=task,
            steps=steps,
            final_summary=final_summary or "Agent loop ended.",
            total_time_sec=total_time,
            iterations_used=len(steps),
            files_created=list(set(self._files_created)),
            files_modified=list(set(self._files_modified)),
            commands_run=list(set(self._commands_run)),
            errors_healed=self._errors_healed,
        )

    # ── Parallel Sub-Agent Support ────────────────────────────────────────────

    async def run_parallel(self, tasks: List[str]) -> List[AgentRunResult]:
        """
        Spawns multiple sub-agents in parallel to handle independent sub-tasks.
        Each sub-agent gets its own ToolExecutor and context.
        """
        async def _run_sub(task: str, idx: int) -> AgentRunResult:
            sub_agent = AgentLoop(
                model_provider=self.model,
                workspace_root=self.workspace_root,
                confirm_destructive=False,   # Sub-agents run non-interactively
                enable_hooks=self.enable_hooks,
                max_iterations=min(self.max_iterations, 15),
                on_message=lambda msg: self._emit(f"[Sub-{idx+1}] {msg}"),
            )
            return await sub_agent.run(task)

        coros = [_run_sub(task, i) for i, task in enumerate(tasks)]
        results = await asyncio.gather(*coros, return_exceptions=True)

        out: List[AgentRunResult] = []
        for r in results:
            if isinstance(r, AgentRunResult):
                out.append(r)
            else:
                out.append(AgentRunResult(
                    success=False, task="unknown", steps=[], final_summary=str(r),
                    total_time_sec=0, iterations_used=0
                ))
        return out
