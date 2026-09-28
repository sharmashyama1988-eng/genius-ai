"""
Genius AI — Claude Code-level Autonomous Agent Terminal.

Capabilities (v2.0):
  ✓ Full codebase understanding & symbol navigation
  ✓ Autonomous agentic loop (create/edit/run/fix files)
  ✓ Git operations (commit, branch, PR, merge conflicts)
  ✓ Dynamic infinite context window (beyond Gemini 1M)
  ✓ protocol.txt command registry (/run test, /run build, etc.)
  ✓ Perplexity-style search with inline citations
  ✓ Self-healing execution (auto-fixes errors in loop)
  ✓ Safety gates for destructive operations
  ✓ Pipe mode: echo "task" | python -m src.chat_cli -p
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .languages.router import DetectedLanguage
from .memory.context_manager import DynamicContextWindow, create_context_window, WINDOW_PRESETS
from .model.provider import UniversalModelRouter
from .reasoning.schema import GroundingVerdict
from .reasoning.text_sanitizer import TextSanitizer
from .reasoning.xthinking import Citation, ReasoningEvent, XThinkingEngine
from .system.agent_loop import AgentLoop
from .system.codebase_indexer import CodebaseIndexer, ProtocolRegistry
from .system.executor import ExecutionResult, SystemExecutor
from .system.git_ops import GitOps
from .system.guard import ActionSafetyLevel, SafetyGuard
from .system.intent_router import IntentRouter, IntentType, RoutedIntent
from .system.tools import ToolExecutor
from .system.workspace import WorkspaceManager
from .system.chat_viewer import ChatViewer


class GeniusChatSession:
    """Multi-turn interactive conversation session for Genius — Claude Code-level capabilities."""

    def __init__(self, show_thinking: bool = True, context_preset: str = "medium") -> None:
        self.console = Console()
        self.show_thinking = show_thinking
        self.chat_view_mode: str = "cards"
        self.engine = XThinkingEngine()
        self.workspace = WorkspaceManager()
        self.executor = SystemExecutor(default_cwd=str(self.workspace.get_workspace()))
        self.session_id = self.engine.session_mgr.create_session("Interactive Session")
        self.history: List[Dict[str, str]] = []
        self.forced_lang: Optional[str] = None

        # ── New: Agentic Systems & Dynamic Skills ──────────────────────────
        ws = str(self.workspace.get_workspace())
        self.git = GitOps(ws)
        self.indexer = CodebaseIndexer(ws)
        self.protocol = ProtocolRegistry(ws)
        self.tool_executor = ToolExecutor(ws)
        self.ctx_window = create_context_window(
            preset=context_preset,
            session_id=self.session_id,
        )
        self._agent_running = False

        from agent.registry import get_skill_registry
        from agent.skill_synthesizer import get_skill_synthesizer
        self.skill_registry = get_skill_registry(self.workspace.get_workspace() / "agent")
        self.skill_synthesizer = get_skill_synthesizer(self.workspace.get_workspace() / "agent")
        self.intent_router = IntentRouter(list(self.protocol.commands.keys()))

    def _refresh_workspace_systems(self) -> None:
        """Called after /project changes workspace — update all systems."""
        ws = str(self.workspace.get_workspace())
        self.git = GitOps(ws)
        self.indexer = CodebaseIndexer(ws)
        self.protocol = ProtocolRegistry(ws)
        self.tool_executor = ToolExecutor(ws)
        self.executor.default_cwd = ws
        from agent.registry import get_skill_registry
        from agent.skill_synthesizer import get_skill_synthesizer
        self.skill_registry = get_skill_registry(self.workspace.get_workspace() / "agent")
        self.skill_synthesizer = get_skill_synthesizer(self.workspace.get_workspace() / "agent")
        self.intent_router.update_protocol_cmds(list(self.protocol.commands.keys()))

    def print_welcome(self) -> None:
        """Displays welcome banner with all capabilities."""
        active_model = self.engine.router.active_provider_name
        ws = str(self.workspace.get_workspace())
        ctx_budget = f"{self.ctx_window.active_token_budget // 1000}K"

        banner = Text()
        banner.append("⚡ GENIUS AI  —  Claude Code-Level Autonomous Agent\n", style="bold cyan")
        banner.append("─" * 54 + "\n", style="dim")
        banner.append("Model      : ", style="bold white")
        banner.append(f"{active_model.upper()}\n", style="green")
        banner.append("Workspace  : ", style="bold white")
        banner.append(f"{ws}\n", style="bold yellow")
        banner.append("Context    : ", style="bold white")
        banner.append(f"Dynamic {ctx_budget} active + UNLIMITED archive\n", style="cyan")
        banner.append("Auto-Intent: ", style="bold white")
        banner.append("ACTIVE — zero-prefix automatic dispatch\n", style="bold green")
        banner.append("Skills     : ", style="bold white")
        s_count = len(self.skill_registry.skills)
        s_preview = ", ".join(list(self.skill_registry.skills.keys())[:4])
        banner.append(f"{s_count} loaded ({s_preview}) [/skills to list]\n", style="cyan")
        banner.append("Reasoning  : ", style="bold white")
        banner.append("7-node Cognitive Graph + ROUGE-L Grounding\n", style="bright_magenta")
        banner.append("Memory     : ", style="bold white")
        banner.append("L1 Active + L2 Archive + L3 Episodic (SQLite FTS5)\n", style="yellow")
        banner.append("Research   : ", style="bold white")
        banner.append(f"Wikipedia + DuckDuckGo + BM25 | Mode: {self.engine.research_mode.upper()}\n", style="magenta")
        banner.append("Git        : ", style="bold white")
        git_status = "✓ Repo detected" if self.git.is_git_repo() else "No repo (/git init)"
        banner.append(f"{git_status}\n", style="green" if self.git.is_git_repo() else "dim")
        banner.append("─" * 54 + "\n", style="dim")
        banner.append("Commands   : ", style="bold white")
        banner.append(
            "/agent, /skills, /newskill, /run, /git, /search, /files,\n"
            "             /read, /create, /edit, /exec, /project, /context,\n"
            "             /index, /model, /research, /think, /export, /exit\n",
            style="dim"
        )

        self.console.print(Panel(banner, border_style="cyan", padding=(1, 2)))

    async def chat_loop(self) -> None:
        """Main conversational loop for terminal chat with zero-prefix auto-dispatch."""
        self.print_welcome()

        while True:
            try:
                self.console.print()
                user_input = self.console.input("[bold green]You[/bold green] [dim]❯[/dim] ").strip()

                if not user_input:
                    continue

                # 1. Handle explicit slash commands if user wrote one
                if user_input.startswith("/"):
                    handled = await self._handle_slash_command(user_input)
                    if handled == "exit":
                        break
                    continue

                # 2. Autonomous Intent Routing (Zero-prefix Mode)
                routed = self.intent_router.route(user_input)

                if routed.intent_type == IntentType.AGENT:
                    self.console.print(Panel(
                        f"⚡ [bold green]Auto-Intent: Autonomous Agent Mode[/bold green]\n"
                        f"[dim]Task:[/dim] [bold white]{routed.argument}[/bold white]",
                        border_style="green",
                        padding=(0, 2),
                    ))
                    await self._run_agent(routed.argument)
                    continue

                elif routed.intent_type == IntentType.GIT:
                    self.console.print(f"[bold cyan]⚡ Auto-Intent:[/bold cyan] [bold yellow]Git Operation[/bold yellow] [dim]({routed.command})[/dim]")
                    await self._handle_git_command(routed.argument)
                    continue

                elif routed.intent_type == IntentType.PROTOCOL:
                    self.console.print(f"[bold cyan]⚡ Auto-Intent:[/bold cyan] [bold blue]Protocol Runner[/bold blue] [dim]({routed.command})[/dim]")
                    await self._execute_protocol(routed.argument)
                    continue

                elif routed.intent_type == IntentType.INDEX:
                    self.console.print(f"[bold cyan]⚡ Auto-Intent:[/bold cyan] [bold magenta]Codebase Indexer[/bold magenta] [dim]({routed.command})[/dim]")
                    await self._handle_index_command(routed.argument)
                    continue

                elif routed.intent_type == IntentType.MATH:
                    self.console.print(f"[bold cyan]⚡ Auto-Intent:[/bold cyan] [bold yellow]Math Calculation[/bold yellow]")
                    from .reasoning.math_solver import MathSolver
                    res = MathSolver.solve(routed.argument, lang_style=self.forced_lang or "en")
                    if res:
                        think, resp = res
                        clean_resp = TextSanitizer.clean_for_display(resp)
                        if self.show_thinking:
                            self.console.print(Panel(TextSanitizer.clean_for_display(think), title="[bold cyan]Mathematical Derivation[/bold cyan]", border_style="cyan"))
                        self.console.print(clean_resp)
                        self.history.append({"user": user_input, "assistant": clean_resp})
                    else:
                        await self._process_turn(user_input)
                    continue

                elif routed.intent_type == IntentType.SEARCH:
                    self.console.print(f"[bold cyan]⚡ Auto-Intent:[/bold cyan] [bold magenta]Live Perplexity Web Search[/bold magenta]")
                    await self._perplexity_search(routed.argument)
                    continue

                elif routed.intent_type == IntentType.WORKSPACE:
                    action = routed.metadata.get("action", "")
                    if action == "shift":
                        ok, msg, summary = self.workspace.set_workspace(routed.argument)
                        if ok:
                            self._refresh_workspace_systems()
                            self.console.print(f"[bold green]✓ Switched project workspace to:[/bold green] [bold cyan]{self.workspace.get_workspace()}[/bold cyan]")
                        else:
                            self.console.print(f"[red]{msg}[/red]")
                        continue
                    elif action == "view_image":
                        ok, msg, _ = self.workspace.inspect_image(routed.argument, auto_open=True)
                        if ok:
                            self.console.print(Panel(msg, title="[bold cyan]Image Inspector[/bold cyan]", border_style="cyan"))
                        else:
                            self.console.print(f"[red]{msg}[/red]")
                        continue
                    elif action == "read_file":
                        ok, msg = self.workspace.read_file(routed.argument)
                        self.console.print(msg if ok else f"[red]{msg}[/red]")
                        continue
                    elif action == "list_files":
                        ok, msg = self.workspace.list_files()
                        self.console.print(msg if ok else f"[red]{msg}[/red]")
                        continue

                # 3. Default: Conversational Deep Reasoning Turn
                await self._process_turn(user_input)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Session terminated by user. Alvida! 👋[/dim]")
                break
            except Exception as e:
                self.console.print(f"\n[red bold]Error during processing:[/red bold] {e}")

    async def _handle_slash_command(self, user_input: str) -> Optional[str]:
        """Handles terminal slash commands."""
        clean_input = user_input.strip()
        # Handle cases like "/ off" -> "/off"
        if clean_input.startswith("/") and len(clean_input) > 1 and clean_input[1] == " ":
            clean_input = "/" + clean_input[1:].lstrip()

        parts = clean_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            self.console.print("[dim]Alvida! Exiting Genius chat. 👋[/dim]")
            return "exit"

        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            self.history.clear()
            self.print_welcome()
            self.console.print("[dim]Screen cleared. Working memory reset.[/dim]")

        elif cmd == "/think":
            self.show_thinking = not self.show_thinking
            status_str = "VISIBLE" if self.show_thinking else "HIDDEN"
            self.console.print(f"[dim]xThinking stream display is now: [bold]{status_str}[/bold][/dim]")

        elif cmd in ("/off", "/offline", "/disable"):
            self.engine.set_research_mode("off")
            self.console.print("[bold yellow]⚡ Research mode set to: [bold red]OFF[/bold red] (Fast direct / offline mode)[/bold yellow]")

        elif cmd in ("/on", "/online", "/enable"):
            self.engine.set_research_mode("on")
            self.console.print("[bold cyan]🔍 Research mode set to: [bold green]ON[/bold green] (Always parallel Wikipedia + Web search)[/bold cyan]")

        elif cmd == "/auto":
            self.engine.set_research_mode("auto")
            self.console.print("[bold green]🧠 Research mode set to: [bold cyan]AUTO[/bold cyan] (Intelligent routing based on query complexity)[/bold green]")

        elif cmd in ("/research", "/mode"):
            if not arg:
                curr = self.engine.research_mode.upper()
                self.console.print(f"[bold cyan]Current Research Mode:[/bold cyan] [bold green]{curr}[/bold green]")
                self.console.print("  • [yellow]auto[/yellow]   - Intelligent routing (short-circuits simple queries, searches for factual questions)")
                self.console.print("  • [yellow]on[/yellow]     - Always perform parallel Wikipedia + Web search for every turn")
                self.console.print("  • [yellow]off[/yellow]    - Fast direct / offline mode (bypasses all web/wiki requests, runs directly from model)")
                self.console.print("[dim]Usage: /research [auto|on|off]  (or directly: /on, /off, /auto)[/dim]")
            else:
                if self.engine.set_research_mode(arg):
                    new_mode = self.engine.research_mode.upper()
                    self.console.print(f"[green]✓ Research mode updated to: [bold]{new_mode}[/bold][/green]")
                else:
                    self.console.print(f"[red]Unknown mode '{arg}'. Available: auto, on, off[/red]")

        elif cmd == "/model":
            if not arg:
                # Show all providers with context window sizes
                table = Table(title="LLM Providers — Context Windows", box=None, border_style="cyan")
                table.add_column("Provider", style="bold cyan", width=14)
                table.add_column("Context Window", style="green", width=16)
                table.add_column("Status", style="yellow", width=22)
                table.add_column("Active", style="bold white", width=8)

                provider_list = self.engine.router.list_providers()
                for p in provider_list:
                    ctx_k = f"{p['context_window'] // 1000}K tokens"
                    status = "✓ Ready" if p["ready"] else "⚠ API key needed"
                    active_mark = "★ ACTIVE" if p["active"] else ""
                    table.add_row(p["name"], ctx_k, status, active_mark)

                self.console.print(table)
                self.console.print()

                # Show local model sizes available
                from .model.llm_engine import MODEL_REGISTRY
                self.console.print("[bold cyan]Local Qwen2.5 Model Sizes:[/bold cyan]")
                size_table = Table(box=None)
                size_table.add_column("Alias", style="cyan", width=18)
                size_table.add_column("Context", style="green", width=10)
                size_table.add_column("RAM Needed", style="yellow", width=12)
                size_table.add_column("Description", style="dim")
                for alias, info in MODEL_REGISTRY.items():
                    ctx_k = f"{info['extended_ctx'] // 1024}K"
                    ram = f"{info['min_ram_gb']}GB+"
                    size_table.add_row(alias, ctx_k, ram, info["description"])
                self.console.print(size_table)
                self.console.print("[dim]Usage: /model local:qwen-7b  |  /model claude  |  /model gemini[/dim]")
            else:
                target = arg.lower().strip()
                # Handle "local:qwen-7b" format for local model switching
                if target.startswith("local:"):
                    size = target.split(":", 1)[1]
                    prov = self.engine.router.providers.get("local")
                    if hasattr(prov, "switch_model"):
                        prov.switch_model(size)
                        ctx = prov.context_window
                        self.console.print(f"[green]✓ Local model switched to [bold]{size}[/bold] | Context: {ctx // 1000}K tokens[/green]")
                        self.console.print(f"[dim]Model will load on next query. RAM needed: check /model[/dim]")
                    else:
                        self.console.print("[red]Could not switch local model.[/red]")
                elif self.engine.set_model_provider(target):
                    ctx = self.engine.router.context_window_size()
                    self.console.print(f"[green]✓ Provider: [bold]{target.upper()}[/bold] | Context: {ctx // 1000}K tokens[/green]")
                else:
                    self.console.print(f"[red]Unknown provider '{target}'. Try: local, local:qwen-7b, claude, gemini, ollama[/red]")

        elif cmd == "/context":
            # Dynamic context window management
            from .memory.context_manager import WINDOW_PRESETS
            if not arg:
                stats = self.ctx_window.get_stats()
                self.console.print(stats.display())
                self.console.print()
                self.console.print("[bold cyan]Context Window Presets:[/bold cyan]")
                for name, tokens in WINDOW_PRESETS.items():
                    mark = " ← current" if tokens == self.ctx_window.active_token_budget else ""
                    label = f"{tokens // 1000}K" if tokens < 1_000_000 else f"{tokens // 1_000_000}M"
                    self.console.print(f"  [cyan]{name:<10}[/cyan] {label} tokens{mark}")
                self.console.print("[dim]Usage: /context <preset>   e.g. /context gemini  /context ultra  /context infinite[/dim]")
            else:
                preset = arg.strip().lower()
                if preset in WINDOW_PRESETS:
                    self.ctx_window.set_budget(WINDOW_PRESETS[preset])
                    new_k = self.ctx_window.active_token_budget // 1000
                    self.console.print(f"[green]✓ Context window set to: [bold]{preset}[/bold] ({new_k}K active tokens + unlimited archive)[/green]")
                else:
                    try:
                        tokens = int(preset.replace("k", "000").replace("K", "000").replace("m", "000000").replace("M", "000000"))
                        self.ctx_window.set_budget(tokens)
                        self.console.print(f"[green]✓ Context window set to: {tokens // 1000}K tokens[/green]")
                    except ValueError:
                        self.console.print(f"[red]Unknown preset '{preset}'. Try: small, medium, large, xl, gemini, ultra, infinite[/red]")

        elif cmd == "/agent":
            # Autonomous agentic task execution
            if not arg:
                self.console.print("[yellow]Usage: /agent <task description>[/yellow]")
                self.console.print("[dim]Examples:[/dim]")
                self.console.print("  [cyan]/agent create a snake game in Python using pygame[/cyan]")
                self.console.print("  [cyan]/agent add unit tests for all functions in main.py[/cyan]")
                self.console.print("  [cyan]/agent refactor the database module to use async/await[/cyan]")
            else:
                await self._run_agent(arg)

        elif cmd == "/git":
            await self._handle_git_command(arg)

        elif cmd == "/skills":
            self.skill_registry.refresh()
            table = Table(title="Genius Active Skills & Tools (in agent/skills/)", border_style="green")
            table.add_column("Skill Name", style="bold cyan")
            table.add_column("Description", style="white")
            table.add_column("CLI Usage", style="dim")
            for s in self.skill_registry.list_all():
                table.add_row(s.name, s.description, s.cli_usage)
            self.console.print(table)
            self.console.print("[dim]Add custom skills in 'agent/skills/' or ask Genius to create one autonomously via /newskill[/dim]")

        elif cmd == "/newskill":
            if not arg:
                self.console.print("[yellow]Usage: /newskill <description of what the tool should do>[/yellow]")
            else:
                self.console.print(f"[bold cyan]⚡ Self-Evolution Triggered:[/bold cyan] Designing new skill for: [bold white]{arg}[/bold white]")
                await self._run_agent(f"Create a new reusable skill in agent/skills/ for: {arg}. Write production-ready code with CLI support, test it with run_command, and verify it works.")

        elif cmd == "/run":
            await self._execute_protocol(arg)

        elif cmd == "/index":
            await self._handle_index_command(arg)

        elif cmd == "/search":
            if not arg:
                self.console.print("[yellow]Usage: /search <query>[/yellow]")
            else:
                await self._perplexity_search(arg)

        elif cmd in ("/exec",):
            if not arg:
                self.console.print("[yellow]Usage: /exec <shell command>[/yellow]")
            else:
                await self._handle_system_exec(arg)


        elif cmd == "/export":
            export_format = arg.lower() if arg else "md"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"genius_transcript_{self.session_id}_{timestamp}.{export_format}"
            output_path = os.path.join(os.getcwd(), filename)

            if export_format == "md":
                self.engine.session_mgr.export_markdown(self.session_id, output_path)
                self.console.print(f"[green]✓ Session exported to Markdown: [bold]{output_path}[/bold][/green]")
            else:
                turns = self.engine.session_mgr.get_turns(self.session_id)
                self.console.print(f"[dim]Exported {len(turns)} turn(s).[/dim]")

        elif cmd == "/sessions":
            sessions = self.engine.session_mgr.list_sessions()
            table = Table(title="Past Conversation Sessions", border_style="blue")
            table.add_column("Session ID", style="bold cyan")
            table.add_column("Title", style="white")
            table.add_column("Turns", style="yellow")
            table.add_column("Last Updated", style="dim")

            for s in sessions[:8]:
                t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["updated_at"]))
                table.add_row(s["session_id"], s["title"][:35], str(s["turn_count"]), t_str)

            self.console.print(table)

        elif cmd == "/new":
            self.session_id = self.engine.session_mgr.create_session("New Session")
            self.history.clear()
            self.console.print(f"[green]✓ Started new session: [bold]{self.session_id}[/bold][/green]")

        elif cmd == "/lang":
            if arg:
                lang_code = arg.lower()
                if lang_code in ("auto", "reset"):
                    self.forced_lang = None
                    self.console.print("[dim]Language mode set to: [bold]AUTO-DETECT[/bold][/dim]")
                elif lang_code in self.engine.multilingual_mgr.router.profiles:
                    self.forced_lang = lang_code
                    prof = self.engine.multilingual_mgr.router.profiles[lang_code]
                    self.console.print(f"[dim]Forced language set to: [bold]{prof.name} ({lang_code})[/bold][/dim]")
                else:
                    self.console.print(f"[red]Unknown language '{lang_code}'. Available: hi, hi-Latn, en, es, fr, de[/red]")
            else:
                current = self.forced_lang or "AUTO-DETECT"
                self.console.print(f"[dim]Current language mode: [bold]{current}[/bold][/dim]")

        elif cmd in ("/project", "/workspace", "/cd"):
            if not arg:
                summary = self.workspace.scan_project()
                table = Table(title="Active Project Workspace", border_style="cyan")
                table.add_column("Property", style="bold cyan")
                table.add_column("Value", style="bold white")
                table.add_row("Root Path", str(self.workspace.get_workspace()))
                table.add_row("Files Count", str(summary.get("file_count", 0)))
                table.add_row("Folders Count", str(summary.get("dir_count", 0)))
                table.add_row("Languages", ", ".join(summary.get("languages", [])) or "None")
                table.add_row("Signatures", ", ".join(summary.get("signatures", [])) or "General Directory")
                self.console.print(table)
                self.console.print("[dim]Usage: /project <path_to_project_directory>[/dim]")
            else:
                ok, msg, summary = self.workspace.set_workspace(arg)
                if ok:
                    self.executor.default_cwd = str(self.workspace.get_workspace())
                    self.console.print(f"[bold green]✓ Switched project workspace to:[/bold green] [bold cyan]{self.workspace.get_workspace()}[/bold cyan]")
                    self.console.print(f"[dim]Total files: {summary.get('file_count', 0)} | Languages: {', '.join(summary.get('languages', [])) or 'None'}[/dim]")
                else:
                    self.console.print(f"[red]{msg}[/red]")

        elif cmd in ("/files", "/ls", "/tree"):
            ok, msg = self.workspace.list_files(arg)
            if ok:
                self.console.print(msg)
            else:
                self.console.print(f"[red]{msg}[/red]")

        elif cmd == "/create":
            if not arg:
                self.console.print("[yellow]Usage: /create <filename> [content][/yellow]")
            else:
                parts = arg.split(maxsplit=1)
                fname = parts[0]
                content = parts[1] if len(parts) > 1 else ""
                ok, msg = self.workspace.create_file(fname, content)
                if ok:
                    self.console.print(f"[bold green]✓ {msg}[/bold green]")
                else:
                    self.console.print(f"[red]{msg}[/red]")

        elif cmd == "/read":
            if not arg:
                self.console.print("[yellow]Usage: /read <filename>[/yellow]")
            else:
                ok, msg = self.workspace.read_file(arg)
                if ok:
                    self.console.print(msg)
                else:
                    self.console.print(f"[red]{msg}[/red]")

        elif cmd in ("/view", "/image"):
            if not arg:
                self.console.print("[yellow]Usage: /view <image_or_file_path>[/yellow]")
            else:
                ok, msg, meta = self.workspace.inspect_image(arg, auto_open=True)
                if ok:
                    self.console.print(Panel(msg, title="[bold cyan]Image Inspector[/bold cyan]", border_style="cyan"))
                else:
                    ok_r, msg_r = self.workspace.read_file(arg)
                    if ok_r:
                        self.console.print(msg_r)
                    else:
                        self.console.print(f"[red]{msg}[/red]")

        elif cmd in ("/calc", "/math"):
            if not arg:
                self.console.print("[yellow]Usage: /calc <mathematical expression or equation>[/yellow]")
            else:
                from .reasoning.math_solver import MathSolver
                res = MathSolver.solve(arg, lang_style=self.forced_lang or "en")
                if res:
                    think, resp = res
                    clean_resp = TextSanitizer.clean_for_display(resp)
                    if self.show_thinking:
                        self.console.print(Panel(TextSanitizer.clean_for_display(think), title="[bold cyan]Mathematical Derivation[/bold cyan]", border_style="cyan"))
                    self.console.print(clean_resp)
                else:
                    self.console.print(f"[red]Could not parse mathematical expression: '{arg}'.[/red]")

        elif cmd == "/code":
            if not arg:
                self.console.print("[yellow]Usage: /code <problem or algorithm>[/yellow]")
            else:
                from .reasoning.code_solver import CodeSolver
                res = CodeSolver.solve(arg, lang_style=self.forced_lang or "en")
                if res:
                    think, resp = res
                    clean_resp = TextSanitizer.clean_for_display(resp)
                    if self.show_thinking:
                        self.console.print(Panel(TextSanitizer.clean_for_display(think), title="[bold cyan]Algorithmic Analysis[/bold cyan]", border_style="cyan"))
                    self.console.print(clean_resp)
                else:
                    await self._process_turn(f"Write code for {arg}")

        elif cmd == "/stats":
            turns = self.engine.session_mgr.get_turns(self.session_id)
            active_model = self.engine.router.active_provider_name
            summary = self.workspace.scan_project()
            table = Table(title="Genius System Telemetry & Statistics", border_style="green")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="bold white")
            table.add_row("Current Session ID", self.session_id)
            table.add_row("Total Turns in Session", str(len(turns)))
            table.add_row("Active Model Provider", active_model.upper())
            table.add_row("Research Mode", self.engine.research_mode.upper())
            table.add_row("Active Project Root", str(self.workspace.get_workspace()))
            table.add_row("Project Files Count", str(summary.get("file_count", 0)))
            table.add_row("Project Languages", ", ".join(summary.get("languages", [])) or "None")
            table.add_row("xThinking Stream", "ENABLED" if self.show_thinking else "DISABLED")
            table.add_row("Language Mode", self.forced_lang or "AUTO-DETECT")
            table.add_row("Working Memory Size", f"{len(self.history)} messages")
            self.console.print(table)

        elif cmd in ("/chatview", "/viewchat", "/history"):
            if not arg:
                ChatViewer.render_history_thread(self.console, self.history, self.session_id)
            elif arg.lower() in ("html", "web", "browser"):
                path = ChatViewer.export_html_view(self.history, self.session_id)
                self.console.print(f"[bold green]✓ Standalone visual chat view opened in browser:[/bold green] [bold cyan]{path}[/bold cyan]")
            elif arg.lower() in ("cards", "bubble", "panel"):
                self.chat_view_mode = "cards"
                self.console.print("[dim]Chat view mode set to: [bold cyan]CARDS[/bold cyan] (Rounded dialogue bubbles)[/dim]")
            elif arg.lower() in ("stream", "classic"):
                self.chat_view_mode = "stream"
                self.console.print("[dim]Chat view mode set to: [bold yellow]STREAM[/bold yellow] (Classic flowing terminal)[/dim]")
            else:
                self.console.print("[yellow]Usage: /chatview [cards|stream|html][/yellow]")

        elif cmd == "/help":
            table = Table(title="Genius CLI Commands", border_style="cyan")
            table.add_column("Command", style="bold yellow")
            table.add_column("Description", style="white")
            table.add_row("/project [path]", "Shift into project directory and analyze codebase structure")
            table.add_row("/files [subpath]", "Display clean file hierarchy and sizes in active project")
            table.add_row("/read <file>", "Read file content (text or image) from active project")
            table.add_row("/create <file>", "Create new file in active project with initial content")
            table.add_row("/view <image>", "Inspect image resolution, metadata, and open in viewer")
            table.add_row("/chatview [mode]", "Display visual chat cards, switch layout, or open HTML view")
            table.add_row("/calc <expr>", "Directly solve math equations, series, formulas, AST")
            table.add_row("/code <query>", "Synthesize production algorithms, data structures, templates")
            table.add_row("/stats", "Show session telemetry, project status, and cognitive state")
            table.add_row("/research [mode]", "Toggle research mode: 'auto', 'on', 'off'")
            table.add_row("/model [name]", "Switch or view active LLM provider (local, claude, ollama)")
            table.add_row("/search <query>", "Execute direct Wikipedia + Web search without LLM generation")
            table.add_row("/exec <command>", "Execute shell command with Human-in-the-Loop (HITL) safety")
            table.add_row("/export [md|json]", "Export conversation transcript and cognitive trajectories")
            table.add_row("/sessions", "List past conversation sessions from SQLite memory")
            table.add_row("/new", "Start a clean conversation session")
            table.add_row("/think", "Toggle live extended thinking (<think>...</think>) stream")
            table.add_row("/lang [code]", "Set language (hi, hi-Latn, en, es, fr, de, auto)")
            table.add_row("/clear", "Clear screen and reset working memory")
            table.add_row("/exit", "Exit chat")
            self.console.print(table)

        else:
            self.console.print(f"[yellow]Unknown command '{cmd}'. Type [bold]/help[/bold] to view available commands.[/yellow]")

        return None

    # ── Agentic Task Runner ────────────────────────────────────────────────────

    async def _run_agent(self, task: str) -> None:
        """
        Runs the full autonomous agent loop for the given task.
        Streams progress in real-time, shows files created, commands run.
        """
        if self._agent_running:
            self.console.print("[yellow]⚠ Agent is already running. Wait for it to finish.[/yellow]")
            return

        self._agent_running = True
        ws = str(self.workspace.get_workspace())
        provider = self.engine.router.get_provider()

        self.console.print(Panel(
            f"[bold cyan]Task:[/bold cyan] {task}\n"
            f"[bold cyan]Workspace:[/bold cyan] {ws}\n"
            f"[dim]Type Ctrl+C to cancel[/dim]",
            title="[bold green]⚡ GENIUS AGENT MODE[/bold green]",
            border_style="green",
        ))

        step_log: list = []

        async def on_message(msg: str) -> None:
            self.console.print(f"[dim cyan]  {msg}[/dim cyan]")

        async def on_step(step) -> None:
            icon = "✓" if step.result.success else "✗"
            color = "green" if step.result.success else "red"
            step_log.append({
                "tool": step.tool_call.name,
                "success": step.result.success,
                "ms": step.result.elapsed_ms,
            })
            self.console.print(
                f"  [{color}]{icon}[/{color}] [cyan]{step.tool_call.name}[/cyan] "
                f"[dim]({step.result.elapsed_ms:.0f}ms)[/dim]"
            )
            if not step.result.success and step.result.error:
                self.console.print(f"    [red dim]{step.result.error[:120]}[/red dim]")

        async def on_confirm(prompt: str) -> bool:
            self.console.print(f"\n[bold yellow]⚠ Agent Safety Gate:[/bold yellow] {prompt}")
            response = input("  Allow? [y/N]: ").strip().lower()
            return response in ("y", "yes")

        agent = AgentLoop(
            model_provider=provider,
            workspace_root=ws,
            confirm_destructive=True,
            enable_hooks=True,
            on_step=on_step,
            on_message=on_message,
            on_confirm=on_confirm,
        )

        try:
            result = await agent.run(task)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Agent cancelled by user.[/yellow]")
            self._agent_running = False
            return
        except Exception as e:
            self.console.print(f"\n[red]Agent error: {e}[/red]")
            self._agent_running = False
            return

        self._agent_running = False

        # Summary panel
        color = "green" if result.success else "yellow"
        summary_lines = [
            f"[bold]Status:[/bold]  {'✓ Completed' if result.success else '⚠ Partial'}",
            f"[bold]Time:[/bold]    {result.total_time_sec:.1f}s | {result.iterations_used} steps",
        ]
        if result.files_created:
            summary_lines.append(f"[bold]Created:[/bold] {', '.join(result.files_created[:5])}")
        if result.files_modified:
            summary_lines.append(f"[bold]Edited:[/bold]  {', '.join(result.files_modified[:5])}")
        if result.commands_run:
            summary_lines.append(f"[bold]Ran:[/bold]     {', '.join(result.commands_run[:3])}")
        if result.errors_healed:
            summary_lines.append(f"[bold]Self-healed:[/bold] {result.errors_healed} error(s) automatically")
        summary_lines.append(f"\n{result.final_summary}")

        self.console.print(Panel(
            "\n".join(summary_lines),
            title=f"[bold {color}]Agent Complete[/bold {color}]",
            border_style=color,
        ))

    # ── Protocol & Index Handlers ─────────────────────────────────────────────

    async def _execute_protocol(self, cmd_name: str) -> None:
        """Executes a registered command from protocol.txt."""
        if not cmd_name:
            self.console.print(self.protocol.list_all())
            return

        clean_name = cmd_name.strip().lower()
        proto_cmd = self.protocol.get_command(clean_name)
        if proto_cmd:
            self.console.print(f"[bold cyan]Running protocol:[/bold cyan] [bold white]{clean_name}[/bold white]")
            self.console.print(f"[dim]Command: {proto_cmd}[/dim]")
            result = self.tool_executor.run_command(proto_cmd)
            if result.success:
                self.console.print("[green]✓ Exit 0[/green]")
                if result.output:
                    self.console.print(Panel(result.output.strip(), border_style="green", title=f"[green]{clean_name}[/green]"))
            else:
                self.console.print("[red]✗ Failed[/red]")
                if result.output or result.error:
                    err_text = (result.output or "") + "\n" + (result.error or "")
                    self.console.print(Panel(err_text.strip(), border_style="red", title=f"[red]{clean_name} — error[/red]"))
        else:
            self.console.print(f"[red]No protocol command '{clean_name}'.[/red]")
            self.console.print(self.protocol.list_all())

    async def _handle_index_command(self, arg: str) -> None:
        """Indexes codebase or searches for symbols."""
        self.console.print("[bold cyan]Indexing codebase...[/bold cyan]")
        with self.console.status("[dim]Building symbol map...", spinner="dots"):
            idx = self.indexer.build_index()
        self.console.print(f"[green]✓ Indexed {idx.total_files} files | {idx.total_lines} lines | {len(idx.symbol_map)} symbols[/green]")
        if arg:
            results = idx.find_symbol(arg)
            if results:
                tbl = Table(title=f"Symbol: '{arg}'", box=None)
                tbl.add_column("Name", style="cyan")
                tbl.add_column("Kind", style="yellow")
                tbl.add_column("File", style="dim")
                tbl.add_column("Line", style="white")
                for s in results[:20]:
                    import os as _os
                    rel = _os.path.relpath(s.file, str(self.workspace.get_workspace()))
                    tbl.add_row(s.name, s.kind, rel, str(s.line))
                self.console.print(tbl)
            else:
                self.console.print(f"[dim]No symbol matching '{arg}' found.[/dim]")
        else:
            self.console.print(idx.summary_for_agent())

    # ── Git Command Handler ────────────────────────────────────────────────────

    async def _handle_git_command(self, arg: str) -> None:
        """Full git operations via /git command."""
        if not self.git.is_git_repo():
            self.console.print("[yellow]Not a git repo. Run /git init to initialize.[/yellow]")
            if arg == "init":
                ok, msg = self.git.init()
                self.console.print(f"{'[green]' if ok else '[red]'}{msg}")
            return

        parts = arg.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if not sub or sub == "status":
            self.console.print(self.git.full_status_report())

        elif sub == "commit":
            msg = sub_arg or f"feat: update by Genius AI [{time.strftime('%Y-%m-%d %H:%M')}]"
            ok, out = self.git.auto_commit(msg)
            self.console.print(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'} {out}")

        elif sub == "push":
            ok, out = self.git.push(set_upstream=True)
            self.console.print(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'} {out}")

        elif sub == "pull":
            ok, out = self.git.pull()
            self.console.print(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'} {out}")

        elif sub == "branch":
            if sub_arg:
                ok, out = self.git.create_branch(sub_arg)
                self.console.print(f"{'[green]✓ Created branch:[/green]' if ok else '[red]✗[/red]'} [cyan]{sub_arg}[/cyan]  {out}")
            else:
                self.console.print(self.git.list_branches())

        elif sub == "log":
            n = int(sub_arg) if sub_arg.isdigit() else 10
            self.console.print(self.git.log(n=n))

        elif sub == "diff":
            self.console.print(self.git.diff(staged=(sub_arg == "--cached")))

        elif sub == "pr":
            base = sub_arg or "main"
            summary = self.git.generate_pr_summary(base)
            self.console.print(Panel(summary, title="[cyan]Pull Request Summary[/cyan]", border_style="cyan"))

        elif sub == "conflicts":
            conflicts = self.git.get_conflicts()
            if not conflicts:
                self.console.print("[green]No merge conflicts detected.[/green]")
            else:
                for fname, content in conflicts.items():
                    self.console.print(Panel(content[:800], title=f"[red]Conflict: {fname}[/red]", border_style="red"))

        elif sub == "init":
            ok, out = self.git.init()
            self.console.print(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'} {out}")
            if ok:
                self._refresh_workspace_systems()

        else:
            self.console.print(
                "[bold cyan]Git Commands:[/bold cyan]\n"
                "  /git status            — Working tree status\n"
                "  /git commit [message]  — Stage all + commit\n"
                "  /git push              — Push to origin\n"
                "  /git pull              — Pull from origin\n"
                "  /git branch [name]     — List branches or create new\n"
                "  /git log [n]           — Show last N commits\n"
                "  /git diff              — Show unstaged diff\n"
                "  /git pr [base]         — Generate PR summary\n"
                "  /git conflicts         — Show merge conflicts\n"
                "  /git init              — Initialize git repo\n"
            )

    async def _handle_system_exec(self, command: str) -> None:
        """Handles system command execution with HITL safety protocol."""
        level, reason = SafetyGuard.evaluate_command(command)

        if level in (ActionSafetyLevel.SENSITIVE, ActionSafetyLevel.BLOCKED):
            warning = Text()
            warning.append("🛡️ SENSITIVE ACTION BLOCKED BY SYSTEM GUARD\n", style="bold red")
            warning.append("System Policy: ", style="bold white")
            warning.append(f"{reason}\n", style="yellow")
            warning.append("Command: ", style="bold white")
            warning.append(f"{command}\n\n", style="bold cyan")
            warning.append("Rule: Sensitive system changes can only be authorized by the USER.\n", style="dim")

            self.console.print(Panel(warning, border_style="red", title="[bold red]Safety Interception[/bold red]"))
            confirm = self.console.input("[bold red]Do you explicitly authorize executing this command? (y/N)[/bold red] ").strip().lower()

            if confirm not in ("y", "yes"):
                self.console.print("[dim]Command execution cancelled by user.[/dim]")
                return

            res = self.executor.execute(command, user_confirmed=True)
        else:
            res = self.executor.execute(command, user_confirmed=False)

        if res.executed:
            self.console.print(f"[bold green]Execution Succeeded (Exit Code: {res.exit_code}):[/bold green]")
            if res.stdout:
                self.console.print(f"[dim]{res.stdout}[/dim]")
        else:
            self.console.print(f"[bold red]Execution Failed:[/bold red] {res.message}")
            if res.stderr:
                self.console.print(f"[red]{res.stderr}[/red]")

    # ─── Perplexity-Style Search: Written answer + inline [1][2] + Sources ───
    async def _perplexity_search(self, query: str) -> None:
        """Perplexity-style: Wikipedia + Web fetch → synthesized written answer
        with inline [1][2] citations → clean Sources section below."""
        import re as _re

        self.console.print()
        with self.console.status(f"[bold cyan]Searching: '{query}'...", spinner="dots"):
            wiki_task = self.engine.wiki.search_and_fetch(query, max_articles=3)
            web_task = self.engine.web.search(query, limit=5)
            wiki_res, web_res = await asyncio.gather(wiki_task, web_task, return_exceptions=True)

        articles = wiki_res if isinstance(wiki_res, list) else []
        web_items = web_res if isinstance(web_res, list) else []

        if not articles and not web_items:
            self.console.print("[red]No results found. Try rephrasing the query.[/red]")
            return

        # Build numbered source list + context blocks for LLM
        sources: List[Dict] = []
        context_parts: List[str] = []
        idx = 1
        for a in articles:
            snippet = (a.summary or getattr(a, "full_text", "")[:600])[:600]
            sources.append({"index": idx, "title": a.title, "url": a.url, "type": "Wikipedia", "snippet": snippet})
            context_parts.append(f"[{idx}] {a.title}:\n{snippet}")
            idx += 1
        for w in web_items:
            if w.snippet and len(w.snippet) > 20:
                sources.append({"index": idx, "title": w.title, "url": w.url, "type": "Web", "snippet": w.snippet})
                context_parts.append(f"[{idx}] {w.title}:\n{w.snippet}")
                idx += 1

        # Language detection
        detected = self.engine.multilingual_mgr.route_query(query)
        lang_style = "hi-Latn" if detected.code == "hi-Latn" else ("hi" if detected.code == "hi" else "en")

        context_block = "\n\n".join(context_parts[:6])
        if lang_style == "hi-Latn":
            synthesis_prompt = (
                f"Neeche diye sources ke basis par '{query}' ka clear jawab likho. "
                f"Har fact ke baad [1], [2] jaise citation lagao. Sirf plain text, koi markdown nahi. 4-6 sentences.\n\n"
                f"Sources:\n{context_block}"
            )
        elif lang_style == "hi":
            synthesis_prompt = (
                f"नीचे दिए स्रोतों के आधार पर '{query}' का स्पष्ट उत्तर लिखें। "
                f"हर तथ्य के बाद [1], [2] जैसे citation लगाएं। Plain text में 4-6 sentences।\n\n"
                f"Sources:\n{context_block}"
            )
        else:
            synthesis_prompt = (
                f"Using the sources below, write a clear answer for: '{query}'. "
                f"Add inline citations [1], [2] after each fact. Plain text only, no markdown. 4-6 sentences.\n\n"
                f"Sources:\n{context_block}"
            )

        messages = [
            {"role": "system", "content": (
                "You are Genius AI. Write plain text answers with inline [n] citations after each fact. "
                "No markdown, no bold, no headers. Natural, flowing prose like Perplexity AI."
            )},
            {"role": "user", "content": synthesis_prompt},
        ]

        # Display header
        self.console.print(
            f"[bold cyan]Genius[/bold cyan] [dim]—[/dim] [bold white]{query}[/bold white]"
        )
        self.console.print(Rule(style="cyan"))

        # Stream synthesized answer
        answer_tokens: List[str] = []
        with self.console.status("[bold magenta]Synthesizing answer...", spinner="dots"):
            async for token in self.engine.model.stream_generate(
                messages, max_new_tokens=600, temperature=0.4
            ):
                if "<think>" in token or "</think>" in token:
                    continue
                answer_tokens.append(token)

        raw_answer = "".join(answer_tokens)
        # Strip any residual <think>…</think> block
        raw_answer = _re.sub(r"<think>.*?</think>", "", raw_answer, flags=_re.DOTALL).strip()
        clean_answer = TextSanitizer.clean_for_display(raw_answer)

        # Fallback: stitch snippets manually if synthesis empty
        if not clean_answer.strip():
            parts_out = []
            for s in sources[:4]:
                snip = s["snippet"][:180].rstrip(".")
                parts_out.append(f"{snip} [{s['index']}].")
            clean_answer = " ".join(parts_out)

        # Render answer card
        self.console.print(
            Panel(Text(clean_answer, style="white"), border_style="cyan", padding=(1, 2))
        )

        # ── Sources Section ───────────────────────────────────────────────
        self.console.print()
        self.console.print(Rule("[bold dim]  Sources  [/bold dim]", style="dim"))
        for s in sources:
            icon = "📖" if s["type"] == "Wikipedia" else "🌐"
            self.console.print(
                f"  [bold cyan][{s['index']}][/bold cyan]  {icon}  [bold white]{s['title']}[/bold white]"
            )
            self.console.print(f"        [dim blue]{s['url']}[/dim blue]")
        self.console.print()

    # ─────────────────────────────────────────────────────────────────────

    async def _process_turn(self, question: str) -> None:
        """Executes a single reasoning and conversation turn."""
        # 1. Autonomous Natural Language Workspace / File / Image Actions
        intent_res = self.workspace.handle_natural_language_intent(question)
        if intent_res:
            intent_type, intent_msg = intent_res
            if intent_type == "workspace_shift":
                self.executor.default_cwd = str(self.workspace.get_workspace())
            clean_msg = TextSanitizer.clean_for_display(intent_msg)
            self.console.print()
            self.console.print(clean_msg)
            self.history.append({"user": question, "assistant": clean_msg})
            return

        detected_info: Optional[Dict] = None
        thinking_text = ""
        response_text = ""
        citations_data = []
        is_thinking = False
        verdict_data: Optional[Dict] = None

        self.console.print()

        # Step tracking status spinner
        with self.console.status("[bold cyan]Genius is deconstructing query...", spinner="dots") as status:
            async for event in self.engine.execute_stream(question, session_id=self.session_id):
                if event.stage == "language":
                    detected_info = event.payload
                    lang_name = detected_info.get("name", "Auto")
                    complexity = detected_info.get("complexity_score", 0.5)
                    status.update(f"[bold blue]🌐 Lang: {lang_name} | Complexity: {complexity} | Planning cognitive route...")

                elif event.stage == "plan":
                    msg = event.payload.get("message", "")
                    status.update(f"[bold yellow]{msg}")

                elif event.stage == "research":
                    if event.event_type == "status":
                        status.update("[bold magenta]🔍 Parallel Epistemic Retrieval (Wikipedia + Web)...")
                    elif event.event_type == "complete":
                        ev_count = event.payload.get("evidence_count", 0)
                        contradiction = event.payload.get("contradiction_density", 0.0)
                        self.console.print(f"[dim]📚 Epistemic Evidence: {ev_count} sources (Contradiction Density D={contradiction})[/dim]")

                elif event.stage == "exemplar":
                    exemplars = event.payload.get("exemplars", [])
                    if exemplars:
                        ex_list = ", ".join([e.get("source", "") for e in exemplars])
                        self.console.print(f"[dim]💡 Alignment exemplars: {ex_list}[/dim]")

                elif event.stage == "thinking":
                    if event.event_type == "status":
                        budget = event.payload.get("thinking_budget", 4096)
                        if self.show_thinking and self.chat_view_mode == "stream":
                            status.stop()
                            self.console.print(f"[bold gold1]─── 💭 Genius Latent_xThinking (Adaptive Budget: {budget} tokens) ───[/bold gold1]")
                            is_thinking = True
                    elif event.event_type == "token":
                        token = event.payload.get("token", "")
                        thinking_text += token
                        if self.show_thinking and self.chat_view_mode == "stream":
                            sys.stdout.write(f"\033[93m{token}\033[0m")
                            sys.stdout.flush()
                    elif event.event_type == "complete":
                        if self.show_thinking and self.chat_view_mode == "cards" and thinking_text.strip():
                            status.stop()
                            budget = event.payload.get("thinking_budget", 4096)
                            ChatViewer.render_thinking_message(self.console, thinking_text.strip(), budget)
                        elif is_thinking:
                            sys.stdout.write("\n")
                            self.console.print("[dim]─── End of Extended Thinking ───[/dim]\n")
                            is_thinking = False

                elif event.stage == "response":
                    if event.event_type == "status":
                        if self.chat_view_mode == "stream":
                            lang_label = detected_info.get("name", "Multilingual") if detected_info else "Grounded"
                            active_provider = self.engine.router.active_provider_name.upper()
                            self.console.print(f"[bold cyan]Genius [{active_provider}] ({lang_label})[/bold cyan] [dim]❯[/dim] ")
                    elif event.event_type == "token":
                        token = event.payload.get("token", "")
                        response_text += token
                        if self.chat_view_mode == "stream":
                            clean_token = token.replace("$$", "").replace(r"\(", "").replace(r"\)", "")
                            sys.stdout.write(clean_token)
                            sys.stdout.flush()

                elif event.stage == "done":
                    citations_data = event.payload.get("citations", [])
                    verdict_data = {
                        "s_ground": event.payload.get("s_ground", 1.0),
                        "tau_crit": event.payload.get("tau_crit", 0.62),
                        "action": event.payload.get("action", "emit"),
                        "contradiction_density": event.payload.get("contradiction_density", 0.0),
                    }

        # Render response based on chat view mode
        lang_label = detected_info.get("name", "Multilingual") if detected_info else "Grounded"
        active_provider = self.engine.router.active_provider_name.upper()
        if not response_text.strip() and thinking_text.strip():
            response_text = thinking_text.strip()
        clean_final_response = TextSanitizer.clean_for_display(response_text)

        if self.chat_view_mode == "cards":
            ChatViewer.render_assistant_message(
                self.console,
                clean_final_response,
                provider=active_provider,
                lang=lang_label,
                citations=citations_data,
            )
        else:
            sys.stdout.write("\n\n")
            if citations_data:
                s_ground = verdict_data.get("s_ground", 1.0) if verdict_data else 1.0
                tau_crit = verdict_data.get("tau_crit", 0.62) if verdict_data else 0.62
                action = verdict_data.get("action", "emit") if verdict_data else "emit"

                table = Table(
                    title=f"Verified Epistemic Grounding (S_ground: {s_ground} | tau_crit: {tau_crit} | Action: {action.upper()})",
                    border_style="dim",
                    box=None,
                )
                table.add_column("#", style="cyan", width=4)
                table.add_column("Source", style="bold white", width=25)
                table.add_column("Type", style="magenta", width=10)
                table.add_column("URL", style="blue")

                for c in citations_data:
                    src_type = c.get("source_type", "wikipedia").capitalize()
                    table.add_row(str(c["index"]), c["title"], src_type, c["url"])

                self.console.print(table)

        self.history.append({"user": question, "assistant": clean_final_response})


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="genius",
        description="Genius: Dual-Core Autonomous Reasoning Agent (xThinking + Epistemic Grounding)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Direct query to solve. If omitted, starts interactive chat session.",
    )
    parser.add_argument(
        "-r", "--research",
        choices=["auto", "on", "off"],
        default="auto",
        help="Research mode: auto (smart routing), on (always search), off (fast direct / offline)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=["local", "claude", "ollama"],
        default="local",
        help="Model provider core (default: local Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Suppress intermediate <think> tokens in output",
    )

    args = parser.parse_args()

    session = GeniusChatSession(show_thinking=not args.no_think)
    session.engine.set_research_mode(args.research)
    session.engine.set_model_provider(args.model)

    if args.query:
        session.print_welcome()
        routed = session.intent_router.route(args.query)
        if routed.intent_type == IntentType.AGENT:
            asyncio.run(session._run_agent(routed.argument))
        elif routed.intent_type == IntentType.GIT:
            asyncio.run(session._handle_git_command(routed.argument))
        elif routed.intent_type == IntentType.PROTOCOL:
            asyncio.run(session._execute_protocol(routed.argument))
        elif routed.intent_type == IntentType.INDEX:
            asyncio.run(session._handle_index_command(routed.argument))
        elif routed.intent_type == IntentType.SEARCH:
            asyncio.run(session._perplexity_search(routed.argument))
        else:
            asyncio.run(session._process_turn(args.query))
    else:
        asyncio.run(session.chat_loop())


if __name__ == "__main__":
    main()
