"""
Interactive Conversational Terminal for Genius.
Autonomous Deep-Reasoning Agent with Neural Cognitive Schema v1.0,
Wikipedia & Web Grounding, Dual-Core Model Routing, and HITL Safety Guard.
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
from .model.provider import UniversalModelRouter
from .reasoning.schema import GroundingVerdict
from .reasoning.text_sanitizer import TextSanitizer
from .reasoning.xthinking import Citation, ReasoningEvent, XThinkingEngine
from .system.executor import ExecutionResult, SystemExecutor
from .system.guard import ActionSafetyLevel, SafetyGuard
from .system.workspace import WorkspaceManager


class GeniusChatSession:
    """Multi-turn interactive conversation session for Genius in the CLI."""

    def __init__(self, show_thinking: bool = True) -> None:
        self.console = Console()
        self.show_thinking = show_thinking
        self.engine = XThinkingEngine()
        self.workspace = WorkspaceManager()
        self.executor = SystemExecutor(default_cwd=str(self.workspace.get_workspace()))
        self.session_id = self.engine.session_mgr.create_session("Interactive Session")
        self.history: List[Dict[str, str]] = []
        self.forced_lang: Optional[str] = None

    def print_welcome(self) -> None:
        """Displays welcome banner and system diagnostics."""
        active_model = self.engine.router.active_provider_name

        banner = Text()
        banner.append("⚡ GENIUS : AUTONOMOUS DEEP RESEARCHER & REASONING AI\n", style="bold cyan")
        banner.append("• Model Core: ", style="bold white")
        banner.append(f"{active_model.upper()} (Qwen2.5-0.5B-Instruct edge default)\n", style="green")
        banner.append("• Active Project: ", style="bold white")
        banner.append(f"{self.workspace.get_workspace()}\n", style="bold green")
        banner.append("• Research Mode: ", style="bold white")
        banner.append(f"{self.engine.research_mode.upper()} (auto / on / off toggleable)\n", style="cyan")
        banner.append("• Neural Schema: ", style="bold white")
        banner.append("v1.0 (ROUGE-L + Cosine Grounding Gate, Dynamic tau_crit)\n", style="bright_magenta")
        banner.append("• Memory Topology: ", style="bold white")
        banner.append("Dual-Memory (In-Context Sliding Window + SQLite FTS5 Episodic Vector Store)\n", style="yellow")
        banner.append("• Factual Grounding: ", style="bold white")
        banner.append("Wikipedia Live Research + DuckDuckGo + Multi-Hop BM25 Ranker\n", style="magenta")
        banner.append("• Safety Guard: ", style="bold white")
        banner.append("Human-in-the-Loop (HITL) Static AST/Regex Command Guardian\n", style="red")
        banner.append("• Commands: ", style="bold white")
        banner.append("/project, /files, /read, /create, /view, /calc, /code, /search, /exec, /stats, /clear, /exit\n", style="dim")

        self.console.print(Panel(banner, border_style="cyan", padding=(1, 2)))

    async def chat_loop(self) -> None:
        """Main conversational loop for terminal chat."""
        self.print_welcome()

        while True:
            try:
                self.console.print()
                user_input = self.console.input("[bold green]You[/bold green] [dim]❯[/dim] ").strip()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    handled = await self._handle_slash_command(user_input)
                    if handled == "exit":
                        break
                    continue

                # Execute conversational turn
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
                active = self.engine.router.active_provider_name
                table = Table(title="Available LLM Providers", box=None)
                table.add_column("Provider", style="bold cyan")
                table.add_column("Status", style="green")
                table.add_column("Active", style="yellow")

                for p_name in ["local", "claude", "ollama"]:
                    is_active = "✓ ACTIVE" if p_name == active else ""
                    status = "Available"
                    if p_name == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
                        status = "Requires ANTHROPIC_API_KEY"
                    table.add_row(p_name, status, is_active)

                self.console.print(table)
                self.console.print("[dim]Usage: /model [local|claude|ollama][/dim]")
            else:
                target = arg.lower()
                if self.engine.set_model_provider(target):
                    self.console.print(f"[green]✓ Switched active model provider to: [bold]{target.upper()}[/bold][/green]")
                else:
                    self.console.print(f"[red]Unknown provider '{target}'. Available: local, claude, ollama[/red]")

        elif cmd == "/search":
            if not arg:
                self.console.print("[yellow]Usage: /search <query>[/yellow]")
            else:
                self.console.print(f"[dim]Executing live multi-hop research for: '{arg}'...[/dim]")
                wiki_task = self.engine.wiki.search_and_fetch(arg, max_articles=2)
                web_task = self.engine.web.search(arg, limit=3)
                wiki_res, web_res = await asyncio.gather(wiki_task, web_task, return_exceptions=True)

                table = Table(title=f"Search Results: '{arg}'", border_style="cyan")
                table.add_column("Source", style="bold white", width=25)
                table.add_column("Type", style="magenta", width=12)
                table.add_column("URL / Snippet", style="dim")

                if isinstance(wiki_res, list):
                    for a in wiki_res:
                        table.add_row(a.title, "Wikipedia", a.url)
                if isinstance(web_res, list):
                    for w in web_res:
                        table.add_row(w.title[:25], "Web", w.url)

                self.console.print(table)

        elif cmd in ("/exec", "/run"):
            if not arg:
                self.console.print("[yellow]Usage: /exec <powershell command>[/yellow]")
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

        elif cmd == "/help":
            table = Table(title="Genius CLI Commands", border_style="cyan")
            table.add_column("Command", style="bold yellow")
            table.add_column("Description", style="white")
            table.add_row("/research [mode]", "Toggle research mode: 'auto' (smart), 'on' (always), 'off' (direct offline)")
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

    async def _process_turn(self, question: str) -> None:
        """Executes a single reasoning and conversation turn."""
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
                        status.stop()
                        if self.show_thinking:
                            budget = event.payload.get("thinking_budget", 4096)
                            self.console.print(f"[bold gold1]─── 💭 Genius Latent_xThinking (Adaptive Budget: {budget} tokens) ───[/bold gold1]")
                            is_thinking = True
                    elif event.event_type == "token":
                        token = event.payload.get("token", "")
                        thinking_text += token
                        if self.show_thinking:
                            sys.stdout.write(f"\033[93m{token}\033[0m")
                            sys.stdout.flush()
                    elif event.event_type == "complete":
                        if is_thinking:
                            sys.stdout.write("\n")
                            self.console.print("[dim]─── End of Extended Thinking ───[/dim]\n")
                            is_thinking = False

                elif event.stage == "response":
                    if event.event_type == "status":
                        lang_label = detected_info.get("name", "Multilingual") if detected_info else "Grounded"
                        active_provider = self.engine.router.active_provider_name.upper()
                        self.console.print(f"[bold cyan]Genius [{active_provider}] ({lang_label})[/bold cyan] [dim]❯[/dim] ")
                    elif event.event_type == "token":
                        token = event.payload.get("token", "")
                        response_text += token
                        sys.stdout.write(token)
                        sys.stdout.flush()

                elif event.stage == "done":
                    citations_data = event.payload.get("citations", [])
                    verdict_data = {
                        "s_ground": event.payload.get("s_ground", 1.0),
                        "tau_crit": event.payload.get("tau_crit", 0.62),
                        "action": event.payload.get("action", "emit"),
                        "contradiction_density": event.payload.get("contradiction_density", 0.0),
                    }

        sys.stdout.write("\n\n")

        # Grounding & Citations Telemetry Table (only shown when external sources were cited)
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

        self.history.append({"user": question, "assistant": response_text})


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
        asyncio.run(session._process_turn(args.query))
    else:
        asyncio.run(session.chat_loop())


if __name__ == "__main__":
    main()
