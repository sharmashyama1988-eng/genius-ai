"""Interactive Conversational Terminal for Genius: Qwen2.5-0.5B-Chat with Wikipedia & xThinking."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .languages.router import DetectedLanguage
from .model.llm_engine import QwenEngine
from .reasoning.xthinking import ReasoningEvent, XThinkingEngine


class GeniusChatSession:
    """Multi-turn interactive conversation session for Genius in the CLI."""

    def __init__(self, show_thinking: bool = True) -> None:
        self.console = Console()
        self.show_thinking = show_thinking
        self.engine = XThinkingEngine(model_engine=QwenEngine(model_id="Qwen2.5-0.5B-Chat"))
        self.history: List[Dict[str, str]] = []
        self.forced_lang: Optional[str] = None

    def print_welcome(self) -> None:
        """Displays welcome banner and system diagnostics."""
        banner = Text()
        banner.append("⚡ GENIUS : AUTONOMOUS DEEP RESEARCHER & REASONING AI\n", style="bold cyan")
        banner.append("• Model: ", style="bold white")
        banner.append("Qwen2.5-0.5B-Chat (Ultra-Fast 0.5B)\n", style="green")
        banner.append("• Alignment & Persona: ", style="bold white")
        banner.append("LIMA (Conversational) + CodeAlpaca (Coding) + Alpaca (Instruction)\n", style="yellow")
        banner.append("• Factual Grounding: ", style="bold white")
        banner.append("Wikipedia Live Research & Multi-Hop BM25 Ranker\n", style="magenta")
        banner.append("• Multilingual Engine: ", style="bold white")
        banner.append("Real-Time Auto-Detection (Hinglish, Hindi, English, Spanish, etc.)\n", style="blue")
        banner.append("• Commands: ", style="bold white")
        banner.append("/think (toggle thinking), /lang [code], /clear, /exit\n", style="dim")

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
                    cmd_parts = user_input.split()
                    cmd = cmd_parts[0].lower()

                    if cmd in ("/exit", "/quit", "/q"):
                        self.console.print("[dim]Alvida! Exiting Genius chat. 👋[/dim]")
                        break

                    elif cmd == "/clear":
                        os.system("cls" if os.name == "nt" else "clear")
                        self.history.clear()
                        self.print_welcome()
                        self.console.print("[dim]Memory reset. Clean session started.[/dim]")
                        continue

                    elif cmd == "/think":
                        self.show_thinking = not self.show_thinking
                        status_str = "VISIBLE" if self.show_thinking else "HIDDEN"
                        self.console.print(f"[dim]xThinking stream display is now: [bold]{status_str}[/bold][/dim]")
                        continue

                    elif cmd == "/lang":
                        if len(cmd_parts) > 1:
                            lang_code = cmd_parts[1].lower()
                            if lang_code in ("auto", "reset"):
                                self.forced_lang = None
                                self.console.print("[dim]Language mode set to: [bold]AUTO-DETECT[/bold][/dim]")
                            elif lang_code in self.engine.multilingual_mgr.router.profiles:
                                self.forced_lang = lang_code
                                prof = self.engine.multilingual_mgr.router.profiles[lang_code]
                                self.console.print(f"[dim]Forced language set to: [bold]{prof.name} ({lang_code})[/bold][/dim]")
                            else:
                                self.console.print(f"[red]Unknown language code '{lang_code}'. Available: hi, hi-Latn, en, es, fr, de[/red]")
                        else:
                            current = self.forced_lang or "AUTO-DETECT"
                            self.console.print(f"[dim]Current language mode: [bold]{current}[/bold][/dim]")
                        continue

                    elif cmd == "/help":
                        self.console.print("[bold cyan]Commands:[/bold cyan]")
                        self.console.print("  /think         - Toggle live internal thinking display")
                        self.console.print("  /lang [code]   - Set language (hi, hi-Latn, en, es, fr, de, auto)")
                        self.console.print("  /clear         - Reset conversation and clear screen")
                        self.console.print("  /exit          - Exit chat")
                        continue

                # Execute conversational turn
                await self._process_turn(user_input)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Session terminated by user. Alvida! 👋[/dim]")
                break
            except Exception as e:
                self.console.print(f"\n[red bold]Error during processing:[/red bold] {e}")

    async def _process_turn(self, question: str) -> None:
        """Executes a single reasoning and conversation turn."""
        detected_info: Optional[Dict] = None
        thinking_text = ""
        response_text = ""
        citations_data = []
        is_thinking = False

        self.console.print()

        # Step tracking status spinner
        with self.console.status("[bold cyan]Genius is preparing...", spinner="dots") as status:
            async for event in self.engine.execute_stream(question):
                if event.stage == "language":
                    detected_info = event.payload
                    lang_name = detected_info.get("name", "Auto")
                    status.update(f"[bold blue]🌐 Lang: {lang_name} | Planning research...")

                elif event.stage == "plan":
                    msg = event.payload.get("message", "")
                    status.update(f"[bold yellow]Planning: {msg}")

                elif event.stage == "research":
                    if event.event_type == "status":
                        status.update(f"[bold magenta]Researching Wikipedia: {event.payload.get('search_query', '')}")
                    elif event.event_type == "data":
                        articles = event.payload.get("articles_found", [])
                        if articles:
                            names = ", ".join([f"[cyan]{a['title']}[/cyan]" for a in articles])
                            self.console.print(f"[dim]📚 Wikipedia Sources: {names}[/dim]")
                    elif event.event_type == "complete":
                        passages = event.payload.get("ranked_passages", [])
                        self.console.print(f"[dim]🔍 Extracted {len(passages)} relevant passages with BM25[/dim]")

                elif event.stage == "exemplar":
                    exemplars = event.payload.get("exemplars", [])
                    if exemplars:
                        ex_list = ", ".join([e.get("source", "") for e in exemplars])
                        self.console.print(f"[dim]💡 In-context exemplars: {ex_list}[/dim]")

                elif event.stage == "thinking":
                    if event.event_type == "status":
                        status.stop()
                        if self.show_thinking:
                            self.console.print("[bold gold1]─── 💭 Genius xThinking (Extended Reasoning) ───[/bold gold1]")
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
                        self.console.print(f"[bold cyan]Genius ({lang_label})[/bold cyan] [dim]❯[/dim] ")
                    elif event.event_type == "token":
                        token = event.payload.get("token", "")
                        response_text += token
                        sys.stdout.write(token)
                        sys.stdout.flush()

                elif event.stage == "done":
                    citations_data = event.payload.get("citations", [])
                    duration = event.payload.get("duration_sec", 0)

        sys.stdout.write("\n\n")

        # Citations table if facts were cited
        if citations_data:
            table = Table(title="Verified Wikipedia Grounding", border_style="dim", box=None)
            table.add_column("#", style="cyan", width=4)
            table.add_column("Source Article", style="bold white", width=25)
            table.add_column("Wikipedia URL", style="blue")

            for c in citations_data:
                table.add_row(str(c["index"]), c["title"], c["url"])

            self.console.print(table)

        self.history.append({"user": question, "assistant": response_text})


def main():
    session = GeniusChatSession(show_thinking=True)
    asyncio.run(session.chat_loop())


if __name__ == "__main__":
    main()
