"""Terminal CLI for Qwen2.5 + Wikipedia xThinking Deep-Reasoning System."""

from __future__ import annotations

import argparse
import asyncio
import sys
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .reasoning.xthinking import XThinkingEngine


async def run_query(question: str):
    console = Console()

    console.print(
        Panel.fit(
            f"[bold cyan]Meet Genius: Deep-Reasoning AI (xThinking + Wikipedia)[/bold cyan]\n"
            f"[white]Question:[/white] [italic]{question}[/italic]",
            border_style="cyan",
            title="🧠 Genius Autonomous Researcher",
        )
    )

    engine = XThinkingEngine()
    thinking_text = ""
    response_text = ""
    citations_data = []

    with console.status("[bold green]Initializing reasoning engine...", spinner="dots") as status:
        async for event in engine.execute_stream(question):
            if event.stage == "plan":
                status.update(f"[bold yellow]Planning: {event.payload.get('message')}")

            elif event.stage == "research":
                if event.event_type == "status":
                    status.update(f"[bold magenta]Research: {event.payload.get('message')}")
                elif event.event_type == "data":
                    articles = event.payload.get("articles_found", [])
                    if articles:
                        art_list = ", ".join([f"[bold]{a['title']}[/bold]" for a in articles])
                        console.print(f"[dim]📚 Wikipedia Articles Found: {art_list}[/dim]")
                elif event.event_type == "complete":
                    passages = event.payload.get("ranked_passages", [])
                    console.print(f"[dim]🔍 Extracted and ranked top {len(passages)} factual passages via BM25[/dim]")

            elif event.stage == "exemplar":
                if event.event_type == "data":
                    exemplars = event.payload.get("exemplars", [])
                    if exemplars:
                        ex_str = ", ".join([e.get("source", "") for e in exemplars])
                        console.print(f"[dim]💡 Matched instruction exemplars: {ex_str}[/dim]")

            elif event.stage == "thinking":
                if event.event_type == "status":
                    status.stop()
                    console.print("\n[bold gold1]─── 💭 xThinking: Deep Reasoning & Verification ───[/bold gold1]")
                elif event.event_type == "token":
                    token = event.payload.get("token", "")
                    thinking_text += token
                    sys.stdout.write(f"\033[93m{token}\033[0m")
                    sys.stdout.flush()
                elif event.event_type == "complete":
                    sys.stdout.write("\n")
                    console.print("[dim]─── End of Extended Thinking ───[/dim]\n")

            elif event.stage == "response":
                if event.event_type == "status":
                    console.print("[bold green]─── 📝 Grounded Response ───[/bold green]")
                elif event.event_type == "token":
                    token = event.payload.get("token", "")
                    response_text += token
                    sys.stdout.write(token)
                    sys.stdout.flush()

            elif event.stage == "done":
                citations_data = event.payload.get("citations", [])
                duration = event.payload.get("duration_sec", 0)

    console.print("\n")
    if citations_data:
        table = Table(title="📚 Verified Wikipedia Grounding & Citations", border_style="dim")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Article", style="bold white", width=25)
        table.add_column("Section", style="yellow", width=20)
        table.add_column("Wikipedia URL", style="blue")

        for c in citations_data:
            table.add_row(str(c["index"]), c["title"], c["section"], c["url"])

        console.print(table)

    console.print(f"[bold dim]✨ Completed in {duration} seconds.[/bold dim]\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B + Wikipedia xThinking Deep Reasoning")
    parser.add_argument("question", nargs="?", default="Who was Alan Turing and why is he considered the father of modern computing?", help="Question to research and answer")
    args = parser.parse_args()

    asyncio.run(run_query(args.question))


if __name__ == "__main__":
    main()
