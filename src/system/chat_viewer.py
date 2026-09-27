"""Chat View & Visual Dialogue Interface for Genius CLI.

Provides modern rounded chat bubble panels for the terminal and an instant
standalone HTML chat viewer with dark obsidian theme and responsive layout.
"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..reasoning.text_sanitizer import TextSanitizer


class ChatViewer:
    """Renders dialogue in modern visual terminal cards and generates standalone HTML chat view."""

    @staticmethod
    def render_user_message(console: Console, text: str, timestamp: Optional[str] = None) -> None:
        """Displays user input in a styled rounded chat card."""
        ts = timestamp or time.strftime("%H:%M:%S")
        title = f"[bold green]👤 You[/bold green] [dim]({ts})[/dim]"
        clean_text = TextSanitizer.clean_for_display(text)
        panel = Panel(
            clean_text,
            title=title,
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print()
        console.print(panel)

    @staticmethod
    def render_thinking_message(console: Console, thoughts: str, budget: int = 4096) -> None:
        """Displays latent reasoning trace in an amber thinking card."""
        clean_thoughts = TextSanitizer.clean_for_display(thoughts)
        title = f"[bold yellow]💭 Latent_xThinking[/bold yellow] [dim](Adaptive Budget: {budget} tokens)[/dim]"
        panel = Panel(
            clean_thoughts,
            title=title,
            title_align="left",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print()
        console.print(panel)

    @staticmethod
    def render_assistant_message(
        console: Console,
        text: str,
        provider: str = "LOCAL",
        lang: str = "English",
        timestamp: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Displays assistant response in a modern cyan/blue chat card."""
        ts = timestamp or time.strftime("%H:%M:%S")
        title = f"[bold cyan]⚡ Genius [{provider.upper()}] • {lang}[/bold cyan] [dim]({ts})[/dim]"
        clean_text = TextSanitizer.clean_for_display(text)

        panel = Panel(
            clean_text,
            title=title,
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print()
        console.print(panel)

        if citations:
            table = Table(
                title=f"[dim]📚 Epistemic Evidence ({len(citations)} sources cited)[/dim]",
                border_style="dim",
                box=None,
            )
            table.add_column("#", style="cyan", width=4)
            table.add_column("Source", style="bold white", width=25)
            table.add_column("Type", style="magenta", width=10)
            table.add_column("URL", style="blue")
            for c in citations:
                src_type = c.get("source_type", "wikipedia").capitalize()
                table.add_row(str(c["index"]), c["title"], src_type, c["url"])
            console.print(table)

    @classmethod
    def render_history_thread(cls, console: Console, history: List[Dict[str, str]], session_id: str) -> None:
        """Renders the entire conversation thread as visual message cards."""
        if not history:
            console.print("[dim]No messages in current session history.[/dim]")
            return

        console.print(f"[bold cyan]─── 💬 Session Dialogue Log ({session_id}) ───[/bold cyan]")
        for i, turn in enumerate(history, start=1):
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")

            cls.render_user_message(console, f"[Turn {i}]\n{user_msg}")
            cls.render_assistant_message(console, assistant_msg)

    @classmethod
    def export_html_view(cls, history: List[Dict[str, str]], session_id: str, output_path: Optional[str] = None) -> str:
        """Generates a standalone, dark-themed HTML chat viewer and opens it in browser."""
        out_file = Path(output_path or f"genius_chat_view_{session_id}.html").resolve()

        messages_html = []
        for i, turn in enumerate(history, 1):
            user_msg = html.escape(TextSanitizer.clean_for_display(turn.get("user", ""))).replace("\n", "<br>")
            assistant_msg = html.escape(TextSanitizer.clean_for_display(turn.get("assistant", ""))).replace("\n", "<br>")

            messages_html.append(f"""
            <div class="message user-message">
                <div class="message-header">
                    <span class="avatar">👤</span>
                    <span class="sender">You</span>
                    <span class="turn-badge">Turn {i}</span>
                </div>
                <div class="message-body">{user_msg}</div>
            </div>
            <div class="message genius-message">
                <div class="message-header">
                    <span class="avatar">⚡</span>
                    <span class="sender">Genius AI</span>
                    <span class="badge">Autonomous Deep-Reasoning</span>
                </div>
                <div class="message-body">{assistant_msg}</div>
            </div>
            """)

        all_msgs = "\n".join(messages_html) if messages_html else "<div class='empty'>No messages in session.</div>"

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genius AI — Visual Chat View</title>
    <style>
        :root {{
            --bg-primary: #0b0b0d;
            --bg-secondary: #16161a;
            --bg-card: #1f1f24;
            --user-bubble: #1a2736;
            --user-border: #2563eb;
            --genius-bubble: #121e24;
            --genius-border: #06b6d4;
            --text-main: #f3f4f6;
            --text-dim: #9ca3af;
            --accent: #00f2fe;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 24px 16px;
        }}
        .chat-container {{
            width: 100%;
            max-width: 860px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .header {{
            background: var(--bg-secondary);
            border: 1px solid #2d2d35;
            border-radius: 12px;
            padding: 18px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        .brand {{ display: flex; align-items: center; gap: 12px; }}
        .brand-icon {{ font-size: 28px; }}
        .brand-title {{ font-size: 20px; font-weight: 700; color: #fff; letter-spacing: 0.5px; }}
        .brand-sub {{ font-size: 13px; color: var(--text-dim); }}
        .session-tag {{
            font-size: 12px;
            background: #23232c;
            padding: 6px 12px;
            border-radius: 20px;
            color: var(--accent);
            border: 1px solid rgba(0, 242, 254, 0.2);
        }}
        .message {{
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.25);
            transition: transform 0.15s ease;
        }}
        .user-message {{
            background: var(--user-bubble);
            border: 1px solid var(--user-border);
            align-self: flex-end;
            width: 92%;
        }}
        .genius-message {{
            background: var(--genius-bubble);
            border: 1px solid var(--genius-border);
            align-self: flex-start;
            width: 96%;
        }}
        .message-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding-bottom: 6px;
        }}
        .sender {{ font-weight: 600; font-size: 14px; }}
        .turn-badge {{ font-size: 11px; color: var(--text-dim); margin-left: auto; }}
        .badge {{
            font-size: 11px;
            background: rgba(6, 182, 212, 0.15);
            color: #22d3ee;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: auto;
        }}
        .message-body {{
            font-size: 15px;
            white-space: pre-wrap;
            word-break: break-word;
            color: #e5e7eb;
        }}
        .empty {{
            text-align: center;
            color: var(--text-dim);
            padding: 40px;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #6b7280;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <div class="brand">
                <span class="brand-icon">⚡</span>
                <div>
                    <div class="brand-title">GENIUS AI</div>
                    <div class="brand-sub">Neural Cognitive Schema v1.0 • Epistemic Grounding</div>
                </div>
            </div>
            <div class="session-tag">Session: {session_id[:12]}</div>
        </div>

        {all_msgs}

        <div class="footer">
            Generated autonomously by Genius AI • Dual-Core Cognitive Reasoning Engine
        </div>
    </div>
</body>
</html>"""

        out_file.write_text(html_template, encoding="utf-8")

        # Open in default browser on Windows
        try:
            if sys.platform == "win32":
                os.startfile(str(out_file))
        except Exception:
            pass

        return str(out_file)
