"""SQLite-backed conversational memory and session export manager."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StoredTurn:
    """Represents a single conversational exchange."""
    id: int
    session_id: str
    user_query: str
    assistant_response: str
    thoughts: str
    citations: List[Dict[str, Any]]
    language: str
    timestamp: float


class SessionManager:
    """Manages multi-session conversation history, SQLite persistence, and exports."""

    def __init__(self, db_path: str | Path = "genius_memory.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes tables for sessions and message turns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_query TEXT,
                    assistant_response TEXT,
                    thoughts TEXT,
                    citations_json TEXT,
                    language TEXT,
                    timestamp REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.commit()

    def create_session(self, title: str = "New Conversation") -> str:
        """Creates a new session and returns its ID."""
        session_id = str(uuid.uuid4())[:8]
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
        return session_id

    def save_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        thoughts: str = "",
        citations: Optional[List[Dict[str, Any]]] = None,
        language: str = "en",
    ) -> None:
        """Appends a completed chat turn to the active session."""
        now = time.time()
        citations_json = json.dumps(citations or [])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Ensure session exists
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            if not cursor.fetchone():
                title = user_query[:40] + "..." if len(user_query) > 40 else user_query
                cursor.execute(
                    "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, title, now, now),
                )

            # Insert turn
            cursor.execute(
                """
                INSERT INTO turns (session_id, user_query, assistant_response, thoughts, citations_json, language, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_query, assistant_response, thoughts, citations_json, language, now),
            )
            # Update session timestamp
            cursor.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
            conn.commit()

    def get_turns(self, session_id: str, limit: int = 10) -> List[StoredTurn]:
        """Fetches the last N turns for the session in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, user_query, assistant_response, thoughts, citations_json, language, timestamp
                FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()

        turns = []
        for r in reversed(rows):
            turns.append(
                StoredTurn(
                    id=r[0],
                    session_id=r[1],
                    user_query=r[2],
                    assistant_response=r[3],
                    thoughts=r[4],
                    citations=json.loads(r[5]) if r[5] else [],
                    language=r[6],
                    timestamp=r[7],
                )
            )
        return turns

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns all sessions ordered by most recent activity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at, COUNT(t.id) as turn_count
                FROM sessions s LEFT JOIN turns t ON s.session_id = t.session_id
                GROUP BY s.session_id ORDER BY s.updated_at DESC
                """
            )
            rows = cursor.fetchall()

        return [
            {
                "session_id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
                "turn_count": r[4],
            }
            for r in rows
        ]

    def export_markdown(self, session_id: str, output_path: str | Path) -> Path:
        """Exports session transcript as formatted Markdown."""
        turns = self.get_turns(session_id, limit=100)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Genius Conversation Transcript (Session: {session_id})",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        for idx, t in enumerate(turns, start=1):
            lines.append(f"### Turn {idx} ({t.language.upper()})")
            lines.append(f"**User**: {t.user_query}")
            lines.append("")
            if t.thoughts:
                lines.append("<details>")
                lines.append("<summary>🧠 Extended Thinking (xThinking)</summary>")
                lines.append("")
                lines.append(f"```text\n{t.thoughts}\n```")
                lines.append("</details>")
                lines.append("")
            lines.append(f"**Genius**: {t.assistant_response}")
            lines.append("")
            if t.citations:
                lines.append("#### Sources & Citations:")
                for c in t.citations:
                    lines.append(f"- [{c.get('index', '#')}] **{c.get('title', '')}**: {c.get('url', '')}")
                lines.append("")
            lines.append("---")
            lines.append("")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p
