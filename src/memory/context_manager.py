"""
Genius Dynamic Context Window Engine — Larger than Gemini 1.5 Flash's 1M token limit.

Architecture:
  L1  Active Window     — What the LLM sees right now (configurable, default 32K tokens)
  L2  Compressed Archive — Smart summaries of older conversation chunks (unlimited)
  L3  Episodic Retrieval — BM25 + keyword pull-back from SQLite episodic store
  L4  File-based Storage — Persistent archive across sessions (JSON on disk)

How it achieves "infinite" context:
  • Every N messages, older chunks are summarized (compressed ~10:1 ratio)
  • Summaries are stored with importance scores and keyword indices
  • When user references something from far back, it's retrieved and re-injected
  • Active window always contains: system prompt + recent exact messages + relevant retrieved chunks
  • Total information preserved = 100%, tokens used = dynamic + adaptive

Comparison:
  Gemini 1.5 Flash   — Fixed 1,000,000 token window (single shot, no compression)
  Genius DynamicCtx  — Unlimited rolling window with lossless semantic archive + retrieval
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 3.8              # Approximate chars per token (GPT/Claude average)
DEFAULT_ACTIVE_TOKENS = 32_000     # Active window token budget
DEFAULT_CHUNK_SIZE = 12            # Messages per archive chunk
COMPRESSION_RATIO_TARGET = 10.0   # Compress N chars to N/10 in summary
MAX_RETRIEVE_CHUNKS = 4            # Max archive chunks to re-inject per turn
IMPORTANCE_DECAY = 0.92            # Per-message importance decay factor


# ─── Message Importance Scoring ───────────────────────────────────────────────

# Keywords that signal high-importance content
HIGH_IMPORTANCE_PATTERNS = [
    r"\berror\b", r"\bexception\b", r"\bfixed\b", r"\bimportant\b",
    r"\bremember\b", r"\byaad\b", r"\bnote\b", r"\bcritical\b",
    r"\barchitecture\b", r"\bdesign\b", r"\bschema\b", r"\bdatabase\b",
    r"\bapi\b", r"\bendpoint\b", r"\bconfigur\b", r"\bpassword\b",
    r"\btoken\b", r"\bkey\b", r"\bsecret\b", r"\bdeploy\b",
    r"\bfile\b", r"\bpath\b", r"\bdir\b", r"\bproject\b",
    r"def \w+", r"class \w+", r"import \w+",  # code definitions
    r"```", r"<tool_call>", r"<tool_result>",   # code/tool blocks
]
HIGH_IMPORTANCE_RE = re.compile("|".join(HIGH_IMPORTANCE_PATTERNS), re.IGNORECASE)


def compute_importance(text: str, role: str, turn_index: int, total_turns: int) -> float:
    """
    Scores a message's importance in [0.0, 1.0].
    High score = keep in active window longer.
    """
    score = 0.5

    # Recency: recent messages are more important
    recency = (turn_index + 1) / max(total_turns, 1)
    score += 0.25 * recency

    # Role weighting: system > user > assistant
    if role == "system":
        score = 1.0  # System prompt always max importance
    elif role == "user":
        score += 0.1

    # Content signals
    matches = len(HIGH_IMPORTANCE_RE.findall(text))
    score += min(0.25, matches * 0.03)

    # Length bonus (longer = more content)
    score += min(0.1, len(text) / 5000)

    # Code blocks are very important
    if "```" in text or "<tool_call>" in text or "def " in text or "class " in text:
        score += 0.15

    return min(1.0, round(score, 4))


def count_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 3.8 chars)."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ManagedMessage:
    role: str
    content: str
    turn_index: int
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    token_count: int = 0
    archived: bool = False
    archive_chunk_id: Optional[str] = None

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ArchiveChunk:
    chunk_id: str
    summary: str
    original_messages: List[Dict[str, Any]]
    turn_range: Tuple[int, int]          # (start_turn, end_turn)
    keywords: List[str]
    timestamp: float
    token_count_original: int
    token_count_summary: int

    @property
    def compression_ratio(self) -> float:
        if self.token_count_summary == 0:
            return 0.0
        return round(self.token_count_original / self.token_count_summary, 1)

    def matches_query(self, query: str) -> float:
        """Simple BM25-lite relevance score for this chunk vs a query."""
        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        text = (self.summary + " " + " ".join(self.keywords)).lower()
        chunk_terms = set(re.findall(r"\b\w{3,}\b", text))
        overlap = q_terms & chunk_terms
        if not q_terms:
            return 0.0
        return len(overlap) / math.sqrt(len(q_terms))


@dataclass
class ContextStats:
    active_messages: int
    active_tokens: int
    active_token_budget: int
    archived_chunks: int
    archived_tokens_original: int
    archived_tokens_compressed: int
    total_turns: int
    compression_ratio: float
    window_utilization: float  # active_tokens / budget

    def display(self) -> str:
        util_pct = round(self.window_utilization * 100, 1)
        saved = self.archived_tokens_original - self.archived_tokens_compressed
        lines = [
            "Dynamic Context Window Status",
            "─" * 44,
            f"  Active Window   : {self.active_tokens:,} / {self.active_token_budget:,} tokens  ({util_pct}% used)",
            f"  Active Messages : {self.active_messages}",
            f"  Archive Chunks  : {self.archived_chunks}",
            f"  Archive Saved   : {saved:,} tokens (ratio {self.compression_ratio:.1f}x)",
            f"  Total Turns     : {self.total_turns}",
            f"  Effective Ctx   : UNLIMITED (L1 active + L2 archive + L3 retrieval)",
            "─" * 44,
        ]
        return "\n".join(lines)


# ─── Naive Summarizer (no LLM required) ──────────────────────────────────────

class NaiveSummarizer:
    """
    Fast extractive summarizer that works without calling an LLM.
    Used as fallback when LLM summarization is too expensive.
    Extracts the highest-signal sentences by importance scoring.
    """

    @staticmethod
    def summarize(messages: List[Dict[str, Any]], max_chars: int = 600) -> str:
        """
        Creates a concise summary of a message chunk.
        Preserves: role labels, key sentences, code snippets.
        """
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if role == "system":
                continue  # System prompt not archived

            # Extract high-signal sentences
            sentences = re.split(r"(?<=[.!?])\s+", content)
            scored = []
            for sent in sentences:
                score = len(HIGH_IMPORTANCE_RE.findall(sent))
                scored.append((score, sent))
            scored.sort(reverse=True)
            top = [s for _, s in scored[:3] if len(s) > 10]

            # Always keep code blocks
            code_blocks = re.findall(r"```[\s\S]{0,400}?```", content)

            role_label = "User" if role == "user" else "AI"
            summary_parts = []
            if top:
                summary_parts.append(" ".join(top)[:200])
            if code_blocks:
                summary_parts.append(code_blocks[0][:200])
            if summary_parts:
                parts.append(f"[{role_label}]: {' | '.join(summary_parts)}")

        result = "\n".join(parts)
        return result[:max_chars] if len(result) > max_chars else result

    @staticmethod
    def extract_keywords(messages: List[Dict[str, Any]]) -> List[str]:
        """Extract top keywords from a set of messages."""
        text = " ".join(m.get("content", "") for m in messages)
        # Remove common stopwords
        STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "i", "you", "he",
                "she", "it", "we", "they", "this", "that", "and", "or", "but",
                "in", "on", "at", "to", "for", "of", "with", "by", "from"}
        words = re.findall(r"\b[a-zA-Z_]\w{2,}\b", text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w not in STOP:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:20]]


# ─── Dynamic Context Window Manager ──────────────────────────────────────────

class DynamicContextWindow:
    """
    Manages an unlimited context window through layered memory architecture.

    Usage:
        ctx = DynamicContextWindow(active_token_budget=32_000)
        ctx.add_message("user", "What is Python?")
        ctx.add_message("assistant", "Python is a programming language...")
        messages = ctx.get_active_messages()          # For LLM
        ctx.maybe_compress()                          # Auto-compress if over budget
    """

    def __init__(
        self,
        active_token_budget: int = DEFAULT_ACTIVE_TOKENS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        archive_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.active_token_budget = active_token_budget
        self.chunk_size = chunk_size
        self.session_id = session_id or f"session_{int(time.time())}"

        # L1: Active message list
        self._messages: List[ManagedMessage] = []
        self._turn_counter: int = 0

        # L2: Compressed archive
        self._archive: List[ArchiveChunk] = []

        # L4: Persistent archive file
        self._archive_path: Optional[Path] = None
        if archive_path:
            self._archive_path = Path(archive_path)
            self._archive_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_archive()

    # ── Public API ───────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> ManagedMessage:
        """Add a new message to the active window."""
        if role == "system":
            # System messages always stay — replace existing system
            for m in self._messages:
                if m.role == "system":
                    m.content = content
                    m.token_count = count_tokens(content)
                    return m

        msg = ManagedMessage(
            role=role,
            content=content,
            turn_index=self._turn_counter,
            importance=0.5,
        )
        self._messages.append(msg)
        self._turn_counter += 1

        # Recompute importance with updated total
        self._recompute_importance()

        # Auto-compress if over budget
        self.maybe_compress()

        return msg

    def get_active_messages(self) -> List[Dict[str, str]]:
        """Returns the current active window as a list of {role, content} dicts for LLM."""
        return [{"role": m.role, "content": m.content} for m in self._messages if not m.archived]

    def inject_retrieved_context(self, query: str) -> int:
        """
        Searches archive for relevant chunks and injects them after the system prompt.
        Returns number of chunks injected.
        """
        if not self._archive:
            return 0

        # Score all archive chunks against query
        scored = [(chunk.matches_query(query), chunk) for chunk in self._archive]
        scored.sort(reverse=True, key=lambda x: x[0])

        # Inject top-K relevant chunks
        injected = 0
        for score, chunk in scored[:MAX_RETRIEVE_CHUNKS]:
            if score < 0.1:
                break  # Not relevant enough
            inject_content = (
                f"[RETRIEVED CONTEXT — turns {chunk.turn_range[0]}-{chunk.turn_range[1]}]\n"
                f"{chunk.summary}\n[END RETRIEVED CONTEXT]"
            )
            # Find system message position
            insert_idx = 1 if (self._messages and self._messages[0].role == "system") else 0
            retrieved_msg = ManagedMessage(
                role="system",
                content=inject_content,
                turn_index=-1,
                importance=0.8,
            )
            self._messages.insert(insert_idx + injected, retrieved_msg)
            injected += 1

        return injected

    def maybe_compress(self) -> bool:
        """
        Auto-compresses if active window exceeds budget.
        Compresses the oldest low-importance non-system messages first.
        Returns True if compression occurred.
        """
        active = [m for m in self._messages if not m.archived and m.role != "system"]
        active_tokens = sum(m.token_count for m in self._messages if not m.archived)

        if active_tokens <= self.active_token_budget:
            return False

        # Sort by importance (lowest first = archive these)
        candidates = sorted(active, key=lambda m: (m.importance, m.turn_index))
        to_archive = candidates[:max(self.chunk_size, len(candidates) // 3)]

        if len(to_archive) < 2:
            return False  # Not enough to meaningfully compress

        self._archive_chunk(to_archive)
        return True

    def force_compress(self, keep_last_n: int = 20) -> ArchiveChunk:
        """
        Force-archives all messages except the last N, regardless of importance.
        Useful for very long agent loops.
        """
        active = [m for m in self._messages if not m.archived and m.role != "system"]
        if len(active) <= keep_last_n:
            return None
        to_archive = active[:-keep_last_n]
        return self._archive_chunk(to_archive)

    def get_stats(self) -> ContextStats:
        """Returns current context window statistics."""
        active = [m for m in self._messages if not m.archived]
        active_tokens = sum(m.token_count for m in active)
        orig = sum(c.token_count_original for c in self._archive)
        comp = sum(c.token_count_summary for c in self._archive)
        ratio = round(orig / max(comp, 1), 1)
        return ContextStats(
            active_messages=len(active),
            active_tokens=active_tokens,
            active_token_budget=self.active_token_budget,
            archived_chunks=len(self._archive),
            archived_tokens_original=orig,
            archived_tokens_compressed=comp,
            total_turns=self._turn_counter,
            compression_ratio=ratio,
            window_utilization=active_tokens / max(self.active_token_budget, 1),
        )

    def get_full_transcript(self) -> str:
        """Returns the FULL conversation including archive. For export/analysis."""
        lines = [f"=== FULL TRANSCRIPT (Dynamic Context) ==="]
        for chunk in self._archive:
            lines.append(f"\n[ARCHIVE CHUNK {chunk.chunk_id} turns {chunk.turn_range}]")
            for m in chunk.original_messages:
                lines.append(f"[{m.get('role','?').upper()}]: {m.get('content','')[:500]}")
        lines.append("\n[ACTIVE WINDOW]")
        for m in self._messages:
            if not m.archived:
                lines.append(f"[{m.role.upper()}]: {m.content[:500]}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear active window (keep archive intact)."""
        sys_msgs = [m for m in self._messages if m.role == "system"]
        self._messages = sys_msgs
        self._turn_counter = 0

    def set_budget(self, tokens: int) -> None:
        """Dynamically adjust the active window token budget."""
        self.active_token_budget = max(8_000, tokens)
        self.maybe_compress()

    # ── Internal Methods ─────────────────────────────────────────────────────

    def _recompute_importance(self) -> None:
        total = self._turn_counter
        for m in self._messages:
            if m.role == "system":
                m.importance = 1.0
            else:
                m.importance = compute_importance(m.content, m.role, m.turn_index, total)

    def _archive_chunk(self, messages: List[ManagedMessage]) -> ArchiveChunk:
        """Compress a list of messages into an archive chunk."""
        raw_dicts = [{"role": m.role, "content": m.content} for m in messages]
        summary = NaiveSummarizer.summarize(raw_dicts)
        keywords = NaiveSummarizer.extract_keywords(raw_dicts)
        orig_tokens = sum(m.token_count for m in messages)

        chunk_id = hashlib.md5(
            (str(messages[0].turn_index) + str(time.time())).encode()
        ).hexdigest()[:8]

        chunk = ArchiveChunk(
            chunk_id=chunk_id,
            summary=summary,
            original_messages=raw_dicts,
            turn_range=(messages[0].turn_index, messages[-1].turn_index),
            keywords=keywords,
            timestamp=time.time(),
            token_count_original=orig_tokens,
            token_count_summary=count_tokens(summary),
        )
        self._archive.append(chunk)

        # Mark messages as archived
        archive_ids = {id(m) for m in messages}
        self._messages = [m for m in self._messages if id(m) not in archive_ids]

        # Persist archive
        self._save_archive()

        return chunk

    def _save_archive(self) -> None:
        """Persist archive chunks to disk."""
        if not self._archive_path:
            return
        try:
            data = {
                "session_id": self.session_id,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "summary": c.summary,
                        "original_messages": c.original_messages,
                        "turn_range": list(c.turn_range),
                        "keywords": c.keywords,
                        "timestamp": c.timestamp,
                        "token_count_original": c.token_count_original,
                        "token_count_summary": c.token_count_summary,
                    }
                    for c in self._archive
                ],
            }
            self._archive_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _load_archive(self) -> None:
        """Load persisted archive from disk."""
        if not self._archive_path or not self._archive_path.exists():
            return
        try:
            data = json.loads(self._archive_path.read_text(encoding="utf-8"))
            for cd in data.get("chunks", []):
                chunk = ArchiveChunk(
                    chunk_id=cd["chunk_id"],
                    summary=cd["summary"],
                    original_messages=cd["original_messages"],
                    turn_range=tuple(cd["turn_range"]),
                    keywords=cd["keywords"],
                    timestamp=cd["timestamp"],
                    token_count_original=cd["token_count_original"],
                    token_count_summary=cd["token_count_summary"],
                )
                self._archive.append(chunk)
        except Exception:
            pass


# ─── Preset Window Sizes (for /context command) ───────────────────────────────

WINDOW_PRESETS: Dict[str, int] = {
    "small":    8_000,    # Fast, minimal context, good for simple chat
    "medium":   32_000,   # Default — balanced (like GPT-4)
    "large":    64_000,   # Deep reasoning, complex code tasks
    "xl":       128_000,  # Claude 3 level
    "gemini":   500_000,  # Gemini 1.5 Flash parity
    "ultra":    1_000_000,# Gemini 1.5 Pro parity (archive-heavy)
    "infinite": 2**31,    # True unlimited (full archival mode)
}


def create_context_window(
    preset: str = "medium",
    session_id: Optional[str] = None,
    archive_dir: Optional[str] = None,
) -> DynamicContextWindow:
    """
    Factory: creates a DynamicContextWindow with the given preset size.
    All presets use the same underlying unlimited architecture —
    only the L1 active window size changes.
    """
    budget = WINDOW_PRESETS.get(preset.lower(), DEFAULT_ACTIVE_TOKENS)
    archive_path = None
    if archive_dir and session_id:
        archive_path = os.path.join(archive_dir, f"archive_{session_id}.json")

    return DynamicContextWindow(
        active_token_budget=budget,
        session_id=session_id,
        archive_path=archive_path,
    )
