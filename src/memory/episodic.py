"""
Long-Term Persistent Episodic Memory for Genius.
Implements §4.2 of Neural Cognitive Schema v1.0:
- Hybrid SQLite store: FTS5 BM25 exact lexical + dense vectors with cosine similarity
- Reciprocal Rank Fusion (RRF, k=60)
- Salience score decay: salience(t) = salience_0 * exp(-delta*(t - t_0)) + eta * log(1 + access_count)
- Salience floor pruning (0.05) and write-through transactional persistence
"""

from __future__ import annotations

import math
import sqlite3
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..reasoning.schema import NeuralStateSnapshot


@dataclass
class EpisodicSearchResult:
    """Retrieved episodic memory result with hybrid RRF score."""
    episode_id: str
    session_id: str
    content_summary: str
    language: str
    salience_score: float
    rrf_score: float
    bm25_rank: Optional[int] = None
    cosine_rank: Optional[int] = None


class EpisodicMemoryManager:
    """Manages SQLite FTS5 lexical indexing, dense embedding vectors, and RRF retrieval."""

    DELTA_DECAY = 1.0 / (30.0 * 86400.0)  # 30-day half-decay rate
    ETA_REINFORCE = 0.05                   # Reinforcement per access
    SALIENCE_FLOOR = 0.05                  # Minimum threshold before pruning
    RRF_K = 60                             # Reciprocal Rank Fusion constant

    def __init__(self, db_path: str | Path = "genius_episodic.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            from ..system.resource_manager import SQLiteOptimizer
            SQLiteOptimizer.optimize_connection(conn)
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        """Initializes tables for episodic FTS5 and vector storage."""
        with self._get_connection() as conn:
            # Check if FTS5 is available
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
                        episode_id UNINDEXED,
                        content,
                        tokenize = 'porter unicode61'
                    );
                    """
                )
            except sqlite3.OperationalError:
                # Fallback to standard FTS4 or regular table if FTS5 is not loaded in SQLite build
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodic_fts (
                        episode_id TEXT PRIMARY KEY,
                        content TEXT
                    );
                    """
                )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_vectors (
                    episode_id      TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    embedding_blob  BLOB NOT NULL,
                    embedding_dim   INTEGER NOT NULL,
                    content_summary TEXT NOT NULL,
                    language        TEXT NOT NULL,
                    salience_score  REAL NOT NULL,
                    created_at      REAL NOT NULL,
                    last_accessed   REAL NOT NULL,
                    access_count    INTEGER DEFAULT 0
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_salience ON episodic_vectors(salience_score DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_vectors(session_id);")
            conn.commit()

    @staticmethod
    def _compute_text_embedding(text: str, dim: int = 128) -> np.ndarray:
        """Lightweight, deterministic feature hashing embedding vector for local CPU efficiency."""
        tokens = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h = hash(token)
            slot = abs(h) % dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            pos_weight = 1.0 / math.sqrt(idx + 1)
            vec[slot] += sign * pos_weight

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def write_through_snapshot(self, snapshot: NeuralStateSnapshot) -> Optional[str]:
        """Atomically persists converged, non-degraded reasoning trajectories into episodic memory."""
        if snapshot.degraded_mode:
            return None

        # Check if trajectories converged
        if not snapshot.trajectories or not any(t.converged for t in snapshot.trajectories):
            return None

        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        now = time.time()

        # Build content summary from top thought steps and final response
        summary_parts = [f"Query: {snapshot.user_query_raw}"]
        if snapshot.final_response_text:
            summary_parts.append(f"Response: {snapshot.final_response_text}")

        # Extract top-salience hypotheses from thought steps
        for traj in snapshot.trajectories:
            for step in traj.thought_steps[:4]:
                summary_parts.append(f"Thought: {step.hypothesis}")

        content_full = "\n".join(summary_parts)
        embedding = self._compute_text_embedding(content_full, dim=128)
        blob = embedding.tobytes()

        primary_lang = "en"
        if snapshot.trajectories and snapshot.trajectories[-1].language_profile:
            primary_lang = snapshot.trajectories[-1].language_profile.primary.value

        initial_salience = 1.0

        with self._get_connection() as conn:
            # Insert into episodic_fts
            try:
                conn.execute(
                    "INSERT INTO episodic_fts (episode_id, content) VALUES (?, ?)",
                    (episode_id, content_full),
                )
            except Exception:
                pass

            # Insert into episodic_vectors
            conn.execute(
                """
                INSERT INTO episodic_vectors
                    (episode_id, session_id, embedding_blob, embedding_dim, content_summary,
                     language, salience_score, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    snapshot.session_id,
                    blob,
                    128,
                    content_full[:500],
                    primary_lang,
                    initial_salience,
                    now,
                    now,
                    1,
                ),
            )
            conn.commit()

        return episode_id

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[EpisodicSearchResult]:
        """Executes Reciprocal Rank Fusion (RRF, k=60) over lexical (BM25) and dense vector cosine."""
        now = time.time()
        with self._get_connection() as conn:
            # 1. Lexical search via FTS5 or LIKE
            bm25_ranks: Dict[str, int] = {}
            try:
                # Clean query for FTS5 match
                clean_q = ' OR '.join([w for w in query.replace('"', '').split() if len(w) > 2])
                if clean_q:
                    fts_rows = conn.execute(
                        "SELECT episode_id FROM episodic_fts WHERE content MATCH ? LIMIT 30",
                        (clean_q,),
                    ).fetchall()
                    for rank, row in enumerate(fts_rows, start=1):
                        bm25_ranks[row["episode_id"]] = rank
            except Exception:
                pass

            # 2. Vector search & salience decay application
            query_vec = self._compute_text_embedding(query, dim=128)
            all_vecs = conn.execute(
                """
                SELECT episode_id, session_id, embedding_blob, content_summary, language,
                       salience_score, created_at, last_accessed, access_count
                FROM episodic_vectors
                """
            ).fetchall()

            cosine_scores: List[Tuple[str, float, sqlite3.Row]] = []
            for row in all_vecs:
                # Compute decayed salience: salience_0 * exp(-delta*(t - t_0)) + eta * log(1 + access_count)
                t_diff = now - row["created_at"]
                decayed = row["salience_score"] * math.exp(-self.DELTA_DECAY * t_diff)
                reinforce = self.ETA_REINFORCE * math.log(1.0 + row["access_count"])
                current_salience = max(0.0, decayed + reinforce)

                if current_salience < self.SALIENCE_FLOOR:
                    continue

                # Cosine similarity
                blob = row["embedding_blob"]
                doc_vec = np.frombuffer(blob, dtype=np.float32)
                sim = float(np.dot(query_vec, doc_vec))
                # Weight cosine by current salience
                weighted_sim = sim * (0.5 + 0.5 * min(current_salience, 1.0))
                cosine_scores.append((row["episode_id"], weighted_sim, row))

            # Rank by cosine similarity
            cosine_scores.sort(key=lambda x: x[1], reverse=True)
            cosine_ranks: Dict[str, int] = {item[0]: rank for rank, item in enumerate(cosine_scores[:30], start=1)}

            # 3. Reciprocal Rank Fusion: RRF(d) = sum_{r} 1 / (k + rank_r(d))
            candidate_ids = set(bm25_ranks.keys()) | set(cosine_ranks.keys())
            rrf_results: List[EpisodicSearchResult] = []

            row_map = {item[0]: item[2] for item in cosine_scores}

            for ep_id in candidate_ids:
                rrf = 0.0
                r_bm25 = bm25_ranks.get(ep_id)
                r_cos = cosine_ranks.get(ep_id)

                if r_bm25 is not None:
                    rrf += 1.0 / (self.RRF_K + r_bm25)
                if r_cos is not None:
                    rrf += 1.0 / (self.RRF_K + r_cos)

                row = row_map.get(ep_id)
                if not row:
                    raw_row = conn.execute(
                        "SELECT * FROM episodic_vectors WHERE episode_id = ?", (ep_id,)
                    ).fetchone()
                    if raw_row:
                        row = raw_row

                if row:
                    rrf_results.append(
                        EpisodicSearchResult(
                            episode_id=ep_id,
                            session_id=row["session_id"],
                            content_summary=row["content_summary"],
                            language=row["language"],
                            salience_score=row["salience_score"],
                            rrf_score=round(rrf, 6),
                            bm25_rank=r_bm25,
                            cosine_rank=r_cos,
                        )
                    )

            rrf_results.sort(key=lambda x: x.rrf_score, reverse=True)
            top_results = rrf_results[:top_k]

            # Update access counts for retrieved items
            for res in top_results:
                conn.execute(
                    """
                    UPDATE episodic_vectors
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE episode_id = ?
                    """,
                    (now, res.episode_id),
                )
            conn.commit()

            return top_results

    def prune_below_floor(self) -> int:
        """Prunes stale memories below the salience floor."""
        now = time.time()
        pruned_count = 0
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT episode_id, salience_score, created_at, access_count FROM episodic_vectors"
            ).fetchall()
            for r in rows:
                t_diff = now - r["created_at"]
                decayed = r["salience_score"] * math.exp(-self.DELTA_DECAY * t_diff)
                reinforce = self.ETA_REINFORCE * math.log(1.0 + r["access_count"])
                current_salience = max(0.0, decayed + reinforce)
                if current_salience < self.SALIENCE_FLOOR:
                    conn.execute("DELETE FROM episodic_vectors WHERE episode_id = ?", (r["episode_id"],))
                    try:
                        conn.execute("DELETE FROM episodic_fts WHERE episode_id = ?", (r["episode_id"],))
                    except Exception:
                        pass
                    pruned_count += 1
            conn.commit()
        return pruned_count
