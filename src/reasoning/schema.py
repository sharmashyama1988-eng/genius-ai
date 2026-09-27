"""
genius_schema.py — Neural Cognitive Schema for the Genius Dual-Core Engine.
Python 3.12 / Pydantic v2. Complete production implementation based on Neural Cognitive Schema v1.0.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# 3.0 — Shared primitives
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    """Returns current UTC timestamp with timezone."""
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    """Generates unique prefixed identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class LanguageCode(str, Enum):
    EN = "en"
    HI = "hi"
    HINGLISH = "hi-en"
    ES = "es"
    FR = "fr"
    DE = "de"
    OTHER = "other"


class SourceReliability(str, Enum):
    """Static tier assigned per-domain; overridden per-fetch by staleness checks."""
    TIER_1_ENCYCLOPEDIC = "tier_1_encyclopedic"   # Wikipedia, primary docs
    TIER_2_JOURNALISTIC = "tier_2_journalistic"   # Established news / academic
    TIER_3_WEB_GENERAL = "tier_3_web_general"     # DuckDuckGo general web
    TIER_4_UNVERIFIED = "tier_4_unverified"


class EngineTier(str, Enum):
    EDGE = "edge_qwen2_5_0_5b"
    CLOUD_SONNET = "cloud_claude_3_7_sonnet"
    CLOUD_OPUS = "cloud_claude_opus"
    OLLAMA = "ollama_local"


# ---------------------------------------------------------------------------
# 3.1 — EpistemicEvidence
# ---------------------------------------------------------------------------

class EpistemicEvidence(BaseModel):
    """A single verified evidentiary unit retrieved during Epistemic_Retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(default_factory=lambda: _new_id("ev"))
    source_url: str = Field(description="URL of source")
    source_reliability: SourceReliability = SourceReliability.TIER_1_ENCYCLOPEDIC
    title: str = Field(min_length=1, max_length=512)
    section_heading: Optional[str] = Field(default=None, max_length=256)
    excerpt: str = Field(min_length=1, max_length=8192)
    retrieved_at: datetime = Field(default_factory=_utc_now)
    language: LanguageCode = LanguageCode.EN
    embedding_hash: str = Field(
        default="",
        description="SHA-256 of the excerpt's vector or text representation, for dedup + cache keys.",
    )
    freshness_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Age of the underlying fact at fetch time; used for live-search decay weighting.",
    )

    @field_validator("excerpt")
    @classmethod
    def _strip_excerpt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("excerpt cannot be blank after normalization")
        return v

    @model_validator(mode="before")
    @classmethod
    def _ensure_embedding_hash(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("embedding_hash") and data.get("excerpt"):
                data["embedding_hash"] = hashlib.sha256(data["excerpt"].strip().encode("utf-8")).hexdigest()
        return data

    @computed_field  # type: ignore[misc]
    @property
    def decay_weight(self) -> float:
        """Exponential recency decay for time-sensitive claims. Half-life = 6h for web tier."""
        half_life = 21_600.0 if self.source_reliability == SourceReliability.TIER_3_WEB_GENERAL else 604_800.0
        return round(0.5 ** (self.freshness_seconds / half_life), 4)


# ---------------------------------------------------------------------------
# 3.2 — CognitiveThoughtStep
# ---------------------------------------------------------------------------

class CognitiveThoughtStep(BaseModel):
    """A single atom of reasoning inside the Latent_xThinking recursive loop."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(default_factory=lambda: _new_id("step"))
    step_index: int = Field(ge=0)
    hypothesis: str = Field(min_length=1, max_length=4096)
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="List of EpistemicEvidence.evidence_id values supporting this hypothesis.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    critique: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Self-critique text generated in the Self_Critique sub-node; null on first pass.",
    )
    revised_from: Optional[str] = Field(
        default=None, description="step_id of the CognitiveThoughtStep this one revises, if any."
    )
    token_cost: int = Field(default=0, ge=0)
    is_terminal: bool = Field(
        default=False, description="True if this step is the final accepted hypothesis for its claim slot."
    )

    @model_validator(mode="after")
    def _validate_revision_chain(self) -> Self:
        if self.revised_from and self.revised_from == self.step_id:
            raise ValueError("a thought step cannot revise itself")
        return self


# ---------------------------------------------------------------------------
# 3.3 — Grounding output (feeds Hallucination_Pruning / Grounding_Gate math)
# ---------------------------------------------------------------------------

class ClaimGroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str
    lexical_overlap: float = Field(ge=0.0, le=1.0)
    semantic_overlap: float = Field(ge=0.0, le=1.0)
    entity_preservation: float = Field(ge=0.0, le=1.0)
    salience_weight: float = Field(ge=0.0, default=1.0)
    supporting_evidence_ids: List[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def combined_grounding(self) -> float:
        """Weighted harmonic mean G(c_i) penalized by entity preservation P(c_i)."""
        beta_sq = 0.7 ** 2
        eps = 1e-6
        num = (1 + beta_sq) * self.lexical_overlap * self.semantic_overlap
        den = beta_sq * self.lexical_overlap + self.semantic_overlap + eps
        g = num / den
        return round(float(g * self.entity_preservation), 6)


class GroundingVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[ClaimGroundingResult] = Field(default_factory=list)
    contradiction_density: float = Field(ge=0.0, le=1.0, default=0.0)
    tau_base: float = 0.62
    lambda_coef: float = 0.25
    retries_used: int = Field(ge=0, default=0)
    max_retries: int = Field(ge=0, default=2)

    @computed_field  # type: ignore[misc]
    @property
    def tau_crit(self) -> float:
        """Dynamic threshold rising with contradiction density."""
        return round(self.tau_base + self.lambda_coef * self.contradiction_density, 6)

    @computed_field  # type: ignore[misc]
    @property
    def s_ground(self) -> float:
        """Weighted geometric mean across atomic claims with log-space accumulation."""
        if not self.claims:
            return 1.0
        total_w = sum(c.salience_weight for c in self.claims) or 1.0
        log_acc = sum(
            c.salience_weight * math.log(max(c.combined_grounding, 1e-9))
            for c in self.claims
        )
        return round(math.exp(log_acc / total_w), 6)

    @computed_field  # type: ignore[misc]
    @property
    def action(self) -> Literal["emit", "re_retrieve", "hitl_fallback"]:
        if self.s_ground >= self.tau_crit:
            return "emit"
        if self.retries_used < self.max_retries:
            return "re_retrieve"
        return "hitl_fallback"

    @computed_field  # type: ignore[misc]
    @property
    def unsupported_claims(self) -> List[str]:
        return [c.claim_text for c in self.claims if c.combined_grounding < self.tau_crit]


# ---------------------------------------------------------------------------
# 3.4 — ReasoningTrajectory
# ---------------------------------------------------------------------------

class LanguageProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: LanguageCode
    secondary: Optional[LanguageCode] = None
    code_switch_ratio: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Fraction of tokens attributable to secondary language (Hinglish detection).",
    )
    dialect_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class LatencyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_deconstruction_ms: float = Field(ge=0.0, default=0.0)
    retrieval_ms: float = Field(ge=0.0, default=0.0)
    thinking_ms: float = Field(ge=0.0, default=0.0)
    pruning_ms: float = Field(ge=0.0, default=0.0)
    synthesis_ms: float = Field(ge=0.0, default=0.0)

    @computed_field  # type: ignore[misc]
    @property
    def total_ms(self) -> float:
        return round(
            self.query_deconstruction_ms
            + self.retrieval_ms
            + self.thinking_ms
            + self.pruning_ms
            + self.synthesis_ms,
            3,
        )


class ReasoningTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str = Field(default_factory=lambda: _new_id("traj"))
    session_id: str
    engine_used: EngineTier = EngineTier.EDGE
    thought_steps: List[CognitiveThoughtStep] = Field(default_factory=list)
    evidence_pool: List[EpistemicEvidence] = Field(default_factory=list)
    grounding_verdict: GroundingVerdict = Field(default_factory=GroundingVerdict)
    language_profile: LanguageProfile = Field(
        default_factory=lambda: LanguageProfile(primary=LanguageCode.EN)
    )
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    thinking_tokens: int = Field(ge=0, default=0)
    thinking_budget: int = Field(ge=0, default=4096)
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("thought_steps")
    @classmethod
    def _steps_sequential(cls, v: List[CognitiveThoughtStep]) -> List[CognitiveThoughtStep]:
        indices = [s.step_index for s in v]
        if indices != sorted(indices):
            raise ValueError("thought_steps must be ordered by ascending step_index")
        return v

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.thinking_tokens

    @computed_field  # type: ignore[misc]
    @property
    def budget_utilization(self) -> float:
        return round(self.thinking_tokens / self.thinking_budget, 4) if self.thinking_budget else 0.0

    @computed_field  # type: ignore[misc]
    @property
    def converged(self) -> bool:
        return self.grounding_verdict.action == "emit"


# ---------------------------------------------------------------------------
# 3.5 — NeuralStateSnapshot (SQLite-serializable session state)
# ---------------------------------------------------------------------------

class HITLGuardEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: _new_id("hitl"))
    triggering_action: str = Field(description="e.g. 'rm_rf', 'db_drop', 'destructive_shell_op'")
    risk_tier: Literal["low", "medium", "high", "critical"]
    human_ack_required: bool = True
    human_ack_received: bool = False
    ack_timestamp: Optional[datetime] = None


class NeuralStateSnapshot(BaseModel):
    """Complete, persistable session state. Round-trips through SQLite via
    model_dump_json() / model_validate_json()."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    session_id: str = Field(default_factory=lambda: _new_id("sess"))
    user_query_raw: str
    trajectories: List[ReasoningTrajectory] = Field(default_factory=list)
    hitl_events: List[HITLGuardEvent] = Field(default_factory=list)
    final_response_text: Optional[str] = None
    citation_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps inline index labels e.g. '[1]' to EpistemicEvidence.evidence_id.",
    )
    degraded_mode: bool = Field(
        default=False,
        description="True if the response emitted below tau_crit after exhausting retries.",
    )
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _sync_updated_at(self) -> Self:
        object.__setattr__(self, "updated_at", _utc_now())
        return self

    @computed_field  # type: ignore[misc]
    @property
    def state_hash(self) -> str:
        payload = self.model_dump_json(exclude={"updated_at"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # -- SQLite persistence ------------------------------------------------

    @staticmethod
    def ddl() -> str:
        return """
        CREATE TABLE IF NOT EXISTS neural_state_snapshots (
            session_id      TEXT PRIMARY KEY,
            state_hash      TEXT NOT NULL,
            payload_json    TEXT NOT NULL,
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_updated
            ON neural_state_snapshots(updated_at DESC);
        """

    def persist(self, conn: sqlite3.Connection) -> None:
        conn.executescript(self.ddl())
        conn.execute(
            """
            INSERT INTO neural_state_snapshots
                (session_id, state_hash, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                state_hash = excluded.state_hash,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (
                self.session_id,
                self.state_hash,
                self.model_dump_json(),
                self.created_at.timestamp(),
                time.time(),
            ),
        )
        conn.commit()

    @classmethod
    def recover(cls, conn: sqlite3.Connection, session_id: str) -> Optional[NeuralStateSnapshot]:
        cls.ddl()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT payload_json FROM neural_state_snapshots WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return cls.model_validate_json(row[0])
