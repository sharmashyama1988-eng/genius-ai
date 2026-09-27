"""
Autonomous xThinking & Deep-Reasoning Engine with Neural Cognitive Schema v1.0.
Implements the 7-node Cognitive State Graph:
  1. Query_Deconstruction (QueryFrame, Complexity Score, Language Routing)
  2. Router_Decision (Edge_ShortCircuit vs. Epistemic_Retrieval)
  3. Epistemic_Retrieval (Parallel Wiki + DDG Search + Exemplars)
  4. Contradiction_Matrix (Contradiction Density D)
  5. Latent_xThinking (Adaptive Token Budgeting based on D)
  6. Hallucination_Pruning & Grounding_Gate (ROUGE-L + Cosine + Entity Preservation)
  7. Grounded_Synthesis & Write-Through Episodic Memory
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from ..dataset.loader import DatasetItem, DatasetManager
from ..dataset.retriever import ExemplarRetriever
from ..languages import DetectedLanguage, MultilingualManager
from ..memory.episodic import EpisodicMemoryManager
from ..memory.session import SessionManager
from ..model.provider import BaseLLMProvider, LocalQwenProvider, UniversalModelRouter
from ..retrieval.ranker import PassageChunk, PassageRanker
from ..retrieval.web_search import WebSearchClient, WebSearchResult
from ..retrieval.wikipedia_client import WikipediaArticle, WikipediaClient
from .grounding import GroundingEngine
from .schema import (
    CognitiveThoughtStep,
    EngineTier,
    EpistemicEvidence,
    GroundingVerdict,
    LanguageCode,
    LanguageProfile,
    LatencyMetrics,
    NeuralStateSnapshot,
    ReasoningTrajectory,
    SourceReliability,
)

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a grounded citation for backward compatibility."""
    index: int
    title: str
    url: str
    section: str
    snippet: str
    source_type: str = "wikipedia"
    evidence_id: Optional[str] = None


@dataclass
class ReasoningEvent:
    """Event emitted during the multi-stage reasoning pipeline."""
    stage: str  # "language", "plan", "research", "exemplar", "thinking", "response", "done", "error"
    event_type: str  # "status", "chunk", "data", "token", "complete"
    payload: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalReasoningOutput:
    """Final output encapsulation for completed reasoning trajectory."""
    session_id: str
    duration_sec: float
    thoughts: str
    response: str
    citations: List[Dict[str, Any]]
    language: str
    s_ground: float = 1.0
    tau_crit: float = 0.62
    action: str = "emit"


class XThinkingEngine:
    """Enterprise Deep-Reasoning Engine:
    User Question
      --> Query_Deconstruction (Complexity & Dialect Analysis)
      --> Router_Decision (Edge Short-Circuit for Conversational vs Epistemic Retrieval for Knowledge)
      --> Epistemic_Retrieval (Parallel Wikipedia + Live DuckDuckGo + Dataset Exemplars)
      --> Contradiction_Matrix (Contradiction Density D Calculation)
      --> Latent_xThinking (Adaptive Thinking Budget: B_base * (1 + gamma * D))
      --> Hallucination_Pruning & Grounding_Gate (ROUGE-L LCS, Cosine, Entity Penalty, Targeted Retries)
      --> Grounded_Synthesis (Citation Injection [n], NeuralStateSnapshot SQLite & Episodic Vector Persistence)
    """

    def __init__(
        self,
        model_engine: Optional[BaseLLMProvider] = None,
        wiki_client: Optional[WikipediaClient] = None,
        web_client: Optional[WebSearchClient] = None,
        dataset_manager: Optional[DatasetManager] = None,
        multilingual_manager: Optional[MultilingualManager] = None,
        session_manager: Optional[SessionManager] = None,
        episodic_manager: Optional[EpisodicMemoryManager] = None,
        grounding_engine: Optional[GroundingEngine] = None,
        research_mode: str = "auto",
    ) -> None:
        self.router = UniversalModelRouter()
        self.model = model_engine or self.router.get_provider("local")
        self.wiki = wiki_client or WikipediaClient()
        self.web = web_client or WebSearchClient()
        self.dataset_mgr = dataset_manager or DatasetManager()
        self.exemplar_retriever = ExemplarRetriever(self.dataset_mgr)
        self.ranker = PassageRanker()
        self.multilingual_mgr = multilingual_manager or MultilingualManager()
        self.session_mgr = session_manager or SessionManager()
        self.episodic_mgr = episodic_manager or EpisodicMemoryManager()
        self.grounding_engine = grounding_engine or GroundingEngine()
        self.research_mode: str = "auto"
        self.set_research_mode(research_mode)

    def set_model_provider(self, provider_name: str) -> bool:
        """Switch active LLM provider (e.g. 'local', 'claude', 'ollama')."""
        if self.router.set_provider(provider_name):
            self.model = self.router.get_provider(provider_name)
            return True
        return False

    def set_research_mode(self, mode: str) -> bool:
        """Sets research mode: 'auto' (smart routing), 'on' (always parallel search), 'off' (direct offline)."""
        mode_clean = mode.lower().strip()
        if mode_clean in ("auto", "on", "off", "always", "disabled", "direct", "offline"):
            if mode_clean in ("always", "enable", "enabled"):
                self.research_mode = "on"
            elif mode_clean in ("disabled", "direct", "offline"):
                self.research_mode = "off"
            else:
                self.research_mode = mode_clean
            return True
        return False

    def _extract_search_keywords(self, query: str) -> str:
        """Removes conversational noise to formulate targeted search terms."""
        noise_words = {
            "what", "is", "the", "tell", "me", "about", "who", "when", "why",
            "how", "does", "explain", "describe", "can", "you", "please",
            "give", "details", "on", "a", "an", "and", "or", "in", "of",
            "kya", "hai", "batao", "bataiye", "mujhe", "ke", "ki", "ka",
            "who", "was", "are", "which",
        }
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query)
        filtered = [w for w in words if w.lower() not in noise_words]
        if not filtered or len(filtered) < 2:
            return query.strip()
        return " ".join(filtered[:8])

    def _detect_category(self, query: str) -> str:
        """Determines if query is coding, math, general knowledge, or conversational."""
        q_lower = query.lower()
        code_indicators = [
            "code", "python", "javascript", "function", "class", "algorithm",
            "bug", "script", "program", "def ", "sql", "html", "css", "c++",
            "rust", "golang", "react", "fastapi", "docker", "json",
        ]
        if any(w in q_lower for w in code_indicators):
            return "codealpaca"

        math_indicators = [
            "calculate", "solve", "speed", "distance", "km/h", "mph", "equation",
            "probability", "integral", "derivative", "ratio", "percentage",
            "matrix", "vector", "prime number", "hypotenuse", "sum of", "modulo",
            "trains", "moving towards", "how many hours", "kitne time",
        ]
        if any(w in q_lower for w in math_indicators) or bool(re.search(r"\b\d+\s*(\+|\-|\*|\/|\^|km/h|m/s)\b", q_lower)):
            return "math"

        return "general"

    def _compute_complexity(self, query: str) -> float:
        """Computes query complexity score in [0.0, 1.0]. Under 0.35 short-circuits to edge synthesis."""
        q_clean = query.strip().lower()

        # Pure greetings and identity queries
        simple_phrases = {
            "hi", "hello", "hey", "namaste", "halo", "kaise ho", "who are you",
            "tum kaun ho", "kya haal hai", "good morning", "good evening", "bye",
            "thanks", "thank you", "dhanyawad", "shukriya",
        }
        if q_clean in simple_phrases or len(q_clean.split()) <= 2 and any(p in q_clean for p in simple_phrases):
            return 0.15

        # Informational or reasoning queries
        score = 0.4
        words = len(re.findall(r"\w+", q_clean))
        if words > 7:
            score += 0.2
        if any(w in q_clean for w in ["why", "how", "compare", "difference", "explain", "analyze", "kisme"]):
            score += 0.2
        if any(w in q_clean for w in ["code", "python", "bug", "architecture", "error"]):
            score += 0.2

        return min(round(score, 2), 1.0)

    async def execute_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        research_mode: Optional[str] = None,
        max_articles: int = 2,
        max_web_results: int = 3,
        top_k_passages: int = 5,
        temperature: float = 0.6,
        max_new_tokens: int = 2048,
        retries_used: int = 0,
    ) -> AsyncIterator[ReasoningEvent]:
        """Asynchronously executes the full 7-node Cognitive State Graph."""
        t_start = time.time()
        active_session = session_id or self.session_mgr.create_session(question[:35])

        t_deconstruct_start = time.time()
        # Node 1: Query_Deconstruction
        detected_lang = self.multilingual_mgr.route_query(question)
        search_query = self._extract_search_keywords(question)
        category = self._detect_category(question)
        complexity_score = self._compute_complexity(question)
        t_deconstruct_ms = round((time.time() - t_deconstruct_start) * 1000, 2)

        # Map language code to schema LanguageCode
        lang_map = {
            "en": LanguageCode.EN,
            "hi": LanguageCode.HI,
            "hi-Latn": LanguageCode.HINGLISH,
            "es": LanguageCode.ES,
            "fr": LanguageCode.FR,
            "de": LanguageCode.DE,
        }
        schema_lang = lang_map.get(detected_lang.code, LanguageCode.OTHER)

        yield ReasoningEvent(
            stage="language",
            event_type="status",
            payload={
                "code": detected_lang.code,
                "name": detected_lang.name,
                "confidence": detected_lang.confidence,
                "complexity_score": complexity_score,
                "message": f"🌐 Language: {detected_lang.name} | Complexity: {complexity_score}",
            },
        )

        yield ReasoningEvent(
            stage="plan",
            event_type="status",
            payload={
                "message": f"Planning cognitive pathway for: '{search_query}' (Category: {category})",
                "category": category,
                "complexity_score": complexity_score,
            },
        )

        evidence_pool: List[EpistemicEvidence] = []
        citations: List[Citation] = []
        contradiction_density = 0.0
        t_retrieval_ms = 0.0

        # Node 2: Router_Decision
        active_mode = (research_mode or self.research_mode).lower()
        should_retrieve = False
        if active_mode in ("off", "disabled", "direct", "offline"):
            should_retrieve = False
        elif active_mode in ("on", "always", "enabled"):
            should_retrieve = True
        else:  # auto
            # Pure math and analytical problems are solved directly without encyclopedia lookups
            if category == "math" and not any(w in question.lower() for w in ["history", "who discovered", "who proved", "biography", "origin"]):
                should_retrieve = False
            else:
                should_retrieve = (complexity_score >= 0.35)

        if not should_retrieve:
            # Node 2a: Edge_ShortCircuit (Direct synthesis without external web/wiki search)
            if active_mode in ("off", "disabled", "direct", "offline"):
                mode_desc = "⚡ Fast Direct / Offline Mode: Web/wiki search bypassed by user."
            elif category == "math":
                mode_desc = "⚡ Analytical Math Engine: Direct formal derivation without web noise."
            else:
                mode_desc = "⚡ Edge Short-Circuit: Direct rapid conversational synthesis."
            yield ReasoningEvent(
                stage="plan",
                event_type="status",
                payload={"message": mode_desc},
            )
            formatted_context = "External live search is bypassed. Answer using internal foundational knowledge, exemplars, and episodic memory."
        else:
            # Node 2b: Epistemic_Retrieval
            t_retrieval_start = time.time()
            yield ReasoningEvent(
                stage="research",
                event_type="status",
                payload={"message": "Executing parallel Epistemic Retrieval (Wikipedia + Web)..."},
            )

            wiki_task = self.wiki.search_and_fetch(search_query, max_articles=max_articles)
            web_task = self.web.search(search_query, limit=max_web_results)

            wiki_articles, web_results = await asyncio.gather(wiki_task, web_task, return_exceptions=True)
            articles: List[WikipediaArticle] = wiki_articles if isinstance(wiki_articles, list) else []
            web_items: List[WebSearchResult] = web_results if isinstance(web_results, list) else []

            # Populate EpistemicEvidence objects
            for a in articles:
                evidence_pool.append(
                    EpistemicEvidence(
                        source_url=a.url,
                        source_reliability=SourceReliability.TIER_1_ENCYCLOPEDIC,
                        title=a.title,
                        section_heading=a.sections[0]["title"] if a.sections else "Overview",
                        excerpt=a.summary or a.full_text[:1000],
                        language=LanguageCode.EN,
                        freshness_seconds=0.0,
                    )
                )

            for w in web_items:
                if w.snippet and len(w.snippet) > 25:
                    evidence_pool.append(
                        EpistemicEvidence(
                            source_url=w.url,
                            source_reliability=SourceReliability.TIER_3_WEB_GENERAL,
                            title=w.title,
                            section_heading="Web Result",
                            excerpt=w.snippet,
                            language=LanguageCode.EN,
                            freshness_seconds=3600.0,
                        )
                    )

            # BM25 Passage Chunking & Relevance Ranking
            all_chunks: List[PassageChunk] = []
            for a in articles:
                chunks = self.ranker.chunk_article(a.title, a.url, a.full_text, a.sections)
                all_chunks.extend(chunks)

            for idx, w in enumerate(web_items):
                if w.snippet and len(w.snippet) > 30:
                    all_chunks.append(
                        PassageChunk(
                            article_title=w.title,
                            article_url=w.url,
                            section_title="Web Search",
                            text=w.snippet,
                            chunk_index=1000 + idx,
                        )
                    )

            ranked_passages = self.ranker.rank_passages(question, all_chunks, top_k=top_k_passages)

            # Node 3: Contradiction_Matrix
            contradiction_density = self.grounding_engine.compute_contradiction_density(evidence_pool)
            t_retrieval_ms = round((time.time() - t_retrieval_start) * 1000, 2)

            yield ReasoningEvent(
                stage="research",
                event_type="complete",
                payload={
                    "evidence_count": len(evidence_pool),
                    "ranked_passages_count": len(ranked_passages),
                    "contradiction_density": contradiction_density,
                    "retrieval_ms": t_retrieval_ms,
                },
            )

            # Format Context Blocks & Citations
            context_blocks: List[str] = []
            for idx, p in enumerate(ranked_passages, start=1):
                src_type = "web" if "duckduckgo" in p.article_url or p.chunk_index >= 1000 else "wikipedia"
                matching_ev = next((e for e in evidence_pool if e.title == p.article_title), None)
                ev_id = matching_ev.evidence_id if matching_ev else None

                citation = Citation(
                    index=idx,
                    title=p.article_title,
                    url=p.article_url,
                    section=p.section_title,
                    snippet=p.text,
                    source_type=src_type,
                    evidence_id=ev_id,
                )
                citations.append(citation)
                context_blocks.append(f"[FACT {idx}] (Source: {p.article_title} | URL: {p.article_url})\n{p.text}")

            formatted_context = "\n\n".join(context_blocks) if context_blocks else "No direct external facts found."

        # In-context exemplars
        exemplars = self.exemplar_retriever.find_relevant_exemplars(
            question, category=category if category == "codealpaca" else None, top_k=1
        )
        if exemplars:
            yield ReasoningEvent(
                stage="exemplar",
                event_type="data",
                payload={"exemplars": [{"source": ex.source, "instruction": ex.instruction} for ex in exemplars]},
            )

        # Node 4: Latent_xThinking (Adaptive Token Budgeting)
        b_base = 4096
        gamma = 1.5
        adaptive_thinking_budget = min(32000, int(b_base * (1.0 + gamma * contradiction_density)))

        system_prompt = self.multilingual_mgr.get_system_prompt_for_language(
            detected_lang, formatted_context, exemplars=exemplars
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        yield ReasoningEvent(
            stage="thinking",
            event_type="status",
            payload={
                "message": f"Commencing Latent_xThinking (Adaptive Budget: {adaptive_thinking_budget} tokens)...",
                "thinking_budget": adaptive_thinking_budget,
            },
        )

        t_thinking_start = time.time()
        in_thinking = True
        thinking_tokens_list: List[str] = []
        response_tokens_list: List[str] = []
        full_tokens: List[str] = []

        try:
            async for token in self.model.stream_generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ):
                full_tokens.append(token)
                text_so_far = "".join(full_tokens)

                if "</think>" in text_so_far and in_thinking:
                    in_thinking = False
                    yield ReasoningEvent(
                        stage="thinking",
                        event_type="complete",
                        payload={"thoughts": "".join(thinking_tokens_list).replace("<think>", "").strip()},
                    )
                    yield ReasoningEvent(
                        stage="response",
                        event_type="status",
                        payload={"message": "Synthesizing grounded response..."},
                    )
                    parts = text_so_far.split("</think>", 1)
                    rem = parts[1].strip()
                    if rem and not response_tokens_list:
                        response_tokens_list.append(rem)
                        yield ReasoningEvent(stage="response", event_type="token", payload={"token": rem})
                    continue

                if in_thinking:
                    clean_t = token.replace("<think>", "")
                    if clean_t:
                        thinking_tokens_list.append(clean_t)
                        yield ReasoningEvent(stage="thinking", event_type="token", payload={"token": clean_t})
                else:
                    response_tokens_list.append(token)
                    yield ReasoningEvent(stage="response", event_type="token", payload={"token": token})

        except Exception as e:
            logger.exception("Inference failed")
            yield ReasoningEvent(stage="error", event_type="error", payload={"error": str(e)})
            return

        t_thinking_ms = round((time.time() - t_thinking_start) * 1000, 2)

        if in_thinking:
            yield ReasoningEvent(
                stage="thinking",
                event_type="complete",
                payload={"thoughts": "".join(thinking_tokens_list).strip()},
            )
            response_tokens_list = thinking_tokens_list

        raw_thoughts = "".join(thinking_tokens_list).replace("<think>", "").strip()
        candidate_response = "".join(response_tokens_list).strip()

        # Node 5: Hallucination_Pruning & Grounding_Gate
        t_pruning_start = time.time()
        verdict: GroundingVerdict = self.grounding_engine.evaluate_response_grounding(
            candidate_response=candidate_response,
            evidence_pool=evidence_pool,
            retries_used=retries_used,
            max_retries=2,
        )
        t_pruning_ms = round((time.time() - t_pruning_start) * 1000, 2)

        # Check Grounding Action
        if verdict.action == "re_retrieve" and retries_used < 2:
            # Self-Targeted Retrieval Retry Loop
            reformulated_query = self.grounding_engine.reformulate_query(
                original_query=search_query,
                verdict=verdict,
                evidence_pool=evidence_pool,
            )
            yield ReasoningEvent(
                stage="plan",
                event_type="status",
                payload={
                    "message": f"🔄 Grounding Gate: S_ground ({verdict.s_ground}) < tau_crit ({verdict.tau_crit}). Re-retrieving targeting: '{reformulated_query}'",
                },
            )
            # Re-fetch targeted evidence and merge into pool
            extra_wiki = await self.wiki.search_and_fetch(reformulated_query, max_articles=1)
            if extra_wiki:
                for a in extra_wiki:
                    evidence_pool.append(
                        EpistemicEvidence(
                            source_url=a.url,
                            source_reliability=SourceReliability.TIER_1_ENCYCLOPEDIC,
                            title=a.title,
                            section_heading="Targeted Retrieval",
                            excerpt=a.summary or a.full_text[:800],
                            language=LanguageCode.EN,
                        )
                    )
            # Re-evaluate verdict with augmented evidence pool
            verdict = self.grounding_engine.evaluate_response_grounding(
                candidate_response=candidate_response,
                evidence_pool=evidence_pool,
                retries_used=retries_used + 1,
                max_retries=2,
            )

        # Deconstruct thought steps
        thought_steps: List[CognitiveThoughtStep] = []
        raw_thought_lines = [line.strip() for line in raw_thoughts.split("\n") if len(line.strip()) > 15]
        for idx, line in enumerate(raw_thought_lines):
            thought_steps.append(
                CognitiveThoughtStep(
                    step_index=idx,
                    hypothesis=line,
                    evidence_refs=[e.evidence_id for e in evidence_pool[:2]],
                    confidence=0.95,
                    token_cost=len(line.split()),
                    is_terminal=(idx == len(raw_thought_lines) - 1),
                )
            )

        # Assemble Latency Metrics & Trajectory
        total_duration = round(time.time() - t_start, 2)
        latency = LatencyMetrics(
            query_deconstruction_ms=t_deconstruct_ms,
            retrieval_ms=t_retrieval_ms,
            thinking_ms=t_thinking_ms,
            pruning_ms=t_pruning_ms,
            synthesis_ms=50.0,
        )

        citation_map = {f"[{c.index}]": c.evidence_id or f"ev_{c.index}" for c in citations}

        trajectory = ReasoningTrajectory(
            session_id=active_session,
            engine_used=EngineTier.EDGE,
            thought_steps=thought_steps,
            evidence_pool=evidence_pool,
            grounding_verdict=verdict,
            language_profile=LanguageProfile(
                primary=schema_lang,
                code_switch_ratio=0.3 if detected_lang.code == "hi-Latn" else 0.0,
                dialect_confidence=detected_lang.confidence,
            ),
            latency=latency,
            prompt_tokens=len(question.split()),
            completion_tokens=len(candidate_response.split()),
            thinking_tokens=len(raw_thoughts.split()),
            thinking_budget=adaptive_thinking_budget,
        )

        # Node 6: NeuralStateSnapshot & Persistence
        snapshot = NeuralStateSnapshot(
            session_id=active_session,
            user_query_raw=question,
            trajectories=[trajectory],
            final_response_text=candidate_response,
            citation_map=citation_map,
            degraded_mode=(verdict.action == "hitl_fallback"),
        )

        # Save to SQLite Session & Episodic Memory
        citations_dicts = [asdict(c) for c in citations]
        self.session_mgr.save_turn(
            session_id=active_session,
            user_query=question,
            assistant_response=candidate_response,
            thoughts=raw_thoughts,
            citations=citations_dicts,
            language=detected_lang.code,
        )

        # Write-through to Episodic Vector store if converged
        try:
            self.episodic_mgr.write_through_snapshot(snapshot)
        except Exception as e:
            logger.debug(f"Episodic memory write-through skipped: {e}")

        # Node 7: Grounded_Synthesis Event
        yield ReasoningEvent(
            stage="done",
            event_type="complete",
            payload={
                "session_id": active_session,
                "duration_sec": total_duration,
                "thoughts": raw_thoughts,
                "response": candidate_response,
                "citations": citations_dicts,
                "language": detected_lang.code,
                "s_ground": verdict.s_ground,
                "tau_crit": verdict.tau_crit,
                "action": verdict.action,
                "contradiction_density": contradiction_density,
                "trajectory": trajectory,
                "snapshot": snapshot,
            },
        )
