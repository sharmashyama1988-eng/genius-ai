"""
Mathematical Grounding & Confidence Engine for Genius.
Implements §2 of the Neural Cognitive Schema v1.0:
- Atomic claim extraction
- Longest Common Subsequence (ROUGE-L) lexical overlap L(c_i)
- Semantic cosine overlap Sem(c_i)
- Entity preservation penalty P(c_i)
- Salience weight calculation w_i
- Dynamic threshold tau_crit and aggregate grounding score S_ground
- Contradiction matrix and contradiction density D
- Targeted query reformulation for self-targeted retrieval loops
"""

from __future__ import annotations

import collections
import math
import re
from typing import Dict, List, Set, Tuple

from .schema import (
    ClaimGroundingResult,
    EpistemicEvidence,
    GroundingVerdict,
)


class GroundingEngine:
    """Evaluates factual grounding of candidate responses against retrieved epistemic evidence."""

    def __init__(self, tau_base: float = 0.62, lambda_coef: float = 0.25) -> None:
        self.tau_base = tau_base
        self.lambda_coef = lambda_coef

    @staticmethod
    def extract_atomic_claims(text: str) -> List[str]:
        """Decomposes a candidate response into atomic declarative claims (sentences/clauses)."""
        # Remove markdown tags, code blocks, or thinking traces
        cleaned = re.sub(r"```[\s\S]*?```", "", text)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\[\d+\]", "", cleaned)  # remove existing citations for scoring

        # Split into sentences based on punctuation (. ! ? \n and Hindi purna viram ।)
        raw_sentences = re.split(r"(?<=[.!?।\n])\s+", cleaned)
        claims: List[str] = []

        for s in raw_sentences:
            s_clean = s.strip()
            # Filter out questions, greetings, empty or very short conversational fragments
            if len(s_clean) > 20 and not s_clean.endswith("?"):
                # If compound sentence with coordinating conjunction, can split or keep
                claims.append(s_clean)

        if not claims and cleaned.strip():
            claims.append(cleaned.strip())

        return claims

    @staticmethod
    def extract_named_entities(text: str) -> Set[str]:
        """Extracts candidate entities: capitalized multi-word phrases, alphanumeric terms, dates, and numbers."""
        entities: Set[str] = set()

        # Capitalized noun phrases (e.g., 'Albert Einstein', 'New Delhi', 'Qwen2.5')
        cap_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)*\b", text)
        for p in cap_phrases:
            if len(p) > 2 and p.lower() not in {"the", "this", "that", "there", "what", "which", "when", "here"}:
                entities.add(p.lower())

        # Specific numbers, dates, version strings, identifiers (e.g. 1947, 0.5B, v1.0)
        numerics = re.findall(r"\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b", text)
        for n in numerics:
            entities.add(n.lower())

        return entities

    @staticmethod
    def compute_lcs(s1: str, s2: str) -> int:
        """Computes length of Longest Common Subsequence between two word sequences."""
        tokens1 = re.findall(r"\w+", s1.lower())
        tokens2 = re.findall(r"\w+", s2.lower())
        m, n = len(tokens1), len(tokens2)
        if m == 0 or n == 0:
            return 0

        # Space-optimized DP table
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def compute_lexical_overlap(self, claim: str, evidence_pool: List[EpistemicEvidence]) -> Tuple[float, str]:
        """ROUGE-L based lexical overlap: L(c_i) = max_{e_j} LCS(c_i, e_j) / max(|c_i|, |e_j|)."""
        if not evidence_pool:
            return 0.0, ""

        claim_tokens = len(re.findall(r"\w+", claim))
        if claim_tokens == 0:
            return 0.0, ""

        max_l = 0.0
        best_ev_id = ""

        for ev in evidence_pool:
            lcs_len = self.compute_lcs(claim, ev.excerpt)
            score = lcs_len / max(claim_tokens, 1)
            # Factor in evidence freshness decay weight
            decayed_score = min(score * ev.decay_weight, 1.0)
            if decayed_score > max_l:
                max_l = decayed_score
                best_ev_id = ev.evidence_id

        return round(min(max_l, 1.0), 6), best_ev_id

    @staticmethod
    def compute_semantic_overlap(claim: str, evidence_pool: List[EpistemicEvidence]) -> Tuple[float, str]:
        """Calculates cosine semantic overlap using token character/word n-gram TF vectors."""
        if not evidence_pool:
            return 0.0, ""

        def get_term_vector(text: str) -> Dict[str, float]:
            words = re.findall(r"\w+", text.lower())
            vec: Dict[str, float] = collections.defaultdict(float)
            for w in words:
                vec[w] += 1.0
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            return {k: v / norm for k, v in vec.items()}

        claim_vec = get_term_vector(claim)
        max_sem = 0.0
        best_ev_id = ""

        for ev in evidence_pool:
            ev_vec = get_term_vector(ev.excerpt)
            # Cosine dot product
            dot = sum(claim_vec[k] * ev_vec.get(k, 0.0) for k in claim_vec)
            if dot > max_sem:
                max_sem = dot
                best_ev_id = ev.evidence_id

        return round(min(max_sem, 1.0), 6), best_ev_id

    def compute_entity_preservation(self, claim: str, evidence_pool: List[EpistemicEvidence]) -> Tuple[float, Set[str]]:
        r"""P(c_i) = 1 - (|Ent(c_i) \ Ent(E)| / (|Ent(c_i)| + 1)). Returns score and unsupported entities."""
        claim_entities = self.extract_named_entities(claim)
        if not claim_entities:
            return 1.0, set()

        all_evidence_entities: Set[str] = set()
        for ev in evidence_pool:
            all_evidence_entities.update(self.extract_named_entities(ev.excerpt))
            all_evidence_entities.update(self.extract_named_entities(ev.title))

        unsupported = claim_entities - all_evidence_entities
        penalty = len(unsupported) / (len(claim_entities) + 1.0)
        preservation = max(0.0, 1.0 - penalty)
        return round(preservation, 6), unsupported

    @staticmethod
    def compute_contradiction_density(evidence_pool: List[EpistemicEvidence]) -> float:
        """Computes contradiction density D in [0, 1] across evidence sources."""
        if len(evidence_pool) < 2:
            return 0.0

        # Scan for conflicting polarity or contradictory numeric claims
        negation_terms = {"not", "never", "no", "false", "disputed", "rejected", "contrary", "denied"}
        polarities = []
        for ev in evidence_pool:
            words = set(re.findall(r"\w+", ev.excerpt.lower()))
            has_neg = any(term in words for term in negation_terms)
            polarities.append(has_neg)

        true_count = sum(1 for p in polarities if p)
        false_count = len(polarities) - true_count

        # Density based on polarity disparity
        divergence = (2 * min(true_count, false_count)) / len(evidence_pool)
        return round(min(divergence, 1.0), 4)

    def evaluate_response_grounding(
        self,
        candidate_response: str,
        evidence_pool: List[EpistemicEvidence],
        retries_used: int = 0,
        max_retries: int = 2,
    ) -> GroundingVerdict:
        """Full execution of §2 Grounding & Confidence Function."""
        claims = self.extract_atomic_claims(candidate_response)

        if not claims or not evidence_pool:
            # If no evidence or no claims, emit baseline zero verdict
            return GroundingVerdict(
                claims=[],
                contradiction_density=0.0,
                retries_used=retries_used,
                max_retries=max_retries,
            )

        contradiction_density = self.compute_contradiction_density(evidence_pool)
        claim_results: List[ClaimGroundingResult] = []

        # Calculate word frequencies for salience weights
        full_tokens = re.findall(r"\w+", candidate_response.lower())
        total_tokens = len(full_tokens) or 1
        term_freq = collections.Counter(full_tokens)

        # Salience weights w_i
        raw_salience = []
        for c in claims:
            c_tokens = re.findall(r"\w+", c.lower())
            mass = sum(term_freq[t] / total_tokens for t in c_tokens)
            raw_salience.append(mass)

        avg_salience = (sum(raw_salience) / len(raw_salience)) if raw_salience else 1.0
        normalized_weights = [round(max(0.2, (s / avg_salience)), 4) for s in raw_salience]

        for idx, claim in enumerate(claims):
            lex_score, lex_ev = self.compute_lexical_overlap(claim, evidence_pool)
            sem_score, sem_ev = self.compute_semantic_overlap(claim, evidence_pool)
            ent_pres, _ = self.compute_entity_preservation(claim, evidence_pool)

            supporting_ids = list(set(filter(None, [lex_ev, sem_ev])))

            result = ClaimGroundingResult(
                claim_text=claim,
                lexical_overlap=lex_score,
                semantic_overlap=sem_score,
                entity_preservation=ent_pres,
                salience_weight=normalized_weights[idx],
                supporting_evidence_ids=supporting_ids,
            )
            claim_results.append(result)

        verdict = GroundingVerdict(
            claims=claim_results,
            contradiction_density=contradiction_density,
            tau_base=self.tau_base,
            lambda_coef=self.lambda_coef,
            retries_used=retries_used,
            max_retries=max_retries,
        )
        return verdict

    def reformulate_query(
        self,
        original_query: str,
        verdict: GroundingVerdict,
        evidence_pool: List[EpistemicEvidence],
    ) -> str:
        """Targeted query reformulation: query' = query ⊕ {unsupported entities in c_i : G(c_i) < tau_crit}."""
        unsupported_entities: Set[str] = set()

        for claim in verdict.claims:
            if claim.combined_grounding < verdict.tau_crit:
                _, unsupp = self.compute_entity_preservation(claim.claim_text, evidence_pool)
                unsupported_entities.update(unsupp)

        if not unsupported_entities:
            return original_query

        # Append targeted entities to original query
        entity_str = " ".join(list(unsupported_entities)[:5])
        return f"{original_query} {entity_str}".strip()
