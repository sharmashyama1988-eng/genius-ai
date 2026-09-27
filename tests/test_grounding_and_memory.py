"""Unit tests for Mathematical Grounding Engine and Episodic Memory."""

import math
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.episodic import EpisodicMemoryManager
from src.reasoning.grounding import GroundingEngine
from src.reasoning.schema import (
    EngineTier,
    EpistemicEvidence,
    LanguageCode,
    LanguageProfile,
    LatencyMetrics,
    NeuralStateSnapshot,
    ReasoningTrajectory,
    SourceReliability,
)


class TestGroundingAndMemory(unittest.TestCase):

    def setUp(self):
        self.grounding = GroundingEngine(tau_base=0.62, lambda_coef=0.25)
        self.evidence = [
            EpistemicEvidence(
                source_url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                source_reliability=SourceReliability.TIER_1_ENCYCLOPEDIC,
                title="Python (programming language)",
                excerpt="Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
            ),
            EpistemicEvidence(
                source_url="https://duckduckgo.com/q=Python",
                source_reliability=SourceReliability.TIER_3_WEB_GENERAL,
                title="Python Overview",
                excerpt="Guido van Rossum began working on Python in the late 1980s at CWI in the Netherlands.",
            ),
        ]

    def test_lcs_computation(self):
        s1 = "Python was created by Guido van Rossum in 1991"
        s2 = "Guido van Rossum created Python and released it in 1991"
        lcs_val = self.grounding.compute_lcs(s1, s2)
        self.assertGreaterEqual(lcs_val, 4)

    def test_entity_preservation(self):
        # Supported claim
        claim_valid = "Guido van Rossum created Python in 1991."
        pres_valid, unsupp_valid = self.grounding.compute_entity_preservation(claim_valid, self.evidence)
        self.assertGreaterEqual(pres_valid, 0.75)
        self.assertEqual(len(unsupp_valid), 0)

        # Hallucinated entity claim
        claim_hallucinated = "James Gosling created Python in Tokyo with Steve Jobs."
        pres_hallu, unsupp_hallu = self.grounding.compute_entity_preservation(claim_hallucinated, self.evidence)
        self.assertLess(pres_hallu, pres_valid)
        self.assertTrue(any("gosling" in e or "jobs" in e or "tokyo" in e for e in unsupp_hallu))

    def test_grounding_verdict_and_action(self):
        valid_response = "Python was created by Guido van Rossum in 1991. It emphasizes readability."
        verdict = self.grounding.evaluate_response_grounding(valid_response, self.evidence)
        self.assertGreaterEqual(verdict.s_ground, 0.4)
        self.assertEqual(verdict.action, "emit")

    def test_targeted_query_reformulation(self):
        bad_response = "Guido van Rossum created Python in 1991 alongside Dennis Ritchie and Bjarne Stroustrup."
        verdict = self.grounding.evaluate_response_grounding(bad_response, self.evidence)
        re_query = self.grounding.reformulate_query("Python creator", verdict, self.evidence)
        self.assertIn("Python creator", re_query)
        self.assertTrue("ritchie" in re_query.lower() or "stroustrup" in re_query.lower())

    def test_episodic_memory_write_through_and_search(self):
        mem_mgr = EpisodicMemoryManager(db_path="test_episodic.db")

        # Create converged snapshot
        verdict = self.grounding.evaluate_response_grounding("Python was created by Guido van Rossum.", self.evidence)
        traj = ReasoningTrajectory(
            session_id="test_sess_01",
            engine_used=EngineTier.EDGE,
            thought_steps=[],
            evidence_pool=self.evidence,
            grounding_verdict=verdict,
            language_profile=LanguageProfile(primary=LanguageCode.EN),
            latency=LatencyMetrics(),
        )
        snapshot = NeuralStateSnapshot(
            session_id="test_sess_01",
            user_query_raw="Who created Python?",
            trajectories=[traj],
            final_response_text="Python was created by Guido van Rossum in 1991.",
            degraded_mode=False,
        )

        ep_id = mem_mgr.write_through_snapshot(snapshot)
        self.assertIsNotNone(ep_id)

        # Hybrid search
        results = mem_mgr.search_hybrid("Guido van Rossum Python creator", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].session_id, "test_sess_01")
        self.assertIn("Guido", results[0].content_summary)


if __name__ == "__main__":
    unittest.main()
