"""Unit tests for xThinking Engine, reasoning event streams, and Genius identity."""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reasoning.xthinking import XThinkingEngine, ReasoningEvent, Citation
from src.retrieval.wikipedia_client import WikipediaArticle
from src.retrieval.ranker import PassageChunk


class TestReasoningEngine(unittest.IsolatedAsyncioTestCase):

    def test_keyword_extraction(self):
        engine = XThinkingEngine()
        query = "What is the history of Python programming language and who created it?"
        keywords = engine._extract_search_keywords(query)
        self.assertTrue("Python" in keywords or "programming" in keywords)
        self.assertNotIn("what", keywords.lower())

    def test_category_detection(self):
        engine = XThinkingEngine()
        self.assertEqual(engine._detect_category("Write a python function to sort a list"), "codealpaca")
        self.assertEqual(engine._detect_category("What year did World War 2 end?"), "general")

    async def test_xthinking_stream_events_and_identity(self):
        # Mock Model Engine
        mock_model = MagicMock()

        async def mock_stream_gen(*args, **kwargs):
            tokens = [
                "<think>",
                "\nAnalyzing question regarding Alan Turing...",
                "\nVerifying Wikipedia evidence: Alan Turing was born in 1912 and broke Enigma codes.",
                "\nChecking self: Is this fully grounded? Yes.",
                "\n</think>",
                "\nHello, I am Genius.",
                "\nAlan Turing was a British mathematician and pioneering computer scientist [1].",
            ]
            for t in tokens:
                yield t

        mock_model.stream_generate = mock_stream_gen

        # Mock Wikipedia Client
        mock_wiki = MagicMock()
        mock_wiki.search_and_fetch = AsyncMock(
            return_value=[
                WikipediaArticle(
                    title="Alan Turing",
                    url="https://en.wikipedia.org/wiki/Alan_Turing",
                    summary="Alan Mathison Turing was an English mathematician...",
                    full_text="Alan Mathison Turing was an English mathematician, computer scientist...",
                    page_id=123,
                    sections=[{"title": "Early life", "content": "Alan Turing was born in 1912."}],
                )
            ]
        )

        engine = XThinkingEngine(model_engine=mock_model, wiki_client=mock_wiki)

        events = []
        async for event in engine.execute_stream("Who was Alan Turing?"):
            events.append(event)

        stages = [e.stage for e in events]
        self.assertIn("language", stages)
        self.assertIn("plan", stages)
        self.assertIn("research", stages)
        self.assertIn("thinking", stages)
        self.assertIn("response", stages)
        self.assertIn("done", stages)

        done_event = [e for e in events if e.stage == "done"][0]
        self.assertIn("Genius", done_event.payload["response"])
        self.assertIn("Alan Turing was born in 1912", done_event.payload["thoughts"])
        self.assertGreater(len(done_event.payload["citations"]), 0)

    def test_claude_reasoning_exemplar_matching(self):
        from src.dataset.retriever import ExemplarRetriever
        retriever = ExemplarRetriever()
        retriever.load_index()
        # Test batch 1 item
        results = retriever.find_relevant_exemplars("quorum consistency distributed nodes", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].source, "claude_reasoning")
        self.assertIn("<think>", results[0].output)

        # Test batch 2 items
        raft_res = retriever.find_relevant_exemplars("Raft cluster network partition stale leader", top_k=1)
        self.assertGreater(len(raft_res), 0)
        self.assertEqual(raft_res[0].source, "claude_reasoning")

        bloom_res = retriever.find_relevant_exemplars("Bloom filter false positive probability derivation", top_k=1)
        self.assertGreater(len(bloom_res), 0)
        self.assertEqual(bloom_res[0].source, "claude_reasoning")

        lru_res = retriever.find_relevant_exemplars("concurrent lock-free LRU cache in Go", top_k=1)
        self.assertGreater(len(lru_res), 0)
        self.assertEqual(lru_res[0].source, "claude_reasoning")

    async def test_offline_research_mode_bypass(self):
        mock_model = MagicMock()

        async def mock_stream_gen(*args, **kwargs):
            yield "<think>\nDirect internal retrieval...</think>\nHello! I am Genius."

        mock_model.stream_generate = mock_stream_gen
        mock_wiki = MagicMock()
        mock_wiki.search_and_fetch = AsyncMock()

        engine = XThinkingEngine(model_engine=mock_model, wiki_client=mock_wiki, research_mode="off")
        events = []
        async for ev in engine.execute_stream("Hi, who are you?"):
            events.append(ev)

        # Wikipedia client should NOT be called in offline/direct mode
        mock_wiki.search_and_fetch.assert_not_called()
        stages = [e.stage for e in events]
        self.assertNotIn("research", stages)
        self.assertIn("response", stages)

    def test_math_solver_identities(self):
        from src.reasoning.math_solver import MathSolver
        # Test (a+b)^2
        res = MathSolver.solve("a+b whole square =")
        self.assertIsNotNone(res)
        think, resp = res
        self.assertTrue("a² + 2ab + b²" in resp or "a^2 + 2ab + b^2" in resp)
        self.assertIn("distributive", think.lower())

        # Test (a-b)^2
        res_sub = MathSolver.solve("(a-b)^2")
        self.assertIsNotNone(res_sub)
        self.assertTrue("a² - 2ab + b²" in res_sub[1] or "a^2 - 2ab + b^2" in res_sub[1])

        # Test linear equation
        res_eq = MathSolver.solve("solve 2x + 5 = 15")
        self.assertIsNotNone(res_eq)
        self.assertIn("x = 5", res_eq[1])

        # Test circle area
        res_geom = MathSolver.solve("area of circle")
        self.assertIsNotNone(res_geom)
        self.assertTrue("πr²" in res_geom[1] or "π r²" in res_geom[1])

    def test_text_sanitizer(self):
        from src.reasoning.text_sanitizer import TextSanitizer
        raw = "### Title\n$$\\mathbf{(a + b)^2 = a^2 + 2ab + b^2}$$\n* Item 1\n**bold**"
        clean = TextSanitizer.clean_for_display(raw)
        self.assertNotIn("$$", clean)
        self.assertNotIn("\\mathbf", clean)
        self.assertNotIn("###", clean)
        self.assertNotIn("**", clean)
        self.assertIn("(a + b)² = a² + 2ab + b²", clean)
        self.assertIn("• Item 1", clean)

    def test_workspace_manager_and_intents(self):
        import tempfile
        from src.system.workspace import WorkspaceManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorkspaceManager(initial_path=tmpdir)
            ok, msg = wm.create_file("test.py", "print('hello from workspace')")
            self.assertTrue(ok)

            ok, rmsg = wm.read_file("test.py")
            self.assertTrue(ok)
            self.assertIn("hello from workspace", rmsg)

            ok, lmsg = wm.list_files()
            self.assertTrue(ok)
            self.assertIn("test.py", lmsg)

            # Test Natural Language Intent
            res = wm.handle_natural_language_intent("create file note.txt with content Important Note")
            self.assertIsNotNone(res)
            self.assertEqual(res[0], "file_create")

            read_res = wm.handle_natural_language_intent("read file note.txt")
            self.assertIsNotNone(read_res)
            self.assertEqual(read_res[0], "file_read")
            self.assertIn("Important Note", read_res[1])

    def test_code_and_concept_solvers(self):
        from src.reasoning.code_solver import CodeSolver
        from src.reasoning.concept_synthesizer import ConceptSynthesizer

        code_res = CodeSolver.solve("binary search in python")
        self.assertIsNotNone(code_res)
        self.assertIn("def binary_search", code_res[1])

        concept_res = ConceptSynthesizer.synthesize("kafka architecture")
        self.assertIsNotNone(concept_res)
        self.assertIn("Kafka", concept_res[1])


if __name__ == "__main__":
    unittest.main()
