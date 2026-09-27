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


if __name__ == "__main__":
    unittest.main()
