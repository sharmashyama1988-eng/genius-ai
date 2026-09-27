"""Test runner using standard library unittest."""

import asyncio
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.wikipedia_client import WikipediaClient, WikipediaArticle
from src.retrieval.ranker import PassageRanker, PassageChunk
from src.dataset.loader import DatasetManager, DatasetItem
from src.dataset.retriever import ExemplarRetriever
from src.languages.router import LanguageRouter


class TestGeniusComponents(unittest.TestCase):

    def test_ranker_bm25(self):
        ranker = PassageRanker(k1=1.5, b=0.75, chunk_size_words=100)
        text = (
            "Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. "
            "Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm "
            "and computation with the Turing machine.\n\n"
            "During the Second World War, Turing was a leading participant in war-time codebreaking at Bletchley Park. "
            "He cracked German Enigma cipher machines."
        )
        chunks = ranker.chunk_article("Alan Turing", "https://en.wikipedia.org/wiki/Alan_Turing", text)
        self.assertGreaterEqual(len(chunks), 1)

        ranked = ranker.rank_passages("Enigma codebreaking at Bletchley Park", chunks, top_k=2)
        self.assertGreaterEqual(len(ranked), 1)
        self.assertIn("Enigma", ranked[0].text)

    def test_language_router(self):
        router = LanguageRouter()

        # Hinglish detection
        res_hinglish = router.detect("kya haal hai bhai, Alan Turing ke baare mein batao")
        self.assertEqual(res_hinglish.code, "hi-Latn")

        # Devanagari Hindi detection
        res_hindi = router.detect("भारत की राजधानी क्या है और इसका इतिहास क्या है?")
        self.assertEqual(res_hindi.code, "hi")

        # English detection
        res_en = router.detect("Explain how quantum superposition works in physics.")
        self.assertEqual(res_en.code, "en")

        # Spanish detection
        res_es = router.detect("Hola, por favor explicame como funciona la inteligencia artificial")
        self.assertEqual(res_es.code, "es")

    def test_wikipedia_cache(self):
        client = WikipediaClient(cache_db="test_cache.db")
        mock = WikipediaArticle(
            title="Artificial intelligence",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            summary="AI is intelligence demonstrated by machines.",
            full_text="Artificial intelligence (AI) is the intelligence of machines or software.",
            page_id=1164,
        )
        client.cache.save_article(mock)
        fetched = client.cache.get_article("Artificial intelligence")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Artificial intelligence")

    def test_system_safety_guard(self):
        from src.system.guard import SafetyGuard, ActionSafetyLevel
        from src.system.executor import SystemExecutor

        # 1. Safe command evaluation
        lvl, _ = SafetyGuard.evaluate_command("Get-ChildItem -Path .")
        self.assertEqual(lvl, ActionSafetyLevel.SAFE)

        # 2. Sensitive command evaluation (deletion)
        lvl_del, reason = SafetyGuard.evaluate_command("Remove-Item -Path 'C:\\temp' -Recurse -Force")
        self.assertEqual(lvl_del, ActionSafetyLevel.SENSITIVE)
        self.assertIn("Only the user is permitted", reason)

        # 3. System executor blocks sensitive command without user confirmation
        executor = SystemExecutor()
        res_blocked = executor.execute("Remove-Item -Path 'C:\\critical_file.txt'")
        self.assertFalse(res_blocked.executed)
        self.assertEqual(res_blocked.exit_code, -1)
        self.assertIn("SENSITIVE ACTION BLOCKED", res_blocked.message)

        # 4. Safe command execution
        res_safe = executor.execute("Write-Output 'Genius System Online'")
        self.assertTrue(res_safe.executed)
        self.assertEqual(res_safe.stdout, "Genius System Online")


def suite():
    s = unittest.TestSuite()
    loader = unittest.TestLoader()
    s.addTests(loader.loadTestsFromTestCase(TestGeniusComponents))

    from tests.test_reasoning import TestReasoningEngine
    s.addTests(loader.loadTestsFromTestCase(TestReasoningEngine))

    from tests.test_grounding_and_memory import TestGroundingAndMemory
    s.addTests(loader.loadTestsFromTestCase(TestGroundingAndMemory))

    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
