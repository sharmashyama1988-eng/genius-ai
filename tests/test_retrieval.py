"""Unit tests for Wikipedia Client and Passage Ranker."""

import pytest
from src.retrieval.wikipedia_client import WikipediaClient, WikipediaArticle
from src.retrieval.ranker import PassageRanker, PassageChunk


@pytest.fixture
def ranker():
    return PassageRanker(k1=1.5, b=0.75, chunk_size_words=100)


def test_chunking_and_ranking(ranker):
    sample_text = (
        "Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. "
        "Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm "
        "and computation with the Turing machine, which can be considered a model of a general-purpose computer.\n\n"
        "During the Second World War, Turing was a leading participant in war-time codebreaking at Bletchley Park. "
        "He played a pivotal role in cracking intercepted coded messages that enabled the Allies to defeat the Axis powers in many critical engagements."
    )

    chunks = ranker.chunk_article(
        title="Alan Turing",
        url="https://en.wikipedia.org/wiki/Alan_Turing",
        full_text=sample_text,
    )

    assert len(chunks) >= 1
    assert chunks[0].article_title == "Alan Turing"

    # Test ranking with targeted query
    ranked = ranker.rank_passages("Bletchley Park codebreaking during war", chunks, top_k=2)
    assert len(ranked) >= 1
    assert "Bletchley" in ranked[0].text or "codebreaking" in ranked[0].text
    assert ranked[0].score > 0


def test_wikipedia_client_search_cache(tmp_path):
    import asyncio
    cache_db = tmp_path / "test_wiki.db"
    client = WikipediaClient(cache_db=cache_db)

    # Mock article save and fetch
    mock_art = WikipediaArticle(
        title="Python (programming language)",
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        summary="Python is a high-level, general-purpose programming language.",
        full_text="Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands.",
        page_id=23862,
    )

    client.cache.save_article(mock_art)
    cached = client.cache.get_article("Python (programming language)")

    assert cached is not None
    assert cached.title == "Python (programming language)"
    assert "Guido van Rossum" in cached.full_text

    asyncio.run(client.close())
