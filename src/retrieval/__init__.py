"""Wikipedia Retrieval and Passage Extraction Module."""

from .wikipedia_client import WikipediaClient, WikipediaArticle
from .ranker import PassageRanker, PassageChunk

__all__ = ["WikipediaClient", "WikipediaArticle", "PassageRanker", "PassageChunk"]
