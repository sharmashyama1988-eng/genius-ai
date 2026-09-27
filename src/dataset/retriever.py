"""In-context exemplar and pattern retrieval from LIMA, Alpaca, and CodeAlpaca."""

from __future__ import annotations

import logging
import math
import re
from typing import List, Optional

from .loader import DatasetItem, DatasetManager

logger = logging.getLogger(__name__)


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # Hinglish & Hindi common function words
    "kya", "hai", "hain", "ho", "mein", "ko", "se", "ka", "ki", "ke", "yeh",
    "woh", "ek", "aur", "toh", "bhi", "tha", "thi", "the", "batao", "mujhe",
    "karo", "karein", "aap", "tum",
}


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()) if len(w) > 1]


class ExemplarRetriever:
    """Retrieves relevant high-quality instruction and code exemplars."""

    def __init__(self, manager: Optional[DatasetManager] = None) -> None:
        self.manager = manager or DatasetManager()
        self.items: List[DatasetItem] = []
        self._is_indexed = False

    def load_index(self, max_per_source: int = 500) -> None:
        """Loads and indexes items in memory for sub-millisecond retrieval."""
        self.items = []
        seen_paths = set()
        for src in ["genius_reasoning", "genius_code", "genius_general", "claude_reasoning", "codealpaca", "lima", "alpaca"]:
            if self.manager.is_cached(src):  # type: ignore
                p = self.manager.get_local_path(src)  # type: ignore
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                try:
                    loaded = self.manager.load_dataset(src, limit=max_per_source)  # type: ignore
                    self.items.extend(loaded)
                except Exception as e:
                    logger.warning(f"Could not load dataset {src} for exemplar index: {e}")
        self._is_indexed = True

    def find_relevant_exemplars(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 2,
        min_score: float = 0.12,
    ) -> List[DatasetItem]:
        """Finds the most contextually relevant exemplars matching the user query."""
        if not self._is_indexed:
            self.load_index()

        # Extract content tokens, ignoring stop words
        raw_tokens = tokenize(query)
        content_tokens = [w for w in raw_tokens if w not in STOP_WORDS]
        if not content_tokens:
            return []

        candidates = self.items
        if category:
            candidates = [item for item in self.items if item.category == category or item.source == category]

        if not candidates:
            return []

        query_tokens = set(content_tokens)
        scored: List[tuple[float, DatasetItem]] = []
        for item in candidates:
            item_raw = tokenize(item.instruction + " " + item.input)
            item_tokens = [w for w in item_raw if w not in STOP_WORDS]
            if not item_tokens:
                continue

            intersection = query_tokens.intersection(item_tokens)
            if not intersection:
                continue

            # Jaccard / Overlap score with length penalty
            score = len(intersection) / (math.sqrt(len(query_tokens) * len(item_tokens)) + 1e-5)
            # Boost high-fidelity synthetic reasoning chains from Claude
            if item.source == "claude_reasoning":
                score *= 1.30

            if score >= min_score:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
