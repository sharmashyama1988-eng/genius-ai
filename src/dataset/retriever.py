"""In-context exemplar and pattern retrieval from LIMA, Alpaca, and CodeAlpaca."""

from __future__ import annotations

import logging
import math
import re
from typing import List, Optional

from .loader import DatasetItem, DatasetManager

logger = logging.getLogger(__name__)


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
        for src in ["codealpaca", "lima", "alpaca"]:
            if self.manager.is_cached(src):  # type: ignore
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
    ) -> List[DatasetItem]:
        """Finds the most contextually relevant exemplars matching the user query."""
        if not self._is_indexed:
            self.load_index()

        candidates = self.items
        if category:
            candidates = [item for item in self.items if item.category == category or item.source == category]

        if not candidates:
            return []

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return candidates[:top_k]

        scored: List[tuple[float, DatasetItem]] = []
        for item in candidates:
            item_tokens = tokenize(item.instruction + " " + item.input)
            if not item_tokens:
                continue

            intersection = query_tokens.intersection(item_tokens)
            if not intersection:
                continue

            # Jaccard / Overlap score with length penalty
            score = len(intersection) / (math.sqrt(len(query_tokens) * len(item_tokens)) + 1e-5)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
