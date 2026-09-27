"""Passage chunking and BM25 lexical ranking engine."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Set


@dataclass
class PassageChunk:
    """A chunked passage from an article with relevance scoring."""
    article_title: str
    article_url: str
    section_title: str
    text: str
    chunk_index: int
    score: float = 0.0


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer with lowercasing."""
    return [word for word in re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()) if len(word) > 1]


class PassageRanker:
    """Passage chunker and BM25 relevance ranker."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, chunk_size_words: int = 150) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_size_words = chunk_size_words

    def chunk_article(
        self,
        title: str,
        url: str,
        full_text: str,
        sections: List[dict] | None = None,
    ) -> List[PassageChunk]:
        """Split article text into clean, contextual passage chunks."""
        chunks: List[PassageChunk] = []
        chunk_idx = 0

        if sections:
            for sec in sections:
                sec_title = sec.get("title", "General")
                # Skip references, see also, external links, bibliography
                if sec_title.lower() in [
                    "references",
                    "external links",
                    "see also",
                    "further reading",
                    "notes",
                    "sources",
                    "bibliography",
                ]:
                    continue

                content = sec.get("content", "").strip()
                if not content:
                    continue

                paragraphs = content.split("\n\n")
                for p in paragraphs:
                    p_clean = p.strip()
                    if len(p_clean) < 25:  # Skip trivial fragments
                        continue

                    # If paragraph is very large, split into chunks
                    words = p_clean.split()
                    if len(words) > self.chunk_size_words * 1.5:
                        for i in range(0, len(words), self.chunk_size_words):
                            sub_chunk = " ".join(words[i : i + self.chunk_size_words])
                            if len(sub_chunk.strip()) > 50:
                                chunks.append(
                                    PassageChunk(
                                        article_title=title,
                                        article_url=url,
                                        section_title=sec_title,
                                        text=sub_chunk,
                                        chunk_index=chunk_idx,
                                    )
                                )
                                chunk_idx += 1
                    else:
                        chunks.append(
                            PassageChunk(
                                article_title=title,
                                article_url=url,
                                section_title=sec_title,
                                text=p_clean,
                                chunk_index=chunk_idx,
                            )
                        )
                        chunk_idx += 1
        else:
            # Fallback to splitting full_text by paragraphs
            paragraphs = full_text.split("\n\n")
            for p in paragraphs:
                p_clean = p.strip()
                if len(p_clean) >= 60:
                    chunks.append(
                        PassageChunk(
                            article_title=title,
                            article_url=url,
                            section_title="Overview",
                            text=p_clean,
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1

        return chunks

    def rank_passages(
        self,
        query: str,
        passages: List[PassageChunk],
        top_k: int = 5,
    ) -> List[PassageChunk]:
        """Rank passages using BM25 scoring algorithm against the query."""
        if not passages:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return passages[:top_k]

        # Calculate passage lengths and corpus stats
        corpus_tokenized = [tokenize(p.text) for p in passages]
        total_docs = len(passages)
        total_len = sum(len(doc) for doc in corpus_tokenized)
        avg_doc_len = total_len / total_docs if total_docs > 0 else 1.0

        # Calculate Document Frequency (DF) for each query term
        doc_freq: dict[str, int] = {}
        for q_token in set(query_tokens):
            doc_freq[q_token] = sum(1 for doc in corpus_tokenized if q_token in doc)

        # Compute BM25 scores
        scored_passages: List[PassageChunk] = []
        for p, doc_tokens in zip(passages, corpus_tokenized):
            doc_len = len(doc_tokens)
            if doc_len == 0:
                continue

            # Term frequency in this passage
            tf_dict: dict[str, int] = {}
            for t in doc_tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            score = 0.0
            for q_term in query_tokens:
                if q_term not in tf_dict:
                    continue
                tf = tf_dict[q_term]
                df = doc_freq.get(q_term, 0)
                # Robertson-Spärck Jones IDF
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                # BM25 TF component
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                score += idf * (num / denom)

            # Title matching boost: if query term appears in article title or section title
            title_tokens = set(tokenize(p.article_title) + tokenize(p.section_title))
            overlap = sum(1 for qt in query_tokens if qt in title_tokens)
            score += overlap * 1.2

            p.score = round(score, 4)
            scored_passages.append(p)

        # Sort descending by score
        scored_passages.sort(key=lambda x: x.score, reverse=True)
        return scored_passages[:top_k]
