"""Asynchronous Wikipedia Client with persistent SQLite caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "QwenWikipediaDeepResearcher/1.0 (https://github.com/example/llm-researcher; research-bot@local)"
WIKIPEDIA_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"


@dataclass
class WikipediaArticle:
    """Represents an extracted Wikipedia article with metadata."""
    title: str
    url: str
    summary: str
    full_text: str
    page_id: int
    sections: List[Dict[str, str]] = field(default_factory=list)


class WikipediaCache:
    """SQLite-based cache for Wikipedia articles and queries."""

    def __init__(self, db_path: str | Path = "wiki_cache.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create cache tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS article_cache (
                    title TEXT PRIMARY KEY,
                    url TEXT,
                    summary TEXT,
                    full_text TEXT,
                    page_id INTEGER,
                    sections_json TEXT,
                    cached_at REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    results_json TEXT,
                    cached_at REAL
                )
                """
            )
            conn.commit()

    def get_article(self, title: str) -> Optional[WikipediaArticle]:
        """Fetch article from cache if present."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT title, url, summary, full_text, page_id, sections_json FROM article_cache WHERE LOWER(title) = LOWER(?)",
                    (title,),
                )
                row = cursor.fetchone()
                if row:
                    sections = json.loads(row[5]) if row[5] else []
                    return WikipediaArticle(
                        title=row[0],
                        url=row[1],
                        summary=row[2],
                        full_text=row[3],
                        page_id=row[4],
                        sections=sections,
                    )
        except Exception as e:
            logger.warning(f"Cache read error for article '{title}': {e}")
        return None

    def save_article(self, article: WikipediaArticle) -> None:
        """Save article to cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO article_cache
                    (title, url, summary, full_text, page_id, sections_json, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article.title,
                        article.url,
                        article.summary,
                        article.full_text,
                        article.page_id,
                        json.dumps(article.sections),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache write error for article '{article.title}': {e}")

    def get_search_results(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch search results from cache."""
        try:
            q_hash = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT results_json FROM search_cache WHERE query_hash = ?",
                    (q_hash,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Cache read error for search query '{query}': {e}")
        return None

    def save_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Save search results to cache."""
        try:
            q_hash = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO search_cache (query_hash, results_json, cached_at) VALUES (?, ?, ?)",
                    (q_hash, json.dumps(results), time.time()),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache write error for search query '{query}': {e}")


class WikipediaClient:
    """Async Wikimedia API client with high concurrency, caching, and rate handling."""

    def __init__(
        self,
        cache_db: str | Path = "wiki_cache.db",
        timeout: float = 12.0,
    ) -> None:
        self.cache = WikipediaCache(cache_db)
        self.timeout = timeout
        self.headers = {"User-Agent": USER_AGENT}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.headers,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia for matching titles and snippets."""
        cached = self.cache.get_search_results(query)
        if cached is not None:
            return cached[:limit]

        client = await self._get_client()
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "format": "json",
            "utf8": "1",
        }

        try:
            resp = await client.get(WIKIPEDIA_API_ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
            search_items = data.get("query", {}).get("search", [])

            results = [
                {
                    "title": item.get("title", ""),
                    "page_id": item.get("pageid", 0),
                    "snippet": item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", ""),
                    "size": item.get("size", 0),
                    "wordcount": item.get("wordcount", 0),
                }
                for item in search_items
            ]

            self.cache.save_search_results(query, results)
            return results
        except Exception as e:
            logger.error(f"Failed to search Wikipedia for '{query}': {e}")
            return []

    async def fetch_article(self, title: str) -> Optional[WikipediaArticle]:
        """Fetch article plaintext, summary, and canonical URL."""
        cached = self.cache.get_article(title)
        if cached:
            return cached

        client = await self._get_client()
        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "inprop": "url",
            "titles": title,
            "redirects": "1",
            "format": "json",
            "utf8": "1",
        }

        try:
            resp = await client.get(WIKIPEDIA_API_ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if int(page_id) < 0:
                    continue  # Missing or not found

                article_title = page_data.get("title", title)
                url = page_data.get("canonicalurl", f"https://en.wikipedia.org/wiki/{article_title.replace(' ', '_')}")
                full_text = page_data.get("extract", "").strip()

                if not full_text:
                    continue

                # Extract summary (lead paragraph before first major section heading)
                lead_parts = full_text.split("\n\n==")
                summary = lead_parts[0].strip() if lead_parts else full_text[:1000]

                # Parse sections
                sections = self._parse_sections(full_text)

                article = WikipediaArticle(
                    title=article_title,
                    url=url,
                    summary=summary,
                    full_text=full_text,
                    page_id=int(page_id),
                    sections=sections,
                )
                self.cache.save_article(article)
                return article

        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia article '{title}': {e}")

        return None

    def _parse_sections(self, text: str) -> List[Dict[str, str]]:
        """Parses Wikipedia plain text into named sections."""
        sections: List[Dict[str, str]] = []
        lines = text.split("\n")
        current_title = "Overview"
        current_lines: List[str] = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("==") and line_stripped.endswith("=="):
                # Save previous section
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append({"title": current_title, "content": content})
                current_title = line_stripped.strip("= ")
                current_lines = []
            else:
                current_lines.append(line)

        # Append last section
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"title": current_title, "content": content})

        return sections

    async def search_and_fetch(self, query: str, max_articles: int = 3) -> List[WikipediaArticle]:
        """Search Wikipedia and concurrently fetch the top matching articles."""
        search_results = await self.search(query, limit=max_articles)
        if not search_results:
            return []

        tasks = [self.fetch_article(item["title"]) for item in search_results]
        articles = await asyncio.gather(*tasks, return_exceptions=True)

        valid_articles: List[WikipediaArticle] = []
        for a in articles:
            if isinstance(a, WikipediaArticle):
                valid_articles.append(a)
        return valid_articles
