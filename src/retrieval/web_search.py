"""Live Web and Knowledge Retrieval Provider."""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "GeniusDeepResearcher/2.0 (Worldwide Production; AI Research Agent)"


@dataclass
class WebSearchResult:
    """Represents a search result from web or knowledge sources."""
    title: str
    url: str
    snippet: str
    source: str = "web"


class WebSearchClient:
    """Async web search and page text extraction client."""

    def __init__(self, timeout: float = 10.0) -> None:
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
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search(self, query: str, limit: int = 5) -> List[WebSearchResult]:
        """Queries DuckDuckGo Instant Answer API and related topics."""
        client = await self._get_client()
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "0",
        }

        results: List[WebSearchResult] = []
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()

                # 1. Main Abstract
                abstract = data.get("AbstractText", "")
                abstract_url = data.get("AbstractURL", "")
                heading = data.get("Heading", query)
                if abstract and abstract_url:
                    results.append(
                        WebSearchResult(
                            title=heading,
                            url=abstract_url,
                            snippet=abstract,
                            source="duckduckgo_abstract",
                        )
                    )

                # 2. Related Topics
                topics = data.get("RelatedTopics", [])
                for t in topics:
                    if len(results) >= limit:
                        break
                    # Sometimes topic is a group with 'Topics'
                    if "Text" in t and "FirstURL" in t:
                        results.append(
                            WebSearchResult(
                                title=t.get("Text", "")[:60] + "...",
                                url=t.get("FirstURL", ""),
                                snippet=t.get("Text", ""),
                                source="duckduckgo_topic",
                            )
                        )
                    elif "Topics" in t:
                        for sub_t in t.get("Topics", []):
                            if len(results) >= limit:
                                break
                            if "Text" in sub_t and "FirstURL" in sub_t:
                                results.append(
                                    WebSearchResult(
                                        title=sub_t.get("Text", "")[:60] + "...",
                                        url=sub_t.get("FirstURL", ""),
                                        snippet=sub_t.get("Text", ""),
                                        source="duckduckgo_topic",
                                    )
                                )

        except Exception as e:
            logger.warning(f"DuckDuckGo API search error for '{query}': {e}")

        return results

    async def fetch_page_content(self, url: str, max_chars: int = 4000) -> Optional[str]:
        """Fetches and extracts clean visible text from any URL."""
        client = await self._get_client()
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None

            raw_html = resp.text
            # Remove scripts and styles
            clean = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
            # Remove tags
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = html.unescape(clean)
            clean = " ".join(clean.split())
            return clean[:max_chars] if clean else None
        except Exception as e:
            logger.warning(f"Failed to fetch content from '{url}': {e}")
            return None
