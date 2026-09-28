"""Live Web and Knowledge Retrieval Provider with Real-Time Web Scraping."""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


@dataclass
class WebSearchResult:
    """Represents a search result from web or knowledge sources."""
    title: str
    url: str
    snippet: str
    source: str = "web"


class WebSearchClient:
    """Async real-time web search and page text extraction client."""

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.7,hi;q=0.5",
        }
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
        """
        Executes real-time web search using multi-tier fallback:
        1. DuckDuckGo HTML search (full snippets + fresh rankings)
        2. DuckDuckGo Lite search
        3. DuckDuckGo Instant Answer API
        """
        clean_q = query.strip()
        if not clean_q:
            return []

        # Tier 1: DuckDuckGo HTML POST search
        results = await self._search_duckduckgo_html(clean_q, limit=limit)
        if results:
            return results

        # Tier 2: DuckDuckGo Lite POST search
        results = await self._search_duckduckgo_lite(clean_q, limit=limit)
        if results:
            return results

        # Tier 3: Instant Answer API fallback
        return await self._search_instant_answer(clean_q, limit=limit)

    async def _search_duckduckgo_html(self, query: str, limit: int = 5) -> List[WebSearchResult]:
        client = await self._get_client()
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query, "b": ""}

        results: List[WebSearchResult] = []
        try:
            resp = await client.post(url, data=data)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for r in soup.select(".result"):
                    if len(results) >= limit:
                        break
                    title_elem = r.select_one(".result__title a")
                    snippet_elem = r.select_one(".result__snippet")
                    if not title_elem:
                        continue

                    raw_url = title_elem.get("href", "")
                    actual_url = raw_url
                    if "uddg=" in raw_url:
                        match = re.search(r"uddg=([^&]+)", raw_url)
                        if match:
                            actual_url = unquote(match.group(1))

                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if title and actual_url and not actual_url.startswith("/"):
                        results.append(
                            WebSearchResult(
                                title=title,
                                url=actual_url,
                                snippet=html.unescape(snippet),
                                source="duckduckgo_web",
                            )
                        )
        except Exception as e:
            logger.debug(f"DuckDuckGo HTML search error for '{query}': {e}")

        return results

    async def _search_duckduckgo_lite(self, query: str, limit: int = 5) -> List[WebSearchResult]:
        client = await self._get_client()
        url = "https://lite.duckduckgo.com/lite/"
        data = {"q": query}

        results: List[WebSearchResult] = []
        try:
            resp = await client.post(url, data=data)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                links = soup.select("a.result-link")
                snippets = soup.select("td.result-snippet")

                for i, link in enumerate(links):
                    if len(results) >= limit:
                        break
                    title = link.get_text(strip=True)
                    actual_url = link.get("href", "")
                    if "uddg=" in actual_url:
                        match = re.search(r"uddg=([^&]+)", actual_url)
                        if match:
                            actual_url = unquote(match.group(1))

                    snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
                    if title and actual_url:
                        results.append(
                            WebSearchResult(
                                title=title,
                                url=actual_url,
                                snippet=html.unescape(snippet),
                                source="duckduckgo_lite",
                            )
                        )
        except Exception as e:
            logger.debug(f"DuckDuckGo Lite search error for '{query}': {e}")

        return results

    async def _search_instant_answer(self, query: str, limit: int = 5) -> List[WebSearchResult]:
        client = await self._get_client()
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "0"}

        results: List[WebSearchResult] = []
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
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

                topics = data.get("RelatedTopics", [])
                for t in topics:
                    if len(results) >= limit:
                        break
                    if "Text" in t and "FirstURL" in t:
                        results.append(
                            WebSearchResult(
                                title=t.get("Text", "")[:60] + "...",
                                url=t.get("FirstURL", ""),
                                snippet=t.get("Text", ""),
                                source="duckduckgo_topic",
                            )
                        )
        except Exception as e:
            logger.debug(f"DuckDuckGo Instant Answer error for '{query}': {e}")

        return results

    async def fetch_page_content(self, url: str, max_chars: int = 4000) -> Optional[str]:
        """Fetches and extracts clean visible text from any URL."""
        client = await self._get_client()
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None

            raw_html = resp.text
            # Remove scripts, styles, navigation, headers, footers
            clean = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
            # Remove tags
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = html.unescape(clean)
            clean = " ".join(clean.split())
            return clean[:max_chars] if clean else None
        except Exception as e:
            logger.warning(f"Failed to fetch content from '{url}': {e}")
            return None
