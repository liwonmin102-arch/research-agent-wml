"""Tavily search API wrapper — produces SearchResult objects matching src/models.py."""

import os
import time
from urllib.parse import urlparse

from rich.console import Console
from tavily import TavilyClient

from src.models import SearchResult


class TavilySearcher:
    """Wrapper around the Tavily search API."""

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set. Add it to your environment or .env file."
            )
        self.client = TavilyClient(api_key=api_key)
        self.console = Console()

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc[4:] if netloc.startswith("www.") else netloc
        except Exception:
            return ""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run one Tavily query; return structured SearchResult list. [] on failure."""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            results: list[SearchResult] = []
            for r in response.get("results", []):
                try:
                    results.append(
                        SearchResult(
                            url=r["url"],
                            title=r.get("title", "Untitled"),
                            content_snippet=r.get("content", "")[:2000],
                            published_date=r.get("published_date"),
                            source_domain=self._extract_domain(r["url"]),
                            tavily_score=r.get("score"),
                        )
                    )
                except Exception as e:
                    self.console.print(f"[yellow]Skipped malformed result: {e}[/]")
            return results
        except Exception as e:
            self.console.print(f"[yellow]Search failed for '{query}': {e}[/]")
            return []

    def batch_search(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
        pause_seconds: float = 1.0,
    ) -> list[SearchResult]:
        """Run multiple plain-string queries, deduplicate by URL, return a flat list."""
        seen_urls: set[str] = set()
        all_results: list[SearchResult] = []
        for i, q in enumerate(queries):
            self.console.print(f"[bold blue]Searching[/] ({i+1}/{len(queries)}): {q}")
            for result in self.search(q, max_results=max_results_per_query):
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
            if i < len(queries) - 1:
                time.sleep(pause_seconds)
        self.console.print(
            f"[dim]Collected {len(all_results)} unique sources across {len(queries)} queries.[/]"
        )
        return all_results
