"""Tavily search API wrapper."""

import os
import time

from rich.console import Console
from tavily import TavilyClient

from src.models import SearchQuery, SearchResult


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

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            return [
                SearchResult(
                    url=r["url"],
                    title=r["title"],
                    content=r["content"],
                    relevance_score=r.get("score", 0.0),
                )
                for r in response.get("results", [])
            ]
        except Exception as e:
            self.console.print(f"[yellow]Search failed for '{query}': {e}[/]")
            return []

    def batch_search(self, queries: list[SearchQuery]) -> dict[str, list[SearchResult]]:
        results: dict[str, list[SearchResult]] = {}
        for i, query in enumerate(queries):
            self.console.print(f"[bold blue]Searching:[/] {query.query}")
            results[query.query] = self.search(query.query)
            if i < len(queries) - 1:
                time.sleep(1)
        return results
