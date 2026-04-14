"""Core agent loop: plan → act → observe → reflect → synthesize."""

import time
from typing import Callable

from rich.console import Console

from src.llm import ResearchLLM
from src.models import (
    Contradiction,
    ResearchReport,
    SearchQuery,
    SearchResult,
    SourceSummary,
)
from src.search import TavilySearcher


class ResearchAgent:
    """Orchestrates the full research pipeline."""

    def __init__(self, on_progress: Callable[[str], None] | None = None):
        self.console = Console()
        self.searcher = TavilySearcher()
        self.llm = ResearchLLM()
        self.on_progress = on_progress

    def _emit(self, message: str) -> None:
        """Emit a plain-text progress message to console and optional callback."""
        self.console.print(message)
        if self.on_progress is not None:
            self.on_progress(message)

    def _plan(self, topic: str) -> list[SearchQuery]:
        """PLAN step: Generate diverse search queries for the topic."""
        self._emit(f"Researching: {topic}")
        self._emit("Step 1/5: Planning search strategy...")

        queries = self.llm.generate_search_queries(topic)

        for i, q in enumerate(queries, 1):
            self._emit(f"  {i}. {q.query} — {q.rationale}")

        return queries

    def _act(self, queries: list[SearchQuery]) -> dict[str, list[SearchResult]]:
        """ACT step: Execute all search queries via Tavily."""
        self._emit("Step 2/5: Searching the web...")

        results = self.searcher.batch_search(queries)

        total_results = sum(len(v) for v in results.values())
        self._emit(
            f"Found {total_results} results across {len(results)} queries."
        )
        return results

    def _observe(
        self, search_results: dict[str, list[SearchResult]], topic: str
    ) -> list[SourceSummary]:
        """OBSERVE step: Summarize each unique source using the LLM."""
        self._emit("Step 3/5: Analyzing sources...")

        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for results_list in search_results.values():
            for result in results_list:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)

        self._emit(f"{len(unique_results)} unique sources after deduplication.")

        summaries: list[SourceSummary] = []
        for i, result in enumerate(unique_results, 1):
            self._emit(
                f"  [{i}/{len(unique_results)}] Analyzing: {result.title}"
            )
            summary = self.llm.summarize_source(result.content, topic)
            # LLM may hallucinate url/title — trust the search result instead.
            summary.url = result.url
            summary.title = result.title
            summaries.append(summary)

        self._emit(f"Analyzed {len(summaries)} sources.")
        return summaries

    def _reflect(
        self, summaries: list[SourceSummary], topic: str
    ) -> tuple[list[SourceSummary], list[Contradiction]]:
        """REFLECT step: Identify contradictions and do one round of follow-up searches if found."""
        self._emit("Step 4/5: Identifying contradictions...")

        contradictions = self.llm.identify_contradictions(summaries)

        if contradictions:
            self._emit(
                f"Found {len(contradictions)} contradiction(s). Running follow-up searches..."
            )

            follow_up_topic = (
                f"{topic} — resolving these contradictions: "
                + "; ".join(c.explanation[:100] for c in contradictions[:2])
            )
            follow_up_queries = self.llm.generate_search_queries(
                follow_up_topic, num_queries=2
            )

            follow_up_results = self.searcher.batch_search(follow_up_queries)

            new_summaries = self._observe(follow_up_results, topic)
            summaries = summaries + new_summaries

            contradictions = self.llm.identify_contradictions(summaries)
            self._emit(
                f"After follow-up: {len(contradictions)} contradiction(s) remain."
            )
        else:
            self._emit("No contradictions found across sources.")

        return summaries, contradictions

    def _synthesize(
        self,
        topic: str,
        summaries: list[SourceSummary],
        contradictions: list[Contradiction],
    ) -> ResearchReport:
        """SYNTHESIZE step: Produce the final structured research report."""
        self._emit("Step 5/5: Synthesizing final report...")

        report = self.llm.synthesize_report(topic, summaries, contradictions)

        self._emit(
            f"Research complete! Sources: {len(summaries)}, "
            f"Contradictions: {len(contradictions)}, "
            f"Follow-up questions: {len(report.follow_up_questions)}, "
            f"API calls: {self.llm.api_call_count}"
        )

        return report

    def research(self, topic: str) -> ResearchReport:
        """Run the full research pipeline: plan → act → observe → reflect → synthesize."""
        start_time = time.time()

        queries = self._plan(topic)
        search_results = self._act(queries)
        summaries = self._observe(search_results, topic)
        summaries, contradictions = self._reflect(summaries, topic)
        report = self._synthesize(topic, summaries, contradictions)

        elapsed = time.time() - start_time
        self._emit(f"Total time: {elapsed:.1f} seconds")

        return report
