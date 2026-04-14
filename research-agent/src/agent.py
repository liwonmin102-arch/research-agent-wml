"""Core agent loop: plan → act → observe → reflect → synthesize."""

import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

    def __init__(self):
        self.console = Console()
        self.searcher = TavilySearcher()
        self.llm = ResearchLLM()

    def _plan(self, topic: str) -> list[SearchQuery]:
        """PLAN step: Generate diverse search queries for the topic."""
        self.console.print(
            Panel(
                f"[bold]Researching:[/] {topic}",
                title="Research Agent",
                border_style="blue",
            )
        )
        self.console.print("[bold cyan]Step 1/5:[/] Planning search strategy...")

        queries = self.llm.generate_search_queries(topic)

        table = Table(title="Search Plan", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Query", style="bold")
        table.add_column("Rationale", style="italic")
        for i, q in enumerate(queries, 1):
            table.add_row(str(i), q.query, q.rationale)
        self.console.print(table)

        return queries

    def _act(self, queries: list[SearchQuery]) -> dict[str, list[SearchResult]]:
        """ACT step: Execute all search queries via Tavily."""
        self.console.print("\n[bold cyan]Step 2/5:[/] Searching the web...")

        results = self.searcher.batch_search(queries)

        total_results = sum(len(v) for v in results.values())
        self.console.print(
            f"[green]Found {total_results} results across {len(results)} queries.[/]\n"
        )

        return results

    def _observe(
        self, search_results: dict[str, list[SearchResult]], topic: str
    ) -> list[SourceSummary]:
        """OBSERVE step: Summarize each unique source using the LLM."""
        self.console.print("[bold cyan]Step 3/5:[/] Analyzing sources...")

        seen_urls: set[str] = set()
        unique_results: list[SearchResult] = []
        for results_list in search_results.values():
            for result in results_list:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)

        self.console.print(
            f"[dim]{len(unique_results)} unique sources after deduplication.[/]"
        )

        summaries: list[SourceSummary] = []
        for i, result in enumerate(unique_results, 1):
            self.console.print(
                f"  [{i}/{len(unique_results)}] Analyzing: "
                f"[link={result.url}]{result.title}[/link]"
            )
            summary = self.llm.summarize_source(result.content, topic)
            # LLM may hallucinate url/title — trust the search result instead.
            summary.url = result.url
            summary.title = result.title
            summaries.append(summary)

        self.console.print(f"[green]Analyzed {len(summaries)} sources.[/]\n")
        return summaries

    def _reflect(
        self, summaries: list[SourceSummary], topic: str
    ) -> tuple[list[SourceSummary], list[Contradiction]]:
        """REFLECT step: Identify contradictions and do one round of follow-up searches if found."""
        self.console.print("[bold cyan]Step 4/5:[/] Identifying contradictions...")

        contradictions = self.llm.identify_contradictions(summaries)

        if contradictions:
            self.console.print(
                f"[yellow]Found {len(contradictions)} contradiction(s). "
                "Running follow-up searches...[/]"
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
            self.console.print(
                f"[dim]After follow-up: {len(contradictions)} contradiction(s) remain.[/]\n"
            )
        else:
            self.console.print("[green]No contradictions found across sources.[/]\n")

        return summaries, contradictions

    def _synthesize(
        self,
        topic: str,
        summaries: list[SourceSummary],
        contradictions: list[Contradiction],
    ) -> ResearchReport:
        """SYNTHESIZE step: Produce the final structured research report."""
        self.console.print("[bold cyan]Step 5/5:[/] Synthesizing final report...")

        report = self.llm.synthesize_report(topic, summaries, contradictions)

        self.console.print(
            Panel(
                f"[bold green]Research complete![/]\n"
                f"Sources analyzed: {len(summaries)}\n"
                f"Contradictions found: {len(contradictions)}\n"
                f"Follow-up questions: {len(report.follow_up_questions)}\n"
                f"Total API calls: {self.llm.api_call_count}",
                title="Summary",
                border_style="green",
            )
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
        self.console.print(f"[dim]Total time: {elapsed:.1f} seconds[/]\n")

        return report
