"""End-to-end research pipeline orchestrator."""

from collections.abc import Callable
from datetime import datetime

from rich.console import Console

from src.agents.analyst import analyze_sources
from src.agents.critic import critique_sources
from src.agents.planner import create_plan
from src.agents.synthesizer import synthesize_report
from src.models import (
    CriticReport,
    DraftReport,
    ResearchPlan,
    ResearchSession,
    SourceAnalysis,
)
from src.search import TavilySearcher

# Threshold below which we run a reflection round to fill gaps / resolve contradictions.
REFLECTION_CONFIDENCE_THRESHOLD = 70
# Hard cap on total sources analyzed across all rounds (safety against runaway cost).
MAX_TOTAL_SOURCES = 30


def run_research(
    topic: str,
    max_results_per_query: int = 3,
    enable_reflection: bool = True,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> ResearchSession:
    """Run the full multi-agent research pipeline.

    Args:
        topic: User's research topic.
        max_results_per_query: How many Tavily results per search query.
        enable_reflection: If True, run a second round when confidence is low.
        progress_callback: Optional callback fn(stage_name, data_dict) for streaming.

    Returns:
        Completed ResearchSession with the final DraftReport populated.
    """
    console = Console()
    session = ResearchSession(topic=topic)

    def _emit(stage: str, data: dict) -> None:
        if progress_callback is not None:
            progress_callback(stage, data)

    # --- STAGE 1: Plan ---
    console.print("\n[bold cyan]━━ STAGE 1: Planner[/]")
    _emit("planner_start", {"topic": topic})
    plan: ResearchPlan = create_plan(topic)
    session.plan = plan
    console.print(
        f"Effort: [bold]{plan.effort_level}[/]  |  "
        f"Target sources: [bold]{plan.target_source_count}[/]  |  "
        f"{len(plan.queries)} queries"
    )
    _emit(
        "planner_done",
        {
            "effort_level": str(plan.effort_level),
            "target_source_count": plan.target_source_count,
            "sub_questions": plan.sub_questions,
            "queries": plan.queries,
        },
    )

    # --- STAGE 2: Search ---
    console.print("\n[bold cyan]━━ STAGE 2: Searcher[/]")
    _emit("search_start", {"query_count": len(plan.queries)})
    searcher = TavilySearcher()
    search_results = searcher.batch_search(
        plan.queries, max_results_per_query=max_results_per_query
    )
    # Trim to target_source_count (highest Tavily score first)
    search_results.sort(key=lambda r: r.tavily_score or 0.0, reverse=True)
    search_results = search_results[: plan.target_source_count]
    session.search_results = search_results
    _emit("search_done", {"source_count": len(search_results)})

    # --- STAGE 3: Analyst ---
    console.print("\n[bold cyan]━━ STAGE 3: Analyst[/]")
    _emit("analyst_start", {"source_count": len(search_results)})
    analyses: list[SourceAnalysis] = analyze_sources(search_results, max_concurrency=5)
    session.source_analyses = analyses
    console.print(
        f"Analyzed [bold]{len(analyses)}/{len(search_results)}[/] sources successfully"
    )
    _emit("analyst_done", {"analyzed_count": len(analyses)})

    if not analyses:
        raise RuntimeError("Analyst produced zero successful analyses; cannot continue.")

    # --- STAGE 4: Critic (first pass) ---
    console.print("\n[bold cyan]━━ STAGE 4: Critic[/]")
    _emit("critic_start", {"round": 1})
    critic_report: CriticReport = critique_sources(analyses, topic)
    session.critic_report = critic_report
    console.print(
        f"Confidence: [bold]{critic_report.overall_confidence}/100[/]  |  "
        f"Contradictions: {len(critic_report.contradictions)}  |  "
        f"Gaps: {len(critic_report.gaps_identified)}"
    )
    _emit(
        "critic_done",
        {
            "round": 1,
            "overall_confidence": critic_report.overall_confidence,
            "contradictions": len(critic_report.contradictions),
            "gaps": len(critic_report.gaps_identified),
        },
    )

    # --- STAGE 4b: Reflection round (optional) ---
    should_reflect = (
        enable_reflection
        and critic_report.overall_confidence < REFLECTION_CONFIDENCE_THRESHOLD
        and critic_report.follow_up_queries
        and len(session.source_analyses) < MAX_TOTAL_SOURCES
    )

    if should_reflect:
        console.print(
            f"\n[bold magenta]━━ STAGE 4b: Reflection "
            f"(confidence below {REFLECTION_CONFIDENCE_THRESHOLD})[/]"
        )
        _emit(
            "reflection_start",
            {"follow_up_queries": critic_report.follow_up_queries},
        )

        remaining_budget = MAX_TOTAL_SOURCES - len(session.source_analyses)
        follow_up_queries = critic_report.follow_up_queries[:3]

        new_results = searcher.batch_search(follow_up_queries, max_results_per_query=2)
        existing_urls = {r.url for r in session.search_results}
        new_results = [r for r in new_results if r.url not in existing_urls][
            :remaining_budget
        ]
        console.print(f"Fetched {len(new_results)} additional unique sources")

        if new_results:
            new_analyses = analyze_sources(new_results, max_concurrency=5)
            session.search_results.extend(new_results)
            session.source_analyses.extend(new_analyses)
            console.print(
                f"Total sources after reflection: [bold]{len(session.source_analyses)}[/]"
            )

            _emit("critic_start", {"round": 2})
            critic_report = critique_sources(session.source_analyses, topic)
            session.critic_report = critic_report
            console.print(
                f"Round 2 confidence: [bold]{critic_report.overall_confidence}/100[/]"
            )
            _emit(
                "critic_done",
                {
                    "round": 2,
                    "overall_confidence": critic_report.overall_confidence,
                    "contradictions": len(critic_report.contradictions),
                    "gaps": len(critic_report.gaps_identified),
                },
            )

        session.iteration_count = 1

    # --- STAGE 5: Synthesizer ---
    console.print("\n[bold cyan]━━ STAGE 5: Synthesizer[/]")
    _emit("synthesizer_start", {})
    draft: DraftReport = synthesize_report(
        topic, plan, session.source_analyses, critic_report
    )
    session.draft_report = draft
    session.completed_at = datetime.now()
    _emit("synthesizer_done", {"title": draft.title})

    console.print(
        f"\n[bold green]✓ Research complete.[/] "
        f"[dim]{len(session.source_analyses)} sources | "
        f"{len(draft.key_findings)} findings | "
        f"{len(draft.references)} references[/]"
    )

    return session


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print(
            "Both ANTHROPIC_API_KEY and TAVILY_API_KEY must be set — skipping live test."
        )
        raise SystemExit(0)

    session = run_research(
        "The efficacy-effectiveness gap in AI medical imaging tools",
        enable_reflection=True,
    )
    print(f"\n{'='*70}")
    assert session.draft_report is not None
    assert session.critic_report is not None
    assert session.completed_at is not None
    print(f"TITLE: {session.draft_report.title}")
    print(f"Sources: {len(session.source_analyses)}")
    print(f"Findings: {len(session.draft_report.key_findings)}")
    print(f"Confidence: {session.critic_report.overall_confidence}/100")
    print(f"Iterations: {session.iteration_count}")
    print(
        f"Duration: {(session.completed_at - session.started_at).total_seconds():.1f}s"
    )
