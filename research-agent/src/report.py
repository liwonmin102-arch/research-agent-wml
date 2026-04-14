"""Report rendering in markdown and JSON formats."""

from src.models import ResearchReport


def render_markdown(report: ResearchReport) -> str:
    frontmatter = (
        "---\n"
        f"topic: {report.topic}\n"
        f"generated: {report.generated_at.isoformat()}\n"
        f"sources: {len(report.source_summaries)}\n"
        f"contradictions: {len(report.contradictions)}\n"
        "---\n\n"
    )
    return frontmatter + report.to_markdown()


def render_json(report: ResearchReport) -> str:
    return report.model_dump_json(indent=2)
