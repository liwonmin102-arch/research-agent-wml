"""Report rendering in markdown, JSON, and HTML formats."""

import html

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


def render_html(report: ResearchReport) -> str:
    """Convert a ResearchReport into a styled HTML fragment for the web UI."""
    e = html.escape
    parts: list[str] = []

    parts.append("<h2>Executive Summary</h2>")
    parts.append(f"<p>{e(report.summary)}</p>")

    parts.append(f"<h2>Sources Analyzed ({len(report.source_summaries)})</h2>")
    for s in report.source_summaries:
        parts.append('<div class="source-card">')
        parts.append(
            f'<h3><a href="{e(s.url)}" target="_blank" rel="noopener">{e(s.title)}</a></h3>'
        )
        if s.key_claims:
            claims = "".join(f"<li>{e(c)}</li>" for c in s.key_claims)
            parts.append(f"<ul>{claims}</ul>")
        if s.bias_or_perspective:
            parts.append(f'<p class="bias"><em>{e(s.bias_or_perspective)}</em></p>')
        parts.append("</div>")

    parts.append(f"<h2>Contradictions ({len(report.contradictions)})</h2>")
    if report.contradictions:
        for c in report.contradictions:
            parts.append('<div class="contradiction-callout">')
            parts.append(
                f'<div class="claim-a"><strong>{e(c.source_a)}:</strong> {e(c.claim_a)}</div>'
            )
            parts.append(
                f'<div class="claim-b"><strong>{e(c.source_b)}:</strong> {e(c.claim_b)}</div>'
            )
            parts.append(f'<p class="explanation">{e(c.explanation)}</p>')
            parts.append("</div>")
    else:
        parts.append("<p><em>No contradictions identified across sources.</em></p>")

    parts.append("<h2>Follow-up Questions</h2>")
    if report.follow_up_questions:
        items = "".join(f"<li>{e(q)}</li>" for q in report.follow_up_questions)
        parts.append(f"<ol>{items}</ol>")

    parts.append(
        f"<footer><small>Generated at {e(report.generated_at.isoformat())} — "
        f"{report.total_sources_searched} sources, {report.total_api_calls} API calls</small></footer>"
    )

    return "".join(parts)
