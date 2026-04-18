"""CLI entrypoint for the multi-agent research agent."""

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from src.models import DraftReport, ResearchSession
from src.orchestrator import run_research


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:max_len].rstrip("-")


def render_markdown(session: ResearchSession) -> str:
    """Render a ResearchSession as a full formal markdown report."""
    r: DraftReport | None = session.draft_report
    if r is None:
        raise ValueError("Cannot render markdown: session has no draft_report.")

    lines: list[str] = []
    lines.append(f"# {r.title}")
    lines.append("")
    lines.append(f"*Research topic: {session.topic}*  ")
    lines.append(
        f"*Generated: {(session.completed_at or datetime.now()):%Y-%m-%d %H:%M}*  "
    )
    if session.critic_report:
        lines.append(
            f"*Sources analyzed: {len(session.source_analyses)} | "
            f"Overall confidence: {session.critic_report.overall_confidence}/100*"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(r.executive_summary)
    lines.append("")

    lines.append("## Introduction")
    lines.append("")
    lines.append(r.introduction)
    lines.append("")

    lines.append("## Key Findings and Analysis")
    lines.append("")
    for f in r.key_findings:
        lines.append(f"### {f.theme}")
        lines.append("")
        lines.append(f.content)
        lines.append("")

    lines.append("## Contradictions, Debates, and Uncertainties")
    lines.append("")
    lines.append(r.contradictions_section)
    lines.append("")

    lines.append("## Implications and Applications")
    lines.append("")
    lines.append(r.implications)
    lines.append("")

    lines.append("## Limitations of Current Knowledge")
    lines.append("")
    lines.append(r.limitations)
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(r.conclusion)
    lines.append("")

    lines.append("## References")
    lines.append("")
    for i, ref in enumerate(r.references, 1):
        author = f" — {ref.author}" if ref.author else ""
        date = f" ({ref.publication_date})" if ref.publication_date else ""
        lines.append(f"{i}. [{ref.title}]({ref.url}){author}{date}")
    lines.append("")

    lines.append("## Study Guide for Deep Learning")
    lines.append("")
    lines.append("### Key Takeaways")
    lines.append("")
    for t in r.study_guide.key_takeaways:
        lines.append(f"- {t}")
    lines.append("")
    lines.append("### Critical Thinking Questions")
    lines.append("")
    for q in r.study_guide.critical_thinking_questions:
        lines.append(f"- {q}")
    lines.append("")
    lines.append("### Suggested Further Reading")
    lines.append("")
    for ref in r.study_guide.further_reading:
        author = f" — {ref.author}" if ref.author else ""
        lines.append(f"- [{ref.title}]({ref.url}){author}")
    lines.append("")

    return "\n".join(lines)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Multi-agent research agent: autonomous web research with bias detection, "
            "contradiction analysis, and formal report output."
        )
    )
    parser.add_argument("topic", type=str, help="The research topic")
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Override the Planner's target source count (default: Planner decides)",
    )
    parser.add_argument(
        "--output-format", choices=["markdown", "json"], default="markdown"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Skip the reflection round (faster, less thorough)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show full tracebacks on errors"
    )
    args = parser.parse_args()

    console = Console()

    try:
        session = run_research(
            topic=args.topic,
            enable_reflection=not args.no_reflection,
        )
    except Exception as e:
        console.print(f"[bold red]Research failed:[/] {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(args.topic)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.output_format == "markdown":
        output_path = output_dir / f"{slug}-{timestamp}.md"
        output_path.write_text(render_markdown(session), encoding="utf-8")
    else:
        output_path = output_dir / f"{slug}-{timestamp}.json"
        output_path.write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

    console.print(f"\n[bold green]Report saved:[/] {output_path}")


if __name__ == "__main__":
    main()
