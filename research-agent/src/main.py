"""CLI entry point for the research agent."""

import argparse
import datetime
import os
import re
import sys

from dotenv import load_dotenv
from rich.console import Console

from src.agent import ResearchAgent
from src.report import render_json, render_markdown


def main():
    load_dotenv()
    console = Console()

    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Autonomous research agent that searches, summarizes, and produces structured reports.",
    )
    parser.add_argument("topic", type=str, help="The research topic to investigate")
    parser.add_argument(
        "--max-sources",
        type=int,
        default=10,
        help="Maximum number of sources to analyze",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for the report",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save reports",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if len(args.topic.strip()) < 3:
        console.print("[red]Error: Topic must be at least 3 characters.[/]")
        sys.exit(1)
    if len(args.topic) > 500:
        console.print("[red]Error: Topic must be under 500 characters.[/]")
        sys.exit(1)

    missing_keys = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")
    if missing_keys:
        console.print(
            f"[red]Error: Missing environment variables: {', '.join(missing_keys)}[/]"
        )
        console.print("[dim]Copy .env.example to .env and fill in your API keys.[/]")
        sys.exit(1)

    console.print("\n[bold blue]═══ Research Agent ═══[/bold blue]\n")

    try:
        agent = ResearchAgent()
        report = agent.research(args.topic)

        os.makedirs(args.output_dir, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower()).strip("-")[:50]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.output_format == "markdown":
            content = render_markdown(report)
            filename = f"{slug}_{timestamp}.md"
        else:
            content = render_json(report)
            filename = f"{slug}_{timestamp}.json"

        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)

        console.print(f"\n[bold green]Report saved to:[/] {filepath}")

        console.print("\n[bold]Executive Summary:[/]")
        console.print(report.summary)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/]")
        console.print("[dim]Partial results were not saved.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        if args.verbose:
            console.print_exception()
        else:
            console.print("[dim]Run with --verbose for full traceback.[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
