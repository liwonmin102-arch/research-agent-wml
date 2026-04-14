"""Claude LLM wrapper for structured research tasks."""

import json
import os

import anthropic
from rich.console import Console

from src.models import Contradiction, ResearchReport, SearchQuery, SourceSummary


class ResearchLLM:
    """LLM client for the research agent's reasoning steps."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. Set it in your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.console = Console()
        self.api_call_count = 0

    def _call_llm(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 4096
    ) -> str:
        self.api_call_count += 1
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            self.console.print(f"[red]LLM API call failed: {e}[/]")
            raise

    def _parse_json(self, text: str) -> dict | list:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Strip opening fence (``` or ```json)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}") from e

    def generate_search_queries(
        self, topic: str, num_queries: int = 5
    ) -> list[SearchQuery]:
        system_prompt = (
            "You are a research planning assistant. Generate diverse search queries "
            "to thoroughly research a topic. Cover different angles: factual overview, "
            "recent developments, expert opinions, criticisms/controversies, and "
            "statistical data. Respond with ONLY a JSON array of objects, each with "
            "'query' and 'rationale' keys. No other text."
        )
        user_prompt = f"Generate {num_queries} search queries to research this topic: {topic}"

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                text = self._call_llm(system_prompt, user_prompt)
                data = self._parse_json(text)
                return [SearchQuery(**item) for item in data]
            except Exception as e:
                last_error = e
                if attempt == 0:
                    self.console.print(
                        "[yellow]Retrying query generation after parse failure...[/]"
                    )
        assert last_error is not None
        raise last_error

    def summarize_source(self, content: str, topic: str) -> SourceSummary:
        system_prompt = (
            "You are a research analyst. Given text content from a web source, "
            "extract the key factual claims as a list and identify any bias or "
            "perspective. Respond with ONLY a JSON object with these keys: "
            "'url' (string, use 'unknown' if not apparent), 'title' (string, "
            "infer from content or use 'Untitled'), 'key_claims' (array of strings "
            "— each should be one specific factual claim, not vague summaries), "
            "'bias_or_perspective' (string or null — note any political lean, "
            "industry affiliation, or editorial slant). No other text."
        )
        # Truncate to 3000 chars to cap token costs on long pages.
        user_prompt = f"Research topic: {topic}\n\nSource content to analyze:\n{content[:3000]}"

        for attempt in range(2):
            try:
                text = self._call_llm(system_prompt, user_prompt)
                data = self._parse_json(text)
                return SourceSummary(**data)
            except Exception:
                if attempt == 0:
                    self.console.print(
                        "[yellow]Retrying source summarization after parse failure...[/]"
                    )

        return SourceSummary(
            url="unknown",
            title="Untitled",
            key_claims=["Failed to parse source"],
            bias_or_perspective=None,
        )

    def identify_contradictions(
        self, summaries: list[SourceSummary]
    ) -> list[Contradiction]:
        if len(summaries) < 2:
            return []

        system_prompt = (
            "You are a critical research analyst. Given summaries from multiple "
            "sources about a topic, identify factual contradictions — places where "
            "sources directly disagree on facts, numbers, dates, or conclusions. "
            "Only flag genuine contradictions, not minor differences in phrasing. "
            "If no real contradictions exist, return an empty array. Respond with "
            "ONLY a JSON array of objects, each with keys: 'claim_a', 'source_a', "
            "'claim_b', 'source_b', 'explanation'. No other text."
        )

        blocks = []
        for s in summaries:
            claims = "".join(f"- {c}\n" for c in s.key_claims)
            blocks.append(f"Source: {s.title} ({s.url})\nClaims:\n{claims}")
        user_prompt = "\n---\n".join(blocks)

        try:
            text = self._call_llm(system_prompt, user_prompt)
            data = self._parse_json(text)
            return [Contradiction(**item) for item in data]
        except Exception as e:
            self.console.print(
                f"[yellow]Failed to parse contradictions, returning empty list: {e}[/]"
            )
            return []

    def synthesize_report(
        self,
        topic: str,
        summaries: list[SourceSummary],
        contradictions: list[Contradiction],
    ) -> ResearchReport:
        system_prompt = (
            "You are a senior research analyst writing a final research briefing. "
            "Synthesize all source summaries and contradictions into a coherent "
            "report. Respond with ONLY a JSON object with these keys: 'summary' "
            "(string — a 2-3 paragraph executive summary that synthesizes key "
            "findings, don't just list what sources said), 'follow_up_questions' "
            "(array of 3-5 strings — questions that remain unanswered or need "
            "deeper investigation). No other text."
        )

        source_blocks = []
        for s in summaries:
            claims = "".join(f"- {c}\n" for c in s.key_claims)
            source_blocks.append(f"Title: {s.title}\nURL: {s.url}\nClaims:\n{claims}")
        sources_section = "\n---\n".join(source_blocks)

        if contradictions:
            contradiction_blocks = [
                f"- {c.claim_a} ({c.source_a}) vs. {c.claim_b} ({c.source_b}): {c.explanation}"
                for c in contradictions
            ]
            contradictions_section = "\n".join(contradiction_blocks)
        else:
            contradictions_section = "No contradictions were identified."

        user_prompt = (
            f"Topic: {topic}\n\n"
            f"SOURCES:\n{sources_section}\n\n"
            f"CONTRADICTIONS FOUND:\n{contradictions_section}"
        )

        text = self._call_llm(system_prompt, user_prompt)
        data = self._parse_json(text)

        return ResearchReport(
            topic=topic,
            summary=data["summary"],
            source_summaries=summaries,
            contradictions=contradictions,
            follow_up_questions=data["follow_up_questions"],
            total_sources_searched=len(summaries),
            total_api_calls=self.api_call_count,
        )
