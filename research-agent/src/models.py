"""Pydantic data models for the research agent pipeline."""

from datetime import datetime

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="The search query string")
    rationale: str = Field(description="Why this search angle is useful for the research")


class SearchResult(BaseModel):
    url: str
    title: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)


class SourceSummary(BaseModel):
    url: str
    title: str
    key_claims: list[str] = Field(description="Main factual claims extracted from this source")
    bias_or_perspective: str | None = Field(
        default=None,
        description="Any detected bias, slant, or perspective in the source",
    )


class Contradiction(BaseModel):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    explanation: str


class ResearchReport(BaseModel):
    topic: str
    summary: str
    source_summaries: list[SourceSummary]
    contradictions: list[Contradiction]
    follow_up_questions: list[str]
    generated_at: datetime = Field(default_factory=datetime.now)
    total_sources_searched: int = 0
    total_api_calls: int = 0

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Research Report: {self.topic}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self.summary)
        lines.append("")

        lines.append(f"## Sources Analyzed ({len(self.source_summaries)})")
        lines.append("")
        for source in self.source_summaries:
            lines.append(f"### {source.title}")
            lines.append("")
            lines.append(f"<{source.url}>")
            lines.append("")
            for claim in source.key_claims:
                lines.append(f"- {claim}")
            lines.append("")
            if source.bias_or_perspective:
                lines.append(f"*{source.bias_or_perspective}*")
                lines.append("")

        lines.append(f"## Contradictions Found ({len(self.contradictions)})")
        lines.append("")
        for c in self.contradictions:
            lines.append(f"> **Claim A** ({c.source_a}): {c.claim_a}")
            lines.append(">")
            lines.append(f"> **Claim B** ({c.source_b}): {c.claim_b}")
            lines.append(">")
            lines.append(f"> {c.explanation}")
            lines.append("")

        lines.append("## Suggested Follow-up Questions")
        lines.append("")
        for i, q in enumerate(self.follow_up_questions, start=1):
            lines.append(f"{i}. {q}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            f"*Generated at {self.generated_at.isoformat()} — "
            f"{self.total_sources_searched} sources searched, "
            f"{self.total_api_calls} API calls*"
        )

        return "\n".join(lines)
