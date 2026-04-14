"""Tests for Pydantic models in src.models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models import (
    Contradiction,
    ResearchReport,
    SearchQuery,
    SearchResult,
    SourceSummary,
)


def test_search_query_valid():
    q = SearchQuery(query="climate tipping points", rationale="core topic")
    assert q.query == "climate tipping points"
    assert q.rationale == "core topic"


def test_search_result_score_bounds():
    SearchResult(url="https://x.com", title="t", content="c", relevance_score=0.5)

    with pytest.raises(ValidationError):
        SearchResult(url="https://x.com", title="t", content="c", relevance_score=-0.1)

    with pytest.raises(ValidationError):
        SearchResult(url="https://x.com", title="t", content="c", relevance_score=1.5)


def test_source_summary_optional_bias():
    s = SourceSummary(url="https://x.com", title="t", key_claims=["a", "b"])
    assert s.bias_or_perspective is None


def test_contradiction_all_fields_required():
    fields = {
        "claim_a": "A says X",
        "source_a": "src-a",
        "claim_b": "B says not X",
        "source_b": "src-b",
        "explanation": "they disagree",
    }
    Contradiction(**fields)

    for missing in fields:
        partial = {k: v for k, v in fields.items() if k != missing}
        with pytest.raises(ValidationError):
            Contradiction(**partial)


def test_research_report_defaults():
    before = datetime.now()
    report = ResearchReport(
        topic="t",
        summary="s",
        source_summaries=[],
        contradictions=[],
        follow_up_questions=[],
    )
    after = datetime.now()

    assert before <= report.generated_at <= after
    assert report.total_sources_searched == 0
    assert report.total_api_calls == 0


def test_research_report_to_markdown():
    sources = [
        SourceSummary(
            url="https://a.example",
            title="Source Alpha",
            key_claims=["claim a1", "claim a2"],
            bias_or_perspective="industry-funded",
        ),
        SourceSummary(
            url="https://b.example",
            title="Source Beta",
            key_claims=["claim b1"],
        ),
    ]
    contradictions = [
        Contradiction(
            claim_a="X is true",
            source_a="https://a.example",
            claim_b="X is false",
            source_b="https://b.example",
            explanation="direct factual disagreement between sources",
        )
    ]
    report = ResearchReport(
        topic="Quantum Widgets",
        summary="exec summary here",
        source_summaries=sources,
        contradictions=contradictions,
        follow_up_questions=["Who funded Alpha?", "Is Beta peer reviewed?"],
        total_sources_searched=5,
        total_api_calls=12,
    )

    md = report.to_markdown()

    assert "# Research Report: Quantum Widgets" in md
    assert "Source Alpha" in md
    assert "Source Beta" in md
    assert "direct factual disagreement between sources" in md
    assert "1. Who funded Alpha?" in md
    assert "2. Is Beta peer reviewed?" in md
    assert report.generated_at.isoformat() in md
