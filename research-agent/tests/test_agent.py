"""Integration tests for ResearchAgent with mocked search and LLM."""

from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    Contradiction,
    ResearchReport,
    SearchQuery,
    SearchResult,
    SourceSummary,
)


def _default_summary(url="https://example.com", title="Test Source") -> SourceSummary:
    return SourceSummary(
        url=url,
        title=title,
        key_claims=["claim 1", "claim 2"],
        bias_or_perspective=None,
    )


def _default_report(summaries) -> ResearchReport:
    return ResearchReport(
        topic="test",
        summary="Test summary",
        source_summaries=summaries,
        contradictions=[],
        follow_up_questions=["Q1"],
        total_sources_searched=len(summaries),
        total_api_calls=3,
    )


@pytest.fixture
def mock_dependencies():
    with patch("src.agent.TavilySearcher") as searcher_cls, patch(
        "src.agent.ResearchLLM"
    ) as llm_cls:
        searcher = MagicMock()
        llm = MagicMock()

        searcher.batch_search.return_value = {
            "query1": [
                SearchResult(
                    url="https://example.com",
                    title="Test Source",
                    content="Test content about AI",
                    relevance_score=0.9,
                )
            ]
        }

        llm.generate_search_queries.return_value = [
            SearchQuery(query="query1", rationale="test reason")
        ]
        llm.summarize_source.return_value = _default_summary()
        llm.identify_contradictions.return_value = []
        llm.synthesize_report.return_value = _default_report([_default_summary()])
        llm.api_call_count = 3

        searcher_cls.return_value = searcher
        llm_cls.return_value = llm

        yield {"searcher": searcher, "llm": llm}


def test_research_returns_report(mock_dependencies):
    from src.agent import ResearchAgent

    agent = ResearchAgent()
    report = agent.research("test topic")

    assert isinstance(report, ResearchReport)
    assert report.topic == "test"
    assert report.summary == "Test summary"


def test_research_calls_steps_in_order(mock_dependencies):
    from src.agent import ResearchAgent

    searcher = mock_dependencies["searcher"]
    llm = mock_dependencies["llm"]

    agent = ResearchAgent()
    agent.research("test topic")

    llm.generate_search_queries.assert_called_once()
    searcher.batch_search.assert_called_once()
    llm.summarize_source.assert_called_once()
    llm.identify_contradictions.assert_called_once()
    llm.synthesize_report.assert_called_once()


def test_research_deduplicates_urls(mock_dependencies):
    from src.agent import ResearchAgent

    searcher = mock_dependencies["searcher"]
    llm = mock_dependencies["llm"]

    same_result = SearchResult(
        url="https://dup.example",
        title="Dup",
        content="dup content",
        relevance_score=0.9,
    )
    searcher.batch_search.return_value = {
        "q1": [same_result],
        "q2": [same_result],
    }
    llm.generate_search_queries.return_value = [
        SearchQuery(query="q1", rationale="r1"),
        SearchQuery(query="q2", rationale="r2"),
    ]
    llm.summarize_source.return_value = _default_summary(
        url="https://dup.example", title="Dup"
    )

    agent = ResearchAgent()
    agent.research("test topic")

    assert llm.summarize_source.call_count == 1


def test_reflect_triggers_followup_on_contradictions(mock_dependencies):
    from src.agent import ResearchAgent

    searcher = mock_dependencies["searcher"]
    llm = mock_dependencies["llm"]

    llm.identify_contradictions.side_effect = [
        [
            Contradiction(
                claim_a="A",
                source_a="https://a",
                claim_b="B",
                source_b="https://b",
                explanation="they disagree",
            )
        ],
        [],
    ]

    agent = ResearchAgent()
    agent.research("test topic")

    assert searcher.batch_search.call_count == 2
    assert llm.identify_contradictions.call_count == 2
