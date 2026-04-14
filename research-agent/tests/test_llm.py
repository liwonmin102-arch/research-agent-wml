"""Tests for ResearchLLM in src.llm with mocked Anthropic client."""

from unittest.mock import MagicMock, patch

import pytest

from src.models import Contradiction, ResearchReport, SourceSummary


def _make_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def mock_anthropic():
    with patch("src.llm.anthropic.Anthropic") as cls, patch(
        "src.llm.os.getenv", return_value="fake-key"
    ):
        instance = MagicMock()
        cls.return_value = instance
        yield instance


def _set_response(mock_anthropic, text: str) -> None:
    mock_anthropic.messages.create.return_value = _make_response(text)


def test_generate_search_queries(mock_anthropic):
    from src.llm import ResearchLLM
    from src.models import SearchQuery

    _set_response(
        mock_anthropic,
        '[{"query": "test query 1", "rationale": "reason 1"}, '
        '{"query": "test query 2", "rationale": "reason 2"}]',
    )

    llm = ResearchLLM()
    queries = llm.generate_search_queries("AI safety")

    assert len(queries) == 2
    assert all(isinstance(q, SearchQuery) for q in queries)
    assert queries[0].query == "test query 1"
    assert queries[0].rationale == "reason 1"
    assert queries[1].query == "test query 2"


def test_generate_search_queries_strips_code_fences(mock_anthropic):
    from src.llm import ResearchLLM

    _set_response(
        mock_anthropic,
        '```json\n[{"query": "test", "rationale": "reason"}]\n```',
    )

    llm = ResearchLLM()
    queries = llm.generate_search_queries("topic")

    assert len(queries) == 1
    assert queries[0].query == "test"


def test_summarize_source(mock_anthropic):
    from src.llm import ResearchLLM

    _set_response(
        mock_anthropic,
        '{"url": "https://example.com", "title": "Test", '
        '"key_claims": ["claim 1", "claim 2"], "bias_or_perspective": "neutral"}',
    )

    llm = ResearchLLM()
    summary = llm.summarize_source("some content", "test topic")

    assert isinstance(summary, SourceSummary)
    assert summary.url == "https://example.com"
    assert summary.title == "Test"
    assert summary.key_claims == ["claim 1", "claim 2"]
    assert summary.bias_or_perspective == "neutral"


def test_identify_contradictions_with_few_sources(mock_anthropic):
    from src.llm import ResearchLLM

    llm = ResearchLLM()
    only = SourceSummary(url="u", title="t", key_claims=["c"])
    result = llm.identify_contradictions([only])

    assert result == []
    assert mock_anthropic.messages.create.call_count == 0


def test_identify_contradictions(mock_anthropic):
    from src.llm import ResearchLLM

    _set_response(
        mock_anthropic,
        '[{"claim_a": "X is 5", "source_a": "https://a", '
        '"claim_b": "X is 10", "source_b": "https://b", '
        '"explanation": "Numeric disagreement"}]',
    )

    llm = ResearchLLM()
    summaries = [
        SourceSummary(url="https://a", title="A", key_claims=["X is 5"]),
        SourceSummary(url="https://b", title="B", key_claims=["X is 10"]),
    ]
    result = llm.identify_contradictions(summaries)

    assert len(result) == 1
    assert isinstance(result[0], Contradiction)
    assert result[0].claim_a == "X is 5"
    assert result[0].source_b == "https://b"
    assert result[0].explanation == "Numeric disagreement"


def test_synthesize_report(mock_anthropic):
    from src.llm import ResearchLLM

    _set_response(
        mock_anthropic,
        '{"summary": "Executive summary text", '
        '"follow_up_questions": ["Q1", "Q2"]}',
    )

    llm = ResearchLLM()
    summaries = [
        SourceSummary(url="https://a", title="A", key_claims=["c1"]),
        SourceSummary(url="https://b", title="B", key_claims=["c2"]),
    ]
    contradictions = [
        Contradiction(
            claim_a="X",
            source_a="https://a",
            claim_b="Y",
            source_b="https://b",
            explanation="they disagree",
        )
    ]

    report = llm.synthesize_report("my topic", summaries, contradictions)

    assert isinstance(report, ResearchReport)
    assert report.topic == "my topic"
    assert report.summary == "Executive summary text"
    assert report.source_summaries == summaries
    assert report.contradictions == contradictions
    assert report.follow_up_questions == ["Q1", "Q2"]
    assert report.total_sources_searched == 2
    assert report.total_api_calls == llm.api_call_count


def test_api_call_counter(mock_anthropic):
    from src.llm import ResearchLLM

    llm = ResearchLLM()

    _set_response(mock_anthropic, '[{"query": "q", "rationale": "r"}]')
    llm.generate_search_queries("t")

    _set_response(
        mock_anthropic,
        '{"url": "u", "title": "t", "key_claims": ["c"], "bias_or_perspective": null}',
    )
    llm.summarize_source("content", "topic")

    _set_response(
        mock_anthropic,
        '{"summary": "s", "follow_up_questions": ["q"]}',
    )
    llm.synthesize_report("topic", [], [])

    assert llm.api_call_count == 3
