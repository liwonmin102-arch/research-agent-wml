"""Tests for TavilySearcher in src.search."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub the `tavily` package so `src.search` imports without the real dep installed.
if "tavily" not in sys.modules:
    tavily_stub = types.ModuleType("tavily")
    tavily_stub.TavilyClient = MagicMock()
    sys.modules["tavily"] = tavily_stub

from src.models import SearchQuery, SearchResult  # noqa: E402


FAKE_RESPONSE = {
    "results": [
        {
            "url": "https://example.com/1",
            "title": "Test Result 1",
            "content": "Some content about the topic",
            "score": 0.95,
        },
        {
            "url": "https://example.com/2",
            "title": "Test Result 2",
            "content": "More content here",
            "score": 0.82,
        },
    ]
}


@pytest.fixture
def mock_tavily_client():
    with patch("src.search.TavilyClient") as client_cls, patch(
        "src.search.os.getenv", return_value="fake-api-key"
    ):
        instance = MagicMock()
        instance.search.return_value = FAKE_RESPONSE
        client_cls.return_value = instance
        yield instance


def test_search_returns_search_results(mock_tavily_client):
    from src.search import TavilySearcher

    searcher = TavilySearcher()
    results = searcher.search("test query")

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].url == "https://example.com/1"
    assert results[0].title == "Test Result 1"
    assert results[0].content == "Some content about the topic"
    assert results[0].relevance_score == 0.95
    assert results[1].url == "https://example.com/2"
    assert results[1].relevance_score == 0.82


def test_search_handles_api_error(mock_tavily_client):
    from src.search import TavilySearcher

    mock_tavily_client.search.side_effect = Exception("API Error")
    searcher = TavilySearcher()
    results = searcher.search("test query")

    assert results == []


def test_batch_search_calls_each_query(mock_tavily_client):
    from src.search import TavilySearcher

    queries = [
        SearchQuery(query=f"q{i}", rationale=f"reason {i}") for i in range(3)
    ]

    with patch("src.search.time.sleep"):
        searcher = TavilySearcher()
        results = searcher.batch_search(queries)

    assert set(results.keys()) == {"q0", "q1", "q2"}
    assert mock_tavily_client.search.call_count == 3
    for key in results:
        assert len(results[key]) == 2


def test_missing_api_key():
    with patch("src.search.os.getenv", return_value=None):
        from src.search import TavilySearcher

        with pytest.raises(ValueError):
            TavilySearcher()
