# Research Agent

An autonomous research agent that searches the web on any topic, summarizes findings from multiple sources, detects bias, identifies factual contradictions between sources, and produces structured reports. Built for researchers, analysts, and anyone who needs a thorough multi-source synthesis on a topic without manually reading dozens of pages.

## Features

- **Autonomous multi-query search planning** — generates diverse search queries covering different angles of a topic
- **Source summarization with bias detection** — extracts key factual claims and flags political lean, industry affiliation, or editorial slant
- **Cross-source contradiction identification** — finds places where sources directly disagree on facts, numbers, or conclusions
- **Follow-up research on contradictions** — automatically runs additional searches to clarify unresolved disagreements
- **Structured reports** — outputs in markdown or JSON with executive summary, per-source analysis, contradictions, and follow-up questions
- **CLI and web UI** — run from the terminal or through a browser interface
- **Real-time progress streaming** — web UI shows live progress via SSE as the agent works through each step

## Quick Start

```bash
git clone https://github.com/liwonmin102-arch/research-agent-wml.git
cd research-agent-wml/research-agent
pip install -e .
cp .env.example .env
# Edit .env and add your API keys
```

Run via CLI:

```bash
python -m src.main "your research topic"
```

Run via web UI:

```bash
uvicorn src.web:app --port 8000
```

## Usage

### CLI

```bash
python -m src.main "Impact of AI on healthcare in 2026"
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-sources` | 10 | Maximum number of sources to analyze |
| `--output-format` | markdown | Output format: `markdown` or `json` |
| `--output-dir` | outputs/ | Directory to save reports |
| `--verbose` | off | Show full tracebacks on errors |

### Web UI

```bash
uvicorn src.web:app --reload --port 8000
```

Open http://localhost:8000, enter a topic, and watch the agent work in real time. Reports render in the browser with download buttons for markdown and JSON.

## How It Works

The agent follows a 5-step loop:

1. **Plan** — Given a topic, the LLM generates diverse search queries covering factual overviews, recent developments, expert opinions, controversies, and data.
2. **Search** — Each query is executed against the Tavily search API with advanced search depth, collecting results across multiple sources.
3. **Analyze** — Each unique source is summarized by the LLM, extracting specific factual claims and noting any detectable bias or perspective.
4. **Reflect** — The LLM compares claims across all sources to identify genuine contradictions. If contradictions are found, follow-up searches run automatically and the analysis repeats.
5. **Synthesize** — All summaries and contradictions are combined into a final structured report with an executive summary and suggested follow-up questions.

## Tech Stack

- **Python 3.11+**
- **Anthropic Claude API** — LLM for query generation, summarization, contradiction detection, and report synthesis
- **Tavily Search API** — web search with advanced depth
- **FastAPI** — web server and API endpoints
- **HTMX + SSE** — real-time progress streaming in the browser
- **Pydantic** — data validation and structured models
- **Rich** — terminal output formatting

## API Keys

| Key | Where to get it | Notes |
|-----|----------------|-------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Required for all LLM calls |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Free tier includes 1,000 searches/month |

Add both to your `.env` file before running.

## License

MIT
