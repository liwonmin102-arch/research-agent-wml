"""Analyst sub-agent: one source → validated SourceAnalysis, run in parallel via asyncio."""

import asyncio
import sys

from anthropic import AsyncAnthropic

from src.models import SearchResult, SourceAnalysis

ANALYST_MODEL = "claude-sonnet-4-6"

ANALYST_SYSTEM_PROMPT = """You are the Analyst in a multi-agent research team. You receive ONE source at a time and produce a compact, structured analysis.

Your output is NOT a summary — it is a structured extraction. Other agents will read it, so every field matters and every field is LENGTH-CAPPED.

Extract exactly these things:

1. **core_claims** (1-7 items): The specific factual claims this source makes. Each claim statement must be ≤200 chars. Prefer concrete, checkable claims ("X increased by 47% in 2024") over vague ones ("X is growing fast"). Tag each claim's type (fact/statistic/opinion/prediction/methodology) and your confidence in it (low/medium/high). If the source provides a direct quote supporting the claim, include it in evidence_quote (≤300 chars). Always set source_url to this source's URL.

2. **methodology** (≤300 chars, optional): If the source describes how it arrived at its conclusions (study design, data source, sample size), note it briefly. Skip if not applicable.

3. **bias**: Assess the source's perspective:
   - political_lean: left / center_left / center / center_right / right / none_detected
   - industry_affiliation: if the publisher has industry ties (e.g., "funded by pharma trade group"), state briefly. Otherwise None.
   - editorial_stance: one-line description of the editorial angle if detectable (e.g., "tech-optimist"). Otherwise None.
   - funding_source: if disclosed. Otherwise None.
   Default to 'none_detected' when genuinely unclear — do NOT invent bias.

4. **credibility_score** (0-100): Rate the source. Rubric:
   - 90-100: peer-reviewed research, government data, major medical bodies
   - 75-89: established news orgs with editorial standards, expert think tanks
   - 60-74: trade publications, industry analyst reports, reputable blogs by credentialed experts
   - 40-59: general news, opinion pieces, content marketing
   - 20-39: anonymous blogs, SEO content, affiliate-heavy sites
   - 0-19: known misinformation, satire treated as news, stripped of context
   credibility_reasoning (≤250 chars): one-line justification.

5. **strengths** (0-3 items, each ≤120 chars): What this source does well.

6. **limitations** (0-3 items, each ≤120 chars): Methodological flaws, missing context, conflicts of interest.

7. **unique_contribution** (≤200 chars): What does THIS source add that others likely don't? One sentence.

Compactness rules — violate these and the pipeline breaks:
- Never exceed any length cap.
- Prefer specific over vague.
- If the source is thin (low-content snippet), return fewer claims rather than padding.
- Do not editorialize in the extraction — assess bias in the bias field, not in claim statements.

Output strictly matches the SourceAnalysis schema. No text outside the tool call."""


async def analyze_source_async(
    source: SearchResult,
    client: AsyncAnthropic,
) -> SourceAnalysis:
    """Analyze one source and return a structured SourceAnalysis."""
    schema = SourceAnalysis.model_json_schema()

    tool = {
        "name": "submit_source_analysis",
        "description": "Submit the structured analysis of this source.",
        "input_schema": schema,
    }

    user_content = (
        f"Source URL: {source.url}\n"
        f"Title: {source.title}\n"
        f"Domain: {source.source_domain}\n"
        f"Published: {source.published_date or 'unknown'}\n"
        f"Tavily relevance score: {source.tavily_score if source.tavily_score is not None else 'n/a'}\n\n"
        f"Content:\n{source.content_snippet}\n\n"
        f"Analyze this source now. Remember: all length caps are hard limits."
    )

    response = await client.messages.create(
        model=ANALYST_MODEL,
        max_tokens=2048,
        system=ANALYST_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_source_analysis"},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise ValueError(f"Analyst returned no tool_use block for {source.url}")

    try:
        return SourceAnalysis(**tool_use_block.input)
    except Exception as e:
        raise ValueError(
            f"Analyst output failed validation for {source.url}: {e}\n"
            f"Raw input: {tool_use_block.input}"
        ) from e


async def analyze_sources_parallel(
    sources: list[SearchResult],
    max_concurrency: int = 5,
    client: AsyncAnthropic | None = None,
) -> list[SourceAnalysis]:
    """Analyze multiple sources concurrently with a semaphore capping parallel API calls.

    Failures are logged to stderr and skipped; only successful analyses are returned.
    """
    if client is None:
        client = AsyncAnthropic()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(src: SearchResult) -> SourceAnalysis | None:
        async with semaphore:
            try:
                return await analyze_source_async(src, client)
            except Exception as e:
                print(f"[analyst] FAILED on {src.url}: {e}", file=sys.stderr)
                return None

    results = await asyncio.gather(*(_bounded(s) for s in sources))
    return [r for r in results if r is not None]


def analyze_sources(
    sources: list[SearchResult],
    max_concurrency: int = 5,
) -> list[SourceAnalysis]:
    """Sync wrapper around analyze_sources_parallel."""
    return asyncio.run(analyze_sources_parallel(sources, max_concurrency))


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping live test.")
        raise SystemExit(0)

    fake_sources = [
        SearchResult(
            url="https://www.nature.com/articles/example-ai-radiology",
            title="AI-assisted radiology reduces missed diagnoses by 23% in multi-center trial",
            content_snippet=(
                "A prospective multi-center trial across 12 hospitals found that AI-assisted "
                "radiology interpretation reduced missed diagnoses of early-stage lung cancer by "
                "23% compared to standard reads (n=4,300 patients). Lead author Dr. Chen noted "
                "that performance varied substantially by institution, with the best-performing "
                "site showing 31% improvement and the lowest showing only 9%. The study was "
                "funded by NIH grant R01-CA-xxxxxx and the authors report no conflicts of interest."
            ),
            source_domain="nature.com",
            published_date="2025-08-14",
            tavily_score=0.94,
        ),
        SearchResult(
            url="https://aihealthnow.example.com/blog/ai-will-replace-doctors",
            title="Why AI Will Replace Most Doctors by 2030 — And That's a Good Thing",
            content_snippet=(
                "The era of the human physician is ending. Our proprietary AI system outperforms "
                "human doctors across every benchmark we've tested. Sign up for our waitlist to "
                "get early access. Industry insiders agree — AI is coming for every medical job, "
                "and traditional training is now obsolete. Don't get left behind."
            ),
            source_domain="aihealthnow.example.com",
            published_date="2026-01-03",
            tavily_score=0.41,
        ),
    ]

    print(f"Analyzing {len(fake_sources)} sources in parallel...\n")
    analyses = analyze_sources(fake_sources, max_concurrency=2)

    for a in analyses:
        print("=" * 70)
        print(f"URL: {a.reference.url}")
        print(f"Credibility: {a.credibility_score}/100 — {a.credibility_reasoning}")
        print(f"Bias: lean={a.bias.political_lean}, stance={a.bias.editorial_stance}")
        print(f"Claims ({len(a.core_claims)}):")
        for c in a.core_claims:
            print(f"  [{c.claim_type}/{c.confidence}] {c.statement}")
        print(f"Unique: {a.unique_contribution}")
        if a.limitations:
            print(f"Limitations: {'; '.join(a.limitations)}")
        print()

    print(f"✓ Analyst agent works. {len(analyses)}/{len(fake_sources)} sources analyzed.")
