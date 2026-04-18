"""Critic sub-agent: cross-compares SourceAnalysis list → validated CriticReport."""

import json

from anthropic import Anthropic

from src.models import CriticReport, SourceAnalysis

CRITIC_MODEL = "claude-sonnet-4-6"

CRITIC_SYSTEM_PROMPT = """You are the Critic in a multi-agent research team. You receive structured analyses of every source the team has gathered, and you produce a cross-source critical assessment.

You are NOT summarizing. You are comparing claims across sources to surface:
1. Genuine contradictions
2. Places where sources agree strongly (worth noting as robust findings)
3. Credibility ranking — which sources deserve the most weight
4. Gaps — things the sources collectively fail to address
5. Follow-up queries to resolve contradictions or fill gaps

## Contradiction detection — this is your hardest task

A GENUINE contradiction requires two (or more) sources making incompatible claims on the SAME factual question. Examples of what IS and IS NOT a contradiction:

GENUINE contradiction:
  - Source A: "AI radiology improved diagnostic accuracy by 23%"
  - Source B: "AI radiology showed no statistically significant improvement in diagnostic accuracy"
  → Real disagreement on the same outcome.

NOT a contradiction (different scope):
  - Source A: "AI improved lung cancer detection by 23%"
  - Source B: "AI improved breast cancer detection by 14%"
  → Different cancers. Not in conflict.

NOT a contradiction (different strength, same direction):
  - Source A: "AI significantly improves outcomes"
  - Source B: "AI modestly improves outcomes"
  → Both agree on direction. Note as agreement with uncertainty, not contradiction.

NOT a contradiction (opinion vs. fact):
  - Source A (peer-reviewed study): "AI increased efficiency by 18%"
  - Source B (opinion piece): "AI is overhyped and fails in practice"
  → Different claim types. Note credibility difference; don't treat as factual contradiction.

For each contradiction you find:
- Topic of disagreement: ≤200 chars, state the specific factual point.
- conflicting_claims: pull the actual Claim objects from the source analyses (min 2).
- possible_reasons: up to 4 short explanations — different methodologies, different time periods, selection bias, measurement differences, ideological framing, etc.
- resolution_status: unresolved / partially_resolved / resolved (you can only mark "resolved" if the higher-credibility source clearly supersedes the other on methodological grounds; otherwise default to "unresolved").
- follow_up_queries: up to 3 targeted searches that could resolve the disagreement.
- resolution_notes: ≤300 chars, optional.

## Agreements

Note where 3+ sources converge on the same claim. Keep each agreement ≤200 chars. Max 5 total — only the most important consensus points.

## Credibility ranking

Return source URLs ordered from MOST credible to LEAST, using the credibility_score and credibility_reasoning from each SourceAnalysis, plus your own judgment when scores are close.

## Gaps

What question — that the Planner identified or that naturally emerges — is NOT addressed by any source in this set? Up to 5, each ≤200 chars.

## Follow-up queries

Up to 5 search queries targeted at the biggest contradictions and gaps. These will be fed back to the Searcher for another round.

## Overall confidence

0-100: How confident are you in the current source set to answer the research topic thoroughly? Low if many contradictions unresolved, major gaps, or dominant sources are low-credibility.

## Compactness rules
- Honor ALL length caps.
- Do NOT list every minor disagreement. Focus on contradictions that would change a reader's understanding if resolved.
- Do NOT invent contradictions where sources simply cover different angles.

Output strictly matches the CriticReport schema. No text outside the tool call."""


def critique_sources(
    analyses: list[SourceAnalysis],
    research_topic: str,
    client: Anthropic | None = None,
) -> CriticReport:
    """Generate a cross-source critical report from a list of SourceAnalysis objects.

    Args:
        analyses: Structured analyses from the Analyst, one per source.
        research_topic: The original user topic, for context.
        client: Optional pre-configured Anthropic client.

    Returns:
        Validated CriticReport.
    """
    if client is None:
        client = Anthropic()

    schema = CriticReport.model_json_schema()

    tool = {
        "name": "submit_critic_report",
        "description": "Submit the cross-source critical report.",
        "input_schema": schema,
    }

    analyses_json = json.dumps(
        [a.model_dump(mode="json") for a in analyses],
        indent=2,
    )

    user_content = (
        f"Research topic: {research_topic}\n\n"
        f"You have {len(analyses)} source analyses to compare. "
        f"Cross-compare claims, find contradictions, rank credibility, identify gaps, "
        f"and propose follow-up queries.\n\n"
        f"SOURCE ANALYSES (JSON):\n{analyses_json}\n\n"
        f"Produce the critic report now."
    )

    response = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=4096,
        system=CRITIC_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_critic_report"},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise ValueError(f"Critic returned no tool_use block. Got: {response.content}")

    try:
        return CriticReport(**tool_use_block.input)
    except Exception as e:
        raise ValueError(
            f"Critic output failed validation: {e}\nRaw input: {tool_use_block.input}"
        ) from e


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping live test.")
        raise SystemExit(0)

    from src.models import (
        BiasAssessment,
        Claim,
        ClaimConfidence,
        ClaimType,
        PoliticalLean,
        Reference,
    )

    a1 = SourceAnalysis(
        reference=Reference(
            url="https://journal.example.com/study-a",
            title="Multi-center RCT on AI diagnostic accuracy",
            author="Chen et al.",
            publication_date="2025-08",
        ),
        core_claims=[
            Claim(
                statement="AI-assisted radiology reduced missed diagnoses by 23% in a 4,300-patient RCT.",
                claim_type=ClaimType.STATISTIC,
                confidence=ClaimConfidence.HIGH,
                source_url="https://journal.example.com/study-a",
            ),
            Claim(
                statement="AI tools significantly improve clinical outcomes across diverse hospital settings.",
                claim_type=ClaimType.FACT,
                confidence=ClaimConfidence.HIGH,
                source_url="https://journal.example.com/study-a",
            ),
        ],
        methodology="Prospective multi-center RCT, n=4300, 12 hospitals.",
        bias=BiasAssessment(political_lean=PoliticalLean.NONE_DETECTED),
        credibility_score=92,
        credibility_reasoning="Peer-reviewed, NIH-funded, large sample, no conflicts.",
        strengths=["Large multi-center sample", "Prospective design"],
        limitations=["High between-site variance (9%-31%)"],
        unique_contribution="First large prospective RCT with site-level variance data.",
    )

    a2 = SourceAnalysis(
        reference=Reference(
            url="https://journal.example.com/study-b",
            title="Systematic review: AI diagnostic tools in real-world deployment",
            author="Patel et al.",
            publication_date="2025-11",
        ),
        core_claims=[
            Claim(
                statement="Systematic review of 27 studies found no statistically significant improvement in diagnostic accuracy from AI tools.",
                claim_type=ClaimType.STATISTIC,
                confidence=ClaimConfidence.HIGH,
                source_url="https://journal.example.com/study-b",
            ),
            Claim(
                statement="Reported AI gains in trials often fail to replicate in real-world deployment.",
                claim_type=ClaimType.FACT,
                confidence=ClaimConfidence.HIGH,
                source_url="https://journal.example.com/study-b",
            ),
        ],
        methodology="Systematic review and meta-analysis, 27 studies, total n=180,000.",
        bias=BiasAssessment(political_lean=PoliticalLean.NONE_DETECTED),
        credibility_score=90,
        credibility_reasoning="Peer-reviewed systematic review, large pooled sample, methods registered on PROSPERO.",
        strengths=["Large pooled sample", "Pre-registered methodology"],
        limitations=["Heterogeneous study designs across included papers"],
        unique_contribution="Largest meta-analysis to date of real-world AI diagnostic performance.",
    )

    a3 = SourceAnalysis(
        reference=Reference(
            url="https://industry.example.com/whitepaper",
            title="AI in Radiology: 2026 Market Outlook",
            author="HealthTech Analytics",
            publication_date="2026-01",
        ),
        core_claims=[
            Claim(
                statement="AI radiology tools are projected to reach $12B market size by 2028.",
                claim_type=ClaimType.PREDICTION,
                confidence=ClaimConfidence.MEDIUM,
                source_url="https://industry.example.com/whitepaper",
            ),
            Claim(
                statement="Vendors report diagnostic accuracy improvements of 15-30% across installed deployments.",
                claim_type=ClaimType.STATISTIC,
                confidence=ClaimConfidence.MEDIUM,
                source_url="https://industry.example.com/whitepaper",
            ),
        ],
        methodology=None,
        bias=BiasAssessment(
            political_lean=PoliticalLean.NONE_DETECTED,
            industry_affiliation="Industry analyst firm serving healthtech vendors",
            editorial_stance="Tech-optimist; commercial interest in positive framing",
        ),
        credibility_score=55,
        credibility_reasoning="Industry analyst report; vendor-reported numbers with no independent verification.",
        strengths=["Broad market coverage"],
        limitations=["Self-reported vendor data", "Financial interest in optimistic outlook"],
        unique_contribution="Quantifies commercial adoption and market trajectory.",
    )

    print("Critiquing 3 sources on 'Impact of AI on healthcare in 2026'...\n")
    report = critique_sources([a1, a2, a3], "Impact of AI on healthcare in 2026")

    print(f"Overall confidence: {report.overall_confidence}/100\n")

    print(f"Agreements ({len(report.agreements)}):")
    for ag in report.agreements:
        print(f"  + {ag}")

    print(f"\nContradictions ({len(report.contradictions)}):")
    for c in report.contradictions:
        print(f"  ✗ {c.topic_of_disagreement}")
        print(f"    status: {c.resolution_status}")
        print("    possible reasons:")
        for r in c.possible_reasons:
            print(f"      - {r}")
        print("    follow-ups:")
        for q in c.follow_up_queries:
            print(f"      - {q}")

    print("\nCredibility ranking (high to low):")
    for i, url in enumerate(report.source_credibility_ranking, 1):
        print(f"  {i}. {url}")

    print(f"\nGaps ({len(report.gaps_identified)}):")
    for g in report.gaps_identified:
        print(f"  ? {g}")

    print(f"\nTop-level follow-up queries ({len(report.follow_up_queries)}):")
    for q in report.follow_up_queries:
        print(f"  → {q}")

    print("\n✓ Critic agent works.")
