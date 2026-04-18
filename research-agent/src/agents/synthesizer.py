"""Synthesizer sub-agent: plan + analyses + critic → validated DraftReport."""

import json

from anthropic import Anthropic

from src.models import (
    CriticReport,
    DraftReport,
    ResearchPlan,
    SourceAnalysis,
)

SYNTHESIZER_MODEL = "claude-sonnet-4-6"

SYNTHESIZER_SYSTEM_PROMPT = """You are the Synthesizer in a multi-agent research team. You write the final formal research report that the user reads.

Your audience: a student doing self-directed learning at college level. They want depth and rigor, but they also want to UNDERSTAND — not just be impressed by jargon. Your writing must earn its formality by being clear, not by being dense.

You receive:
- The research topic
- A ResearchPlan with sub-questions the report should address
- A list of SourceAnalysis objects — compact, structured extractions of each source
- A CriticReport with agreements, contradictions, credibility rankings, and gaps

You produce a DraftReport matching the schema exactly. Do not write anything outside the structured fields.

## Voice and style rules — non-negotiable

1. **Clarity over performance.** Short paragraphs (3-6 sentences typical). Short sentences where possible. Active voice. Academic but not bureaucratic.

2. **Define technical terms on first use.** If you introduce a term like "federated learning" or "sensitivity/specificity" or "PROSPERO registration," give a one-phrase definition inline. The reader is learning, not already expert.

3. **Concrete before abstract.** When a finding has a specific number or example from the sources, lead with that number or example. Generalize afterward. "A 2025 multi-center trial (n=4,300) found a 23% reduction in missed diagnoses; this pattern — strong gains at the study level, uneven replication in practice — recurs across the literature."

4. **Cite inline, then list in References.** Use format `[Source N]` inline where N is the reference's position in the references list. Every numerical claim, direct quote, or specific finding must be cited.

5. **Balance is earned, not declared.** Don't say "on the other hand" reflexively. Actually present opposing evidence with equal seriousness. If the evidence genuinely leans one way, say so and explain why — don't manufacture false symmetry.

6. **Flag uncertainty explicitly.** If a finding is disputed or tentative, say "Evidence is mixed" or "This remains unresolved; see the Contradictions section" rather than burying the caveat.

## Per-section guidance

**title**: Descriptive and specific. Not "AI in Healthcare" — something like "AI in Healthcare (2026): Diagnostic Accuracy, Deployment Gaps, and Regulatory Evolution."

**executive_summary** (200-400 words): The stand-alone TL;DR. What did you find? What are the main conclusions? Where is the evidence strongest, where is it weakest? A reader should be able to stop after the executive summary and know the core shape of the topic. Avoid listing every finding — hit the important ones and signal depth.

**introduction**: Frame the topic, state why it matters right now, and preview the sub-questions the report will address. 2-4 short paragraphs.

**key_findings** (list of FindingSection, min 2, typically 3-6): Organized by theme, NOT by source. Each FindingSection has:
- `theme`: short thematic label (e.g., "Diagnostic Accuracy in Clinical Trials vs. Real-World Deployment")
- `content`: multi-paragraph synthesis pulling evidence from multiple sources. This is where depth lives.
- `supporting_sources`: list of URLs that directly inform this theme.

Organize themes around the Planner's sub-questions where possible, but feel free to reorganize if the evidence actually clusters differently. Go DEEP here — this section is the bulk of the report.

**contradictions_section**: Walk through the genuine contradictions from the CriticReport. For each: state the disagreement, present both sides, discuss possible reasons (methodological differences, scope differences, selection bias, etc.), state resolution status honestly. This is a prose version of the CriticReport's contradictions — do not just dump the JSON.

**implications**: What does this mean? Theoretical implications, practical implications, implications for practitioners/patients/policymakers. Ground every implication in the evidence.

**limitations**: Be honest about what THIS REPORT cannot conclude. Gaps in the source set (pull from CriticReport.gaps_identified). Low-credibility sources where the evidence thins out. Topics that moved faster than the research could track.

**conclusion**: What can we say with confidence? What remains open? What would change the picture if new evidence emerged? Avoid restating the executive summary — conclude, don't summarize.

**references**: Full list. Use the Reference objects from the SourceAnalysis inputs. Number them in the order they first appear in the body.

**study_guide**: This is not an afterthought — this is the pedagogical payoff.
- key_takeaways (5-8): Each takeaway is 2-4 sentences (≤500 chars total) and must do TWO things: state the finding AND explain WHY it matters. Format example: "Observation or finding. Why this matters / what it reframes for the learner." Bad: "AI reduced missed diagnoses by 23%." Good: "AI reduced missed diagnoses by 23% in a major RCT, but site-to-site performance ranged from 9% to 31% — meaning the headline number hides more than it reveals, and vendor claims based on the 23% average should be read with that variance in mind." Aim for takeaways that would change how the reader thinks about the topic, not just what they know.
- `critical_thinking_questions` (3-5): Open-ended. Force the reader to reason about the material, not recall it. "If a systematic review and an RCT disagree on the same intervention, which should clinicians trust, and under what conditions?"
- `further_reading` (2-4 Reference objects): Pick the highest-credibility, highest-yield sources from the pool. Include a brief note on each in the title field if useful.

## Length discipline — hard requirement

You have a single response budget and MUST fit all nine required fields within it: title, executive_summary, introduction, key_findings, contradictions_section, implications, limitations, conclusion, references, and study_guide. Running out of space before completing study_guide is a pipeline failure.

Target section lengths (soft guides, adjust for topic complexity):
- executive_summary: 200-400 words
- introduction: 200-400 words
- each key_findings section: 250-450 words (3-6 sections total; DO NOT exceed 6)
- contradictions_section: 300-800 words (cover each contradiction in 100-200 words, not 500+)
- implications: 200-400 words
- limitations: 150-300 words
- conclusion: 150-300 words
- study_guide: key_takeaways ~80-150 words each, questions 1-2 sentences each

If the topic is rich enough to support more depth in key_findings or contradictions_section, compress the less central sections — but NEVER skip a required field. Every field must be present. Finish every section. No field is optional.

If you find yourself approaching the budget mid-way through the report, shorten remaining sections rather than truncating the final ones. The conclusion and study_guide exist to serve the learner; they are not ornamental.

Output strictly matches the DraftReport schema."""


def synthesize_report(
    topic: str,
    plan: ResearchPlan,
    analyses: list[SourceAnalysis],
    critic_report: CriticReport,
    client: Anthropic | None = None,
) -> DraftReport:
    """Produce the final structured research report.

    Args:
        topic: Original research topic from the user.
        plan: The ResearchPlan from the Planner.
        analyses: All SourceAnalysis objects produced by the Analyst.
        critic_report: The CriticReport from the Critic.
        client: Optional pre-configured Anthropic client.

    Returns:
        Validated DraftReport.
    """
    if client is None:
        client = Anthropic()

    schema = DraftReport.model_json_schema()

    tool = {
        "name": "submit_draft_report",
        "description": "Submit the final structured research report.",
        "input_schema": schema,
    }

    payload = {
        "topic": topic,
        "plan": plan.model_dump(mode="json"),
        "source_analyses": [a.model_dump(mode="json") for a in analyses],
        "critic_report": critic_report.model_dump(mode="json"),
    }

    user_content = (
        f"Write the formal research report now.\n\n"
        f"RESEARCH TOPIC: {topic}\n\n"
        f"You have {len(analyses)} analyzed sources and a critic report with "
        f"{len(critic_report.contradictions)} contradictions and "
        f"{len(critic_report.gaps_identified)} gaps identified.\n\n"
        f"FULL INPUT (JSON):\n{json.dumps(payload, indent=2)}\n\n"
        f"Produce the DraftReport now, following every rule in your system prompt."
    )

    response = client.messages.create(
        model=SYNTHESIZER_MODEL,
        max_tokens=16000,
        system=SYNTHESIZER_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_draft_report"},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise ValueError(
            f"Synthesizer returned no tool_use block. Got: {response.content}"
        )

    try:
        return DraftReport(**tool_use_block.input)
    except Exception as e:
        # Attempt ONE recovery retry: feed the partial back and ask for only the missing fields.
        partial = tool_use_block.input if tool_use_block else {}
        required_fields = [
            "title", "executive_summary", "introduction", "key_findings",
            "contradictions_section", "implications", "limitations",
            "conclusion", "references", "study_guide",
        ]
        missing = [
            f for f in required_fields
            if f not in partial or partial.get(f) in (None, "", [])
        ]
        if not missing:
            raise ValueError(
                f"Synthesizer output failed validation despite all fields present: {e}\n"
                f"Raw input: {partial}"
            ) from e

        import sys
        print(
            f"[synthesizer] First attempt missing fields {missing}; retrying to complete.",
            file=sys.stderr,
        )

        retained = {k: partial.get(k) for k in partial if k not in missing}
        retry_user_content = (
            "Your previous attempt produced a partial DraftReport. The following fields are present "
            f"and FINAL — DO NOT rewrite them:\n\n{json.dumps(retained, indent=2)}\n\n"
            f"Complete the report by returning the SAME partial above PLUS these missing fields "
            f"(which you must fully write now): {', '.join(missing)}.\n\n"
            "Rules:\n"
            "1. Return the complete DraftReport including both the previously-written fields and the newly-written ones.\n"
            "2. Do not change or re-phrase the fields already written.\n"
            "3. Keep the missing fields within the target length ranges in the system prompt.\n"
            "4. The report must end with a fully populated study_guide.\n"
        )

        retry_response = client.messages.create(
            model=SYNTHESIZER_MODEL,
            max_tokens=16000,
            system=SYNTHESIZER_SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_draft_report"},
            messages=[{"role": "user", "content": retry_user_content}],
        )
        retry_tool_use = next(
            (block for block in retry_response.content if block.type == "tool_use"),
            None,
        )
        if retry_tool_use is None:
            raise ValueError("Synthesizer retry returned no tool_use block.") from e
        try:
            return DraftReport(**retry_tool_use.input)
        except Exception as e2:
            raise ValueError(
                f"Synthesizer retry ALSO failed validation: {e2}\n"
                f"First attempt missing: {missing}\n"
                f"Retry raw: {retry_tool_use.input}"
            ) from e2


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping live test.")
        raise SystemExit(0)

    from src.agents.critic import critique_sources
    from src.models import (
        BiasAssessment,
        Claim,
        ClaimConfidence,
        ClaimType,
        EffortLevel,
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
        credibility_reasoning="Peer-reviewed systematic review, large pooled sample, PROSPERO-registered.",
        strengths=["Large pooled sample", "Pre-registered methodology"],
        limitations=["Heterogeneous study designs"],
        unique_contribution="Largest meta-analysis of real-world AI diagnostic performance to date.",
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
        credibility_reasoning="Industry report; vendor self-reported numbers without independent validation.",
        strengths=["Broad market coverage"],
        limitations=["Self-reported vendor data", "Financial interest in optimistic outlook"],
        unique_contribution="Quantifies commercial adoption trajectory.",
    )

    plan = ResearchPlan(
        topic="Impact of AI on healthcare in 2026",
        sub_questions=[
            "What does current evidence show about AI diagnostic accuracy in clinical settings?",
            "How do trial results compare to real-world deployment outcomes?",
            "What does the commercial and regulatory landscape look like in 2026?",
        ],
        queries=[
            "AI radiology diagnostic accuracy 2025",
            "AI healthcare real-world deployment 2026",
            "AI medical device regulation 2025",
            "AI radiology market size 2026",
            "AI clinical trial efficacy-effectiveness gap",
        ],
        effort_level=EffortLevel.MEDIUM,
        target_source_count=8,
        angles_covered=[
            "clinical evidence",
            "real-world deployment",
            "commercial trends",
            "methodology",
        ],
    )

    analyses = [a1, a2, a3]

    print("Running Critic → Synthesizer pipeline on 3 fake sources...\n")
    critic_report = critique_sources(analyses, plan.topic)
    print(
        f"Critic finished: {len(critic_report.contradictions)} contradictions, "
        f"{len(critic_report.gaps_identified)} gaps.\n"
    )

    print("Synthesizing final report...\n")
    report = synthesize_report(plan.topic, plan, analyses, critic_report)

    print("=" * 70)
    print(f"TITLE: {report.title}\n")
    print(f"EXECUTIVE SUMMARY ({len(report.executive_summary.split())} words):")
    print(
        report.executive_summary[:500]
        + ("..." if len(report.executive_summary) > 500 else "")
    )
    print()
    print("SECTIONS:")
    print(f"  - Introduction ({len(report.introduction.split())} words)")
    for i, f in enumerate(report.key_findings, 1):
        print(
            f"  - Finding {i}: {f.theme} "
            f"({len(f.content.split())} words, {len(f.supporting_sources)} sources)"
        )
    print(f"  - Contradictions ({len(report.contradictions_section.split())} words)")
    print(f"  - Implications ({len(report.implications.split())} words)")
    print(f"  - Limitations ({len(report.limitations.split())} words)")
    print(f"  - Conclusion ({len(report.conclusion.split())} words)")
    print(f"  - References: {len(report.references)}")
    print()
    print("STUDY GUIDE:")
    print(f"  Key Takeaways ({len(report.study_guide.key_takeaways)}):")
    for t in report.study_guide.key_takeaways[:3]:
        print(f"    • {t[:180]}{'...' if len(t) > 180 else ''}")
    if len(report.study_guide.key_takeaways) > 3:
        print(f"    ... ({len(report.study_guide.key_takeaways) - 3} more)")
    print(
        f"  Critical Thinking Questions ({len(report.study_guide.critical_thinking_questions)}):"
    )
    for q in report.study_guide.critical_thinking_questions:
        print(f"    ? {q[:180]}{'...' if len(q) > 180 else ''}")
    print(f"  Further Reading: {len(report.study_guide.further_reading)} items")
    print()
    print("✓ Synthesizer agent works.")
