"""Planner sub-agent: topic → validated ResearchPlan via Claude tool-use."""

from anthropic import Anthropic

from src.models import ResearchPlan

PLANNER_MODEL = "claude-sonnet-4-6"

PLANNER_SYSTEM_PROMPT = """You are the Planner in a multi-agent research team. Given a research topic, you produce a structured research plan that will guide downstream agents (Searcher, Analyst, Critic, Synthesizer).

Your job:
1. Decompose the topic into 3-8 specific sub-questions that, if answered, would give a thorough understanding.
2. Generate 5-15 DIVERSE search queries covering different angles. You MUST cover at least 4 of these angles: factual background, recent developments, data/evidence/statistics, expert opinion, counterarguments/criticism, controversies/debates, methodology/approaches, real-world applications.
3. Assess topic complexity and set effort_level:
   - 'low' (5-7 sources): narrow factual questions with clear answers
   - 'medium' (8-12 sources): standard research topics with some nuance
   - 'high' (13-18 sources): complex, contested, or rapidly evolving topics
   - 'deep' (19-25 sources): highly controversial, multi-disciplinary, or requires expert-level synthesis
4. Set target_source_count within the range for the chosen effort_level.
5. List the specific angles_covered (min 4) as short labels.

Quality rules:
- Queries should be concrete and searchable — NOT abstract questions. Bad: "What is the philosophy of X?" Good: "X historical origins and key thinkers"
- Queries should be diverse — avoid near-duplicates. Each should target a different angle or sub-question.
- Prefer queries likely to surface primary sources, academic/official data, and expert analysis over SEO content.
- Sub-questions should be open-ended and substantive, not yes/no.

Output strictly matches the ResearchPlan schema. Do not include any text outside the JSON."""


def create_plan(topic: str, client: Anthropic | None = None) -> ResearchPlan:
    """Generate a structured research plan for the given topic.

    Args:
        topic: Raw user research topic (e.g., "Impact of AI on healthcare in 2026").
        client: Optional pre-configured Anthropic client. If None, creates a new one.

    Returns:
        Validated ResearchPlan object.

    Raises:
        ValueError: If the LLM response can't be parsed or fails validation.
    """
    if client is None:
        client = Anthropic()

    schema = ResearchPlan.model_json_schema()

    tool = {
        "name": "submit_research_plan",
        "description": "Submit the structured research plan for the given topic.",
        "input_schema": schema,
    }

    response = client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=2048,
        system=PLANNER_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_research_plan"},
        messages=[
            {
                "role": "user",
                "content": f"Research topic: {topic}\n\nProduce the research plan now.",
            }
        ],
    )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise ValueError(
            f"Planner did not return a tool_use block. Got: {response.content}"
        )

    try:
        plan = ResearchPlan(**tool_use_block.input)
    except Exception as e:
        raise ValueError(
            f"Planner output failed validation: {e}\nRaw: {tool_use_block.input}"
        ) from e

    return plan


if __name__ == "__main__":
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping live test.")
        raise SystemExit(0)

    test_topic = "Impact of AI on healthcare in 2026"
    print(f"Planning for topic: {test_topic!r}\n")
    plan = create_plan(test_topic)

    print(f"Effort level: {plan.effort_level}")
    print(f"Target sources: {plan.target_source_count}")
    print(f"\nSub-questions ({len(plan.sub_questions)}):")
    for q in plan.sub_questions:
        print(f"  - {q}")
    print(f"\nQueries ({len(plan.queries)}):")
    for q in plan.queries:
        print(f"  - {q}")
    print(f"\nAngles covered: {', '.join(plan.angles_covered)}")
    print("\n✓ Planner agent works.")
