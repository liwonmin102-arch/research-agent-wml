"""Pydantic models for the multi-agent research pipeline.

These models are the contracts between stages:
Planner → Analyst → Critic → Synthesizer → Judge
"""

import warnings
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# --- Enums ---------------------------------------------------------------


class EffortLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DEEP = "deep"


class ClaimType(str, Enum):
    FACT = "fact"
    STATISTIC = "statistic"
    OPINION = "opinion"
    PREDICTION = "prediction"
    METHODOLOGY = "methodology"


class ClaimConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResolutionStatus(str, Enum):
    UNRESOLVED = "unresolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    RESOLVED = "resolved"


class PoliticalLean(str, Enum):
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    NONE_DETECTED = "none_detected"


# --- Leaf models ---------------------------------------------------------


class Reference(BaseModel):
    """Bibliographic reference to an external source. Used across all stages."""

    model_config = {"extra": "forbid"}

    title: str
    url: str = Field(description="Must start with http:// or https://")
    author: str | None = None
    publication_date: str | None = Field(
        default=None, description="ISO date string of the source's publication"
    )
    access_date: str = Field(
        default_factory=lambda: date.today().isoformat(),
        description="ISO date string when we accessed the source",
    )

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class Claim(BaseModel):
    """A single atomic claim extracted from a source by the Analyst."""

    model_config = {"extra": "forbid"}

    statement: str = Field(description="The claim itself, <= 200 chars")
    claim_type: ClaimType
    confidence: ClaimConfidence
    evidence_quote: str | None = Field(
        default=None, description="Verbatim supporting quote, <= 300 chars"
    )
    source_url: str

    @field_validator("statement")
    @classmethod
    def statement_length(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("statement must be <= 200 chars")
        return v

    @field_validator("evidence_quote")
    @classmethod
    def quote_length(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 300:
            raise ValueError("evidence_quote must be <= 300 chars")
        return v


class BiasAssessment(BaseModel):
    """Potential bias attached to a source by the Analyst."""

    model_config = {"extra": "forbid"}

    political_lean: PoliticalLean
    industry_affiliation: str | None = None
    editorial_stance: str | None = Field(
        default=None, description="One-line description of editorial slant"
    )
    funding_source: str | None = None


# --- Analyst output ------------------------------------------------------


class SourceAnalysis(BaseModel):
    """Per-source deep analysis produced by the Analyst agent. Strict compactness caps."""

    model_config = {"extra": "forbid"}

    reference: Reference
    core_claims: list[Claim] = Field(description="1-7 atomic claims from this source")
    methodology: str | None = Field(
        default=None, description="How the source arrived at its claims, <= 300 chars"
    )
    bias: BiasAssessment
    credibility_score: int = Field(description="0-100; higher = more credible")
    credibility_reasoning: str = Field(description="Why this score, <= 250 chars")
    strengths: list[str] = Field(description="<=3 items, each <=120 chars")
    limitations: list[str] = Field(description="<=3 items, each <=120 chars")
    unique_contribution: str = Field(
        description="What only this source provides, <=200 chars"
    )

    @field_validator("core_claims")
    @classmethod
    def claims_len(cls, v: list[Claim]) -> list[Claim]:
        if not 1 <= len(v) <= 7:
            raise ValueError("core_claims must have 1-7 items")
        return v

    @field_validator("methodology")
    @classmethod
    def methodology_len(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 300:
            raise ValueError("methodology must be <= 300 chars")
        return v

    @field_validator("credibility_score")
    @classmethod
    def credibility_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("credibility_score must be in [0, 100]")
        return v

    @field_validator("credibility_reasoning")
    @classmethod
    def reasoning_len(cls, v: str) -> str:
        if len(v) > 250:
            raise ValueError("credibility_reasoning must be <= 250 chars")
        return v

    @field_validator("strengths", "limitations")
    @classmethod
    def short_bullet_list(cls, v: list[str]) -> list[str]:
        if len(v) > 3:
            raise ValueError("max 3 items")
        for item in v:
            if len(item) > 120:
                raise ValueError("each item must be <= 120 chars")
        return v

    @field_validator("unique_contribution")
    @classmethod
    def unique_len(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("unique_contribution must be <= 200 chars")
        return v


# --- Search wrapper ------------------------------------------------------


class SearchResult(BaseModel):
    """Raw search result wrapping one Tavily response row."""

    model_config = {"extra": "forbid"}

    url: str
    title: str
    content_snippet: str
    published_date: str | None = None
    source_domain: str
    tavily_score: float | None = Field(
        default=None, description="Tavily's own relevance score, 0-1"
    )


# --- Planner output ------------------------------------------------------


class ResearchPlan(BaseModel):
    """Research plan produced by the Planner before any search runs."""

    model_config = {"extra": "forbid"}

    topic: str
    sub_questions: list[str] = Field(description="3-8 decomposed sub-questions")
    queries: list[str] = Field(description="5-15 concrete search queries")
    effort_level: EffortLevel
    target_source_count: int = Field(description="5-25 sources to collect")
    angles_covered: list[str] = Field(
        description=(
            ">=4 coverage angles, e.g. 'background', 'recent developments', "
            "'expert opinion', 'counterarguments', 'data/evidence', 'controversies'"
        )
    )

    @field_validator("sub_questions")
    @classmethod
    def subq_len(cls, v: list[str]) -> list[str]:
        if not 3 <= len(v) <= 8:
            raise ValueError("sub_questions must have 3-8 items")
        return v

    @field_validator("queries")
    @classmethod
    def queries_len(cls, v: list[str]) -> list[str]:
        if not 5 <= len(v) <= 15:
            raise ValueError("queries must have 5-15 items")
        return v

    @field_validator("target_source_count")
    @classmethod
    def target_range(cls, v: int) -> int:
        if not 5 <= v <= 25:
            raise ValueError("target_source_count must be in [5, 25]")
        return v

    @field_validator("angles_covered")
    @classmethod
    def angles_min(cls, v: list[str]) -> list[str]:
        if len(v) < 4:
            raise ValueError("angles_covered must have >= 4 items")
        return v


# --- Critic output -------------------------------------------------------


class Contradiction(BaseModel):
    """A contradiction between sources, surfaced by the Critic."""

    model_config = {"extra": "forbid"}

    topic_of_disagreement: str = Field(description="<=200 chars")
    conflicting_claims: list[Claim] = Field(description=">=2 directly opposing claims")
    possible_reasons: list[str] = Field(description="<=4 items")
    resolution_status: ResolutionStatus
    follow_up_queries: list[str] = Field(description="<=3 clarifying queries")
    resolution_notes: str | None = Field(default=None, description="<=300 chars")

    @field_validator("topic_of_disagreement")
    @classmethod
    def topic_len(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("topic_of_disagreement must be <= 200 chars")
        return v

    @field_validator("conflicting_claims")
    @classmethod
    def min_two_claims(cls, v: list[Claim]) -> list[Claim]:
        if len(v) < 2:
            raise ValueError("conflicting_claims must have >= 2 items")
        return v

    @field_validator("possible_reasons")
    @classmethod
    def reasons_cap(cls, v: list[str]) -> list[str]:
        if len(v) > 4:
            raise ValueError("possible_reasons max 4 items")
        return v

    @field_validator("follow_up_queries")
    @classmethod
    def follow_cap(cls, v: list[str]) -> list[str]:
        if len(v) > 3:
            raise ValueError("follow_up_queries max 3 items")
        return v

    @field_validator("resolution_notes")
    @classmethod
    def notes_len(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 300:
            raise ValueError("resolution_notes must be <= 300 chars")
        return v


class CriticReport(BaseModel):
    """Cross-source critique produced by the Critic agent."""

    model_config = {"extra": "forbid"}

    agreements: list[str] = Field(description="<=5 items, each <=200 chars")
    contradictions: list[Contradiction]
    source_credibility_ranking: list[str] = Field(
        description="URLs ordered most- to least-credible"
    )
    gaps_identified: list[str] = Field(description="<=5 items, each <=200 chars")
    follow_up_queries: list[str] = Field(description="<=5 items")
    overall_confidence: int = Field(description="0-100")

    @field_validator("agreements", "gaps_identified")
    @classmethod
    def capped_string_list(cls, v: list[str]) -> list[str]:
        if len(v) > 5:
            raise ValueError("max 5 items")
        for item in v:
            if len(item) > 200:
                raise ValueError("each item must be <= 200 chars")
        return v

    @field_validator("follow_up_queries")
    @classmethod
    def follow_cap(cls, v: list[str]) -> list[str]:
        if len(v) > 5:
            raise ValueError("follow_up_queries max 5 items")
        return v

    @field_validator("overall_confidence")
    @classmethod
    def confidence_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("overall_confidence must be in [0, 100]")
        return v


# --- Synthesizer output --------------------------------------------------


class StudyGuide(BaseModel):
    """Reader-facing study guide attached to the final report by the Synthesizer."""

    model_config = {"extra": "forbid"}

    key_takeaways: list[str] = Field(description="5-8 items, each <=500 chars")
    critical_thinking_questions: list[str] = Field(description="3-5 items")
    further_reading: list[Reference] = Field(description="2-4 references")

    @field_validator("key_takeaways")
    @classmethod
    def takeaways_shape(cls, v: list[str]) -> list[str]:
        if not 5 <= len(v) <= 8:
            raise ValueError("key_takeaways must have 5-8 items")
        for item in v:
            if len(item) > 500:
                raise ValueError("each key_takeaway must be <= 500 chars")
        return v

    @field_validator("critical_thinking_questions")
    @classmethod
    def ctq_shape(cls, v: list[str]) -> list[str]:
        if not 3 <= len(v) <= 5:
            raise ValueError("critical_thinking_questions must have 3-5 items")
        return v

    @field_validator("further_reading")
    @classmethod
    def fr_shape(cls, v: list[Reference]) -> list[Reference]:
        if not 2 <= len(v) <= 4:
            raise ValueError("further_reading must have 2-4 items")
        return v


class FindingSection(BaseModel):
    """One thematic finding within the DraftReport — prose-heavy, no length cap."""

    model_config = {"extra": "forbid"}

    theme: str
    content: str = Field(description="Synthesized prose, no length cap")
    supporting_sources: list[str] = Field(description="URLs supporting this finding")


class DraftReport(BaseModel):
    """Full synthesized report produced by the Synthesizer agent."""

    model_config = {"extra": "forbid"}

    title: str
    executive_summary: str = Field(description="Target 200-400 words (soft warning)")
    introduction: str
    key_findings: list[FindingSection] = Field(description=">=2 thematic sections")
    contradictions_section: str
    implications: str
    limitations: str
    conclusion: str
    references: list[Reference]
    study_guide: StudyGuide

    @field_validator("executive_summary")
    @classmethod
    def summary_word_count(cls, v: str) -> str:
        n = len(v.split())
        if not 200 <= n <= 400:
            warnings.warn(
                f"executive_summary has {n} words; target is 200-400 (not failing)",
                stacklevel=2,
            )
        return v

    @field_validator("key_findings")
    @classmethod
    def findings_min(cls, v: list[FindingSection]) -> list[FindingSection]:
        if len(v) < 2:
            raise ValueError("key_findings must have >= 2 sections")
        return v


# --- Judge output --------------------------------------------------------


class JudgeScore(BaseModel):
    """Final quality assessment produced by the Judge agent."""

    model_config = {"extra": "forbid"}

    accuracy: int
    balance: int
    clarity: int
    educational_value: int
    overall: int
    specific_improvements: list[str] = Field(description="<=6 items")
    needs_revision: bool

    @field_validator("accuracy", "balance", "clarity", "educational_value", "overall")
    @classmethod
    def score_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("score must be in [0, 100]")
        return v

    @field_validator("specific_improvements")
    @classmethod
    def improvements_cap(cls, v: list[str]) -> list[str]:
        if len(v) > 6:
            raise ValueError("specific_improvements max 6 items")
        return v


# --- Orchestrator state --------------------------------------------------


class ResearchSession(BaseModel):
    """Orchestrator state carried through the full pipeline, mutated across iterations."""

    model_config = {"extra": "forbid"}

    topic: str
    plan: ResearchPlan | None = None
    search_results: list[SearchResult] = Field(default_factory=list)
    source_analyses: list[SourceAnalysis] = Field(default_factory=list)
    critic_report: CriticReport | None = None
    draft_report: DraftReport | None = None
    judge_score: JudgeScore | None = None
    iteration_count: int = 0
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None


# --- Smoke test ----------------------------------------------------------


if __name__ == "__main__":
    ref = Reference(title="Dummy Source", url="https://example.com/article")
    claim = Claim(
        statement="Sample factual claim.",
        claim_type=ClaimType.FACT,
        confidence=ClaimConfidence.HIGH,
        source_url="https://example.com/article",
    )
    bias = BiasAssessment(political_lean=PoliticalLean.NONE_DETECTED)
    analysis = SourceAnalysis(
        reference=ref,
        core_claims=[claim],
        bias=bias,
        credibility_score=82,
        credibility_reasoning="Peer-reviewed publication with transparent methods.",
        strengths=["Clear methodology"],
        limitations=["Small sample size"],
        unique_contribution="Only longitudinal dataset covering 2010-2024.",
    )
    search = SearchResult(
        url="https://example.com/article",
        title="Dummy Source",
        content_snippet="Snippet…",
        source_domain="example.com",
    )
    plan = ResearchPlan(
        topic="Dummy topic",
        sub_questions=["What?", "Why?", "How?"],
        queries=["q1", "q2", "q3", "q4", "q5"],
        effort_level=EffortLevel.MEDIUM,
        target_source_count=10,
        angles_covered=[
            "background",
            "recent developments",
            "expert opinion",
            "counterarguments",
        ],
    )
    contradiction = Contradiction(
        topic_of_disagreement="Whether X causes Y",
        conflicting_claims=[claim, claim],
        possible_reasons=["Different definitions of Y"],
        resolution_status=ResolutionStatus.UNRESOLVED,
        follow_up_queries=["X causation mechanism"],
    )
    critic = CriticReport(
        agreements=["Most sources agree on baseline trend"],
        contradictions=[contradiction],
        source_credibility_ranking=["https://example.com/article"],
        gaps_identified=["Missing 2025 data"],
        follow_up_queries=["2025 X trend data"],
        overall_confidence=72,
    )
    guide = StudyGuide(
        key_takeaways=[
            "Takeaway one.",
            "Takeaway two.",
            "Takeaway three.",
            "Takeaway four.",
            "Takeaway five.",
        ],
        critical_thinking_questions=[
            "Who benefits from this framing?",
            "What counterevidence exists?",
            "How would this change by 2030?",
        ],
        further_reading=[ref, ref],
    )
    finding = FindingSection(
        theme="Trend",
        content="Synthesized prose content about the trend.",
        supporting_sources=["https://example.com/article"],
    )
    # 300-word executive summary stays inside the 200-400 soft target.
    exec_summary = " ".join(["word"] * 300)
    draft = DraftReport(
        title="Dummy Report",
        executive_summary=exec_summary,
        introduction="Intro prose.",
        key_findings=[finding, finding],
        contradictions_section="Contradictions prose.",
        implications="Implications prose.",
        limitations="Limitations prose.",
        conclusion="Conclusion prose.",
        references=[ref],
        study_guide=guide,
    )
    judge = JudgeScore(
        accuracy=85,
        balance=80,
        clarity=90,
        educational_value=78,
        overall=83,
        specific_improvements=["Add one more counterargument"],
        needs_revision=False,
    )
    session = ResearchSession(
        topic="Dummy topic",
        plan=plan,
        search_results=[search],
        source_analyses=[analysis],
        critic_report=critic,
        draft_report=draft,
        judge_score=judge,
    )

    print("All models validate ✓")
