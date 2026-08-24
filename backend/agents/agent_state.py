from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal
from datetime import datetime, timezone
from __future__ import annotations
from uuid import uuid4


# ──────────────────────────────────────────────
# ATTEMPTS
# ──────────────────────────────────────────────

class AttemptEntry(BaseModel):
    """Single attempt record for a child task."""

    attempt_number: int = Field(
        ...,
        ge=1,
        description="Attempt number starting from 1"
    )
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when this attempt started"
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when this attempt completed"
    )
    output: Optional[dict] = Field(
        default=None,
        description="Raw output produced by sub-agent in this attempt"
    )
    supervisor_feedback: Optional[str] = Field(
        default=None,
        description="Specific feedback from supervisor on rejection — null if approved"
    )
    result: Literal["pending", "approved", "rejected"] = Field(
        default="pending",
        description="Result of this attempt after supervisor review"
    )


class Attempts(BaseModel):
    """Tracks all attempts made by a sub-agent on a child task."""

    count: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Total number of attempts made. Max 5 to prevent infinite loops"
    )
    history: List[AttemptEntry] = Field(
        default_factory=list,
        description="Full history of every attempt with feedback and results"
    )

    @field_validator("count")
    @classmethod
    def count_must_not_exceed_max(cls, v):
        if v > 5:
            raise ValueError("Attempt count cannot exceed 5 — max retry limit reached")
        return v


# ──────────────────────────────────────────────
# SUCCESS CRITERIA
# ──────────────────────────────────────────────

class SuccessCriteria(BaseModel):
    """Defines what a good sub-agent output must contain."""

    must_cover: List[str] = Field(
        ...,
        min_length=1,
        description="Specific points the sub-agent must address — at least one required"
    )
    minimum_depth: str = Field(
        ...,
        min_length=10,
        description="What minimum depth of research looks like — must be specific"
    )
    must_acknowledge: str = Field(
        ...,
        description="Limitations, gaps, or debates the agent must not ignore"
    )
    must_not: str = Field(
        ...,
        description="What the agent must avoid — e.g. surface level explanation only"
    )


# ──────────────────────────────────────────────
# CHILD TASK
# ──────────────────────────────────────────────

class ChildTask(BaseModel):
    """A single decomposed sub-task assigned to one sub-agent."""

    child_task_id: str = Field(
        ...,
        default_factory=lambda: str(uuid4()),
        description="Unique UUID for this child task — auto generated"
    )
    sub_agent_id: Optional[str] = Field(
        ...,
        default=None,
        description="UUID of the sub-agent assigned to this task — null until assigned"
    )
    scope: str = Field(
        ...,
        min_length=10,
        description="Exactly what this sub-agent must research — no overlap with others"
    )
    context: str = Field(
        ...,
        min_length=10,
        description="Why this dimension matters to the parent question"
    )
    task: str = Field(
        ...,
        min_length=10,
        description="Full detailed description of the research task"
    )
    success_criteria: SuccessCriteria = Field(
        ...,
        description="Specific measurable criteria supervisor uses to approve or reject"
    )
    status: Literal[
        "pending",
        "in_progress",
        "submitted",
        "approved",
        "rejected",
        "retry",
        "done"
    ] = Field(
        default="pending",
        description="Current status of this child task in the pipeline"
    )
    attempts: Attempts = Field(
        default_factory=Attempts,
        description="Full attempt history including feedback and results"
    )
    final_output: Optional[dict] = Field(
        default=None,
        description="Final approved output from sub-agent — null until approved"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when this child task was created by supervisor"
    )

    @field_validator("sub_agent_id")
    @classmethod
    def validate_sub_agent_id(cls, v):
        if v is not None and len(v) < 5:
            raise ValueError("sub_agent_id must be a valid UUID string")
        return v

    @model_validator(mode="after")
    def final_output_only_when_done(self) -> ChildTask:
        if self.final_output is not None and self.status not in ("approved", "done"):
            raise ValueError(
                "final_output can only be set when status is 'approved' or 'done'"
            )
        return self


# ──────────────────────────────────────────────
# SUB-AGENT OUTPUT (goes into review queue)
# ──────────────────────────────────────────────

class FindingSection(BaseModel):
    """One research finding on a specific topic."""

    topic: str = Field(
        ...,
        description="The specific topic this finding addresses"
    )
    finding: str = Field(
        ...,
        min_length=50,
        description="Detailed research finding — must be substantive not surface level"
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score 0-100 for this specific finding"
    )
    evidence: str = Field(
        ...,
        description="Specific papers, models, or sources that back this finding"
    )


class SubAgentOutput(BaseModel):
    """Complete output produced by a sub-agent after research."""

    agent_id: str = Field(
        ...,
        description="UUID of the sub-agent that produced this output"
    )
    child_task_id: str = Field(
        ...,
        description="UUID of the child task this output addresses"
    )
    parent_task_id: str = Field(
        ...,
        description="UUID of the parent task — for traceability"
    )
    attempt_number: int = Field(
        ...,
        ge=1,
        description="Which attempt produced this output"
    )
    submitted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when this output was submitted to review queue"
    )
    status: Literal["submitted", "approved", "rejected"] = Field(
        default="submitted",
        description="Status of this output in the review pipeline"
    )
    findings: List[FindingSection] = Field(
        ...,
        min_length=1,
        description="Structured research findings — at least one required"
    )
    sources: List[str] = Field(
        ...,
        min_length=1,
        description="All sources used — papers, reports, articles"
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall confidence score for the entire output 0-100"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="What could not be verified or found — honest gaps"
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Conflicting evidence or disagreements found during research"
    )
    self_check: dict = Field(
        default_factory=dict,
        description="Agent's self evaluation against success criteria before submitting"
    )

    @field_validator("confidence_score")
    @classmethod
    def confidence_must_be_realistic(cls, v):
        if v == 100:
            raise ValueError(
                "Confidence score of 100 is not allowed — "
                "no research finding is ever 100% certain"
            )
        return v


# ──────────────────────────────────────────────
# MASTER SUPERVISOR STATE
# ──────────────────────────────────────────────

class SupervisorState(BaseModel):
    """Complete state of the supervisor agent throughout the entire pipeline."""

    supervisor_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique UUID for this supervisor instance"
    )
    parent_task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique UUID for the parent research task"
    )
    parent_question: str = Field(
        ...,
        min_length=20,
        description="The original raw research question from the user"
    )
    state: Literal[
        "decompose",
        "assign",
        "waiting",
        "review",
        "synthesize",
        "validate",
        "done"
    ] = Field(
        default="decompose",
        description="Current state of the supervisor in the pipeline"
    )
    child_tasks: List[ChildTask] = Field(
        default_factory=list,
        description="All decomposed child tasks — populated after decompose state"
    )
    n_sub_agents: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of sub-agents launched — max 10"
    )
    review_queue: List[SubAgentOutput] = Field(
        default_factory=list,
        description="Queue of submitted sub-agent outputs waiting for supervisor review"
    )
    approved_outputs: List[SubAgentOutput] = Field(
        default_factory=list,
        description="All supervisor approved sub-agent outputs"
    )
    rejected_outputs: List[SubAgentOutput] = Field(
        default_factory=list,
        description="All rejected outputs with supervisor feedback — for retry memory"
    )
    final_report: Optional[str] = Field(
        default=None,
        description="Final synthesized research report — null until synthesize state"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when this supervisor session was created"
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when pipeline completed — null until done"
    )

    @field_validator("parent_question")
    @classmethod
    def question_must_be_a_question(cls, v):
        if len(v.strip()) < 20:
            raise ValueError("Parent question is too short — must be meaningful")
        return v.strip()

    @model_validator(mode="after")
    def n_sub_agents_matches_child_tasks(self) -> SupervisorState:
        if self.child_tasks and self.n_sub_agents != len(self.child_tasks):
            raise ValueError(
                f"n_sub_agents ({self.n_sub_agents}) must match "
                f"number of child_tasks ({len(self.child_tasks)})"
            )
        return self

    @model_validator(mode="after")
    def final_report_only_in_late_states(self) -> SupervisorState:
        if self.final_report is not None and self.state not in ("validate", "done"):
            raise ValueError(
                "final_report can only exist in 'validate' or 'done' state"
            )
        return self