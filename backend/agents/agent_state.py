from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime, timezone
from uuid import uuid4


# ──────────────────────────────────────────────
# ATTEMPT HISTORY
# ──────────────────────────────────────────────

class AttemptEntry(BaseModel):
    attempt_number: int = Field(..., ge=1)
    supervisor_feedback: Optional[str] = Field(default=None)
    result: Literal["pending", "approved", "rejected"] = Field(default="pending")
    submitted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Attempts(BaseModel):
    count: int = Field(default=0, ge=0, le=5)
    history: List[AttemptEntry] = Field(default_factory=list)

    @field_validator("count")
    @classmethod
    def count_must_not_exceed_max(cls, v):
        if v > 5:
            raise ValueError("Max retry limit of 5 reached")
        return v


# ──────────────────────────────────────────────
# SUCCESS CRITERIA
# ──────────────────────────────────────────────

class SuccessCriteria(BaseModel):
    must_cover: List[str] = Field(..., min_length=1)
    must_not: str = Field(...)


# ──────────────────────────────────────────────
# CHILD TASK
# ──────────────────────────────────────────────

class ChildTask(BaseModel):
    child_task_id: str = Field(default_factory=lambda: str(uuid4()))
    sub_agent_id: Optional[str] = Field(default=None)
    task: str = Field(...)
    context: str = Field(...)
    success_criteria: SuccessCriteria = Field(...)
    status: Literal["pending", "in_progress", "submitted", "approved", "rejected", "done"] = Field(default="pending")
    attempts: Attempts = Field(default_factory=Attempts)
    final_output: Optional[str] = Field(default=None)


# ──────────────────────────────────────────────
# QUERY ANALYZER OUTPUT
# ──────────────────────────────────────────────

class QueryAnalyzerOutput(BaseModel):
    is_question: bool = Field(default=False)
    is_research_able: bool = Field(default=False)
    agent_type: Literal["simple_agent", "multi_agent"] = Field(default="simple_agent")


# ──────────────────────────────────────────────
# SUPERVISOR STATE  (single source of truth)
# ──────────────────────────────────────────────

class SupervisorState(BaseModel):
    # ── entry point ──────────────────────────
    parent_question: str = Field(..., min_length=5)

    # ── set by query_analyzer node ───────────
    agent_type: Literal["simple_agent", "multi_agent"] = Field(default="simple_agent")
    is_question: bool = Field(default=False)
    is_research_able: bool = Field(default=False)

    # ── set by supervisor_agent node ─────────
    state: Literal["decompose", "assign", "waiting", "review", "synthesize", "validate", "done"] = Field(default="decompose")
    child_tasks: List[ChildTask] = Field(default_factory=list)
    n_agents: int = Field(default=0)

    # ── set by worker / review nodes ─────────
    review_queue: List[str] = Field(default_factory=list)
    approved_outputs: List[str] = Field(default_factory=list)
    rejected_outputs: List[str] = Field(default_factory=list)

    # ── final output ─────────────────────────
    final_report: Optional[str] = Field(default=None)

    @field_validator("parent_question")
    @classmethod
    def question_must_be_meaningful(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Parent question too short")
        return v.strip()