from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, TypedDict, Annotated
from datetime import datetime, timezone
import operator

# ──────────────────────────────────────────────
# QUERY ANALYZER OUTPUT
# ──────────────────────────────────────────────

class QueryAnalyzerOutput(BaseModel):
    is_question: bool = Field(default=False)
    is_research_able: bool = Field(default=False)
    agent_type: Literal["simple_agent", "multi_agent"] = Field(default="simple_agent")


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
# ReviewOutput STATE
# ──────────────────────────────────────────────

class ReviewOutput(BaseModel):
    result: Literal["approved", "rejected"]
    feedback: str


# ──────────────────────────────────────────────
# CHILD TASK
# ──────────────────────────────────────────────

class ChildTask(BaseModel):
    child_task_id: str
    sub_agent_id: str
    task: str = Field(...)
    context: str = Field(...)
    success_criteria: SuccessCriteria = Field(...)
    status: Literal["pending", "in_progress", "submitted", "approved", "rejected", "done"] = Field(default="pending")
    attempts: Attempts = Field(default_factory=Attempts)
    final_output: Optional[str] = Field(default=None)


class ChildTaskDraft(BaseModel):
    task: str = Field(...)
    context: str = Field(...)
    success_criteria: SuccessCriteria = Field(...)


class WorkerState(TypedDict):
    child_task: ChildTask


# ──────────────────────────────────────────────
# SUPERVISOR STATE  (single source of truth)
# ──────────────────────────────────────────────

class SupervisorState(TypedDict, total=False):
    supervisor_id: str
    child_task_id: List[str]
    sub_agent_id: List[str]
    parent_question: str
    agent_type: Literal["simple_agent", "multi_agent"]
    is_question: bool
    is_research_able: bool
    state: Literal["decompose", "assign", "waiting", "review", "synthesize", "validate", "done"]
    child_tasks: List[ChildTask]
    n_agents: int
    review_queue: Annotated[List[str], operator.add]
    reviewed_count: int
    approved_outputs: Annotated[List[str], operator.add]
    rejected_outputs: Annotated[List[str], operator.add]
    final_report: Optional[str]
    summarized: Optional[str]

