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
# SUCCESS CRITERIA
# ──────────────────────────────────────────────

class SuccessCriteria(BaseModel):
    must_cover: List[str] = Field(..., min_length=1)
    must_not: str = Field(...)


# ──────────────────────────────────────────────
# ATTEMPTS
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


# ──────────────────────────────────────────────
# SUPERVISOR STATE  (single source of truth)
# ──────────────────────────────────────────────

class SupervisorState(TypedDict, total=False):
    supervisor_id: str
    parent_question: str
    is_question: bool
    is_research_able: bool
    agent_type: Literal["simple_agent", "multi_agent"]
    child_tasks: List[ChildTask]
    state: Literal["decompose", "assign", "waiting", "review", "synthesize", "validate", "done"]
    n_agents: int
    final_report: str