from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    name: str = Field(default="Patient", min_length=1, max_length=80)
    age: int = Field(ge=0, le=130)
    gender: str = Field(min_length=1, max_length=32)
    duration: str = Field(min_length=1, max_length=128)
    conditions: str = Field(default="")
    severity: int = Field(ge=0, le=10)
    symptoms: str = Field(min_length=1, max_length=2000)


class LLMTriageOutput(BaseModel):
    urgency: Literal["emergency", "urgent", "routine"]
    department: str
    confidence: float = Field(ge=0, le=1)
    summary: str
    red_flags: list[str]


class TriageResponse(LLMTriageOutput):
    rule_triggered: str | None = None


class TriageLog(BaseModel):
    input_json: str
    llm_output: str
    final_output: str
    rules_triggered: str
    timestamp: datetime


class IntakeState(BaseModel):
    name: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    gender: str | None = None
    duration: str | None = None
    conditions: str | None = ""
    severity: int | None = Field(default=None, ge=0, le=10)
    symptoms: str | None = None


class IntakeTurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    state: IntakeState = Field(default_factory=IntakeState)


class IntakeTurnResponse(BaseModel):
    state: IntakeState
    missing_fields: list[str]
    ready_for_triage: bool
    assistant_reply: str
