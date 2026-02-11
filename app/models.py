"""Pydantic schemas (contracts) for the Drakko Email AI server.

Schema layers

A) Provider input schema
   Accepts the raw Gmail "Message" JSON payload (same as Gmail API returns).

B) LLM output schema (Structured Outputs)
   Forces the model to return JSON matching a Pydantic schema.
   Does NOT include provider IDs or debug metadata — those are injected server-side.

C) API response schema
   What the FastAPI endpoint returns, wrapped in an `output` envelope:
     { "output": { ...all triage fields... } }

"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Gmail provider input schema (mirrors the Gmail "Message" resource)
# ---------------------------------------------------------------------------

class GmailHeader(BaseModel):
    """A single header entry from Gmail's message payload."""
    name: str
    value: str


class GmailBody(BaseModel):
    """MIME part in GMails message payload"""
    model_config = ConfigDict(extra="allow")
    size: int = 0
    data: Optional[str] = None
    attachmentId: Optional[str] = None


class GmailPart(BaseModel):
    """A MIME part in Gmail's message payload."""
    model_config = ConfigDict(extra="allow")

    partId: str = ""
    mimeType: str
    filename: Optional[str] = None
    headers: List[GmailHeader] = Field(default_factory=list)
    body: GmailBody = Field(default_factory=GmailBody)
    parts: Optional[List["GmailPart"]] = None


class GmailPayload(GmailPart):
    """Alias for the top-level Gmail payload (same shape as a part)."""
    pass


class GmailMessageInput(BaseModel):
    """Raw Gmail message JSON object."""
    model_config = ConfigDict(extra="allow")

    provider: str = "gmail"
    id: str
    threadId: Optional[str] = None
    labelIds: Optional[List[str]] = None
    snippet: Optional[str] = None
    historyId: Optional[str] = None
    internalDate: Optional[str] = None
    payload: GmailPayload
    sizeEstimate: Optional[int] = None


# ---------------------------------------------------------------------------
# Output taxonomy
# ---------------------------------------------------------------------------

class MajorCategory(str, Enum):
    """High-level semantic bucket."""
    core_communication = "core_communication"
    decisions_and_approvals = "decisions_and_approvals"
    schedule_and_time = "schedule_and_time"
    documents_and_review = "documents_and_review"
    financial_and_admin = "financial_and_admin"
    people_and_process = "people_and_process"
    information_and_org = "information_and_org"
    learning_and_awareness = "learning_and_awareness"
    social_and_people = "social_and_people"
    meta_and_systems = "meta_and_systems"
    other = "other"


# ---------------------------------------------------------------------------
# Entity schemas
# ---------------------------------------------------------------------------

class PersonRef(BaseModel):
    email: str
    role: str

class DateRef(BaseModel):
    text: str
    iso: Optional[str] = None
    type: str = "other"

class MoneyRef(BaseModel):
    text: str
    amount: Optional[float] = None
    currency: Optional[str] = None

class DocRef(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None

class MeetingRef(BaseModel):
    topic: Optional[str] = None
    start_at: Optional[str] = None
    tz: Optional[str] = None

class Entities(BaseModel):
    people: List[PersonRef] = Field(default_factory=list)
    dates: List[DateRef] = Field(default_factory=list)
    money: List[MoneyRef] = Field(default_factory=list)
    docs: List[DocRef] = Field(default_factory=list)
    meeting: Optional[MeetingRef] = None


# ---------------------------------------------------------------------------
# Triage output schemas
# ---------------------------------------------------------------------------

class TaskProposal(BaseModel):
    """A fully-formed task object ready for a task manager."""
    type: Optional[str] = None
    title: str = ""
    description: str = ""
    priority: str = "medium"
    status: str = "open"
    scheduled_for: Optional[str] = None
    due_at: Optional[str] = None
    waiting_on: Optional[str] = None


class RecommendedAction(BaseModel):
    """A ranked UI action button."""
    key: str
    label: str
    kind: str = "SECONDARY"
    rank: int = 1


class UrgencySignals(BaseModel):
    """Structured urgency assessment."""
    urgency: str = "medium"
    deadline_detected: bool = False
    deadline_text: Optional[str] = None
    reply_by: Optional[str] = None
    reason: str = ""


class ExtractedSummary(BaseModel):
    """Executive-assistant-style summary of what the email needs."""
    ask: str = ""
    success_criteria: str = ""
    missing_info: List[str] = Field(default_factory=list)


class DebugInfo(BaseModel):
    """Observability metadata injected server-side."""
    analysis_timestamp: str = ""
    model_version: str = ""
    prompt_version: str = ""


# ---------------------------------------------------------------------------
# LLM output schema (Structured Outputs) — what the model returns
# ---------------------------------------------------------------------------

class LLMTriageOutput(BaseModel):
    """The model's decision output.

    Does NOT include debug metadata — that is injected server-side in postprocessing.
    """
    major_category: MajorCategory
    sub_action_key: str

    explicit_task: bool
    confidence: float = Field(ge=0.0, le=1.0)

    suggested_reply_action: List[str] = Field(default_factory=list)

    task_proposal: Optional[TaskProposal] = None

    recommended_actions: List[RecommendedAction] = Field(default_factory=list)

    urgency_signals: UrgencySignals = Field(default_factory=UrgencySignals)

    extracted_summary: ExtractedSummary = Field(default_factory=ExtractedSummary)

    entities: Entities = Field(default_factory=Entities)

    evidence: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API response schema — what the endpoint returns
# ---------------------------------------------------------------------------

class TriageOutput(BaseModel):
    """The inner output object containing all triage fields + debug."""
    major_category: MajorCategory
    sub_action_key: str

    explicit_task: bool
    confidence: float = Field(ge=0.0, le=1.0)

    suggested_reply_action: List[str] = Field(default_factory=list)

    task_proposal: Optional[TaskProposal] = None

    recommended_actions: List[RecommendedAction] = Field(default_factory=list)

    urgency_signals: UrgencySignals = Field(default_factory=UrgencySignals)

    extracted_summary: ExtractedSummary = Field(default_factory=ExtractedSummary)

    entities: Entities = Field(default_factory=Entities)

    evidence: List[str] = Field(default_factory=list)

    debug: DebugInfo = Field(default_factory=DebugInfo)


class TriageResponse(BaseModel):
    """Top-level API response envelope: { "output": { ... } }"""
    output: TriageOutput


# Fix forward refs for nested MIME parts.
GmailPart.model_rebuild()
GmailPayload.model_rebuild()
