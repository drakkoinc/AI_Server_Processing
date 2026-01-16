"""Pydantic schemas (contracts) for the Drakko Email AI server.

This repo treats the LLM as a **decision engine** (per your PDFs):
  1) What is this email about?  (major category)
  2) What should the user do?   (sub-action / verb)
  3) How urgent/important is it? (confidence + urgency-ish signals)
  4) What evidence/entities support that decision? (extracted, observable cues)

This file contains *three* conceptual layers:

A) Provider input schema
------------------------
We accept the raw Gmail "Message" JSON payload (the structure returned by Gmail's API)
so you can store the raw payload in your DB and re-run inference later.

B) LLM output schema (Structured Outputs)
----------------------------------------
We use OpenAI Structured Outputs to force the model to return JSON matching a schema.
The LLM output schema intentionally *does not* include provider IDs (message/thread IDs),
so that the server remains the source of truth for:
  - ID consistency
  - future metadata injection (timestamps, reference versions, etc.)

C) API response schema
----------------------
This is what your application receives.

IMPORTANT:
- The user requested a specific response JSON shape with snake_case keys.
  That response shape is implemented by `EmailTriageResponse`.
- If you want to version contracts, add a `reference` field here and bump it when the
  schema changes. (We removed it from the response because the requested structure
  did not include it.)

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
    """Body container for a Gmail message part.

    Gmail uses base64url-encoded `data` for inline part bodies.
    For attachments, Gmail may provide `attachmentId`.

    We allow extra keys because Gmail's schema is wider than what we need.
    """

    model_config = ConfigDict(extra="allow")

    size: int = 0
    data: Optional[str] = None
    attachmentId: Optional[str] = None


class GmailPart(BaseModel):
    """A MIME part in Gmail's message payload.

    Parts can be nested (multipart/*). A single Gmail message payload is represented as the
    top-level part.
    """

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
    """Raw Gmail message JSON object.

    This matches the structure Gmail returns (with minimal fields required).

    Notes:
    - Your input example does not include a provider field; we default to "gmail".
    - `internalDate` is milliseconds since epoch, as a string.
    """

    model_config = ConfigDict(extra="allow")

    provider: str = "gmail"

    # Gmail ids
    id: str
    threadId: Optional[str] = None

    # Common Gmail metadata (optional)
    labelIds: Optional[List[str]] = None
    snippet: Optional[str] = None
    historyId: Optional[str] = None
    internalDate: Optional[str] = None  # ms since epoch

    payload: GmailPayload

    sizeEstimate: Optional[int] = None


# ---------------------------------------------------------------------------
# Output taxonomy
# ---------------------------------------------------------------------------

class MajorCategory(str, Enum):
    """High-level semantic bucket (from your Deep Classification breakdown)."""

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


# We intentionally keep `sub_action_key` as a STRING (not an Enum).
# Why?
# - In early product iterations, you will iterate on these keys rapidly.
# - Enforcing a hard Enum inside Structured Outputs can cause avoidable failures
#   (model produces a new key you haven't whitelisted yet).
#
# We instead:
# - document recommended keys in the prompt
# - allow arbitrary strings at runtime
# - optionally validate/normalize them in postprocessing if you want


# ---------------------------------------------------------------------------
# Entities schema for the requested response object
# ---------------------------------------------------------------------------

class PersonRef(BaseModel):
    """A lightweight person reference.

    The user-provided desired output only includes:
      - email
      - role

    We keep it intentionally minimal to avoid UI/schema bloat.
    """

    email: str
    role: str  # e.g. "sender", "recipient", "mentioned"


class DateRef(BaseModel):
    """A lightweight date/time mention."""

    text: str
    iso: Optional[str] = None
    type: str = "other"  # e.g. "meeting_time", "deadline", "event_time"


class MoneyRef(BaseModel):
    """Money mention. Optional details for downstream automation."""

    text: str
    amount: Optional[float] = None
    currency: Optional[str] = None


class DocRef(BaseModel):
    """Document/artifact mention."""

    title: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None


class MeetingRef(BaseModel):
    """Meeting-specific structured fields.

    Present when the email is about scheduling/confirming a meeting.
    """

    topic: Optional[str] = None
    start_at: Optional[str] = None  # ISO datetime with timezone offset
    tz: Optional[str] = None  # IANA timezone preferred (e.g., "America/Los_Angeles")


class Entities(BaseModel):
    """All extracted entities used to support UI and future automations."""

    people: List[PersonRef] = Field(default_factory=list)
    dates: List[DateRef] = Field(default_factory=list)
    money: List[MoneyRef] = Field(default_factory=list)
    docs: List[DocRef] = Field(default_factory=list)
    meeting: Optional[MeetingRef] = None


# ---------------------------------------------------------------------------
# LLM output schema (Structured Outputs)
# ---------------------------------------------------------------------------

class LLMTriageOutput(BaseModel):
    """The model's decision output (provider IDs are injected server-side)."""

    major_category: MajorCategory
    sub_action_key: str

    reply_required: bool
    explicit_task: bool
    task_type: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)

    reason: str
    evidence: List[str] = Field(default_factory=list)

    entities: Entities = Field(default_factory=Entities)


# ---------------------------------------------------------------------------
# API response schema (what the FastAPI endpoint returns)
# ---------------------------------------------------------------------------

class EmailTriageResponse(BaseModel):
    """Final response matching the user-requested JSON structure.

    Example (shape):
    {
      "message_id": "...",
      "thread_id": "...",
      "major_category": "schedule_and_time",
      "sub_action_key": "SCHEDULE_CONFIRM_TIME",
      "reply_required": true,
      "explicit_task": false,
      "task_type": null,
      "confidence": 0.87,
      "reason": "Sender asks you to confirm the meeting time.",
      "evidence": ["Can you confirm Friday at 2pm PT works?"],
      "entities": { ... }
    }

    NOTE:
    - We keep snake_case field names so the JSON matches exactly.
    """

    message_id: str
    thread_id: Optional[str] = None

    major_category: MajorCategory
    sub_action_key: str

    reply_required: bool
    explicit_task: bool
    task_type: Optional[str] = None

    confidence: float = Field(ge=0.0, le=1.0)

    reason: str
    evidence: List[str] = Field(default_factory=list)

    entities: Entities = Field(default_factory=Entities)


# Fix forward refs for nested MIME parts.
GmailPart.model_rebuild()
GmailPayload.model_rebuild()
