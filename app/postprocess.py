"""Post-processing and consistency checks.

Structured Outputs guarantees the model returns JSON with the right shape, but
does not guarantee:
  - confidence values are clamped
  - obvious entities (sender) are present
  - natural-language time phrases are converted to ISO
  - urgency_signals.deadline_text is parsed into reply_by ISO
  - task_proposal.due_at is filled from deadline detection
  - debug metadata is present

This module runs a pure, deterministic postprocess step.
No network calls. No LLM calls.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from dateutil import parser as dateutil_parser
from dateutil import tz as dateutil_tz

from app.config import settings
from app.gmail import ParsedEmail
from app.models import (
    DebugInfo,
    GmailMessageInput,
    LLMTriageOutput,
    MeetingRef,
    PersonRef,
    TriageOutput,
    TriageResponse,
)
from app.prompt import PROMPT_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY_OF_WEEK = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_TZ_ABBREV_TO_IANA = {
    "pt": "America/Los_Angeles", "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
    "mt": "America/Denver", "mst": "America/Denver", "mdt": "America/Denver",
    "ct": "America/Chicago", "cst": "America/Chicago", "cdt": "America/Chicago",
    "et": "America/New_York", "est": "America/New_York", "edt": "America/New_York",
    "utc": "UTC", "gmt": "UTC",
}

_OFFSET_TO_IANA = {
    timedelta(hours=-8): "America/Los_Angeles",
    timedelta(hours=-7): "America/Los_Angeles",
    timedelta(hours=-6): "America/Chicago",
    timedelta(hours=-5): "America/New_York",
    timedelta(hours=0): "UTC",
}

_TIME_RE_12H = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_TIME_RE_24H = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _normalize_sub_action_key(key: str) -> str:
    if not key:
        return "OTHER"
    key = key.strip()
    if not key:
        return "OTHER"
    key = re.sub(r"[^A-Za-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key.upper() if key else "OTHER"


def _subject_to_topic(subject: str) -> str:
    s = (subject or "").strip()
    while True:
        new = re.sub(r"^(\s*(re|fwd|fw)\s*:\s*)", "", s, flags=re.IGNORECASE)
        if new == s:
            break
        s = new
    return s.strip()


def _extract_timezone(text: str, base_dt: datetime) -> Tuple[timezone | datetime.tzinfo, Optional[str]]:
    lower = (text or "").lower()
    for abbr, iana in _TZ_ABBREV_TO_IANA.items():
        if re.search(rf"\b{re.escape(abbr)}\b", lower):
            if iana == "UTC":
                return timezone.utc, "UTC"
            tzinfo = dateutil_tz.gettz(iana)
            return (tzinfo or timezone.utc), iana

    if base_dt.tzinfo is not None:
        off = base_dt.utcoffset()
        if off in _OFFSET_TO_IANA:
            iana = _OFFSET_TO_IANA[off]
            if iana == "UTC":
                return timezone.utc, "UTC"
            tzinfo = dateutil_tz.gettz(iana)
            return (tzinfo or base_dt.tzinfo), iana

    return (base_dt.tzinfo or timezone.utc), None


def _extract_time_components(text: str) -> Tuple[Optional[int], Optional[int]]:
    lower = (text or "").lower()
    if "noon" in lower:
        return 12, 0
    if "midnight" in lower:
        return 0, 0
    if "eod" in lower or "end of day" in lower:
        return 17, 0

    m = _TIME_RE_12H.search(lower)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).lower()
        if ampm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = hour if hour == 12 else hour + 12
        return hour, minute

    m = _TIME_RE_24H.search(lower)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute

    return None, None


def _infer_datetime(text: str, base_dt: datetime) -> Tuple[Optional[datetime], Optional[str]]:
    if not text or not text.strip():
        return None, None

    base = base_dt
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    tzinfo, tz_name = _extract_timezone(text, base)

    try:
        base_local = base.astimezone(tzinfo)
    except Exception:
        base_local = base

    lower = text.strip().lower()

    day_dt: Optional[datetime] = None
    if "today" in lower:
        day_dt = base_local
    elif "tomorrow" in lower:
        day_dt = base_local + timedelta(days=1)

    if day_dt is None:
        for day_name, target_wd in _DAY_OF_WEEK.items():
            if re.search(rf"\b{day_name}\b", lower):
                base_wd = base_local.weekday()
                delta = (target_wd - base_wd) % 7
                if delta == 0:
                    delta = 7
                if re.search(r"\bnext\b", lower):
                    delta += 7
                day_dt = base_local + timedelta(days=delta)
                break

    if day_dt is not None:
        hour, minute = _extract_time_components(lower)
        if hour is not None:
            day_dt = day_dt.replace(hour=hour, minute=minute or 0, second=0, microsecond=0)
        else:
            day_dt = day_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        if day_dt.tzinfo is None:
            try:
                day_dt = day_dt.replace(tzinfo=tzinfo)
            except Exception:
                day_dt = day_dt.replace(tzinfo=timezone.utc)
        return day_dt, tz_name

    try:
        parsed = dateutil_parser.parse(text, default=base_local, fuzzy=True)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tzinfo)
        else:
            parsed = parsed.astimezone(tzinfo)
        return parsed, tz_name
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Entity fillers
# ---------------------------------------------------------------------------

def _fill_sender(parsed_email: ParsedEmail, entities) -> None:
    """Ensure the sender appears in entities.people as role=sender."""
    sender_email = parsed_email.from_email
    if not sender_email:
        return
    for p in entities.people:
        if (p.email or "").lower() == sender_email.lower():
            if not p.role:
                p.role = "sender"
            return
    entities.people.insert(0, PersonRef(email=sender_email, role="sender"))


def _fill_date_isos(parsed_email: ParsedEmail, entities) -> Tuple[Optional[str], Optional[datetime]]:
    base_dt = parsed_email.sent_at or parsed_email.internal_date or datetime.now(timezone.utc)
    best_dt: Optional[datetime] = None
    best_tz_name: Optional[str] = None

    for d in entities.dates:
        if d.iso and str(d.iso).strip():
            if best_dt is None:
                try:
                    best_dt = dateutil_parser.isoparse(d.iso)
                except Exception:
                    pass
            continue

        dt, tz_name = _infer_datetime(d.text, base_dt)
        if not dt:
            continue

        has_time = _extract_time_components(d.text)[0] is not None or bool(_TIME_RE_24H.search(d.text))
        d.iso = dt.isoformat() if has_time else dt.date().isoformat()

        if best_dt is None and has_time:
            best_dt = dt
            best_tz_name = tz_name

    return best_tz_name, best_dt


def _fill_meeting(parsed_email: ParsedEmail, llm_category: str, llm_sub_action: str, entities, tz_name: Optional[str], best_dt: Optional[datetime]) -> None:
    wants_meeting = (
        llm_category == "schedule_and_time"
        or llm_sub_action.upper().startswith("SCHEDULE_")
    )
    if not wants_meeting:
        return

    if entities.meeting is None:
        entities.meeting = MeetingRef()

    if not (entities.meeting.topic or "").strip():
        entities.meeting.topic = _subject_to_topic(parsed_email.subject)

    if not (entities.meeting.start_at or "").strip() and best_dt is not None:
        entities.meeting.start_at = best_dt.isoformat()

    if not (entities.meeting.tz or "").strip() and tz_name:
        entities.meeting.tz = tz_name


def _clean_evidence(evidence_list) -> list:
    """Trim, de-dupe, cap evidence snippets."""
    cleaned = []
    seen = set()
    for e in (evidence_list or []):
        s = (e or "").strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        s = s[:240]
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        cleaned.append(s)
        if len(cleaned) >= 3:
            break
    return cleaned


# ---------------------------------------------------------------------------
# Postprocessor
# ---------------------------------------------------------------------------

def postprocess_triage(
    *,
    msg: GmailMessageInput,
    parsed_email: ParsedEmail,
    llm: LLMTriageOutput,
    predicted_at: datetime,
) -> TriageResponse:
    """Build the API response from the parsed LLM output."""

    # --- Normalize and clamp
    llm.confidence = _clamp01(llm.confidence)
    llm.sub_action_key = _normalize_sub_action_key(llm.sub_action_key)

    # --- Evidence cleanup
    llm.evidence = _clean_evidence(llm.evidence)

    # --- Urgency signals cleanup
    llm.urgency_signals.reason = (llm.urgency_signals.reason or "").strip()
    llm.urgency_signals.urgency = (llm.urgency_signals.urgency or "medium").strip().lower()
    if llm.urgency_signals.urgency not in ("low", "medium", "high", "critical"):
        llm.urgency_signals.urgency = "medium"

    # If deadline_text is set, try to parse it into reply_by ISO
    if llm.urgency_signals.deadline_text and not llm.urgency_signals.reply_by:
        base_dt = parsed_email.sent_at or parsed_email.internal_date or datetime.now(timezone.utc)
        dt, _ = _infer_datetime(llm.urgency_signals.deadline_text, base_dt)
        if dt:
            llm.urgency_signals.reply_by = dt.isoformat()

    # --- Task proposal cleanup
    if llm.task_proposal is not None:
        llm.task_proposal.type = (llm.task_proposal.type or "").strip() or None
        llm.task_proposal.title = (llm.task_proposal.title or "").strip()
        llm.task_proposal.description = (llm.task_proposal.description or "").strip()
        llm.task_proposal.priority = (llm.task_proposal.priority or "medium").strip().lower()
        if llm.task_proposal.priority not in ("low", "medium", "high", "critical"):
            llm.task_proposal.priority = "medium"
        llm.task_proposal.status = "open"

        # Fill due_at from urgency deadline if not already set
        if not llm.task_proposal.due_at and llm.urgency_signals.reply_by:
            llm.task_proposal.due_at = llm.urgency_signals.reply_by

    # --- Recommended actions: normalize kinds and re-rank
    valid_kinds = {"PRIMARY", "SECONDARY", "DANGER"}
    for action in llm.recommended_actions:
        action.kind = action.kind.upper() if action.kind else "SECONDARY"
        if action.kind not in valid_kinds:
            action.kind = "SECONDARY"
        action.key = (action.key or "").strip()
        action.label = (action.label or "").strip()
    # Ensure ranks are sequential
    for i, action in enumerate(sorted(llm.recommended_actions, key=lambda a: a.rank)):
        action.rank = i + 1

    # --- Extracted summary cleanup
    llm.extracted_summary.ask = (llm.extracted_summary.ask or "").strip()
    llm.extracted_summary.success_criteria = (llm.extracted_summary.success_criteria or "").strip()
    llm.extracted_summary.missing_info = [
        s.strip() for s in (llm.extracted_summary.missing_info or []) if (s or "").strip()
    ][:3]

    # --- Fill deterministic entities
    _fill_sender(parsed_email, llm.entities)
    tz_name, best_dt = _fill_date_isos(parsed_email, llm.entities)
    _fill_meeting(
        parsed_email, llm.major_category.value, llm.sub_action_key,
        llm.entities, tz_name, best_dt,
    )

    # --- Build debug metadata (server-side only)
    debug = DebugInfo(
        analysis_timestamp=predicted_at.isoformat(),
        model_version=settings.model_version,
        prompt_version=PROMPT_VERSION,
    )

    # --- Assemble the output
    output = TriageOutput(
        major_category=llm.major_category,
        sub_action_key=llm.sub_action_key,
        explicit_task=bool(llm.explicit_task),
        confidence=llm.confidence,
        suggested_reply_action=llm.suggested_reply_action or [],
        task_proposal=llm.task_proposal,
        recommended_actions=sorted(llm.recommended_actions, key=lambda a: a.rank),
        urgency_signals=llm.urgency_signals,
        extracted_summary=llm.extracted_summary,
        entities=llm.entities,
        evidence=llm.evidence,
        debug=debug,
    )

    return TriageResponse(output=output)
