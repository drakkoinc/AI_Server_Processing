"""Post-processing and consistency checks.

Why post-process at all if we use Structured Outputs?
----------------------------------------------------
Structured Outputs guarantees the model returns JSON with the *right shape*, but it does not
guarantee:
  - that IDs match the request (models can hallucinate IDs)
  - that confidence values are in a sane range (even with schema constraints, we clamp)
  - that obvious entities (sender) are always present
  - that natural-language time phrases are converted into ISO timestamps consistently

So we run a **pure, deterministic** postprocess step that:
  1) injects `message_id` + `thread_id` from the request
  2) clamps numeric fields
  3) fills missing obvious entities
  4) best-effort parses relative times ("Friday 2pm PT") into ISO datetimes

IMPORTANT:
- This module must remain deterministic and side-effect free.
- No network calls.
- No LLM calls.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from dateutil import parser as dateutil_parser
from dateutil import tz as dateutil_tz

from app.gmail import ParsedEmail
from app.models import (
    EmailTriageResponse,
    GmailMessageInput,
    LLMTriageOutput,
    MeetingRef,
    PersonRef,
)


_DAY_OF_WEEK = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Common timezone abbreviations → IANA.
# NOTE: abbreviations are ambiguous in general; we only handle the common ones
# that show up in scheduling emails.
_TZ_ABBREV_TO_IANA = {
    "pt": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "mt": "America/Denver",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "ct": "America/Chicago",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "et": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "utc": "UTC",
    "gmt": "UTC",
}

# Fallback mapping from UTC offsets to a "reasonable" US timezone.
# This is heuristic (offsets are not unique), but it improves ISO rendering
# for common US-based founders/teams when the email date header includes only
# a numeric offset.
_OFFSET_TO_IANA = {
    timedelta(hours=-8): "America/Los_Angeles",
    timedelta(hours=-7): "America/Los_Angeles",
    timedelta(hours=-6): "America/Chicago",
    timedelta(hours=-5): "America/New_York",
    timedelta(hours=0): "UTC",
}


def _clamp01(x: float) -> float:
    """Clamp a numeric value to [0, 1]."""
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _normalize_sub_action_key(key: str) -> str:
    """Normalize a free-form sub_action_key into SCREAMING_SNAKE_CASE.

    We keep sub_action_key as a string so you can iterate on the taxonomy quickly.
    This normalization step improves downstream consistency.
    """
    if not key:
        return "OTHER"
    key = key.strip()
    if not key:
        return "OTHER"
    key = re.sub(r"[^A-Za-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key.upper() if key else "OTHER"


def _subject_to_topic(subject: str) -> str:
    """Convert an email subject into a meeting topic.

    Example:
      "Re: Contract approval timeline" → "Contract approval timeline"

    This is intentionally conservative; it only strips common reply/forward prefixes.
    """
    s = (subject or "").strip()
    # Remove repeated prefixes like "Re: Re: "
    while True:
        new = re.sub(r"^(\s*(re|fwd|fw)\s*:\s*)", "", s, flags=re.IGNORECASE)
        if new == s:
            break
        s = new
    return s.strip()


def _extract_timezone(text: str, base_dt: datetime) -> Tuple[timezone | datetime.tzinfo, Optional[str]]:
    """Extract timezone info from text.

    Returns:
      (tzinfo, tz_name)

    tz_name is an IANA string when we can infer one, otherwise None.
    """
    lower = (text or "").lower()

    # 1) Explicit abbreviations like "PT", "PST", "ET".
    for abbr, iana in _TZ_ABBREV_TO_IANA.items():
        if re.search(rf"\b{re.escape(abbr)}\b", lower):
            if iana == "UTC":
                return timezone.utc, "UTC"
            tzinfo = dateutil_tz.gettz(iana)
            return (tzinfo or timezone.utc), iana

    # 2) Infer from base_dt offset.
    if base_dt.tzinfo is not None:
        off = base_dt.utcoffset()
        if off in _OFFSET_TO_IANA:
            iana = _OFFSET_TO_IANA[off]
            if iana == "UTC":
                return timezone.utc, "UTC"
            tzinfo = dateutil_tz.gettz(iana)
            return (tzinfo or base_dt.tzinfo), iana

    # 3) Fallback: keep base tz (or UTC).
    return (base_dt.tzinfo or timezone.utc), None


_TIME_RE_12H = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_TIME_RE_24H = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _extract_time_components(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract hour/minute from a time phrase.

    Supports:
      - "2pm", "2:30 pm"
      - "14:00"
      - "noon", "midnight"
      - "EOD" (treated as 17:00)

    Returns (hour, minute) or (None, None) if no time found.
    """
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
    """Best-effort parse of natural-language date/time into a timezone-aware datetime.

    Returns:
      (dt, tz_name)

    We do conservative parsing:
      - relative words (today/tomorrow)
      - weekday names (Friday)
      - time-of-day hints (2pm, 14:00, EOD)
      - timezone abbreviations (PT, ET)
      - fallback to dateutil for explicit dates (Dec 26 2025 2pm)

    If we cannot infer anything, returns (None, None).
    """
    if not text or not text.strip():
        return None, None

    base = base_dt
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    tzinfo, tz_name = _extract_timezone(text, base)

    # Normalize base into the inferred tz to make weekday resolution correct.
    try:
        base_local = base.astimezone(tzinfo)
    except Exception:
        base_local = base

    lower = text.strip().lower()

    # --- 1) Relative day words
    day_dt: Optional[datetime] = None
    if "today" in lower:
        day_dt = base_local
    elif "tomorrow" in lower:
        day_dt = base_local + timedelta(days=1)

    # --- 2) Weekday resolution ("Friday")
    if day_dt is None:
        for day_name, target_wd in _DAY_OF_WEEK.items():
            if re.search(rf"\b{day_name}\b", lower):
                base_wd = base_local.weekday()
                delta = (target_wd - base_wd) % 7
                # Prefer future date when ambiguous ("Friday" on Friday → next Friday).
                if delta == 0:
                    delta = 7
                if re.search(r"\bnext\b", lower):
                    delta += 7
                day_dt = base_local + timedelta(days=delta)
                break

    # --- 3) If we have a day, attach a time if present
    if day_dt is not None:
        hour, minute = _extract_time_components(lower)
        if hour is not None:
            day_dt = day_dt.replace(hour=hour, minute=minute or 0, second=0, microsecond=0)
        else:
            # If the text doesn't specify time, keep date-only semantics by setting
            # time to 00:00; downstream decides whether to show date or datetime.
            day_dt = day_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        # Ensure tzinfo
        if day_dt.tzinfo is None:
            try:
                day_dt = day_dt.replace(tzinfo=tzinfo)
            except Exception:
                day_dt = day_dt.replace(tzinfo=timezone.utc)
        return day_dt, tz_name

    # --- 4) Fallback: dateutil for explicit dates
    try:
        parsed = dateutil_parser.parse(text, default=base_local, fuzzy=True)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tzinfo)
        else:
            # Convert into tzinfo for consistency
            parsed = parsed.astimezone(tzinfo)
        return parsed, tz_name
    except Exception:
        return None, None


def _fill_sender(parsed_email: ParsedEmail, llm: LLMTriageOutput) -> None:
    """Ensure the sender appears in entities.people as role=sender."""
    sender_email = parsed_email.from_email
    if not sender_email:
        return

    # If sender already exists (any role), don't duplicate.
    for p in llm.entities.people:
        if (p.email or "").lower() == sender_email.lower():
            # Ensure sender role if missing/empty.
            if not p.role:
                p.role = "sender"
            return

    llm.entities.people.insert(0, PersonRef(email=sender_email, role="sender"))


def _fill_date_isos(parsed_email: ParsedEmail, llm: LLMTriageOutput) -> Tuple[Optional[str], Optional[datetime]]:
    """Fill missing date ISO strings when possible.

    Returns:
      (best_tz_name, best_dt)

    We also return the first "best" inferred datetime to help meeting.start_at.
    """
    base_dt = parsed_email.sent_at or parsed_email.internal_date or datetime.now(timezone.utc)

    best_dt: Optional[datetime] = None
    best_tz_name: Optional[str] = None

    for d in llm.entities.dates:
        if d.iso and str(d.iso).strip():
            # Try to parse tz name from existing ISO (if any)
            if best_dt is None:
                try:
                    best_dt = dateutil_parser.isoparse(d.iso)
                except Exception:
                    pass
            continue

        dt, tz_name = _infer_datetime(d.text, base_dt)
        if not dt:
            continue

        # Decide whether to output date-only or datetime ISO.
        # If the text includes time, emit full datetime; else emit date.
        has_time = _extract_time_components(d.text)[0] is not None or bool(_TIME_RE_24H.search(d.text))
        d.iso = dt.isoformat() if has_time else dt.date().isoformat()

        if best_dt is None and has_time:
            best_dt = dt
            best_tz_name = tz_name

    return best_tz_name, best_dt


def _fill_meeting(parsed_email: ParsedEmail, llm: LLMTriageOutput, tz_name: Optional[str], best_dt: Optional[datetime]) -> None:
    """Populate entities.meeting when appropriate.

    We only create a meeting object when:
      - major_category is schedule_and_time
      - OR sub_action_key suggests meeting scheduling/confirmation

    The LLM may also provide entities.meeting directly. In that case we only fill
    missing fields.
    """
    wants_meeting = (
        llm.major_category.value == "schedule_and_time"
        or llm.sub_action_key.upper().startswith("SCHEDULE_")
    )
    if not wants_meeting:
        return

    if llm.entities.meeting is None:
        llm.entities.meeting = MeetingRef()

    # Fill topic from subject if missing.
    if not (llm.entities.meeting.topic or "").strip():
        llm.entities.meeting.topic = _subject_to_topic(parsed_email.subject)

    # Fill start_at from the best inferred datetime (if missing).
    if not (llm.entities.meeting.start_at or "").strip() and best_dt is not None:
        llm.entities.meeting.start_at = best_dt.isoformat()

    # Fill tz (IANA preferred).
    if not (llm.entities.meeting.tz or "").strip() and tz_name:
        llm.entities.meeting.tz = tz_name


def postprocess_triage(
    *,
    msg: GmailMessageInput,
    parsed_email: ParsedEmail,
    llm: LLMTriageOutput,
    predicted_at: datetime,
) -> EmailTriageResponse:
    """Build the final API response (snake_case keys) from the parsed LLM output."""

    # --- Normalize and clamp
    llm.confidence = _clamp01(llm.confidence)
    llm.sub_action_key = _normalize_sub_action_key(llm.sub_action_key)

    llm.reason = (llm.reason or "").strip()

    # Evidence: trim, de-dupe, cap length
    cleaned = []
    seen = set()
    for e in (llm.evidence or []):
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
    llm.evidence = cleaned

    # If explicit_task is false, force task_type null.
    if not llm.explicit_task:
        llm.task_type = None
    else:
        llm.task_type = (llm.task_type or "").strip() or None

    # --- Fill deterministic entities
    _fill_sender(parsed_email, llm)
    tz_name, best_dt = _fill_date_isos(parsed_email, llm)
    _fill_meeting(parsed_email, llm, tz_name, best_dt)

    return EmailTriageResponse(
        message_id=msg.id,
        thread_id=msg.threadId,
        major_category=llm.major_category,
        sub_action_key=llm.sub_action_key,
        reply_required=bool(llm.reply_required),
        explicit_task=bool(llm.explicit_task),
        task_type=llm.task_type,
        confidence=llm.confidence,
        reason=llm.reason,
        evidence=llm.evidence,
        entities=llm.entities,
    )
