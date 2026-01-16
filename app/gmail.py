"""Gmail message parsing helpers.

Gmail's API returns message bodies as a nested MIME tree under `payload`.
The content of each body part is typically base64url encoded.

This module turns the raw Gmail JSON object into a normalized representation:

- subject, from/to/cc
- sent_at (best-effort)
- body_text and body_html (best-effort)
- attachments metadata (best-effort; actual attachment fetching is out-of-scope for v1)

Keeping this in one place makes it easy to:
- add support for additional providers later (Outlook, IMAP, etc.)
- add attachment text extraction (e.g., pdf → text) without touching business logic
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import getaddresses
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from dateutil import parser as dateutil_parser

from app.models import GmailHeader, GmailMessageInput, GmailPart
from app.preprocess import html_to_text


@dataclass(frozen=True)
class ParsedEmail:
    """Normalized email fields derived from a GmailMessageInput."""

    provider: str
    message_id: str
    thread_id: Optional[str]

    subject: str
    from_name: Optional[str]
    from_email: Optional[str]
    to: List[str]
    cc: List[str]

    sent_at: Optional[datetime]  # timezone-aware if parsed
    internal_date: Optional[datetime]  # timezone-aware UTC
    snippet: str

    body_text: str
    body_html: Optional[str]


def _headers_to_dict(headers: List[GmailHeader]) -> Dict[str, str]:
    """Convert Gmail headers array to a case-insensitive dict."""
    out: Dict[str, str] = {}
    for h in headers:
        # Gmail can include repeated headers; keep the first seen for simplicity.
        key = (h.name or "").strip().lower()
        if key and key not in out:
            out[key] = (h.value or "").strip()
    return out


def _b64url_decode_to_text(data: str) -> str:
    """Decode Gmail base64url data into UTF-8 text.

    Gmail's `data` is base64url (RFC 4648 §5) and often omits padding.
    """
    if not data:
        return ""
    # Add padding if missing.
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        # Defensive fallback: return empty string instead of failing the whole pipeline.
        return ""


def _walk_parts(part: GmailPart) -> Iterator[GmailPart]:
    """Depth-first traversal over all MIME parts."""
    yield part
    if part.parts:
        for child in part.parts:
            yield from _walk_parts(child)


def _extract_best_bodies(payload: GmailPart) -> Tuple[str, Optional[str]]:
    """Extract best-effort (plain_text, html) from a Gmail MIME payload."""
    plain_candidates: List[str] = []
    html_candidates: List[str] = []

    for p in _walk_parts(payload):
        mt = (p.mimeType or "").lower()
        data = p.body.data if p.body else None
        if not data:
            continue

        if mt == "text/plain":
            plain_candidates.append(_b64url_decode_to_text(data))
        elif mt == "text/html":
            html_candidates.append(_b64url_decode_to_text(data))

    body_html: Optional[str] = None
    body_text: str = ""

    if plain_candidates:
        # Join plain parts with separators; most emails only have one.
        body_text = "\n\n".join([t for t in plain_candidates if t.strip()])
    elif html_candidates:
        body_html = "\n\n".join([h for h in html_candidates if h.strip()])
        body_text = html_to_text(body_html)
    else:
        body_text = ""

    if body_html is None and html_candidates:
        body_html = "\n\n".join([h for h in html_candidates if h.strip()])

    return body_text.strip(), body_html


def parse_gmail_message(msg: GmailMessageInput) -> ParsedEmail:
    """Parse a Gmail message into a normalized ParsedEmail.

    This function is intentionally tolerant of missing/partial data so that
    your pipeline doesn't break when Gmail returns unexpected formats.

    For time:
    - Prefer parsing the RFC2822 'Date' header (keeps sender timezone).
    - Fallback to `internalDate` milliseconds (Gmail internal time, stored as UTC).
    """
    headers = _headers_to_dict(msg.payload.headers or [])

    subject = headers.get("subject", "").strip()

    # Parse From / To / Cc using stdlib email.utils.getaddresses.
    from_value = headers.get("from", "")
    from_pairs = getaddresses([from_value])
    from_name = from_pairs[0][0] if from_pairs else None
    from_email = from_pairs[0][1] if from_pairs else None

    to_value = headers.get("to", "")
    cc_value = headers.get("cc", "")
    to = [addr for _, addr in getaddresses([to_value]) if addr]
    cc = [addr for _, addr in getaddresses([cc_value]) if addr]

    sent_at: Optional[datetime] = None
    date_value = headers.get("date")
    if date_value:
        try:
            sent_at = dateutil_parser.parse(date_value)
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
        except Exception:
            sent_at = None

    internal_date: Optional[datetime] = None
    if msg.internalDate:
        try:
            ms = int(msg.internalDate)
            internal_date = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        except Exception:
            internal_date = None

    body_text, body_html = _extract_best_bodies(msg.payload)

    snippet = (msg.snippet or "").strip()

    return ParsedEmail(
        provider=msg.provider or "gmail",
        message_id=msg.id,
        thread_id=msg.threadId,
        subject=subject,
        from_name=from_name,
        from_email=from_email,
        to=to,
        cc=cc,
        sent_at=sent_at,
        internal_date=internal_date,
        snippet=snippet,
        body_text=body_text,
        body_html=body_html,
    )
