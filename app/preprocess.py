"""Preprocessing utilities.

The LLM works best when the input is:
- normalized
- free of HTML boilerplate
- limited in length (to control cost/latency)

This module converts HTML → plain text and extracts simple *observable* signals
(links, money strings, 'by Friday', etc.) that can be used both:
- as extra context for the LLM
- as a sanity-check if the model outputs something implausible

Important: This preprocessing is intentionally lightweight.
For production you might add:
- robust MIME parsing
- attachment OCR / PDF parsing
- language detection
- PII redaction (if required by your compliance policy)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup


_URL_RE = re.compile(r"""\bhttps?://[^\s<>()]+""", re.IGNORECASE)

# Very lightweight money patterns (covers common invoice/payment emails).
_MONEY_RE = re.compile(
    r"""(?:
        (?:\$|USD\s?)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?  # $1,234.56
        |
        \d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|dollars)   # 1,234 USD
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Very lightweight time/deadline phrases; you will likely replace this with
# dateparser.search.search_dates or Duckling later.
_TIME_PHRASE_RE = re.compile(
    r"""\b(
        asap|
        eod|
        end\s+of\s+day|
        by\s+\w+(?:\s+\d{1,2})?|
        due\s+by\s+[^\n\.]+|
        due\s+on\s+[^\n\.]+|
        tomorrow|
        today|
        next\s+week|
        this\s+week|
        within\s+\d+\s+(?:hours?|days?)|
        in\s+\d+\s+(?:hours?|days?)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


def html_to_text(html: str) -> str:
    """Convert HTML email body to clean-ish plain text.

    We remove script/style tags, then return visible text with normalized whitespace.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    # Normalize newlines and collapse excessive blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\x0b\x0c]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \u00a0]{2,}", " ", text)
    return text.strip()


def extract_links(text: str) -> List[str]:
    return list(dict.fromkeys(_URL_RE.findall(text)))  # de-dupe while keeping order


def extract_money_expressions(text: str) -> List[str]:
    return list(dict.fromkeys([m.group(0).strip() for m in _MONEY_RE.finditer(text)]))


def extract_time_expressions(text: str, limit: int = 10) -> List[str]:
    matches = [m.group(0).strip() for m in _TIME_PHRASE_RE.finditer(text)]
    # de-dupe
    out: List[str] = []
    for x in matches:
        if x.lower() not in {y.lower() for y in out}:
            out.append(x)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class Preprocessed:
    subject: str
    from_line: str
    to_line: str
    cc_line: str
    body_text: str
    thread_text: Optional[str]
    links: List[str]
    money_expressions: List[str]
    time_expressions: List[str]


def _format_party_list(parties) -> str:
    if not parties:
        return ""
    return ", ".join([p.email if not p.name else f"{p.name} <{p.email}>" for p in parties])


def preprocess_email(email, max_body_chars: int = 12000) -> Preprocessed:
    """Normalize a raw EmailInput into a compact context object for the LLM."""
    subject = (email.subject or "").strip()
    from_line = "" if email.from_ is None else (email.from_.email if not email.from_.name else f"{email.from_.name} <{email.from_.email}>")
    to_line = _format_party_list(email.to)
    cc_line = _format_party_list(email.cc)

    # Prefer explicit body_text; fall back to HTML.
    body_text = (email.body_text or "").strip()
    if not body_text and email.body_html:
        body_text = html_to_text(email.body_html)

    # Guardrail: cap body length to control cost + avoid prompt overflow.
    if len(body_text) > max_body_chars:
        body_text = body_text[:max_body_chars] + "\n\n[TRUNCATED]"

    # Thread context (optional): concatenate a few recent messages.
    thread_text = None
    if getattr(email, "thread_messages", None):
        # Keep the last ~5 messages (configurable later).
        msgs = email.thread_messages[-5:]
        chunks: List[str] = []
        for m in msgs:
            hdr = f"From: {m.from_.email} | Subject: {m.subject or ''}"
            bt = (m.body_text or "").strip()
            if not bt and m.body_html:
                bt = html_to_text(m.body_html)
            bt = normalize_whitespace(bt)
            chunks.append(hdr + "\n" + bt)
        thread_text = "\n\n---\n\n".join(chunks)
        if len(thread_text) > max_body_chars:
            thread_text = thread_text[:max_body_chars] + "\n\n[TRUNCATED]"

    links = extract_links(body_text)
    money = extract_money_expressions(body_text)
    times = extract_time_expressions(body_text)

    return Preprocessed(
        subject=subject,
        from_line=from_line,
        to_line=to_line,
        cc_line=cc_line,
        body_text=body_text,
        thread_text=thread_text,
        links=links,
        money_expressions=money,
        time_expressions=times,
    )


def build_prompt_input(email_id: str, thread_id: Optional[str], pre: Preprocessed) -> str:
    """Build the *user* message sent to the LLM.

    We keep it as plain text to be model-agnostic.
    """
    parts = [
        f"EMAIL_ID: {email_id}",
        f"THREAD_ID: {thread_id or ''}",
        f"SUBJECT: {pre.subject}",
        f"FROM: {pre.from_line}",
        f"TO: {pre.to_line}",
        f"CC: {pre.cc_line}",
        "",
        "BODY:",
        pre.body_text,
    ]
    if pre.thread_text:
        parts += ["", "THREAD_CONTEXT:", pre.thread_text]

    if pre.links:
        parts += ["", "LINKS_DETECTED:", "\n".join(pre.links)]
    if pre.money_expressions:
        parts += ["", "MONEY_STRINGS_DETECTED:", "\n".join(pre.money_expressions)]
    if pre.time_expressions:
        parts += ["", "TIME_PHRASES_DETECTED:", "\n".join(pre.time_expressions)]

    return "\n".join(parts).strip()
