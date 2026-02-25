"""Pre-LLM classification gate.

A fast, deterministic classifier that runs *before* Claude sees the email.
It catches obvious spam/automated emails using header analysis, body signals,
and sender patterns — saving LLM cost and latency while improving accuracy
on clear-cut cases where rules outperform probabilistic models.

Architecture:
    Email arrives
        │
        ├─→ Gate (deterministic, <5ms, free)
        │       │
        │       ├─→ SPAM (high confidence)  → build response, skip LLM
        │       ├─→ NOT_SPAM (high conf.)   → send to Claude for full triage
        │       └─→ AMBIGUOUS               → send to Claude with signal hints
        │
        └─→ Claude triages (only emails that need it)

Each check returns a weighted signal. If the combined spam score exceeds the
threshold, the email is classified as spam without calling the LLM.

Signals are intentionally ordered from cheapest to most expensive (headers
first, then body analysis) so we can short-circuit early.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.gmail import ParsedEmail


# ---------------------------------------------------------------------------
# Signal weights — tuned conservatively to minimize false positives.
# A total score >= SPAM_THRESHOLD triggers the gate.
# ---------------------------------------------------------------------------

SPAM_THRESHOLD = 0.70

# Individual signal weights
_W_LIST_HEADER = 0.35          # List-Unsubscribe header present
_W_PRECEDENCE_BULK = 0.30      # Precedence: bulk/list header
_W_NOREPLY_SENDER = 0.25       # noreply@, no-reply@, mailer-daemon@
_W_MARKETING_SENDER = 0.20     # marketing@, promotions@, newsletter@
_W_UNSUBSCRIBE_BODY = 0.30     # Unsubscribe/opt-out language in body
_W_TRACKING_PIXELS = 0.15      # 1x1 images (tracking pixels)
_W_UTM_PARAMS = 0.15           # UTM tracking parameters in links
_W_HIGH_HTML_RATIO = 0.10      # Body is mostly HTML with little text
_W_REPLY_TO_MISMATCH = 0.10    # From ≠ Reply-To (bulk sender pattern)
_W_BCC_RECIPIENT = 0.10        # Sent via BCC (bulk send pattern)
_W_AUTO_SUBMITTED = 0.30       # Auto-Submitted header present
_W_BULK_SENDER_NAME = 0.15     # Sender name contains marketing terms
_W_VIEW_IN_BROWSER = 0.20      # "View in browser" / "View as web page"
_W_FOOTER_PATTERN = 0.15       # Address/company footer pattern

# Negative weights (signals that this is NOT spam)
_W_PERSONAL_REPLY = -0.40      # Starts with "Re:" and has personal tone
_W_DIRECT_ADDRESS = -0.25      # Addresses recipient by name personally
_W_QUESTION_ASKED = -0.15      # Contains direct questions expecting reply


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_NOREPLY_RE = re.compile(
    r"""^(
        no[-_.]?reply |
        do[-_.]?not[-_.]?reply |
        mailer[-_.]?daemon |
        postmaster |
        bounce[s]? |
        auto[-_.]?notify |
        notifications? |
        alerts?
    )@""",
    re.IGNORECASE | re.VERBOSE,
)

_MARKETING_SENDER_RE = re.compile(
    r"""^(
        marketing |
        promotions? |
        newsletters? |
        offers? |
        deals |
        info |
        hello |
        team |
        support |
        updates? |
        news |
        digest |
        communications? |
        campaigns?
    )@""",
    re.IGNORECASE | re.VERBOSE,
)

_BULK_SENDER_NAME_RE = re.compile(
    r"""\b(
        newsletter |
        digest |
        weekly\s+update |
        daily\s+brief |
        marketing |
        promotions? |
        no[-\s]?reply
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Unsubscribe/opt-out language in body (expanded from preprocess.py version)
_UNSUBSCRIBE_BODY_RE = re.compile(
    r"""\b(
        unsubscribe |
        opt[\s-]?out |
        manage\s+(?:your\s+)?(?:email\s+)?preferences |
        update\s+(?:your\s+)?(?:email\s+)?preferences |
        change\s+(?:your\s+)?preferences |
        edit\s+preferences |
        update\s+subscription |
        notification\s+settings |
        email\s+settings |
        communication\s+preferences |
        mailing\s+list |
        stop\s+receiving |
        remove\s+(?:me|yourself)\s+from |
        no\s+longer\s+wish\s+to\s+receive |
        (?:you\s+are|you're)\s+receiving\s+this\s+(?:email|message\s+)?because |
        receiving\s+this\s+because\s+you\s+subscribed |
        sent\s+(?:to|because)\s+you(?:'re|\s+are)\s+(?:a\s+)?(?:subscribed|registered|member|on\s+our\s+list)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# "View in browser" / "View as web page" pattern
_VIEW_IN_BROWSER_RE = re.compile(
    r"""\b(
        view\s+(?:in|this\s+email\s+in)\s+(?:your\s+)?(?:browser|web) |
        view\s+as\s+(?:a\s+)?web\s*page |
        view\s+(?:the\s+)?online\s+version |
        having\s+trouble\s+(?:viewing|reading)\s+this |
        can(?:'t|not)\s+(?:see|view|read)\s+this\s+email |
        email\s+not\s+displaying\s+(?:correctly|properly)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Company footer / physical address pattern (CAN-SPAM compliance)
_FOOTER_ADDRESS_RE = re.compile(
    r"""(?:
        \d{1,5}\s+\w+\s+(?:street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln|way|suite|ste)[\s,]+ |
        ©\s*\d{4} |
        copyright\s*\d{4} |
        all\s+rights\s+reserved |
        you\s+(?:are\s+)?receiving\s+this\s+(?:email|message)\s+because
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Tracking pixel detection (1x1 images)
_TRACKING_PIXEL_RE = re.compile(
    r"""<img[^>]+(?:
        width\s*=\s*["']?1["'\s>] |
        height\s*=\s*["']?1["'\s>] |
        style\s*=\s*["'][^"']*(?:width|height)\s*:\s*1px
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# UTM parameters in URLs
_UTM_RE = re.compile(r"""[?&]utm_(?:source|medium|campaign|content|term)=""", re.IGNORECASE)

# Personal reply indicators
_PERSONAL_REPLY_RE = re.compile(
    r"""(?:
        ^(?:hi|hey|hello|dear|thanks|thank\s+you)\s+\w+ |
        (?:let\s+me\s+know|what\s+do\s+you\s+think|could\s+you|can\s+you|would\s+you)
    )""",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

# Direct questions
_QUESTION_RE = re.compile(
    r"""\b(?:
        can\s+you |
        could\s+you |
        would\s+you |
        will\s+you |
        do\s+you |
        are\s+you |
        have\s+you |
        what\s+(?:do|does|did|is|are|should|would|could) |
        when\s+(?:can|will|should|is|are) |
        how\s+(?:do|does|can|should|would) |
        shall\s+(?:we|i)
    )\b.*\?""",
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateSignal:
    """A single signal from the gate, with its name, weight, and match status."""
    name: str
    weight: float
    fired: bool
    detail: str = ""


@dataclass
class GateResult:
    """The outcome of the pre-LLM gate.

    Attributes:
        verdict:    "spam" | "not_spam" | "ambiguous"
        score:      Combined weighted score (0.0 = clean, 1.0 = definite spam)
        signals:    List of all signals evaluated, with their status
        skip_llm:   True if the gate is confident enough to skip the LLM call
        sub_action_key: Suggested sub_action_key if verdict is spam
    """
    verdict: str
    score: float
    signals: List[GateSignal] = field(default_factory=list)
    skip_llm: bool = False
    sub_action_key: str = "SPAM_MARKETING"

    @property
    def fired_signals(self) -> List[GateSignal]:
        return [s for s in self.signals if s.fired]

    @property
    def signal_summary(self) -> Dict[str, bool]:
        return {s.name: s.fired for s in self.signals}


# ---------------------------------------------------------------------------
# Header extraction helpers
# ---------------------------------------------------------------------------

def _get_headers_dict(parsed_email: ParsedEmail, raw_headers: List) -> Dict[str, str]:
    """Build a case-insensitive header dict from the raw Gmail headers."""
    out: Dict[str, str] = {}
    for h in raw_headers:
        key = (getattr(h, "name", "") or "").strip().lower()
        val = (getattr(h, "value", "") or "").strip()
        if key and key not in out:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Individual signal checks
# ---------------------------------------------------------------------------

def _check_list_unsubscribe(headers: Dict[str, str]) -> GateSignal:
    """Check for List-Unsubscribe header (RFC 2369) — strong spam indicator."""
    has_it = "list-unsubscribe" in headers
    detail = headers.get("list-unsubscribe", "")[:100] if has_it else ""
    return GateSignal("list_unsubscribe_header", _W_LIST_HEADER, has_it, detail)


def _check_precedence_bulk(headers: Dict[str, str]) -> GateSignal:
    """Check for Precedence: bulk/list/junk header."""
    prec = headers.get("precedence", "").lower()
    fired = prec in ("bulk", "list", "junk")
    return GateSignal("precedence_bulk", _W_PRECEDENCE_BULK, fired, prec)


def _check_auto_submitted(headers: Dict[str, str]) -> GateSignal:
    """Check for Auto-Submitted header (RFC 3834) — auto-generated emails."""
    auto = headers.get("auto-submitted", "").lower()
    fired = auto != "" and auto != "no"
    return GateSignal("auto_submitted", _W_AUTO_SUBMITTED, fired, auto)


def _check_noreply_sender(from_email: Optional[str]) -> GateSignal:
    """Check if the sender address matches no-reply patterns."""
    if not from_email:
        return GateSignal("noreply_sender", _W_NOREPLY_SENDER, False)
    fired = bool(_NOREPLY_RE.match(from_email))
    return GateSignal("noreply_sender", _W_NOREPLY_SENDER, fired, from_email if fired else "")


def _check_marketing_sender(from_email: Optional[str]) -> GateSignal:
    """Check if the sender address matches marketing/newsletter patterns."""
    if not from_email:
        return GateSignal("marketing_sender", _W_MARKETING_SENDER, False)
    fired = bool(_MARKETING_SENDER_RE.match(from_email))
    return GateSignal("marketing_sender", _W_MARKETING_SENDER, fired, from_email if fired else "")


def _check_bulk_sender_name(from_name: Optional[str]) -> GateSignal:
    """Check if the sender display name contains marketing terms."""
    if not from_name:
        return GateSignal("bulk_sender_name", _W_BULK_SENDER_NAME, False)
    fired = bool(_BULK_SENDER_NAME_RE.search(from_name))
    return GateSignal("bulk_sender_name", _W_BULK_SENDER_NAME, fired, from_name if fired else "")


def _check_reply_to_mismatch(headers: Dict[str, str], from_email: Optional[str]) -> GateSignal:
    """Check if Reply-To differs from From (common in bulk email)."""
    reply_to = headers.get("reply-to", "").strip().lower()
    from_lower = (from_email or "").strip().lower()
    fired = bool(reply_to) and bool(from_lower) and reply_to != from_lower and from_lower not in reply_to
    detail = f"from={from_lower} reply-to={reply_to}" if fired else ""
    return GateSignal("reply_to_mismatch", _W_REPLY_TO_MISMATCH, fired, detail)


def _check_bcc_recipient(headers: Dict[str, str], to_list: List[str]) -> GateSignal:
    """Check if the recipient appears to be BCC'd (no To header or undisclosed)."""
    to_val = headers.get("to", "").strip().lower()
    fired = (
        not to_val
        or "undisclosed" in to_val
        or (len(to_list) == 0 and not to_val)
    )
    return GateSignal("bcc_recipient", _W_BCC_RECIPIENT, fired, to_val[:80] if fired else "")


def _check_unsubscribe_body(body_text: str) -> GateSignal:
    """Check for unsubscribe/opt-out language in the email body."""
    match = _UNSUBSCRIBE_BODY_RE.search(body_text)
    fired = bool(match)
    detail = match.group(0).strip() if match else ""
    return GateSignal("unsubscribe_body", _W_UNSUBSCRIBE_BODY, fired, detail)


def _check_view_in_browser(body_text: str) -> GateSignal:
    """Check for 'View in browser' / 'View as web page' patterns."""
    match = _VIEW_IN_BROWSER_RE.search(body_text)
    fired = bool(match)
    detail = match.group(0).strip() if match else ""
    return GateSignal("view_in_browser", _W_VIEW_IN_BROWSER, fired, detail)


def _check_footer_pattern(body_text: str) -> GateSignal:
    """Check for CAN-SPAM compliant footer patterns (address, copyright, etc.)."""
    match = _FOOTER_ADDRESS_RE.search(body_text)
    fired = bool(match)
    detail = match.group(0).strip()[:80] if match else ""
    return GateSignal("footer_pattern", _W_FOOTER_PATTERN, fired, detail)


def _check_tracking_pixels(body_html: Optional[str]) -> GateSignal:
    """Check for 1x1 tracking pixel images in HTML body."""
    if not body_html:
        return GateSignal("tracking_pixels", _W_TRACKING_PIXELS, False)
    fired = bool(_TRACKING_PIXEL_RE.search(body_html))
    return GateSignal("tracking_pixels", _W_TRACKING_PIXELS, fired)


def _check_utm_params(body_text: str, body_html: Optional[str]) -> GateSignal:
    """Check for UTM tracking parameters in links."""
    search_text = (body_html or "") + " " + body_text
    fired = bool(_UTM_RE.search(search_text))
    return GateSignal("utm_params", _W_UTM_PARAMS, fired)


def _check_high_html_ratio(body_text: str, body_html: Optional[str]) -> GateSignal:
    """Check if HTML body is much larger than plain text (marketing template indicator)."""
    if not body_html:
        return GateSignal("high_html_ratio", _W_HIGH_HTML_RATIO, False)
    text_len = len(body_text.strip())
    html_len = len(body_html.strip())
    # If HTML is 5x+ larger than plain text, it's likely a marketing template
    fired = html_len > 500 and (text_len == 0 or html_len / max(text_len, 1) > 5)
    detail = f"text={text_len} html={html_len}" if fired else ""
    return GateSignal("high_html_ratio", _W_HIGH_HTML_RATIO, fired, detail)


# --- Negative signals (reduce spam score) ---

def _check_personal_reply(subject: str, body_text: str) -> GateSignal:
    """Check for personal reply indicators that suggest this is NOT spam."""
    is_reply = subject.lower().startswith("re:")
    has_personal = bool(_PERSONAL_REPLY_RE.search(body_text[:500]))
    fired = is_reply and has_personal
    return GateSignal("personal_reply", _W_PERSONAL_REPLY, fired)


def _check_direct_questions(body_text: str) -> GateSignal:
    """Check for direct questions — spam rarely asks real questions."""
    fired = bool(_QUESTION_RE.search(body_text[:1000]))
    return GateSignal("direct_questions", _W_QUESTION_ASKED, fired)


# ---------------------------------------------------------------------------
# Sub-action key inference
# ---------------------------------------------------------------------------

def _infer_spam_sub_action(body_text: str, from_email: Optional[str]) -> str:
    """Pick the most appropriate spam sub_action_key based on content."""
    lower = body_text.lower()

    # Newsletter indicators
    newsletter_terms = ["newsletter", "digest", "weekly update", "daily brief", "roundup", "edition"]
    if any(t in lower for t in newsletter_terms):
        return "SPAM_NEWSLETTER"

    # Explicit unsubscribe language
    if _UNSUBSCRIBE_BODY_RE.search(body_text):
        return "SPAM_UNSUBSCRIBE"

    # Default to marketing
    return "SPAM_MARKETING"


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

def run_gate(
    parsed_email: ParsedEmail,
    raw_headers: List,
    body_html: Optional[str] = None,
) -> GateResult:
    """Run the pre-LLM classification gate on a parsed email.

    Args:
        parsed_email: The normalized email from gmail.py.
        raw_headers:  The raw GmailHeader list from the input payload
                      (needed for headers not exposed on ParsedEmail).
        body_html:    The raw HTML body if available (for tracking pixel detection).

    Returns:
        GateResult with verdict, score, and detailed signal breakdown.
    """
    headers = _get_headers_dict(parsed_email, raw_headers)
    body_text = parsed_email.body_text or ""
    from_email = parsed_email.from_email
    from_name = parsed_email.from_name
    subject = parsed_email.subject or ""

    # --- Collect all signals (cheapest first) ---
    signals: List[GateSignal] = []

    # Header signals (instant, no body parsing)
    signals.append(_check_list_unsubscribe(headers))
    signals.append(_check_precedence_bulk(headers))
    signals.append(_check_auto_submitted(headers))
    signals.append(_check_noreply_sender(from_email))
    signals.append(_check_marketing_sender(from_email))
    signals.append(_check_bulk_sender_name(from_name))
    signals.append(_check_reply_to_mismatch(headers, from_email))
    signals.append(_check_bcc_recipient(headers, parsed_email.to))

    # Body signals (require text scanning)
    signals.append(_check_unsubscribe_body(body_text))
    signals.append(_check_view_in_browser(body_text))
    signals.append(_check_footer_pattern(body_text))
    signals.append(_check_tracking_pixels(body_html))
    signals.append(_check_utm_params(body_text, body_html))
    signals.append(_check_high_html_ratio(body_text, body_html))

    # Negative signals (reduce spam score)
    signals.append(_check_personal_reply(subject, body_text))
    signals.append(_check_direct_questions(body_text))

    # --- Calculate combined score ---
    score = sum(s.weight for s in signals if s.fired)
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]

    # --- Determine verdict ---
    if score >= SPAM_THRESHOLD:
        sub_action = _infer_spam_sub_action(body_text, from_email)
        return GateResult(
            verdict="spam",
            score=round(score, 3),
            signals=signals,
            skip_llm=True,
            sub_action_key=sub_action,
        )
    elif score <= 0.20:
        return GateResult(
            verdict="not_spam",
            score=round(score, 3),
            signals=signals,
            skip_llm=False,
        )
    else:
        # Ambiguous — pass to LLM with extra signal context
        return GateResult(
            verdict="ambiguous",
            score=round(score, 3),
            signals=signals,
            skip_llm=False,
        )
