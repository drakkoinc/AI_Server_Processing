"""Tests for the pre-LLM classification gate.

Covers:
  - Individual signal detection (headers, body, sender patterns)
  - Negative signals (personal replies, questions)
  - Combined scoring and verdict logic
  - Gate short-circuit for obvious spam
  - Gate pass-through for clear non-spam
  - Ambiguous emails that need LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pytest

from app.gate import (
    SPAM_THRESHOLD,
    GateResult,
    run_gate,
    _check_list_unsubscribe,
    _check_precedence_bulk,
    _check_auto_submitted,
    _check_noreply_sender,
    _check_marketing_sender,
    _check_bulk_sender_name,
    _check_reply_to_mismatch,
    _check_bcc_recipient,
    _check_unsubscribe_body,
    _check_view_in_browser,
    _check_footer_pattern,
    _check_tracking_pixels,
    _check_utm_params,
    _check_high_html_ratio,
    _check_personal_reply,
    _check_direct_questions,
    _infer_spam_sub_action,
)
from app.gmail import ParsedEmail
from app.models import GmailHeader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed_email(
    *,
    subject: str = "Test Subject",
    from_name: Optional[str] = "Test Sender",
    from_email: Optional[str] = "test@example.com",
    to: Optional[List[str]] = None,
    body_text: str = "Hello, this is a test email.",
    body_html: Optional[str] = None,
) -> ParsedEmail:
    """Create a minimal ParsedEmail for testing."""
    return ParsedEmail(
        provider="gmail",
        message_id="test-msg-001",
        thread_id="test-thread-001",
        subject=subject,
        from_name=from_name,
        from_email=from_email,
        to=to or ["recipient@example.com"],
        cc=[],
        sent_at=datetime(2026, 2, 20, 10, 0, 0, tzinfo=timezone.utc),
        internal_date=datetime(2026, 2, 20, 10, 0, 0, tzinfo=timezone.utc),
        snippet="Test snippet",
        body_text=body_text,
        body_html=body_html,
    )


def _make_headers(**kwargs) -> List[GmailHeader]:
    """Create GmailHeader list from keyword args."""
    return [GmailHeader(name=k, value=v) for k, v in kwargs.items()]


# ===================================================================
# 1. Individual Signal Tests — Header Signals
# ===================================================================

class TestHeaderSignals:

    def test_list_unsubscribe_present(self):
        headers = {"list-unsubscribe": "<mailto:unsub@example.com>"}
        sig = _check_list_unsubscribe(headers)
        assert sig.fired is True
        assert sig.weight > 0

    def test_list_unsubscribe_absent(self):
        headers = {"from": "test@example.com"}
        sig = _check_list_unsubscribe(headers)
        assert sig.fired is False

    def test_precedence_bulk(self):
        sig = _check_precedence_bulk({"precedence": "bulk"})
        assert sig.fired is True

    def test_precedence_list(self):
        sig = _check_precedence_bulk({"precedence": "list"})
        assert sig.fired is True

    def test_precedence_normal(self):
        sig = _check_precedence_bulk({"precedence": "normal"})
        assert sig.fired is False

    def test_precedence_absent(self):
        sig = _check_precedence_bulk({})
        assert sig.fired is False

    def test_auto_submitted_auto_generated(self):
        sig = _check_auto_submitted({"auto-submitted": "auto-generated"})
        assert sig.fired is True

    def test_auto_submitted_no(self):
        sig = _check_auto_submitted({"auto-submitted": "no"})
        assert sig.fired is False

    def test_auto_submitted_absent(self):
        sig = _check_auto_submitted({})
        assert sig.fired is False

    def test_reply_to_mismatch_fired(self):
        sig = _check_reply_to_mismatch(
            {"reply-to": "marketing@bigcorp.com"},
            "noreply@bigcorp.com",
        )
        assert sig.fired is True

    def test_reply_to_match_not_fired(self):
        sig = _check_reply_to_mismatch(
            {"reply-to": "alice@example.com"},
            "alice@example.com",
        )
        assert sig.fired is False

    def test_bcc_empty_to(self):
        sig = _check_bcc_recipient({"to": ""}, [])
        assert sig.fired is True

    def test_bcc_undisclosed(self):
        sig = _check_bcc_recipient({"to": "undisclosed-recipients:;"}, [])
        assert sig.fired is True

    def test_bcc_normal_to(self):
        sig = _check_bcc_recipient(
            {"to": "user@example.com"},
            ["user@example.com"],
        )
        assert sig.fired is False


# ===================================================================
# 2. Individual Signal Tests — Sender Patterns
# ===================================================================

class TestSenderSignals:

    @pytest.mark.parametrize("email", [
        "noreply@company.com",
        "no-reply@company.com",
        "no_reply@company.com",
        "do-not-reply@company.com",
        "mailer-daemon@company.com",
        "bounce@company.com",
        "notifications@company.com",
        "alert@company.com",
    ])
    def test_noreply_sender_detected(self, email):
        sig = _check_noreply_sender(email)
        assert sig.fired is True

    def test_noreply_sender_personal(self):
        sig = _check_noreply_sender("alice@company.com")
        assert sig.fired is False

    @pytest.mark.parametrize("email", [
        "marketing@company.com",
        "promotions@company.com",
        "newsletter@company.com",
        "offers@company.com",
        "info@company.com",
        "updates@company.com",
        "news@company.com",
        "digest@company.com",
    ])
    def test_marketing_sender_detected(self, email):
        sig = _check_marketing_sender(email)
        assert sig.fired is True

    def test_marketing_sender_personal(self):
        sig = _check_marketing_sender("sarah@company.com")
        assert sig.fired is False

    @pytest.mark.parametrize("name", [
        "Company Newsletter",
        "Weekly Digest",
        "Daily Brief Updates",
        "Marketing Team",
    ])
    def test_bulk_sender_name_detected(self, name):
        sig = _check_bulk_sender_name(name)
        assert sig.fired is True

    def test_bulk_sender_name_personal(self):
        sig = _check_bulk_sender_name("Sarah Chen")
        assert sig.fired is False


# ===================================================================
# 3. Individual Signal Tests — Body Signals
# ===================================================================

class TestBodySignals:

    @pytest.mark.parametrize("text", [
        "Click here to unsubscribe from this list",
        "You can opt out at any time",
        "Manage your email preferences",
        "Update your preferences",
        "Change your preferences here",
        "Edit preferences for your account",
        "Stop receiving these emails",
        "Remove yourself from this mailing list",
        "You are receiving this because you subscribed",
    ])
    def test_unsubscribe_body_detected(self, text):
        sig = _check_unsubscribe_body(text)
        assert sig.fired is True

    def test_unsubscribe_body_personal(self):
        sig = _check_unsubscribe_body("Hi, can you confirm the meeting time for Friday?")
        assert sig.fired is False

    @pytest.mark.parametrize("text", [
        "View in browser",
        "View this email in your browser",
        "View as web page",
        "Having trouble viewing this email?",
        "Can't see this email? View online version",
        "Email not displaying correctly?",
    ])
    def test_view_in_browser_detected(self, text):
        sig = _check_view_in_browser(text)
        assert sig.fired is True

    def test_view_in_browser_absent(self):
        sig = _check_view_in_browser("Please review the attached document.")
        assert sig.fired is False

    @pytest.mark.parametrize("text", [
        "123 Main Street, Suite 100, San Francisco, CA",
        "© 2026 Acme Corp. All rights reserved.",
        "Copyright 2026 Company Inc",
        "You are receiving this email because you signed up for our newsletter",
    ])
    def test_footer_pattern_detected(self, text):
        sig = _check_footer_pattern(text)
        assert sig.fired is True

    def test_footer_pattern_absent(self):
        sig = _check_footer_pattern("Please send me the report by Friday.")
        assert sig.fired is False

    def test_tracking_pixels_detected(self):
        html = '<img src="https://track.example.com/open.gif" width="1" height="1" />'
        sig = _check_tracking_pixels(html)
        assert sig.fired is True

    def test_tracking_pixels_normal_image(self):
        html = '<img src="https://example.com/logo.png" width="200" height="50" />'
        sig = _check_tracking_pixels(html)
        assert sig.fired is False

    def test_tracking_pixels_no_html(self):
        sig = _check_tracking_pixels(None)
        assert sig.fired is False

    def test_utm_params_detected(self):
        sig = _check_utm_params(
            "Check out https://example.com/sale?utm_source=email&utm_medium=newsletter",
            None,
        )
        assert sig.fired is True

    def test_utm_params_absent(self):
        sig = _check_utm_params("Visit https://example.com/docs", None)
        assert sig.fired is False

    def test_high_html_ratio_detected(self):
        sig = _check_high_html_ratio(
            "Short text",
            "<html>" + "<div>" * 200 + "Marketing content" + "</div>" * 200 + "</html>",
        )
        assert sig.fired is True

    def test_high_html_ratio_normal(self):
        sig = _check_high_html_ratio(
            "This is a normal email with reasonable text content.",
            "<p>This is a normal email with reasonable text content.</p>",
        )
        assert sig.fired is False


# ===================================================================
# 4. Negative Signal Tests
# ===================================================================

class TestNegativeSignals:

    def test_personal_reply_detected(self):
        sig = _check_personal_reply(
            "Re: Contract review",
            "Hi Sarah, thanks for the update. Let me know if you need anything else.",
        )
        assert sig.fired is True
        assert sig.weight < 0  # Negative weight

    def test_personal_reply_not_reply(self):
        sig = _check_personal_reply(
            "Weekly Newsletter",
            "Hi Sarah, check out our latest deals.",
        )
        assert sig.fired is False

    def test_direct_questions_detected(self):
        sig = _check_direct_questions("Can you confirm the meeting for Friday at 2pm?")
        assert sig.fired is True
        assert sig.weight < 0  # Negative weight

    def test_direct_questions_absent(self):
        sig = _check_direct_questions("50% off all items this weekend only!")
        assert sig.fired is False


# ===================================================================
# 5. Sub-Action Key Inference
# ===================================================================

class TestSubActionInference:

    def test_newsletter_detected(self):
        assert _infer_spam_sub_action("Your weekly newsletter is here", None) == "SPAM_NEWSLETTER"

    def test_digest_detected(self):
        assert _infer_spam_sub_action("Daily digest for Feb 20", None) == "SPAM_NEWSLETTER"

    def test_unsubscribe_detected(self):
        assert _infer_spam_sub_action(
            "Big sale! Click here to unsubscribe if you no longer wish to receive.",
            None,
        ) == "SPAM_UNSUBSCRIBE"

    def test_marketing_default(self):
        assert _infer_spam_sub_action("Amazing deals just for you!", None) == "SPAM_MARKETING"


# ===================================================================
# 6. Full Gate Integration Tests
# ===================================================================

class TestGateIntegration:

    def test_obvious_spam_skips_llm(self):
        """An email with List-Unsubscribe + noreply + unsubscribe body should be caught."""
        email = _make_parsed_email(
            subject="50% Off Everything!",
            from_name="Deals Newsletter",
            from_email="noreply@marketing.bigstore.com",
            body_text="Amazing sale! 50% off everything. Click here to unsubscribe.",
        )
        headers = _make_headers(
            From="Deals Newsletter <noreply@marketing.bigstore.com>",
            To="user@example.com",
            Subject="50% Off Everything!",
            **{"List-Unsubscribe": "<mailto:unsub@bigstore.com>"},
        )
        # Add Precedence header
        headers.append(GmailHeader(name="Precedence", value="bulk"))

        result = run_gate(email, headers)
        assert result.verdict == "spam"
        assert result.skip_llm is True
        assert result.score >= SPAM_THRESHOLD

    def test_personal_email_passes_through(self):
        """A personal email from a colleague should not be caught."""
        email = _make_parsed_email(
            subject="Re: Contract review",
            from_name="Sarah Chen",
            from_email="sarah@acme.com",
            body_text="Hi Hani, can you confirm Friday at 2pm PT works for the contract review?",
        )
        headers = _make_headers(
            From="Sarah Chen <sarah@acme.com>",
            To="hani@drakko.io",
            Subject="Re: Contract review",
        )

        result = run_gate(email, headers)
        assert result.verdict == "not_spam"
        assert result.skip_llm is False
        assert result.score <= 0.20

    def test_ambiguous_email_passes_to_llm_with_signals(self):
        """An email with some spam signals but also personal traits."""
        email = _make_parsed_email(
            subject="Your weekly project update",
            from_name="Project Bot",
            from_email="updates@internal-tool.com",
            body_text="Hi team, here is this week's project status. Can you review the milestones? Manage your notification settings here.",
        )
        headers = _make_headers(
            From="Project Bot <updates@internal-tool.com>",
            To="team@company.com",
            Subject="Your weekly project update",
        )

        result = run_gate(email, headers)
        # Should have some spam signals but also question signals
        assert result.skip_llm is False
        assert result.score > 0  # Some signals fired

    def test_noreply_with_unsubscribe_is_spam(self):
        """Classic automated email pattern."""
        email = _make_parsed_email(
            subject="Your Monthly Statement",
            from_name="No Reply",
            from_email="no-reply@bank-marketing.com",
            body_text="View your statement online. You are receiving this email because you are a customer. Unsubscribe from marketing emails.",
            body_html='<html><body><p>View your statement</p><img src="https://track.com/pixel.gif" width="1" height="1"></body></html>',
        )
        headers = _make_headers(
            From="No Reply <no-reply@bank-marketing.com>",
            To="user@example.com",
            Subject="Your Monthly Statement",
            **{"List-Unsubscribe": "<https://bank-marketing.com/unsub>"},
        )

        result = run_gate(email, headers, body_html=email.body_html)
        assert result.verdict == "spam"
        assert result.skip_llm is True

    def test_newsletter_sub_action(self):
        """Gate should pick SPAM_NEWSLETTER for newsletter-style emails."""
        email = _make_parsed_email(
            subject="Tech Weekly Newsletter - Issue #142",
            from_name="Tech Weekly Newsletter",
            from_email="newsletter@techweekly.com",
            body_text="This week's newsletter: Top 10 programming languages. To unsubscribe, click here.",
        )
        headers = _make_headers(
            From="Tech Weekly Newsletter <newsletter@techweekly.com>",
            To="user@example.com",
            Subject="Tech Weekly Newsletter - Issue #142",
            **{"List-Unsubscribe": "<mailto:unsub@techweekly.com>"},
            Precedence="bulk",
        )

        result = run_gate(email, headers)
        assert result.verdict == "spam"
        assert result.sub_action_key == "SPAM_NEWSLETTER"

    def test_gate_result_contains_all_signals(self):
        """Every signal should be evaluated and present in the result."""
        email = _make_parsed_email()
        headers = _make_headers(From="test@example.com", To="user@example.com")

        result = run_gate(email, headers)
        signal_names = [s.name for s in result.signals]

        # Verify all expected signals are present
        assert "list_unsubscribe_header" in signal_names
        assert "precedence_bulk" in signal_names
        assert "auto_submitted" in signal_names
        assert "noreply_sender" in signal_names
        assert "marketing_sender" in signal_names
        assert "unsubscribe_body" in signal_names
        assert "personal_reply" in signal_names
        assert "direct_questions" in signal_names

    def test_gate_score_clamped_to_0_1(self):
        """Score should always be between 0.0 and 1.0."""
        email = _make_parsed_email(
            body_text="Just a normal email with no signals.",
        )
        headers = _make_headers(From="alice@example.com", To="bob@example.com")

        result = run_gate(email, headers)
        assert 0.0 <= result.score <= 1.0

    def test_fired_signals_property(self):
        """The fired_signals property should only include signals that fired."""
        email = _make_parsed_email(
            from_email="noreply@company.com",
            body_text="Click to unsubscribe from this list",
        )
        headers = _make_headers(
            From="noreply@company.com",
            To="user@example.com",
        )

        result = run_gate(email, headers)
        for sig in result.fired_signals:
            assert sig.fired is True
        # At least noreply and unsubscribe should fire
        fired_names = [s.name for s in result.fired_signals]
        assert "noreply_sender" in fired_names
        assert "unsubscribe_body" in fired_names

    def test_signal_summary_dict(self):
        """The signal_summary property should return a dict of name -> bool."""
        email = _make_parsed_email()
        headers = _make_headers(From="test@example.com")

        result = run_gate(email, headers)
        summary = result.signal_summary
        assert isinstance(summary, dict)
        assert all(isinstance(v, bool) for v in summary.values())
