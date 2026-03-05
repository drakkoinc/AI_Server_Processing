"""
Pipeline orchestration — two-model architecture with automatic fallback.

Flow:
  GmailMessageInput -> Parse MIME -> Gate -> Triage (Qwen/Ollama) -> Postprocess
                                                                  -> Reply draft (DeepSeek/Ollama, if needed)
                                                                  -> API response

Model hierarchy:
  Primary:   Ollama (local) — Qwen 2.5 72B for triage, DeepSeek-V3.2 for replies
  Fallback:  Anthropic Claude (cloud) — kicks in if Ollama is unavailable

The pre-LLM gate catches obvious spam/automated emails without calling any model.
Only ambiguous or clearly non-spam emails are forwarded for full triage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from app.config import settings
from app.gate import GateResult, run_gate
from app.gmail import parse_gmail_message, ParsedEmail
from app.llm.anthropic_client import AnthropicClient
from app.llm.ollama_client import OllamaClient
from app.models import (
    DebugInfo,
    DraftReply,
    Entities,
    ExtractedSummary,
    GateDebug,
    GmailMessageInput,
    LLMReplyOutput,
    LLMTriageOutput,
    MajorCategory,
    RecommendedAction,
    TriageOutput,
    TriageResponse,
    UrgencySignals,
)
from app.postprocess import postprocess_triage
from app.preprocess import extract_links, extract_money_expressions, extract_time_expressions, extract_unsubscribe_signals
from app.prompt import PROMPT_VERSION, REPLY_SYSTEM_PROMPT, TRIAGE_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client builders
# ---------------------------------------------------------------------------

def _build_triage_client():
    """Build the primary triage client based on config."""
    if settings.llm_provider == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_triage_model,
            temperature=settings.temperature,
            timeout_s=settings.timeout_s,
        )
    # Direct Anthropic mode (no Ollama)
    return AnthropicClient(
        model=settings.anthropic_model,
        temperature=settings.temperature,
        timeout_s=settings.timeout_s,
    )


def _build_reply_client():
    """Build the reply drafting client based on config."""
    if settings.llm_provider == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_reply_model,
            temperature=0.4,  # slightly higher temp for creative reply drafting
            timeout_s=settings.timeout_s,
        )
    return AnthropicClient(
        model=settings.anthropic_model,
        temperature=0.4,
        timeout_s=settings.timeout_s,
    )


def _build_fallback_client():
    """Build the Anthropic fallback client (used when Ollama is down)."""
    return AnthropicClient(
        model=settings.anthropic_model,
        temperature=settings.temperature,
        timeout_s=settings.timeout_s,
    )


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def _build_gate_debug(gate: GateResult) -> GateDebug:
    """Convert gate result to debug metadata."""
    return GateDebug(
        verdict=gate.verdict,
        score=gate.score,
        skip_llm=gate.skip_llm,
        fired_signals=[s.name for s in gate.fired_signals],
    )


def _build_spam_response(
    parsed_email: ParsedEmail,
    gate: GateResult,
    predicted_at: datetime,
) -> TriageResponse:
    """Build a complete TriageResponse for gate-caught spam without calling any model.

    This produces the same response shape as the LLM path, but with deterministic
    values appropriate for spam/automated emails.
    """
    # Build a minimal extracted summary from the email
    subject = parsed_email.subject or "(no subject)"
    from_email = parsed_email.from_email or "unknown"

    # Select evidence from the gate signals
    evidence = []
    for sig in gate.fired_signals:
        if sig.detail:
            evidence.append(f"[{sig.name}] {sig.detail}")
        if len(evidence) >= 3:
            break

    output = TriageOutput(
        major_category=MajorCategory.spam,
        sub_action_key=gate.sub_action_key,
        explicit_task=False,
        confidence=min(gate.score + 0.1, 1.0),  # gate confidence, slightly boosted
        suggested_reply_action=[],
        task_proposal=None,
        recommended_actions=[
            RecommendedAction(key="archive", label="Archive", kind="PRIMARY", rank=1),
            RecommendedAction(key="unsubscribe", label="Unsubscribe", kind="SECONDARY", rank=2),
        ],
        urgency_signals=UrgencySignals(
            urgency="low",
            deadline_detected=False,
            deadline_text=None,
            reply_by=None,
            reason="Automated/subscription email — no action required.",
        ),
        extracted_summary=ExtractedSummary(
            ask=f"Automated email from {from_email}: {subject}",
            success_criteria="Archive or unsubscribe. No response needed.",
            missing_info=[],
        ),
        entities=Entities(people=[], dates=[], money=[], docs=[], meeting=None),
        evidence=evidence,
        draft_reply=None,  # spam never gets a draft reply
        debug=DebugInfo(
            analysis_timestamp=predicted_at.isoformat(),
            model_version=settings.model_version,
            prompt_version=PROMPT_VERSION,
            gate=_build_gate_debug(gate),
        ),
    )

    return TriageResponse(output=output)


# ---------------------------------------------------------------------------
# Reply logic
# ---------------------------------------------------------------------------

# Categories that may warrant a draft reply
_REPLY_CATEGORIES = frozenset({
    "core_communication",
    "decisions_and_approvals",
    "schedule_and_time",
    "documents_and_review",
    "social_and_people",
})


def _needs_reply(llm: LLMTriageOutput) -> bool:
    """Determine if this email warrants a draft reply.

    Returns True when:
    - The category is one that typically requires a response
    - AND either explicit_task is True or urgency is high/critical
    """
    category = llm.major_category.value
    if category not in _REPLY_CATEGORIES:
        return False

    # High-urgency or explicit tasks always get a draft
    if llm.explicit_task:
        return True
    if llm.urgency_signals.urgency in ("high", "critical"):
        return True

    # Core communication usually expects a reply
    if category == "core_communication":
        return True

    return False


def _build_reply_context(parsed_email: ParsedEmail, llm: LLMTriageOutput) -> str:
    """Build the context payload for the reply model.

    Passes the original email + triage classification so the reply model
    can generate an informed draft without re-analyzing the email.
    """
    body = (parsed_email.body_text or "").strip()
    if len(body) > 4000:  # shorter limit for reply context
        body = body[:4000] + "\n...[truncated]"

    context = {
        "original_email": {
            "from_name": parsed_email.from_name,
            "from_email": parsed_email.from_email,
            "to": parsed_email.to,
            "subject": parsed_email.subject,
            "body": body,
            "sent_at": parsed_email.sent_at.isoformat() if parsed_email.sent_at else None,
        },
        "triage": {
            "major_category": llm.major_category.value,
            "sub_action_key": llm.sub_action_key,
            "urgency": llm.urgency_signals.urgency,
            "ask": llm.extracted_summary.ask,
            "success_criteria": llm.extracted_summary.success_criteria,
        },
    }
    return json.dumps(context, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Container for a pipeline response + optional provider metadata."""
    response: Any
    model_info: Dict[str, Any]


class GmailTriagePipeline:
    """Primary pipeline entrypoint for Gmail messages.

    Two-model architecture:
    - Triage client (Qwen 2.5 72B via Ollama) for classification + extraction
    - Reply client (DeepSeek-V3.2 via Ollama) for draft reply generation
    - Fallback client (Anthropic Claude) if Ollama is unavailable
    """

    def __init__(self):
        self._triage_client = _build_triage_client()
        self._reply_client = _build_reply_client()
        # Only build fallback if primary is Ollama (otherwise it IS Anthropic)
        if settings.llm_provider == "ollama":
            self._fallback_client = _build_fallback_client()
        else:
            self._fallback_client = None

    def _call_with_fallback(self, primary_client, **kwargs):
        """Try the primary client; fall back to Anthropic if Ollama is down."""
        try:
            return primary_client.parse(**kwargs)
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError, OSError) as exc:
            if self._fallback_client is None:
                raise
            logger.warning(
                "Primary LLM unavailable (%s), falling back to Anthropic: %s",
                type(primary_client).__name__,
                exc,
            )
            return self._fallback_client.parse(**kwargs)

    def _build_user_content(self, email: ParsedEmail, gate: Optional[GateResult] = None) -> str:
        """Create a compact, model-friendly JSON string from a parsed email.

        If the gate returned an ambiguous verdict, we include the gate signals
        as extra context to help the model make a better decision.
        """
        body = (email.body_text or "").strip()
        if len(body) > settings.max_body_chars:
            body = body[: settings.max_body_chars] + "\n...[truncated]"

        time_phrases = extract_time_expressions(body)
        links = extract_links(email.body_html or body)
        money = extract_money_expressions(body)
        has_unsub = extract_unsubscribe_signals(body)

        signals: Dict[str, Any] = {
            "timePhrases": time_phrases,
            "links": links[:10],
            "moneyStrings": money[:10],
            "hasUnsubscribeSignal": has_unsub,
        }

        # If the gate found some spam signals but wasn't confident enough
        # to skip the LLM, pass them through so the model can use them.
        if gate and gate.verdict == "ambiguous":
            signals["gateVerdict"] = "ambiguous"
            signals["gateScore"] = gate.score
            signals["gateFiredSignals"] = [s.name for s in gate.fired_signals]

        payload = {
            "provider": email.provider,
            "messageId": email.message_id,
            "threadId": email.thread_id,
            "subject": email.subject,
            "from": {"name": email.from_name, "email": email.from_email},
            "to": email.to,
            "cc": email.cc,
            "sentAt": email.sent_at.isoformat() if email.sent_at else None,
            "snippet": email.snippet,
            "bodyText": body,
            "signals": signals,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _generate_reply(self, parsed_email: ParsedEmail, llm: LLMTriageOutput) -> Optional[DraftReply]:
        """Generate a draft reply using the reply model.

        Returns None if reply generation fails (non-fatal — triage still succeeds).
        """
        try:
            reply_context = _build_reply_context(parsed_email, llm)
            result = self._call_with_fallback(
                self._reply_client,
                system_prompt=REPLY_SYSTEM_PROMPT,
                user_content=reply_context,
                output_model=LLMReplyOutput,
            )
            reply_output: LLMReplyOutput = result.parsed
            return DraftReply(
                subject=reply_output.subject,
                body=reply_output.body,
                tone=reply_output.tone,
                confidence=max(0.0, min(1.0, reply_output.confidence)),
            )
        except Exception as exc:
            logger.warning("Reply generation failed (non-fatal): %s", exc)
            return None

    def triage(self, msg: GmailMessageInput) -> PipelineResult:
        """Run the triage pipeline -> TriageResponse.

        Flow:
          1. Parse the raw Gmail message
          2. Run the pre-LLM gate
          3. If gate says "spam" with high confidence -> build response, skip LLM
          4. Otherwise -> triage with primary model (fallback to Anthropic)
          5. Postprocess the triage output
          6. If reply is needed and enabled -> draft reply with reply model
          7. Return combined response
        """
        parsed = parse_gmail_message(msg)

        # --- Step 2: Run the gate ---
        gate = run_gate(
            parsed_email=parsed,
            raw_headers=msg.payload.headers or [],
            body_html=parsed.body_html,
        )

        predicted_at = datetime.now(timezone.utc)

        # --- Step 3: Short-circuit for obvious spam ---
        if gate.skip_llm:
            resp = _build_spam_response(parsed, gate, predicted_at)
            return PipelineResult(
                response=resp,
                model_info={
                    "provider": "gate",
                    "model": "deterministic",
                    "gate_verdict": gate.verdict,
                    "gate_score": gate.score,
                },
            )

        # --- Step 4: Triage with primary model (fallback to Anthropic) ---
        user_content = self._build_user_content(parsed, gate)

        result = self._call_with_fallback(
            self._triage_client,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_content=user_content,
            output_model=LLMTriageOutput,
        )

        # --- Step 5: Postprocess ---
        resp: TriageResponse = postprocess_triage(
            msg=msg,
            parsed_email=parsed,
            llm=result.parsed,
            predicted_at=predicted_at,
            gate_result=gate,
        )

        # --- Step 6: Draft reply if needed ---
        draft_reply = None
        if settings.reply_enabled and _needs_reply(result.parsed):
            draft_reply = self._generate_reply(parsed, result.parsed)

        # Attach the draft reply to the response
        resp.output.draft_reply = draft_reply

        # Build combined model info
        model_info = result.model_info
        if draft_reply is not None:
            model_info["reply_model"] = settings.ollama_reply_model if settings.llm_provider == "ollama" else settings.anthropic_model

        return PipelineResult(response=resp, model_info=model_info)
