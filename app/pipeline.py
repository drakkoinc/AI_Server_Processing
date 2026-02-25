"""
Pipeline orchestration.
GmailMessageInput -> Parse Gmail MIME -> Gate -> (LLM if needed) -> Postprocess -> API response

The pre-LLM gate runs deterministic checks on headers and body content to catch
obvious spam/automated emails without burning an LLM call.  Only ambiguous or
clearly non-spam emails are forwarded to Claude for full triage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings
from app.gate import GateResult, run_gate
from app.gmail import parse_gmail_message, ParsedEmail
from app.llm.anthropic_client import AnthropicClient
from app.llm.local_client import LocalClient
from app.models import (
    DebugInfo,
    Entities,
    ExtractedSummary,
    GateDebug,
    GmailMessageInput,
    LLMTriageOutput,
    MajorCategory,
    RecommendedAction,
    TriageOutput,
    TriageResponse,
    UrgencySignals,
)
from app.postprocess import postprocess_triage
from app.preprocess import extract_links, extract_money_expressions, extract_time_expressions, extract_unsubscribe_signals
from app.prompt import PROMPT_VERSION, TRIAGE_SYSTEM_PROMPT


def _choose_client():
    """Select the LLM provider adapter based on env vars."""
    if settings.llm_provider == "anthropic":
        return AnthropicClient(
            model=settings.anthropic_model,
            temperature=settings.temperature,
            timeout_s=settings.timeout_s,
        )
    if settings.llm_provider == "local":
        return LocalClient()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


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
    """Build a complete TriageResponse for gate-caught spam without calling the LLM.

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
        debug=DebugInfo(
            analysis_timestamp=predicted_at.isoformat(),
            model_version=settings.model_version,
            prompt_version=PROMPT_VERSION,
            gate=_build_gate_debug(gate),
        ),
    )

    return TriageResponse(output=output)


@dataclass
class PipelineResult:
    """Container for a pipeline response + optional provider metadata."""
    response: Any
    model_info: Dict[str, Any]


class GmailTriagePipeline:
    """Primary pipeline entrypoint for Gmail messages."""

    def __init__(self):
        self._client = _choose_client()

    def _build_user_content(self, email: ParsedEmail, gate: Optional[GateResult] = None) -> str:
        """Create a compact, model-friendly JSON string from a parsed email.

        If the gate returned an ambiguous verdict, we include the gate signals
        as extra context to help Claude make a better decision.
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
        # to skip the LLM, pass them through so Claude can use them.
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

    def triage(self, msg: GmailMessageInput) -> PipelineResult:
        """Run the triage pipeline -> TriageResponse.

        Flow:
          1. Parse the raw Gmail message
          2. Run the pre-LLM gate
          3. If gate says "spam" with high confidence → build response, skip LLM
          4. Otherwise → send to Claude (with gate signals if ambiguous)
          5. Postprocess the LLM output
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

        # --- Step 4: Send to LLM (with gate context if ambiguous) ---
        user_content = self._build_user_content(parsed, gate)

        result = self._client.parse(
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
        return PipelineResult(response=resp, model_info=result.model_info)
