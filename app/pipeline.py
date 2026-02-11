"""
Pipeline orchestration.
GmailMessageInput -> Parse Gmail MIME -> Preprocess/Signals -> LLM -> Postprocess -> API response
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from app.config import settings
from app.gmail import parse_gmail_message
from app.llm.local_client import LocalClient
from app.llm.openai_client import OpenAIClient
from app.models import GmailMessageInput, LLMTriageOutput, TriageResponse
from app.postprocess import postprocess_triage
from app.preprocess import extract_links, extract_money_expressions, extract_time_expressions
from app.prompt import TRIAGE_SYSTEM_PROMPT


def _choose_client():
    """Select the LLM provider adapter based on env vars."""
    if settings.llm_provider == "openai":
        return OpenAIClient(
            model=settings.openai_model,
            temperature=settings.temperature,
            timeout_s=settings.timeout_s,
        )
    if settings.llm_provider == "local":
        return LocalClient()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


@dataclass
class PipelineResult:
    """Container for a pipeline response + optional provider metadata."""
    response: Any
    model_info: Dict[str, Any]


class GmailTriagePipeline:
    """Primary pipeline entrypoint for Gmail messages."""

    def __init__(self):
        self._client = _choose_client()

    def _build_user_content(self, email) -> str:
        """Create a compact, model-friendly JSON string from a parsed email."""
        body = (email.body_text or "").strip()
        if len(body) > settings.max_body_chars:
            body = body[: settings.max_body_chars] + "\n...[truncated]"

        time_phrases = extract_time_expressions(body)
        links = extract_links(email.body_html or body)
        money = extract_money_expressions(body)

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
            "signals": {
                "timePhrases": time_phrases,
                "links": links[:10],
                "moneyStrings": money[:10],
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def triage(self, msg: GmailMessageInput) -> PipelineResult:
        """Run the triage pipeline -> TriageResponse."""
        parsed = parse_gmail_message(msg)
        user_content = self._build_user_content(parsed)

        result = self._client.parse(
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_content=user_content,
            output_model=LLMTriageOutput,
        )

        predicted_at = datetime.now(timezone.utc)

        resp: TriageResponse = postprocess_triage(
            msg=msg,
            parsed_email=parsed,
            llm=result.parsed,
            predicted_at=predicted_at,
        )
        return PipelineResult(response=resp, model_info=result.model_info)
