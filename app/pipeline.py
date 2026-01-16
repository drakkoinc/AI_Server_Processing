"""Pipeline orchestration.

This module is the glue that turns:

    GmailMessageInput  →  Parse Gmail MIME  →  Preprocess/Signals  →  LLM  →  Postprocess  →  API response

Why a dedicated pipeline module?
------------------------------
Keeping orchestration separate from FastAPI makes it easy to:
  - unit test the pipeline without running an HTTP server
  - reuse the same logic in batch backfills
  - call the same logic from a queue worker or cron job

In this version, we produce a *single* output object (`EmailTriageResponse`) with:
  - major_category (semantic bucket)
  - sub_action_key (verb/action key)
  - reply_required / explicit_task
  - evidence + entities

This aligns with the "LLM as decision engine" mental model described in your PDFs:
  What is this email?  → major_category
  What should I do?    → sub_action_key + reply_required/explicit_task
  Why?                 → reason + evidence
  What objects matter? → entities
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
from app.models import EmailTriageResponse, GmailMessageInput, LLMTriageOutput
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
        """Create a compact, model-friendly JSON string.

        Why not pass the entire raw Gmail payload to the model?
        - Gmail messages can be large and noisy (headers, MIME boundaries, encoded blobs).
        - We want stable prompts and predictable token usage.
        - We want to provide only the *semantic* content needed to classify.

        We still include lightweight "signals" (time phrases, links, money strings)
        to help both the model and human debugging.
        """
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
        """Run the triage pipeline and return `EmailTriageResponse`."""
        parsed = parse_gmail_message(msg)
        user_content = self._build_user_content(parsed)

        result = self._client.parse(
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_content=user_content,
            output_model=LLMTriageOutput,
        )

        # Even though the user-requested output schema does not include timestamps,
        # it is still valuable to know when inference occurred.
        predicted_at = datetime.now(timezone.utc)

        resp: EmailTriageResponse = postprocess_triage(
            msg=msg,
            parsed_email=parsed,
            llm=result.parsed,
            predicted_at=predicted_at,
        )
        return PipelineResult(response=resp, model_info=result.model_info)
