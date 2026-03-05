"""FastAPI app for the Drakko Gmail Triage AI server.

- A Node service polls Gmail every ~5 minutes and stores new messages in a DB.
- A trigger / queue job fires for each new record.
- A worker calls this AI server with the raw Gmail JSON message.
- This AI server returns a strict JSON response used for UI + downstream automation.

For V3, I have updated ENDPOINTS

  GET  /rd/api/v1/apidata       -> API metadata and endpoint listing
  GET  /rd/api/v1/health        -> Health diagnostics
  GET  /rd/api/v1/ai            -> AI model configuration and stats
  POST /rd/api/v1/ai/triage     -> Full email triage (single large output)

"""

from __future__ import annotations

import logging
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import GmailMessageInput, MajorCategory, TriageResponse
from app.pipeline import GmailTriagePipeline
from app.prompt import PROMPT_VERSION, REPLY_PROMPT_VERSION


logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Server",
    version=settings.api_version,
    description="Email classification, extraction, other returned features on GMAIL"
)

# CORS — allow frontend origins to call the API from the browser.
# allow_origin_regex covers Render preview URLs (e.g. drakko-server-pr-42.onrender.com)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://drakko-server.onrender.com",
        "https://drakko-web-1.onrender.com",
        "https://drakko.ai",
        "https://www.drakko.ai",
        "http://localhost:3000",   # local frontend dev
        "http://localhost:5173",   # Vite default
    ],
    allow_origin_regex=r"^https://(drakko-server|drakko-web-1)(-[a-zA-Z0-9-]+)?\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = GmailTriagePipeline()


# Track server startup time for health diagnostics.
_started_at = datetime.now(timezone.utc)
_start_monotonic = time.monotonic()

# Track recent errors for health endpoint (ring buffer, last 50).
_recent_errors: List[Dict[str, Any]] = []
_MAX_ERRORS = 50

# Track request counts.
_request_counts: Dict[str, int] = {
    "triage": 0,
    "total": 0,
}


def _record_error(endpoint: str, error: str) -> None:
    """Record an error for the health endpoint."""
    _recent_errors.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "error": str(error)[:500],
    })
    while len(_recent_errors) > _MAX_ERRORS:
        _recent_errors.pop(0)

# ALL ENDPOINTS< ADD HERE

@app.get("/rd/api/v1/apidata")
def rd_apidata():
    """Returns full API information: name, version, endpoints, schema, contract."""
    return {
        "name": "Drakko Email AI Server",
        "version": settings.api_version,
        "description": "Email triage AI service — classifies, extracts, and recommends actions for Gmail messages.",
        "schema_version": settings.schema_version,
        "contract_reference": settings.contract_reference,
        "endpoints": [
            {
                "method": "POST",
                "path": "/rd/api/v1/ai/triage",
                "description": "Full email triage — returns the complete output object from classification through debug metadata.",
            },
            {
                "method": "GET",
                "path": "/rd/api/v1/apidata",
                "description": "API metadata, version, and endpoint listing.",
            },
            {
                "method": "GET",
                "path": "/rd/api/v1/health",
                "description": "Health diagnostics: uptime, status, recent errors.",
            },
            {
                "method": "GET",
                "path": "/rd/api/v1/ai",
                "description": "AI model configuration, usage stats, and prompt versioning.",
            },
        ],
        "major_categories": [c.value for c in MajorCategory],
    }


@app.get("/rd/api/v1/health")
def rd_health():
    """Returns health diagnostics: status, uptime, recent errors, system info."""
    uptime_s = round(time.monotonic() - _start_monotonic, 2)

    # Determine overall status from recent errors.
    recent_window = [
        e for e in _recent_errors
        if (datetime.now(timezone.utc) - datetime.fromisoformat(e["timestamp"])).total_seconds() < 300
    ]
    if len(recent_window) >= 10:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "uptime_seconds": uptime_s,
        "started_at": _started_at.isoformat(),
        "checks": {
            "llm_provider": {
                "status": "ok",
                "provider": settings.llm_provider,
                "triage_model": settings.ollama_triage_model if settings.llm_provider == "ollama" else settings.anthropic_model,
                "reply_model": settings.ollama_reply_model if settings.llm_provider == "ollama" else settings.anthropic_model,
                "fallback": "anthropic" if settings.llm_provider == "ollama" else "none",
                "reply_enabled": settings.reply_enabled,
            },
            "python_version": platform.python_version(),
        },
        "request_counts": dict(_request_counts),
        "recent_errors": _recent_errors[-10:],
        "version": settings.api_version,
    }


@app.get("/rd/api/v1/ai")
def rd_ai():
    """Returns AI model configuration, prompt version, and usage stats."""
    is_ollama = settings.llm_provider == "ollama"
    return {
        "provider": settings.llm_provider,
        "triage_model": settings.ollama_triage_model if is_ollama else settings.anthropic_model,
        "reply_model": settings.ollama_reply_model if is_ollama else settings.anthropic_model,
        "fallback_model": settings.anthropic_model if is_ollama else None,
        "reply_enabled": settings.reply_enabled,
        "temperature": settings.temperature,
        "timeout_s": settings.timeout_s,
        "max_body_chars": settings.max_body_chars,
        "schema_version": settings.schema_version,
        "model_version": settings.model_version,
        "prompt_version": PROMPT_VERSION,
        "reply_prompt_version": REPLY_PROMPT_VERSION,
        "contract_reference": settings.contract_reference,
        "capabilities": [
            "email_triage",
            "entity_extraction",
            "urgency_detection",
            "task_proposal",
            "action_recommendation",
            "summary_extraction",
            "reply_drafting",
        ],
        "request_counts": dict(_request_counts),
    }


@app.post("/rd/api/v1/ai/triage", response_model=TriageResponse)
def rd_ai_triage(payload: GmailMessageInput):
    """Full email triage endpoint.

    Accepts a raw Gmail message JSON and returns the complete output:
    classification, task proposal, recommended actions, urgency signals,
    extracted summary, entities, evidence, and debug metadata — all under
    a single { "output": { ... } } envelope.
    """
    _request_counts["triage"] += 1
    _request_counts["total"] += 1

    msg_id = getattr(payload, "id", None) or "unknown"
    logger.info("[triage] request start msg_id=%s", msg_id)

    try:
        out = _pipeline.triage(payload)
        logger.info("[triage] success msg_id=%s", msg_id)
        return out.response
    except NotImplementedError as e:
        logger.exception("[triage] NotImplementedError msg_id=%s: %s", msg_id, e)
        _record_error("/rd/api/v1/ai/triage", str(e))
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception("[triage] error msg_id=%s: %s", msg_id, e)
        _record_error("/rd/api/v1/ai/triage", str(e))
        raise HTTPException(status_code=500, detail=f"failed_to_triage: {e}")