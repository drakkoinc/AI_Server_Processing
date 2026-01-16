"""FastAPI app for the Drakko Gmail Triage AI server.

High-level deployment model (from your docs)
-------------------------------------------
- A Node service polls Gmail every ~5 minutes and stores new messages in a DB.
- A trigger / queue job fires for each new record.
- A worker calls this AI server with the raw Gmail JSON message.
- This AI server returns a strict JSON response used for UI + downstream automation.

In this version, the server exposes ONE primary endpoint:

  POST /v1/gmail/triage

It returns a JSON object with the exact shape requested by the user:
  message_id, thread_id, major_category, sub_action_key, reply_required,
  explicit_task, task_type, confidence, reason, evidence, entities

We also keep a compatibility alias:
  POST /v1/gmail/classify  (same behavior)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.models import EmailTriageResponse, GmailMessageInput
from app.pipeline import GmailTriagePipeline


app = FastAPI(title="Drakko Gmail Triage AI Server", version="2.0.0")

_pipeline = GmailTriagePipeline()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/gmail/triage", response_model=EmailTriageResponse)
def gmail_triage(payload: GmailMessageInput):
    """Primary triage endpoint."""
    try:
        out = _pipeline.triage(payload)
        return out.response
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed_to_triage: {e}")


# Backwards-compatible alias (same behavior).
@app.post("/v1/gmail/classify", response_model=EmailTriageResponse)
def gmail_classify(payload: GmailMessageInput):
    return gmail_triage(payload)
