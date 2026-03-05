"""
Full endpoint tests for the Drakko Email AI Server.

Tests all five v3 endpoints and shows the complete response shape:
  GET  /rd/api/v1/apidata
  GET  /rd/api/v1/health
  GET  /rd/api/v1/ai
  POST /rd/api/v1/ai/triage
  POST /rd/api/v1/ai/triage/batch

The triage endpoint is tested with a mocked pipeline so we can demonstrate
a full successful response without needing a real Anthropic API key.

Strategy:
  - We mock `anthropic.Anthropic` at the SDK level BEFORE `app.main` is imported.
  - `app.main` creates `_pipeline = GmailTriagePipeline()` at module-load time,
    which instantiates `AnthropicClient` → calls `Anthropic(timeout=...)`.
  - By patching `anthropic.Anthropic` first, the pipeline initializes with a fake client.
  - Then we replace the pipeline's `_client.parse` to return our mock LLM output,
    so the full postprocess pipeline runs deterministically.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models import (
    BatchTriageItemResponse,
    Entities,
    ExtractedSummary,
    LLMTriageOutput,
    MajorCategory,
    PersonRef,
    DateRef,
    MeetingRef,
    RecommendedAction,
    TaskProposal,
    TriageResponse,
    UrgencySignals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GMAIL_INPUT = {
    "id": "18c8a3f1b2d0a9a1",
    "threadId": "18c89f3d1a2b4c55",
    "labelIds": ["INBOX", "IMPORTANT"],
    "snippet": "Can you confirm Friday at 2pm PT works?",
    "internalDate": "1706500000000",
    "payload": {
        "mimeType": "multipart/alternative",
        "headers": [
            {"name": "From", "value": "Sarah Chen <sarah@acme.com>"},
            {"name": "To", "value": "hani@drakko.io"},
            {"name": "Subject", "value": "Re: Contract approval timeline"},
            {"name": "Date", "value": "Mon, 29 Jan 2024 10:00:00 -0800"},
        ],
        "body": {"size": 0},
        "parts": [
            {
                "partId": "0",
                "mimeType": "text/plain",
                "headers": [],
                "body": {
                    "size": 120,
                    "data": "SGkgSGFuaSxcblxuQ2FuIHlvdSBjb25maXJtIEZyaWRheSBhdCAycG0gUFQgd29ya3MgZm9yIHRoZSBjb250cmFjdCByZXZpZXc_XG5cblRoYW5rcyxcblNhcmFo",
                },
            }
        ],
    },
}


def _mock_llm_output() -> LLMTriageOutput:
    """Return a realistic LLMTriageOutput for testing."""
    return LLMTriageOutput(
        major_category=MajorCategory.schedule_and_time,
        sub_action_key="SCHEDULE_CONFIRM_TIME",
        explicit_task=False,
        confidence=0.91,
        suggested_reply_action=[
            "Yes, Friday at 2 PM PT works for me.",
            "Can we push to Monday instead?",
        ],
        task_proposal=TaskProposal(
            type="confirm_meeting",
            title="Confirm meeting time with Sarah",
            description="Sender asks you to confirm Friday 2 PM PT for the contract review.",
            priority="high",
            status="open",
            scheduled_for=None,
            due_at=None,
            waiting_on=None,
        ),
        recommended_actions=[
            RecommendedAction(key="reply_confirm", label="Confirm Time", kind="PRIMARY", rank=1),
            RecommendedAction(key="reply_reschedule", label="Propose New Time", kind="SECONDARY", rank=2),
            RecommendedAction(key="snooze_1h", label="Snooze 1 Hour", kind="SECONDARY", rank=3),
        ],
        urgency_signals=UrgencySignals(
            urgency="high",
            deadline_detected=True,
            deadline_text="Friday 2pm PT",
            reply_by=None,
            reason="Sender is waiting on your confirmation for a scheduled meeting.",
        ),
        extracted_summary=ExtractedSummary(
            ask="Confirm whether Friday at 2 PM PT works for the contract review meeting.",
            success_criteria="Reply confirming or proposing an alternative time.",
            missing_info=[],
        ),
        entities=Entities(
            people=[],
            dates=[DateRef(text="Friday 2pm PT", iso=None, type="meeting_time")],
            money=[],
            docs=[],
            meeting=None,
        ),
        evidence=["Can you confirm Friday at 2pm PT works?"],
    )


@pytest.fixture
def client():
    """Create a TestClient with the pipeline mocked so no Anthropic key is needed.

    The trick: we mock `anthropic.Anthropic` (the SDK constructor) BEFORE importing
    `app.main`, because `app.main` creates the pipeline at module-level which
    calls `Anthropic(timeout=...)` during import.

    Steps:
      1. Remove any cached `app.main` and `app.pipeline` from sys.modules
         so they can be re-imported cleanly under the mock.
      2. Patch `anthropic.Anthropic` so the constructor returns a MagicMock.
      3. Import `app.main` — the pipeline initializes with the fake Anthropic.
      4. Replace `_pipeline._client.parse` to return our mock LLM output,
         routing through the real postprocessor for deterministic results.
      5. Yield a TestClient wrapping the FastAPI app.
      6. Clean up sys.modules so subsequent test sessions start fresh.
    """
    # --- Step 1: Remove cached modules so we get a fresh import under our mock
    modules_to_clear = [
        key for key in sys.modules
        if key == "app.main" or key == "app.pipeline"
    ]
    saved_modules = {}
    for key in modules_to_clear:
        saved_modules[key] = sys.modules.pop(key)

    # --- Step 2: Patch anthropic.Anthropic before app.main loads
    mock_anthropic_constructor = MagicMock()
    mock_anthropic_instance = MagicMock()
    mock_anthropic_constructor.return_value = mock_anthropic_instance

    with patch("anthropic.Anthropic", mock_anthropic_constructor):
        # Also patch it in the anthropic_client module (import cache)
        with patch.dict(sys.modules, {}, {}):
            pass  # just ensure clean state
        # Patch at the location where anthropic_client imports it
        with patch("app.llm.anthropic_client.anthropic.Anthropic", mock_anthropic_constructor):
            # --- Step 3: Import app.main fresh — pipeline initializes with our mock
            import app.main as main_module
            importlib.reload(main_module)
            test_app = main_module.app

            # --- Step 4: Wire up the mock parse to return our LLM output
            mock_result = MagicMock()
            mock_result.parsed = _mock_llm_output()
            mock_result.model_info = {
                "provider": "anthropic",
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 1420,
                    "output_tokens": 380,
                },
                "response_id": "msg_mock_test_12345",
            }

            # The pipeline's internal client is the AnthropicClient instance;
            # its _client attribute is our mocked anthropic.Anthropic instance.
            # But we need to mock at the AnthropicClient.parse level instead,
            # since that's what the pipeline calls.
            # Access: main_module._pipeline._client is the AnthropicClient,
            #         and AnthropicClient.parse is the method we need to mock.
            main_module._pipeline._client.parse = MagicMock(return_value=mock_result)

            # --- Step 5: Yield the test client
            yield TestClient(test_app)

    # --- Step 6: Restore modules to avoid polluting other tests
    for key in modules_to_clear:
        if key in saved_modules:
            sys.modules[key] = saved_modules[key]
        elif key in sys.modules:
            del sys.modules[key]


# ===================================================================
# 1. GET /rd/api/v1/apidata
# ===================================================================

class TestApiData:
    """Tests for the API metadata endpoint."""

    def test_apidata_returns_200(self, client):
        r = client.get("/rd/api/v1/apidata")
        assert r.status_code == 200

    def test_apidata_contains_version_and_schema(self, client):
        data = client.get("/rd/api/v1/apidata").json()
        assert data["version"] == "3.0.0"
        assert data["schema_version"] == "v3"
        assert data["contract_reference"] == "drakko.gmail_insights.v3"

    def test_apidata_lists_all_endpoints(self, client):
        data = client.get("/rd/api/v1/apidata").json()
        paths = [ep["path"] for ep in data["endpoints"]]
        assert "/rd/api/v1/ai/triage" in paths
        assert "/rd/api/v1/ai/triage/batch" in paths
        assert "/rd/api/v1/apidata" in paths
        assert "/rd/api/v1/health" in paths
        assert "/rd/api/v1/ai" in paths

    def test_apidata_lists_all_12_major_categories(self, client):
        data = client.get("/rd/api/v1/apidata").json()
        categories = data["major_categories"]
        assert len(categories) == 12
        assert "core_communication" in categories
        assert "schedule_and_time" in categories
        assert "meta_and_systems" in categories
        assert "spam" in categories
        assert "other" in categories

    def test_apidata_full_response_shape(self, client):
        """Show the complete response shape for /rd/api/v1/apidata."""
        data = client.get("/rd/api/v1/apidata").json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "schema_version" in data
        assert "contract_reference" in data
        assert "endpoints" in data
        assert "major_categories" in data
        # Each endpoint entry has method, path, description
        for ep in data["endpoints"]:
            assert "method" in ep
            assert "path" in ep
            assert "description" in ep


# ===================================================================
# 2. GET /rd/api/v1/health
# ===================================================================

class TestHealth:
    """Tests for the health diagnostics endpoint."""

    def test_health_returns_200(self, client):
        r = client.get("/rd/api/v1/health")
        assert r.status_code == 200

    def test_health_reports_healthy_on_startup(self, client):
        data = client.get("/rd/api/v1/health").json()
        assert data["status"] == "healthy"

    def test_health_contains_uptime(self, client):
        data = client.get("/rd/api/v1/health").json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_contains_checks(self, client):
        data = client.get("/rd/api/v1/health").json()
        assert "checks" in data
        assert "llm_provider" in data["checks"]
        assert data["checks"]["llm_provider"]["provider"] == "anthropic"
        assert "python_version" in data["checks"]

    def test_health_contains_request_counts(self, client):
        data = client.get("/rd/api/v1/health").json()
        assert "request_counts" in data
        assert "triage" in data["request_counts"]
        assert "total" in data["request_counts"]

    def test_health_full_response_shape(self, client):
        """Show the complete response shape for /rd/api/v1/health."""
        data = client.get("/rd/api/v1/health").json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "started_at" in data
        assert "checks" in data
        assert "request_counts" in data
        assert "recent_errors" in data
        assert "version" in data
        assert isinstance(data["recent_errors"], list)


# ===================================================================
# 3. GET /rd/api/v1/ai
# ===================================================================

class TestAI:
    """Tests for the AI model configuration endpoint."""

    def test_ai_returns_200(self, client):
        r = client.get("/rd/api/v1/ai")
        assert r.status_code == 200

    def test_ai_contains_model_config(self, client):
        data = client.get("/rd/api/v1/ai").json()
        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-opus-4-6"
        assert data["temperature"] == 0.2
        assert data["timeout_s"] == 90.0

    def test_ai_contains_versioning(self, client):
        data = client.get("/rd/api/v1/ai").json()
        assert data["schema_version"] == "v3"
        assert data["model_version"] == "drakko-email-v3"
        assert data["prompt_version"] == "triage-v3-2026-02"
        assert data["contract_reference"] == "drakko.gmail_insights.v3"

    def test_ai_lists_all_capabilities(self, client):
        data = client.get("/rd/api/v1/ai").json()
        caps = data["capabilities"]
        assert "email_triage" in caps
        assert "entity_extraction" in caps
        assert "urgency_detection" in caps
        assert "task_proposal" in caps
        assert "action_recommendation" in caps
        assert "summary_extraction" in caps

    def test_ai_full_response_shape(self, client):
        """Show the complete response shape for /rd/api/v1/ai."""
        data = client.get("/rd/api/v1/ai").json()
        assert "provider" in data
        assert "model" in data
        assert "temperature" in data
        assert "timeout_s" in data
        assert "max_body_chars" in data
        assert "schema_version" in data
        assert "model_version" in data
        assert "prompt_version" in data
        assert "contract_reference" in data
        assert "capabilities" in data
        assert "request_counts" in data


# ===================================================================
# 4. POST /rd/api/v1/ai/triage — Success
# ===================================================================

class TestTriageSuccess:
    """Tests for a successful triage response showing the full output shape."""

    def test_triage_returns_200(self, client):
        r = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT)
        assert r.status_code == 200

    def test_triage_response_has_output_envelope(self, client):
        data = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()
        assert "output" in data
        assert isinstance(data["output"], dict)

    def test_triage_classification_fields(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        assert output["major_category"] == "schedule_and_time"
        assert output["sub_action_key"] == "SCHEDULE_CONFIRM_TIME"
        assert output["explicit_task"] is False
        assert 0.0 <= output["confidence"] <= 1.0

    def test_triage_suggested_reply_actions(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        replies = output["suggested_reply_action"]
        assert isinstance(replies, list)
        assert len(replies) == 2
        assert all(isinstance(r, str) for r in replies)

    def test_triage_task_proposal(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        tp = output["task_proposal"]
        assert tp is not None
        assert tp["type"] == "confirm_meeting"
        assert tp["title"] == "Confirm meeting time with Sarah"
        assert tp["priority"] == "high"
        assert tp["status"] == "open"
        # due_at should be filled from urgency deadline by postprocessor
        assert "description" in tp
        assert "scheduled_for" in tp
        assert "due_at" in tp
        assert "waiting_on" in tp

    def test_triage_recommended_actions(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        actions = output["recommended_actions"]
        assert isinstance(actions, list)
        assert len(actions) == 3
        # Verify structure and sequential ranking
        for i, action in enumerate(actions):
            assert "key" in action
            assert "label" in action
            assert "kind" in action
            assert action["rank"] == i + 1
            assert action["kind"] in ("PRIMARY", "SECONDARY", "DANGER")
        # First action should be PRIMARY
        assert actions[0]["kind"] == "PRIMARY"
        assert actions[0]["key"] == "reply_confirm"

    def test_triage_urgency_signals(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        urg = output["urgency_signals"]
        assert urg["urgency"] == "high"
        assert urg["deadline_detected"] is True
        assert urg["deadline_text"] == "Friday 2pm PT"
        # reply_by should be filled by postprocessor from deadline_text
        assert urg["reply_by"] is not None
        assert "reason" in urg

    def test_triage_extracted_summary(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        summary = output["extracted_summary"]
        assert summary["ask"] != ""
        assert summary["success_criteria"] != ""
        assert isinstance(summary["missing_info"], list)

    def test_triage_entities(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        entities = output["entities"]

        # People: sender should be backfilled by postprocessor
        assert any(
            p["email"] == "sarah@acme.com" and p["role"] == "sender"
            for p in entities["people"]
        )

        # Dates: ISO should be filled from "Friday 2pm PT"
        assert len(entities["dates"]) >= 1
        assert entities["dates"][0]["text"] == "Friday 2pm PT"
        assert entities["dates"][0]["iso"] is not None
        assert entities["dates"][0]["type"] == "meeting_time"

        # Meeting: should be created for schedule_and_time
        assert entities["meeting"] is not None
        assert entities["meeting"]["topic"] == "Contract approval timeline"
        assert entities["meeting"]["start_at"] is not None

        # Lists present even if empty
        assert isinstance(entities["money"], list)
        assert isinstance(entities["docs"], list)

    def test_triage_evidence(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        evidence = output["evidence"]
        assert isinstance(evidence, list)
        assert len(evidence) >= 1
        assert all(isinstance(e, str) for e in evidence)

    def test_triage_debug_metadata(self, client):
        output = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()["output"]
        debug = output["debug"]
        assert debug["analysis_timestamp"] != ""
        assert debug["model_version"] == "drakko-email-v3"
        assert debug["prompt_version"] == "triage-v3-2026-02"

    def test_triage_validates_against_pydantic_schema(self, client):
        """The full response should validate cleanly against TriageResponse."""
        data = client.post("/rd/api/v1/ai/triage", json=SAMPLE_GMAIL_INPUT).json()
        validated = TriageResponse.model_validate(data)
        assert validated.output.major_category == MajorCategory.schedule_and_time


# ===================================================================
# 5. POST /rd/api/v1/ai/triage — Error Cases
# ===================================================================

class TestTriageErrors:
    """Tests for triage endpoint error handling."""

    def test_empty_body_returns_422(self, client):
        r = client.post("/rd/api/v1/ai/triage", content=b"", headers={"Content-Type": "application/json"})
        assert r.status_code == 422
        errors = r.json()["detail"]
        assert any(e["loc"] == ["body"] for e in errors)

    def test_missing_id_returns_422(self, client):
        r = client.post("/rd/api/v1/ai/triage", json={"payload": {"mimeType": "text/plain"}})
        assert r.status_code == 422
        errors = r.json()["detail"]
        locs = [e["loc"] for e in errors]
        assert ["body", "id"] in locs

    def test_missing_payload_returns_422(self, client):
        r = client.post("/rd/api/v1/ai/triage", json={"id": "abc123"})
        assert r.status_code == 422
        errors = r.json()["detail"]
        locs = [e["loc"] for e in errors]
        assert ["body", "payload"] in locs

    def test_wrong_content_type_returns_422(self, client):
        r = client.post("/rd/api/v1/ai/triage", content=b"hello", headers={"Content-Type": "text/plain"})
        assert r.status_code == 422

    def test_completely_invalid_json_returns_422(self, client):
        r = client.post("/rd/api/v1/ai/triage", json={"bad": "data"})
        assert r.status_code == 422
        errors = r.json()["detail"]
        # Should report both missing 'id' and missing 'payload'
        missing_fields = [e["loc"][-1] for e in errors if e["type"] == "missing"]
        assert "id" in missing_fields
        assert "payload" in missing_fields

    def test_422_error_response_shape(self, client):
        """Show the complete validation error response shape."""
        r = client.post("/rd/api/v1/ai/triage", json={})
        assert r.status_code == 422
        data = r.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        for error in data["detail"]:
            assert "type" in error
            assert "loc" in error
            assert "msg" in error
            assert "input" in error


# ===================================================================
# 6. POST /rd/api/v1/ai/triage/batch — Batch Triage
# ===================================================================

# A second sample email to test batch with multiple distinct inputs.
SAMPLE_GMAIL_INPUT_2 = {
    "id": "18c8b4e2c3d1b0b2",
    "threadId": "18c8b4e2c3d1b0b2",
    "labelIds": ["INBOX"],
    "snippet": "Your AWS bill for February is ready.",
    "internalDate": "1706600000000",
    "payload": {
        "mimeType": "text/plain",
        "headers": [
            {"name": "From", "value": "billing@aws.amazon.com"},
            {"name": "To", "value": "hani@drakko.io"},
            {"name": "Subject", "value": "Your AWS bill for February 2024"},
            {"name": "Date", "value": "Tue, 30 Jan 2024 08:00:00 -0800"},
        ],
        "body": {
            "size": 80,
            "data": "WW91ciBBV1MgYmlsbCBmb3IgRmVicnVhcnkgMjAyNCBpcyAkMSwyMzQuNTYuIFZpZXcgeW91ciBiaWxsIGF0IGh0dHBzOi8vY29uc29sZS5hd3MuYW1hem9uLmNvbS9iaWxsaW5n",
        },
    },
}


class TestBatchTriage:
    """Tests for the batch triage endpoint (streaming NDJSON)."""

    def test_batch_returns_200(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT]},
        )
        assert r.status_code == 200

    def test_batch_content_type_is_ndjson(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT]},
        )
        assert "application/x-ndjson" in r.headers["content-type"]

    def test_batch_single_email_returns_one_line(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT]},
        )
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        assert len(lines) == 1

    def test_batch_single_email_parses_as_valid_item(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT]},
        )
        import json
        line = r.text.strip().split("\n")[0]
        item = json.loads(line)
        assert item["message_id"] == "18c8a3f1b2d0a9a1"
        assert item["index"] == 0
        assert item["status"] == "success"
        assert item["output"] is not None
        assert item["error"] is None

    def test_batch_single_email_output_has_triage_fields(self, client):
        import json
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT]},
        )
        item = json.loads(r.text.strip().split("\n")[0])
        output = item["output"]
        assert "major_category" in output
        assert "sub_action_key" in output
        assert "urgency_signals" in output
        assert "extracted_summary" in output
        assert "debug" in output

    def test_batch_multiple_emails_returns_multiple_lines(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT, SAMPLE_GMAIL_INPUT_2]},
        )
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        assert len(lines) == 2

    def test_batch_multiple_emails_sequential_indexes(self, client):
        import json
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT, SAMPLE_GMAIL_INPUT_2]},
        )
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        items = [json.loads(l) for l in lines]
        assert items[0]["index"] == 0
        assert items[1]["index"] == 1
        assert items[0]["message_id"] == "18c8a3f1b2d0a9a1"
        assert items[1]["message_id"] == "18c8b4e2c3d1b0b2"

    def test_batch_each_item_validates_against_schema(self, client):
        import json
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [SAMPLE_GMAIL_INPUT, SAMPLE_GMAIL_INPUT_2]},
        )
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        for line in lines:
            item = json.loads(line)
            validated = BatchTriageItemResponse.model_validate(item)
            assert validated.status == "success"
            assert validated.output is not None

    def test_batch_empty_array_returns_empty(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": []},
        )
        assert r.status_code == 200
        assert r.text.strip() == ""

    def test_batch_missing_messages_returns_422(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={},
        )
        assert r.status_code == 422

    def test_batch_invalid_email_in_array_returns_422(self, client):
        r = client.post(
            "/rd/api/v1/ai/triage/batch",
            json={"messages": [{"bad": "data"}]},
        )
        assert r.status_code == 422