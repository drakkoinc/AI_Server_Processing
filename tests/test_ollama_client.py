"""
Unit tests for OllamaClient.

Tests the Ollama HTTP adapter in isolation using mocked httpx calls.
Covers: successful parse, JSON extraction, code fence stripping,
error handling, and response metadata.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field

from app.llm.ollama_client import OllamaClient, OllamaResult, _extract_json


# ---------------------------------------------------------------------------
# Test model (simpler than LLMTriageOutput for focused unit testing)
# ---------------------------------------------------------------------------

class SimpleOutput(BaseModel):
    category: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# _extract_json helper tests
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Tests for the _extract_json code fence stripping helper."""

    def test_plain_json_passthrough(self):
        raw = '{"category": "spam", "confidence": 0.95, "tags": []}'
        assert _extract_json(raw) == raw

    def test_strips_json_code_fence(self):
        raw = '```json\n{"category": "spam", "confidence": 0.95, "tags": []}\n```'
        expected = '{"category": "spam", "confidence": 0.95, "tags": []}'
        assert _extract_json(raw) == expected

    def test_strips_bare_code_fence(self):
        raw = '```\n{"category": "spam"}\n```'
        expected = '{"category": "spam"}'
        assert _extract_json(raw) == expected

    def test_strips_whitespace(self):
        raw = '  \n {"category": "spam"} \n  '
        expected = '{"category": "spam"}'
        assert _extract_json(raw) == expected

    def test_multiline_json_in_fence(self):
        raw = '```json\n{\n  "category": "spam",\n  "confidence": 0.9,\n  "tags": ["bulk"]\n}\n```'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["category"] == "spam"
        assert parsed["tags"] == ["bulk"]


# ---------------------------------------------------------------------------
# OllamaClient.parse tests
# ---------------------------------------------------------------------------

class TestOllamaClientParse:
    """Tests for OllamaClient.parse() with mocked httpx."""

    def _make_ollama_response(
        self,
        content: str,
        prompt_eval_count: int = 500,
        eval_count: int = 200,
        total_duration: int = 3_000_000_000,  # 3 seconds in nanoseconds
    ) -> dict:
        """Build a mock Ollama /api/chat response."""
        return {
            "model": "qwen2.5:72b",
            "message": {
                "role": "assistant",
                "content": content,
            },
            "done": True,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration": total_duration,
        }

    @patch("app.llm.ollama_client.httpx.Client")
    def test_successful_parse(self, mock_client_cls):
        """Test a successful parse with valid JSON response."""
        response_json = json.dumps({
            "category": "spam",
            "confidence": 0.95,
            "tags": ["bulk", "marketing"],
        })
        ollama_resp = self._make_ollama_response(response_json)

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        result = client.parse(
            system_prompt="You are a classifier.",
            user_content='{"text": "Buy now!"}',
            output_model=SimpleOutput,
        )

        assert isinstance(result, OllamaResult)
        assert result.parsed.category == "spam"
        assert result.parsed.confidence == 0.95
        assert result.parsed.tags == ["bulk", "marketing"]

    @patch("app.llm.ollama_client.httpx.Client")
    def test_model_info_populated(self, mock_client_cls):
        """Test that model_info contains provider, model, usage, and timing."""
        response_json = json.dumps({
            "category": "inbox",
            "confidence": 0.8,
            "tags": [],
        })
        ollama_resp = self._make_ollama_response(
            response_json,
            prompt_eval_count=1000,
            eval_count=300,
            total_duration=5_000_000_000,
        )

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        result = client.parse(
            system_prompt="test",
            user_content="test",
            output_model=SimpleOutput,
        )

        assert result.model_info["provider"] == "ollama"
        assert result.model_info["model"] == "qwen2.5:72b"
        assert result.model_info["usage"]["input_tokens"] == 1000
        assert result.model_info["usage"]["output_tokens"] == 300
        assert result.model_info["total_duration_ms"] == 5000.0

    @patch("app.llm.ollama_client.httpx.Client")
    def test_handles_code_fenced_json(self, mock_client_cls):
        """Test that JSON wrapped in code fences is parsed correctly."""
        fenced = '```json\n{"category": "inbox", "confidence": 0.7, "tags": ["personal"]}\n```'
        ollama_resp = self._make_ollama_response(fenced)

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        result = client.parse(
            system_prompt="test",
            user_content="test",
            output_model=SimpleOutput,
        )

        assert result.parsed.category == "inbox"
        assert result.parsed.tags == ["personal"]

    @patch("app.llm.ollama_client.httpx.Client")
    def test_empty_content_raises(self, mock_client_cls):
        """Test that empty model response raises ValueError."""
        ollama_resp = {
            "model": "qwen2.5:72b",
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        with pytest.raises(ValueError, match="no content"):
            client.parse(
                system_prompt="test",
                user_content="test",
                output_model=SimpleOutput,
            )

    @patch("app.llm.ollama_client.httpx.Client")
    def test_invalid_json_raises_validation_error(self, mock_client_cls):
        """Test that malformed JSON raises a Pydantic validation error."""
        ollama_resp = self._make_ollama_response("this is not json at all")

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        with pytest.raises(Exception):
            client.parse(
                system_prompt="test",
                user_content="test",
                output_model=SimpleOutput,
            )

    @patch("app.llm.ollama_client.httpx.Client")
    def test_http_error_propagates(self, mock_client_cls):
        """Test that HTTP errors from Ollama are propagated."""
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        with pytest.raises(httpx.HTTPStatusError):
            client.parse(
                system_prompt="test",
                user_content="test",
                output_model=SimpleOutput,
            )

    @patch("app.llm.ollama_client.httpx.Client")
    def test_connection_error_propagates(self, mock_client_cls):
        """Test that connection errors (Ollama down) propagate for fallback handling."""
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        with pytest.raises(httpx.ConnectError):
            client.parse(
                system_prompt="test",
                user_content="test",
                output_model=SimpleOutput,
            )

    @patch("app.llm.ollama_client.httpx.Client")
    def test_sends_correct_payload(self, mock_client_cls):
        """Test that the correct payload is sent to Ollama's API."""
        response_json = json.dumps({"category": "test", "confidence": 0.5, "tags": []})
        ollama_resp = self._make_ollama_response(response_json)

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="qwen2.5:72b",
            temperature=0.3,
        )
        client.parse(
            system_prompt="Be a classifier.",
            user_content='{"email": "test"}',
            output_model=SimpleOutput,
        )

        # Verify the HTTP call
        mock_http.post.assert_called_once()
        call_args = mock_http.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"

        sent_payload = call_args[1]["json"]
        assert sent_payload["model"] == "qwen2.5:72b"
        assert sent_payload["stream"] is False
        assert sent_payload["options"]["temperature"] == 0.3
        assert len(sent_payload["messages"]) == 2
        assert sent_payload["messages"][0]["role"] == "system"
        assert sent_payload["messages"][0]["content"] == "Be a classifier."
        assert sent_payload["messages"][1]["role"] == "user"
        assert sent_payload["messages"][1]["content"] == '{"email": "test"}'

    @patch("app.llm.ollama_client.httpx.Client")
    def test_model_info_without_timing(self, mock_client_cls):
        """Test model_info when Ollama doesn't return timing data."""
        response_json = json.dumps({"category": "test", "confidence": 0.5, "tags": []})
        ollama_resp = {
            "model": "qwen2.5:72b",
            "message": {"role": "assistant", "content": response_json},
            "done": True,
        }

        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = ollama_resp
        mock_response.raise_for_status.return_value = None
        mock_http.post.return_value = mock_response
        mock_client_cls.return_value = mock_http

        client = OllamaClient(model="qwen2.5:72b")
        result = client.parse(
            system_prompt="test",
            user_content="test",
            output_model=SimpleOutput,
        )

        assert result.model_info["provider"] == "ollama"
        assert "total_duration_ms" not in result.model_info
        assert "usage" not in result.model_info


# ---------------------------------------------------------------------------
# Pipeline fallback tests
# ---------------------------------------------------------------------------

class TestPipelineFallback:
    """Tests for the fallback mechanism in the pipeline."""

    def test_needs_reply_core_communication(self):
        """core_communication emails should get a draft reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory, UrgencySignals

        llm = LLMTriageOutput(
            major_category=MajorCategory.core_communication,
            sub_action_key="COMM_REPLY_REQUIRED",
            explicit_task=False,
            confidence=0.9,
        )
        assert _needs_reply(llm) is True

    def test_needs_reply_spam_never(self):
        """spam emails should never get a draft reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory

        llm = LLMTriageOutput(
            major_category=MajorCategory.spam,
            sub_action_key="SPAM_MARKETING",
            explicit_task=False,
            confidence=0.95,
        )
        assert _needs_reply(llm) is False

    def test_needs_reply_meta_systems_never(self):
        """meta_and_systems emails should never get a draft reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory

        llm = LLMTriageOutput(
            major_category=MajorCategory.meta_and_systems,
            sub_action_key="SYSTEM_NOTIFICATION",
            explicit_task=False,
            confidence=0.9,
        )
        assert _needs_reply(llm) is False

    def test_needs_reply_schedule_high_urgency(self):
        """schedule_and_time with high urgency should get a draft reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory, UrgencySignals

        llm = LLMTriageOutput(
            major_category=MajorCategory.schedule_and_time,
            sub_action_key="SCHEDULE_CONFIRM_TIME",
            explicit_task=False,
            confidence=0.9,
            urgency_signals=UrgencySignals(urgency="high"),
        )
        assert _needs_reply(llm) is True

    def test_needs_reply_schedule_low_urgency_no_task(self):
        """schedule_and_time with low urgency and no explicit task should not get reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory, UrgencySignals

        llm = LLMTriageOutput(
            major_category=MajorCategory.schedule_and_time,
            sub_action_key="SCHEDULE_RSVP",
            explicit_task=False,
            confidence=0.8,
            urgency_signals=UrgencySignals(urgency="low"),
        )
        assert _needs_reply(llm) is False

    def test_needs_reply_decisions_with_task(self):
        """decisions_and_approvals with explicit_task should get a draft reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory

        llm = LLMTriageOutput(
            major_category=MajorCategory.decisions_and_approvals,
            sub_action_key="DECISION_APPROVE_REJECT",
            explicit_task=True,
            confidence=0.9,
        )
        assert _needs_reply(llm) is True

    def test_needs_reply_info_fyi_never(self):
        """information_and_org emails should never get a draft reply."""
        from app.pipeline import _needs_reply
        from app.models import LLMTriageOutput, MajorCategory

        llm = LLMTriageOutput(
            major_category=MajorCategory.information_and_org,
            sub_action_key="INFO_FYI",
            explicit_task=False,
            confidence=0.85,
        )
        assert _needs_reply(llm) is False
