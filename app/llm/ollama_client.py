"""Ollama LLM provider client.

Connects to a local Ollama instance via its HTTP API.  Ollama serves models
like Qwen 2.5 72B (triage) and DeepSeek-V3.2 (reply drafting) on a single
machine with a unified ``/api/chat`` endpoint.

The client uses the same ``parse()`` interface as ``AnthropicClient`` so the
pipeline can swap between providers transparently.  The model is instructed to
return raw JSON matching the Pydantic schema; the response text is validated
with ``output_model.model_validate_json()``.

References:
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar

import httpx
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass
class OllamaResult(Generic[T]):
    parsed: T
    model_info: Dict[str, Any]


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, returning the raw JSON string."""
    stripped = text.strip()
    # Handle ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return stripped


class OllamaClient:
    """Thin wrapper around Ollama's HTTP API that returns parsed Pydantic objects.

    Uses the ``/api/chat`` endpoint with ``stream: false`` for single-shot
    JSON responses.  Compatible with any model served by Ollama.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:72b",
        temperature: float = 0.2,
        timeout_s: float = 120,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._http = httpx.Client(timeout=timeout_s)

    def parse(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: Type[T],
    ) -> OllamaResult[T]:
        """Call Ollama and parse the result into ``output_model``.

        Sends a chat completion request with a system message and user message,
        then validates the raw JSON response with Pydantic.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }

        resp = self._http.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()

        data = resp.json()

        # Extract the assistant's message content
        raw_text = data.get("message", {}).get("content", "")
        if not raw_text:
            raise ValueError(
                f"Ollama returned no content for model {self._model}. "
                f"Response: {data}"
            )

        # Parse the JSON into the Pydantic model
        json_str = _extract_json(raw_text)
        parsed: T = output_model.model_validate_json(json_str)

        # Build model info metadata
        model_info: Dict[str, Any] = {
            "provider": "ollama",
            "model": self._model,
        }

        # Ollama returns token counts in the response
        if "prompt_eval_count" in data or "eval_count" in data:
            model_info["usage"] = {
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
            }

        # Include timing info if available
        if "total_duration" in data:
            model_info["total_duration_ms"] = round(
                data["total_duration"] / 1_000_000, 2
            )

        return OllamaResult(parsed=parsed, model_info=model_info)
