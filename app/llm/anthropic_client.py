"""Anthropic LLM provider client.

Uses the Anthropic Messages API with plain JSON output.  The model is
instructed to return raw JSON matching the Pydantic schema; the response
text is then validated with ``output_model.model_validate_json()``.

This avoids Anthropic's constrained-grammar ``messages.parse()`` path,
which rejects schemas that compile to a grammar above the size limit
(our ``LLMTriageOutput`` schema triggers that limit).

References:
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar

import anthropic
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@dataclass
class AnthropicResult(Generic[T]):
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


class AnthropicClient:
    """Thin wrapper around Anthropic that returns parsed Pydantic objects."""

    def __init__(self, model: str, temperature: float = 0.2, timeout_s: float = 30):
        self._client = anthropic.Anthropic(timeout=timeout_s)
        self._model = model
        self._temperature = temperature

    def parse(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: Type[T],
    ) -> AnthropicResult[T]:
        """Call Anthropic and parse the result into ``output_model``.

        We use ``messages.create()`` (plain text generation) and then
        validate the JSON output with Pydantic.  This sidesteps the
        constrained-grammar size limit that ``messages.parse()`` hits
        on large schemas.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content},
            ],
            temperature=self._temperature,
        )

        # Extract text from the first content block
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text = block.text
                break

        if not raw_text:
            raise ValueError("Anthropic returned no text content in the response.")

        # Parse the JSON into the Pydantic model
        json_str = _extract_json(raw_text)
        parsed: T = output_model.model_validate_json(json_str)

        model_info: Dict[str, Any] = {"provider": "anthropic", "model": self._model}

        usage = getattr(response, "usage", None)
        if usage is not None:
            model_info["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }

        rid = getattr(response, "id", None)
        if rid:
            model_info["response_id"] = rid

        return AnthropicResult(parsed=parsed, model_info=model_info)
