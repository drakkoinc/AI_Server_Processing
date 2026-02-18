"""Anthropic LLM provider client.

We use the Anthropic **Messages API** + **Structured Outputs** so the model output can be parsed
directly into a Pydantic model (no fragile "JSON repair" loops).

This client is intentionally generic: you pass:
- a system prompt
- a user content string
- a Pydantic model that defines the required output schema

References:
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
- Structured outputs guide: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar

import anthropic
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@dataclass
class AnthropicResult(Generic[T]):
    parsed: T
    model_info: Dict[str, Any]


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
        """Call Anthropic and parse the result into `output_model`.

        The Anthropic Python SDK supports `messages.parse`, which:
        - sends your prompt
        - attaches a JSON Schema derived from `output_model`
        - returns `response.parsed_output` as an instance of `output_model`

        Any schema mismatch becomes an exception (which is good: it forces determinism).
        """
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content},
            ],
            output_format=output_model,
            temperature=self._temperature,
        )

        parsed: T = response.parsed_output

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
