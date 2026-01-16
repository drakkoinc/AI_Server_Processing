"""OpenAI LLM provider client.

We use the OpenAI **Responses API** + **Structured Outputs** so the model output can be parsed
directly into a Pydantic model (no fragile "JSON repair" loops).

This client is intentionally generic: you pass:
- a system prompt
- a user content string
- a Pydantic model that defines the required output schema

References:
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- Structured outputs guide: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@dataclass
class OpenAIResult(Generic[T]):
    parsed: T
    model_info: Dict[str, Any]


class OpenAIClient:
    """Thin wrapper around OpenAI that returns parsed Pydantic objects."""

    def __init__(self, model: str, temperature: float = 0.2, timeout_s: float = 30):
        self._client = OpenAI(timeout=timeout_s)
        self._model = model
        self._temperature = temperature

    def parse(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: Type[T],
    ) -> OpenAIResult[T]:
        """Call OpenAI and parse the result into `output_model`.

        The OpenAI Python SDK supports `responses.parse`, which:
        - sends your prompt
        - attaches a JSON Schema derived from `output_model`
        - returns `response.output_parsed` as an instance of `output_model`

        Any schema mismatch becomes an exception (which is good: it forces determinism).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self._client.responses.parse(
            model=self._model,
            input=messages,
            text_format=output_model,
            temperature=self._temperature,
        )

        parsed: T = response.output_parsed

        model_info: Dict[str, Any] = {"provider": "openai", "model": self._model}

        usage = getattr(response, "usage", None)
        if usage is not None:
            model_info["usage"] = usage

        rid = getattr(response, "id", None)
        if rid:
            model_info["response_id"] = rid

        return OpenAIResult(parsed=parsed, model_info=model_info)
