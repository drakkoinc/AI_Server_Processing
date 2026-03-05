"""Local model client (DEPRECATED — use OllamaClient instead).

This stub existed as a placeholder for local model inference.  The real
local model path is now ``app.llm.ollama_client.OllamaClient``, which
connects to a local Ollama instance serving Qwen 2.5 72B (triage) and
DeepSeek-V3.2 (reply drafting).

This file is kept for backward compatibility but is no longer used
in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@dataclass
class LocalResult(Generic[T]):
    parsed: T
    model_info: Dict[str, Any]


class LocalClient:
    def __init__(self):
        pass

    def parse(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: Type[T],
    ) -> LocalResult[T]:
        raise NotImplementedError(
            "Local model client is a stub. Implement this with your own inference stack."
        )
