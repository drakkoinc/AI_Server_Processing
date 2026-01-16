"""Local model client (stub).

Your PDFs mention a future setup where you have:
- one AI server backed by OpenAI/ChatGPT
- one AI server backed by a personally trained model

This file is the hook for that second path.

IMPORTANT: Unlike OpenAI Structured Outputs, most local model stacks will NOT naturally
adhere to a strict JSON schema. In practice you'll need one (or more) of:

- constrained decoding (JSON grammar / regex / tokenizer constraints)
- function calling / tool calling style APIs
- post-hoc JSON repair + validation + retry
- fine-tuning to a strict output format

For v1 we raise NotImplementedError so you don't accidentally deploy a stub.
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
