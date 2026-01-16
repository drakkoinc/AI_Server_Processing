"""Configuration helpers.

We keep this very simple: all runtime configuration comes from environment variables.
In production you can replace this with pydantic-settings or your own config system.

Why this exists:
- Keeps model/provider choice out of business logic.
- Makes Docker/Kubernetes deployment straightforward.

"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Which LLM backend to use.
    # - openai: calls OpenAI's Responses API.
    # - local: placeholder for your self-hosted / fine-tuned model.
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI model name (only used when llm_provider == "openai").
    # See OpenAI docs for current model names supported by structured outputs.
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")

    # A constant value included in every response, useful for:
    # - pipeline versioning
    # - contract / schema migrations
    # - downstream debugging
    contract_reference: str = os.getenv("CONTRACT_REFERENCE", "drakko.gmail_insights.v1")

    # Safety / product knobs
    max_body_chars: int = int(os.getenv("MAX_BODY_CHARS", "12000"))  # prompt length guard

    # LLM request options
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))


settings = Settings()
