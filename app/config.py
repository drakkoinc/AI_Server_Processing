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
    #openai: calls OpenAI's Responses API - may change and experiment here
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI model name - change and test for GPT4o
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")

    # A constant value included in every response, useful for:
    # - pipeline versioning
    # - contract / schema migrations
    # - downstream debugging
    contract_reference: str = os.getenv("CONTRACT_REFERENCE", "drakko.gmail_insights.v3")

    # Safety / product knobs
    max_body_chars: int = int(os.getenv("MAX_BODY_CHARS", "12000"))  # prompt length guard

    # LLM request options 
    # Temperature at 0.2, timeout at minute and half
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "90"))

    # Server metadata
    api_version: str = "3.0.0"
    schema_version: str = "v3"
    model_version: str = os.getenv("MODEL_VERSION", "drakko-email-v3")


settings = Settings()
