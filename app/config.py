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
    # "ollama" (primary, local) or "anthropic" (cloud fallback)
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # --- Ollama (local models) ---
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_triage_model: str = os.getenv("OLLAMA_TRIAGE_MODEL", "qwen2.5:72b")
    ollama_reply_model: str = os.getenv("OLLAMA_REPLY_MODEL", "qwen2.5:72b")

    # --- Anthropic (cloud fallback) ---
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

    # --- Reply drafting ---
    reply_enabled: bool = os.getenv("REPLY_ENABLED", "true").lower() in ("true", "1", "yes")

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
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "180"))

    # Server metadata
    api_version: str = "3.0.0"
    schema_version: str = "v3"
    model_version: str = os.getenv("MODEL_VERSION", "drakko-email-v3")


settings = Settings()
