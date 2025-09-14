from __future__ import annotations

from openai import AsyncOpenAI

from app.config import get_settings

# Centralized OpenRouter Async client with required attribution headers.
# This module is imported by services so we create a single shared instance.
_settings = get_settings()

_client = AsyncOpenAI(
    api_key=_settings.openrouter_api_key,
    base_url=_settings.base_url,  # OpenRouter endpoint
    default_headers=_settings.attribution_headers(),  # required attribution headers
    timeout=_settings.api_timeout,  # centralized timeout configuration
)


def get_openrouter_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI client configured for OpenRouter."""
    return _client
