from __future__ import annotations

import time
from typing import TYPE_CHECKING

from services.openrouter_sse_client import OpenRouterMessage, model_supports_reasoning

if TYPE_CHECKING:  # for typing only to satisfy strict settings
    from .openrouter_chat_schemas import Message


def to_openrouter_messages(history: list[Message]) -> list[OpenRouterMessage]:
    """Convert API message history (role 'user'|'model') to OpenRouter messages (user|assistant)."""
    out: list[OpenRouterMessage] = []
    for m in history:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        else:
            out.append({"role": "assistant", "content": m.content})
    return out


# --------- Reasoning support cache (1 hour TTL) ---------

_REASONING_CACHE_TTL = 60 * 60
_reasoning_cache: dict[str, tuple[bool, float]] = {}


async def supports_reasoning_cached(model: str) -> bool:
    now = time.monotonic()
    hit = _reasoning_cache.get(model)
    if hit and (now - hit[1] < _REASONING_CACHE_TTL):
        return hit[0]
    value = await model_supports_reasoning(model)
    _reasoning_cache[model] = (value, now)
    return value
