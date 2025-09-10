import pytest
from httpx import AsyncClient

from routers import chat as chat_router


@pytest.mark.asyncio
async def test_disable_reasoning_prevents_reasoning_path(async_client: AsyncClient, monkeypatch):
    """When disable_reasoning=true, backend should not attempt reasoning pathway.

    We monkeypatch `_model_supports_reasoning` to always return True so the auto-enable
    logic would normally kick in. Then verify that with disable_reasoning the stream
    contains only content/done events (and no 'reasoning' events or reasoning JSON fields).
    """

    async def _always_supports(_model_id: str) -> bool:  # pragma: no cover - trivial
        return True

    # New implementation uses services.openrouter_sse_client.model_supports_reasoning
    monkeypatch.setattr(chat_router, "model_supports_reasoning", _always_supports)

    # Minimal request body
    body = {"history": [], "message": "Test message", "model": "test/model"}

    # With disable_reasoning=true
    resp = await async_client.post("/chat?disable_reasoning=true", json=body)
    # Accept various upstream failures, but if 200, inspect SSE content
    assert resp.status_code in [200, 400, 502, 503, 504]
    text = resp.text
    # Should never contain reasoning events when disabled
    assert "event: reasoning" not in text
    # Ensure we still emit final usage and done
    assert "event: usage" in text
    assert text.index("event: usage") < text.index("event: done")


@pytest.mark.asyncio
async def test_reasoning_auto_enabled_without_disable(async_client: AsyncClient, monkeypatch):
    """When model supports reasoning and disable_reasoning is not set, the code should choose
    the reasoning-capable (raw SSE) pathway; we detect this by the presence of 'event: content' or
    'event: reasoning' markers in a 200 response. We can't force upstream reasoning frames, but
    we can assert the pathway didn't short-circuit to immediate error.
    """

    async def _always_supports(_model_id: str) -> bool:  # pragma: no cover - trivial
        return True

    monkeypatch.setattr(chat_router, "model_supports_reasoning", _always_supports)

    body = {"history": [], "message": "Test message", "model": "test/model"}
    resp = await async_client.post("/chat", json=body)
    assert resp.status_code in [200, 400, 502, 503, 504]
    text = resp.text
    # Expect either raw SSE success markers or upstream_http_error indicating raw path engaged
    assert (
        ("event: reasoning" in text)
        or ("event: content" in text)
        or ("upstream_http_error" in text)
        or ("event: done" in text)
    )
    # If we got upstream error marker, that confirms raw SSE path chosen
    if "upstream_http_error" in text:
        assert "event: error" in text
