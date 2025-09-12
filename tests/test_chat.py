# tests/test_chat.py
from typing import Any
from unittest.mock import patch

import pytest
from httpx import AsyncClient, Response


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Test the health endpoint to validate the app structure works."""
    resp: Response = await async_client.get("/health")
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_endpoint_validation_error(
    async_client: AsyncClient, empty_chat_request: dict[str, object]
) -> None:
    """Test that the chat endpoint validates input correctly."""
    # Test with missing message - should get validation error
    resp: Response = await async_client.post("/chat", json=empty_chat_request)
    assert resp.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_chat_endpoint_with_empty_message(async_client: AsyncClient) -> None:
    """Test chat endpoint with empty message string."""
    request_data = {"history": [], "message": ""}
    resp: Response = await async_client.post("/chat", json=request_data)
    assert resp.status_code == 422  # Validation error for empty message (min_length=1)


@pytest.mark.asyncio
async def test_chat_endpoint_with_valid_request(
    async_client: AsyncClient, sample_chat_request: dict[str, object]
) -> None:
    """Test chat endpoint with valid request structure."""

    # Mock streaming to avoid real network
    async def fake_stream(*_args, **_kwargs):
        yield {"type": "reasoning", "data": {"text": "Think..."}}
        yield {"type": "content", "data": {"text": "Hello"}}
        yield {"type": "done", "data": {"completed": True}}

    with patch(
        "routers.chat.stream_chat_completion",
        new=fake_stream,
    ):
        resp: Response = await async_client.post("/chat", json=sample_chat_request)
        assert resp.status_code == 200
        text = resp.text
        # Order: reasoning -> content -> usage -> done
        assert "event: reasoning" in text
        assert text.index("event: reasoning") < text.index("event: content")
        assert "event: usage" in text
        assert text.index("event: usage") < text.index("event: done")


@pytest.mark.asyncio
async def test_pass_through_knobs(async_client: AsyncClient) -> None:
    """Ensure pass-through knobs are forwarded to service call."""
    captured = {}

    async def capturing_stream(**kwargs):
        nonlocal captured
        captured = kwargs
        yield {"type": "done", "data": {"completed": True}}

    with patch(
        "routers.chat.stream_chat_completion",
        new=capturing_stream,
    ):
        body = {"history": [], "message": "Hi"}
        params = {
            "providers": "openai,anthropic",
            "sort": "throughput_high_to_low",
            "fallbacks": "foo,bar",
            "max_price": "0.01",
        }
        resp: Response = await async_client.post("/chat", json=body, params=params)
        assert resp.status_code == 200
        # Check knobs present
        assert captured.get("providers") == ["openai", "anthropic"]
        assert captured.get("sort") == "throughput_high_to_low"
        assert captured.get("fallbacks") == ["foo", "bar"]
        max_price_val = captured.get("max_price")
        assert max_price_val is not None and float(max_price_val) == 0.01


@pytest.mark.asyncio
async def test_malformed_json_params_emit_error(async_client: AsyncClient) -> None:
    """Params that look like JSON but are invalid should cause SSE error."""
    body = {"history": [], "message": "Hi"}
    # tools is JSON-looking but invalid
    resp: Response = await async_client.post("/chat?tools={invalid}", json=body)
    assert resp.status_code == 200
    text = resp.text
    assert "event: error" in text
    assert "Malformed JSON in 'tools'" in text
    # Should still end with done
    assert "event: done" in text


@pytest.mark.asyncio
async def test_timeout_sse_error(async_client: AsyncClient) -> None:
    """Timeouts in the service should surface as SSE error, not silent disconnects."""

    async def fake_stream_timeout(**_kwargs):
        yield {"type": "error", "data": {"error": "Request timed out", "type": "timeout"}}
        yield {"type": "done", "data": {"completed": False}}

    with patch(
        "routers.chat.stream_chat_completion",
        new=fake_stream_timeout,
    ):
        body = {"history": [], "message": "Hi"}
        resp: Response = await async_client.post("/chat", json=body)
        assert resp.status_code == 200
        text = resp.text
        assert "event: error" in text
        assert "timeout" in text
        assert "event: done" in text


@pytest.mark.asyncio
async def test_models_endpoint_structure(async_client: AsyncClient) -> None:
    """Test that the models endpoint exists and handles errors gracefully."""
    resp: Response = await async_client.get("/models")
    # With test credentials, we expect this to fail gracefully
    assert resp.status_code in [200, 401, 403, 502, 503, 504]
