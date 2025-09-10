# tests/test_models.py
from typing import Any

import pytest
from httpx import AsyncClient, Response


@pytest.mark.asyncio
async def test_models_endpoint_exists(async_client: AsyncClient) -> None:
    """Test that the models endpoint exists."""
    resp: Response = await async_client.get("/models")
    # With test credentials, we expect authentication failure but endpoint should exist
    assert resp.status_code in [200, 401, 403, 502, 503, 504]


@pytest.mark.asyncio
async def test_models_endpoint_returns_json_on_error(async_client: AsyncClient) -> None:
    """Test that models endpoint returns JSON even on error."""
    resp: Response = await async_client.get("/models")
    # Should return JSON content type even on error
    content_type = resp.headers.get("content-type", "")
    assert "application/json" in content_type


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_endpoint_with_real_credentials(async_client: AsyncClient) -> None:
    """Integration test with real credentials (skip in CI)."""
    # This test would only pass with real API credentials
    # Mark as integration test so it can be skipped in CI
    resp: Response = await async_client.get("/models")

    if resp.status_code == 200:
        data: dict[str, Any] = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    else:
        # Expected to fail with test credentials
        assert resp.status_code in [401, 403, 502, 503, 504]


@pytest.mark.asyncio
async def test_models_search_endpoint_exists(async_client: AsyncClient) -> None:
    """Test that the models search endpoint exists."""
    resp: Response = await async_client.get("/models/search")
    # With test credentials, we expect authentication failure but endpoint should exist
    assert resp.status_code in [200, 401, 403, 502, 503, 504]


@pytest.mark.asyncio
async def test_models_search_endpoint_returns_json_on_error(async_client: AsyncClient) -> None:
    """Test that models search endpoint returns JSON even on error."""
    resp: Response = await async_client.get("/models/search")
    # Should return JSON content type even on error
    content_type = resp.headers.get("content-type", "")
    assert "application/json" in content_type


@pytest.mark.asyncio
async def test_models_search_with_filters(async_client: AsyncClient) -> None:
    """Test that search endpoint accepts filter parameters without crashing."""
    params = {
        "search_term": "gpt",
        "input_modalities": "text,image",
        "output_modalities": "text",
        "min_context_length": "4000",
        "max_context_length": "128000",
        "min_price": "0",
        "max_price": "10",
        "free_only": "false",
        "supported_parameters": "temperature,top_p",
        "sort": "pricing_low_to_high",
        "limit": "10",
        "offset": "0",
    }
    resp: Response = await async_client.get("/models/search", params=params)
    # Should not crash, even with authentication failure
    assert resp.status_code in [200, 401, 403, 502, 503, 504]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_search_endpoint_with_real_credentials(async_client: AsyncClient) -> None:
    """Integration test for search endpoint with real credentials (skip in CI)."""
    # This test would only pass with real API credentials
    resp: Response = await async_client.get("/models/search?limit=5&sort=pricing_low_to_high")

    if resp.status_code == 200:
        data: dict[str, Any] = resp.json()
        assert "models" in data
        assert "total_count" in data
        assert "filters_applied" in data
        assert "sort_applied" in data
        assert isinstance(data["models"], list)
        assert isinstance(data["total_count"], int)
        assert data["sort_applied"] == "pricing_low_to_high"
        assert len(data["models"]) <= 5  # Should respect limit
    else:
        # Expected to fail with test credentials
        assert resp.status_code in [401, 403, 502, 503, 504]
