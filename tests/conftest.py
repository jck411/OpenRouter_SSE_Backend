# tests/conftest.py
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up required environment variables for all tests."""
    # Store original values to restore later
    original_env = {}
    test_env_vars = {
        "OPENROUTER_API_KEY": "test-key-12345",
        "REFERER": "http://localhost:8000/test",
        "X_TITLE": "pydanticai_openrouter_backend_test",
        "API_TIMEOUT": "10.0",  # Shorter timeout for tests
    }

    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, Any]:
    """Create an async test client for FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_chat_request() -> dict[str, object]:
    """Sample chat request data for testing."""
    return {
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there!"},
        ],
        "message": "How are you?",
    }


@pytest.fixture
def empty_chat_request() -> dict[str, object]:
    """Empty chat request for validation testing."""
    return {"history": []}
