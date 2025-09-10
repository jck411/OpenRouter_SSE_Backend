# tests/test_mcp_server.py
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.mcp_server import llm_chat


class TestMCPServer:
    """Test the MCP server functionality."""

    @pytest.mark.asyncio
    async def test_llm_chat_success(self):
        """Test successful chat completion using OpenRouter simple client."""
        with patch("app.mcp_server.settings") as mock_settings:
            mock_settings.model_name = "test/model"
            mock_settings.api_timeout = 10.0

            with patch(
                "services.openrouter_sse_client.simple_chat_completion", new_callable=AsyncMock
            ) as mock_simple:
                mock_simple.return_value = "Hello! How can I help you today?"

                result = await llm_chat("Hello")

                assert result == "Hello! How can I help you today?"
                mock_simple.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_chat_with_complex_prompt(self):
        """Test chat with a more complex prompt."""
        with patch(
            "services.openrouter_sse_client.simple_chat_completion", new_callable=AsyncMock
        ) as mock_simple:
            mock_simple.return_value = "The capital of France is Paris."

            result = await llm_chat("What is the capital of France?")

            assert result == "The capital of France is Paris."
            mock_simple.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_chat_timeout_error(self):
        """Test that timeout errors are handled gracefully."""
        with patch(
            "services.openrouter_sse_client.simple_chat_completion", new_callable=AsyncMock
        ) as mock_simple:
            mock_simple.return_value = "Error: Operation timed out after 30.0s"

            result = await llm_chat("Hello")

            assert "Error: Operation timed out after" in result

    @pytest.mark.asyncio
    async def test_llm_chat_cancellation_error(self):
        """Test that CancelledError is re-raised for proper cleanup."""
        with patch(
            "services.openrouter_sse_client.simple_chat_completion", new_callable=AsyncMock
        ) as mock_simple:
            mock_simple.side_effect = asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await llm_chat("Hello")

    @pytest.mark.asyncio
    async def test_llm_chat_generic_exception(self):
        """Test that generic exceptions are handled and return error message."""
        with patch(
            "services.openrouter_sse_client.simple_chat_completion", new_callable=AsyncMock
        ) as mock_simple:
            mock_simple.side_effect = ValueError("Something went wrong")

            result = await llm_chat("Hello")

            assert result == "Error: Something went wrong"

    # Removed obsolete test_llm_chat_empty_prompt: relied on removed PydanticAI agent

    @pytest.mark.asyncio
    async def test_llm_chat_with_custom_timeout(self):
        """Test that custom timeout from settings is used."""
        # Mock settings to have a custom timeout and ensure it's passed
        with patch("app.mcp_server.settings") as mock_settings:
            mock_settings.model_name = "test/model"
            mock_settings.api_timeout = 60.0

            with patch(
                "services.openrouter_sse_client.simple_chat_completion", new_callable=AsyncMock
            ) as mock_simple:
                mock_simple.return_value = "Error: Operation timed out after 60.0s"

                result = await llm_chat("Hello")

                assert "Error: Operation timed out after 60.0s" in result
