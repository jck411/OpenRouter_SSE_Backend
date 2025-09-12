# tests/test_config.py
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings


def _create_test_settings() -> Settings:
    """
    Factory function to create Settings in tests.

    Pylance incorrectly flags this as missing arguments, but BaseSettings
    gets values from environment variables when properly configured.
    """
    return Settings()


class TestSettings:
    """Test the Settings configuration class."""

    def test_settings_with_valid_env_vars(self):
        """Test Settings initialization with valid environment variables."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "https://example.com",
                "X_TITLE": "Test App",
            },
            clear=True,
        ):
            settings = _create_test_settings()
            assert settings.openrouter_api_key == "test-key-123"
            assert settings.referer == "https://example.com"
            assert settings.x_title == "Test App"
            assert settings.model_name == "openai/gpt-4o"  # default
            assert settings.base_url == "https://openrouter.ai/api/v1"  # default

    def test_settings_empty_referer_validation(self):
        """Test that empty referer is rejected."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "   ",  # whitespace only
                "X_TITLE": "Test App",
            },
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                _create_test_settings()
            errors = exc_info.value.errors()
            assert any(error["loc"][0] == "referer" for error in errors)

    def test_settings_empty_x_title_validation(self):
        """Test that empty x_title is rejected."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "https://example.com",
                "X_TITLE": "",  # empty string
            },
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                _create_test_settings()
            errors = exc_info.value.errors()
            assert any(error["loc"][0] == "x_title" for error in errors)

    def test_settings_strips_whitespace(self):
        """Test that referer and x_title whitespace is stripped."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "  https://example.com  ",
                "X_TITLE": "  Test App  ",
            },
            clear=True,
        ):
            settings = _create_test_settings()
            assert settings.referer == "https://example.com"
            assert settings.x_title == "Test App"

    def test_settings_custom_values(self):
        """Test Settings with custom non-default values."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "https://example.com",
                "X_TITLE": "Test App",
                "MODEL_NAME": "anthropic/claude-3-sonnet",
                "BASE_URL": "https://custom.api.com/v1",
                "API_TIMEOUT": "60.0",
                "HOST": "127.0.0.1",
                "PORT": "9000",
                "RELOAD": "true",
            },
            clear=True,
        ):
            settings = _create_test_settings()
            assert settings.model_name == "anthropic/claude-3-sonnet"
            assert settings.base_url == "https://custom.api.com/v1"
            assert settings.api_timeout == 60.0
            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.reload is True

    def test_attribution_headers(self):
        """Test the attribution_headers helper method."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "https://example.com",
                "X_TITLE": "Test App",
            },
            clear=True,
        ):
            settings = _create_test_settings()
            headers = settings.attribution_headers()
            assert headers == {
                "HTTP-Referer": "https://example.com",
                "X-Title": "Test App",
            }

    def test_settings_defaults(self):
        """Test that default values are correctly set."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key-123",
                "REFERER": "https://example.com",
                "X_TITLE": "Test App",
            },
            clear=True,
        ):
            settings = _create_test_settings()
            # Test defaults
            assert settings.model_name == "openai/gpt-4o"
            assert settings.base_url == "https://openrouter.ai/api/v1"
            assert settings.api_timeout == 30.0
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.reload is False
