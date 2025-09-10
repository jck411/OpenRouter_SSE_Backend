# src/app/config.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- OpenRouter ----------
    # Env var auto-maps from OPENROUTER_API_KEY, REFERER, X_TITLE, etc.
    openrouter_api_key: str = Field(..., description="OpenRouter API key")

    # Required by project rules: must be present and non-empty
    referer: str = Field(..., description="HTTP-Referer for OpenRouter attribution")
    x_title: str = Field(..., description="X-Title for OpenRouter attribution")

    model_name: str = Field(default="openai/gpt-4o", description="Default model ID")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")

    # API timeout configuration
    api_timeout: float = Field(default=30.0, description="API request timeout in seconds")

    # ---------- FastAPI ----------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)

    # ---------- Validators ----------
    @field_validator("referer", "x_title")
    @classmethod
    def _non_empty(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be set via environment and non-empty")
        return v.strip()

    # ---------- Helpers ----------
    def attribution_headers(self) -> dict[str, str]:
        """
        Canonical attribution headers required on every request.
        Use this wherever you construct the OpenAI SDK client / transport.
        """
        # OpenRouter expects these exact header names:
        #   - HTTP-Referer
        #   - X-Title
        return {
            "HTTP-Referer": self.referer,
            "X-Title": self.x_title,
        }


@lru_cache
def get_settings() -> Settings:  # pragma: no cover
    return Settings()  # type: ignore[call-arg]
