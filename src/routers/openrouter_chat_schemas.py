from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "model"]
    content: str


class Sampling(BaseModel):
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    repetition_penalty: float | None = Field(None, ge=0)
    min_p: float | None = Field(None, ge=0, le=1)
    top_a: float | None = Field(None, ge=0, le=1)
    seed: int | None = None


class Routing(BaseModel):
    sort: str | None = None
    providers: list[str] | None = None
    fallbacks: list[str] | None = None
    max_price: float | None = Field(None, ge=0)
    require_parameters: bool | None = None


class ChatCompletionPayload(BaseModel):
    # Core
    history: list[Message] = Field(default_factory=list)
    message: str = Field(..., min_length=1)
    model: str | None = Field(None, description="Model ID to use for the chat")
    web_search: bool = Field(False, description="Enable web search for this request")

    # Grouped controls
    sampling: Sampling | None = Field(
        None, description="Sampling and decoding controls (temperature, top_p, etc.)"
    )
    routing: Routing | None = Field(
        None, description="Provider routing and price caps (providers, fallbacks, max_price, etc.)"
    )

    # Output controls
    max_tokens: int | None = Field(None, ge=1, description="Maximum response length")
    stop: str | list[str] | None = Field(None, description="Stop sequences (string or array)")
    logit_bias: dict[str, float] | None = Field(None, description="Token bias as JSON object")
    logprobs: bool | None = Field(None, description="Return log probabilities")
    top_logprobs: int | None = Field(
        None, ge=0, le=20, description="Number of top logprobs to return"
    )
    response_format: dict[str, Any] | None = Field(
        None, description="Response format (JSON schema)"
    )

    # Function calling
    tools: list[dict[str, Any]] | None = Field(None, description="Available tools array")
    tool_choice: str | dict[str, Any] | None = Field(None, description="Tool selection strategy")

    # Reasoning controls
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        None, description="Reasoning effort hint"
    )
    reasoning_max_tokens: int | None = Field(None, ge=1)
    hide_reasoning: bool | None = Field(
        None, description="If true, hide reasoning trace events in the stream"
    )
    disable_reasoning: bool | None = Field(
        None,
        description="Force disable reasoning entirely (overrides any reasoning_* params and auto-detection)",
    )
