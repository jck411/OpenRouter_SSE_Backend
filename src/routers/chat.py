from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logger import log
from services.openrouter_sse_client import (
    OpenRouterMessage,
    ReasoningParams,
    model_supports_reasoning,
    stream_chat_completion,
)

router = APIRouter(prefix="/chat", tags=["chat"])
# OpenAI-compatible alias router (no prefix) for /v1/chat/completions
openai_router = APIRouter(tags=["chat"])
settings = get_settings()

# ---------- Models ----------


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
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="Tool selection strategy"
    )

    # Reasoning controls
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        None, description="Reasoning effort hint"
    )
    reasoning_max_tokens: int | None = Field(None, ge=1)
    # Explicitly control whether to hide reasoning deltas in the stream
    hide_reasoning: bool | None = Field(
        None, description="If true, hide reasoning trace events in the response stream"
    )
    disable_reasoning: bool | None = Field(
        None,
        description="Force disable reasoning entirely (overrides any reasoning_* params and auto-detection)",
    )


# ---------- Utilities ----------

# -------- Constants --------

# Default headers for SSE streams. Adding an explicit content-type prevents
# some proxies/browsers from buffering.
SSE_HEADERS = {
    "Cache-Control": "no-cache",  # Reduce proxy buffering
    "X-Accel-Buffering": "no",  # Disable Nginx buffering
    "Content-Type": "text/event-stream; charset=utf-8",  # Explicit MIME type
}


# (legacy CSV/JSON query parsing helpers removed in favor of typed JSON body via Pydantic)


def _sanitize_sse_line(line: str) -> str:
    """Sanitize a single SSE line.

    Lines starting with ':' are treated by browsers as comments and dropped.
    Prefix with a space to preserve user content starting with ':'.
    """
    return f" {line}" if line.startswith(":") else line


def _format_sse_event(event_type: str, data: Any) -> str:
    """Format data as a Server-Sent Events (SSE) message.

    Converts arbitrary data into proper SSE format with event type and data payload.
    Handles multi-line data by prefixing each line with 'data:' as required by SSE spec.

    Args:
        event_type: The SSE event type (e.g., 'content', 'reasoning', 'error')
        data: Data to include in the event (will be JSON-encoded if dict/list)

    Returns:
        Properly formatted SSE event string ending with double newline

    Example:
        _format_sse_event('content', {'text': 'hello'}) ->
        "event: content\ndata: {\"text\": \"hello\"}\n\n"
    """
    payload = json.dumps(data) if isinstance(data, dict | list) else str(data)
    # SSE requires each line to start with 'data:'; split to be robust to newlines
    lines = payload.splitlines() or [""]
    buf = [f"event: {event_type}"]
    for ln in lines:
        buf.append(f"data: {_sanitize_sse_line(ln)}")
    buf.append("")  # blank line terminator
    return "\n".join(buf) + "\n"


def _format_sse_error(error_message: str, error_type: str = "error") -> str:
    """Format an error message as a structured SSE event.

    Creates a standardized error event with both human-readable message and error type
    for programmatic handling by clients.

    Args:
        error_message: Human-readable error description
        error_type: Error category/type for client logic (default: "error")

    Returns:
        SSE-formatted error event

    Example:
        _format_sse_error("Invalid JSON", "bad_request") ->
        "event: error\ndata: {\"error\": \"Invalid JSON\", \"type\": \"bad_request\"}\n\n"
    """
    return _format_sse_event("error", {"error": error_message, "type": error_type})


def _to_openrouter_messages(history: list[Message]) -> list[OpenRouterMessage]:
    """Convert API message history to OpenRouter message format.

    Transforms the external API message format (with 'model' role) to OpenRouter's
    expected format (with 'assistant' role).

    Args:
        history: List of Message objects from the API request

    Returns:
        List of OpenRouterMessage objects compatible with OpenRouter API

    Note:
        Maps 'model' role to 'assistant' role as required by OpenRouter
    """
    out: list[OpenRouterMessage] = []
    for m in history:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        else:
            out.append({"role": "assistant", "content": m.content})
    return out


# ---------- Route ----------


@router.post(
    "",
    response_model=None,
    summary="OpenAI-compatible chat completion endpoint",
)
async def chat(
    payload: ChatCompletionPayload,
    stream: bool = Query(True, description="If true, stream as SSE; else return JSON response"),
) -> Response:
    """
    Stream AI assistant responses as Server-Sent Events (SSE) with real-time delivery.

    This endpoint handles chat conversations with support for:
    - Streaming content and reasoning traces
    - OpenRouter model routing and cost controls
    - Advanced model parameters (temperature, top-p, etc.)
    - Tool calling and function execution
    - Web search integration (via :online model variants)
    - Automatic reasoning detection and configuration
    The response streams different event types:
    - `reasoning`: Thinking/reasoning process deltas (for capable models)
    - `content`: Actual response content deltas
    - `usage`: Final metrics (timing, tokens, routing info)
    - `error`: Structured error information
    - `done`: Completion signal

    Args:
        req: Chat request with message history and new user message
        sort: OpenRouter provider selection strategy (price, throughput, latency, etc.)
        providers: Comma-separated list of preferred providers
        max_price: Maximum cost limit per request
        fallbacks: Comma-separated list of fallback models
        require_parameters: Ensure all providers support all parameters
        temperature: Response randomness (0.0 = deterministic, 2.0 = very random)
        top_p: Nucleus sampling threshold
        top_k: Limit to top K tokens
        frequency_penalty: Reduce token repetition
        presence_penalty: Encourage topic diversity
        repetition_penalty: Alternative repetition control
        min_p: Minimum probability threshold
        top_a: Alternative sampling method
        seed: Reproducibility seed
        max_tokens: Maximum response length
        stop: Stop sequences (string or JSON array)
        logit_bias: Token probability adjustments (JSON object)
        logprobs: Include log probabilities in response
        top_logprobs: Number of top log probabilities to return
        response_format: Output format constraints (JSON schema)
        tools: Available tools for function calling (JSON array)
        tool_choice: Tool selection strategy
        reasoning_effort: Reasoning depth for capable models
        reasoning_max_tokens: Limit reasoning token usage
        hide_reasoning: Hide reasoning trace in response stream
        disable_reasoning: Force disable all reasoning features

    Returns:
        StreamingResponse with SSE events containing response data

    Raises:
        HTTPException: For invalid requests (400) or server errors (500+)

    Example SSE Events:
        event: reasoning
        data: {"text": "Let me think about this..."}

        event: content
        data: {"text": "Hello! How can I help you?"}

        event: usage
        data: {"model": "anthropic/claude-3-sonnet", "duration_ms": 1500, ...}

        event: done
        data: {"completed": true}
    """
    p = payload
    if not p.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # Convert API messages -> OpenRouter messages
    history_messages = _to_openrouter_messages(p.history)

    # Determine model (optionally ensure :online)
    final_model = p.model or settings.model_name
    if p.web_search and not final_model.endswith(":online"):
        final_model = f"{final_model}:online"

    # Build reasoning dict (normalize flags)
    reasoning: ReasoningParams | None = None
    # Determine explicit hide preference
    _hide: bool | None = p.hide_reasoning

    if p.reasoning_effort or (p.reasoning_max_tokens is not None) or (_hide is not None):
        r: ReasoningParams = {}
        if p.reasoning_effort:
            r["effort"] = p.reasoning_effort
        if p.reasoning_max_tokens is not None:
            r["max_tokens"] = p.reasoning_max_tokens
        if _hide is not None:
            # ReasoningParams uses negative flag 'exclude' (exclude = hide)
            r["exclude"] = _hide
        reasoning = r
    else:
        # If the model supports reasoning (and caller didn't specify), opt-in with a sensible default.
        if not p.disable_reasoning and await _supports_reasoning_cached(final_model):
            reasoning = {"effort": "high"}

    # Explicit override: disable reasoning completely
    if p.disable_reasoning:
        reasoning = None

    # Add user message to conversation
    messages = history_messages + [{"role": "user", "content": p.message}]

    async def iterate() -> AsyncIterator[str]:
        started_monotonic = time.monotonic()
        reasoning_events = 0
        content_events = 0
        # Locally suppress reasoning streaming if requested or fully disabled
        suppress_reasoning_stream = bool(p.hide_reasoning) or bool(p.disable_reasoning)
        # Enhanced usage tracking - capture tokens, costs, and metadata
        # Use OpenRouter's native token fields
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        cost: float | None = None
        provider: str | None = None
        actual_model: str | None = None  # The actual model selected by OpenRouter
        # Detailed token breakdowns
        cached_tokens: int | None = None
        reasoning_tokens: int | None = None
        audio_tokens: int | None = None
        image_tokens: int | None = None
        # Cost breakdowns
        prompt_cost: float | None = None
        completion_cost: float | None = None
        is_byok: bool | None = None
        try:
            async for event in stream_chat_completion(
                messages=messages,
                model=final_model,
                # OpenRouter routing controls
                providers=(p.routing.providers if p.routing else None),
                fallbacks=(p.routing.fallbacks if p.routing else None),
                sort=(p.routing.sort if p.routing else None),
                max_price=(p.routing.max_price if p.routing else None),
                require_parameters=(p.routing.require_parameters if p.routing else None),
                # Reasoning controls
                reasoning=reasoning,
                # Standard LLM parameters
                temperature=(p.sampling.temperature if p.sampling else None),
                top_p=(p.sampling.top_p if p.sampling else None),
                top_k=(p.sampling.top_k if p.sampling else None),
                frequency_penalty=(p.sampling.frequency_penalty if p.sampling else None),
                presence_penalty=(p.sampling.presence_penalty if p.sampling else None),
                repetition_penalty=(p.sampling.repetition_penalty if p.sampling else None),
                min_p=(p.sampling.min_p if p.sampling else None),
                top_a=(p.sampling.top_a if p.sampling else None),
                seed=(p.sampling.seed if p.sampling else None),
                max_tokens=p.max_tokens,
                stop=p.stop,
                logit_bias=p.logit_bias,
                logprobs=p.logprobs,
                top_logprobs=p.top_logprobs,
                response_format=p.response_format,
                tools=p.tools,
                tool_choice=p.tool_choice,
            ):
                if event["type"] == "reasoning":
                    if not suppress_reasoning_stream:
                        reasoning_events += 1
                        yield _format_sse_event("reasoning", event["data"])
                elif event["type"] == "content":
                    content_events += 1
                    yield _format_sse_event("content", event["data"])
                elif event["type"] == "usage":
                    # Capture enhanced usage data from OpenRouter (with usage accounting enabled)
                    usage_data = event.get("data", {})

                    # Basic token counts
                    prompt_tokens = usage_data.get("prompt_tokens")
                    completion_tokens = usage_data.get("completion_tokens")
                    total_tokens = usage_data.get("total_tokens")

                    # Cost information
                    cost = usage_data.get("cost")
                    is_byok = usage_data.get("is_byok")

                    # Provider and actual model information (included from frame level)
                    provider = usage_data.get("provider")
                    actual_model = usage_data.get("actual_model")

                    # Detailed token breakdowns
                    prompt_details = usage_data.get("prompt_tokens_details", {})
                    cached_tokens = prompt_details.get("cached_tokens")
                    audio_tokens = prompt_details.get("audio_tokens")

                    completion_details = usage_data.get("completion_tokens_details", {})
                    reasoning_tokens = completion_details.get("reasoning_tokens")
                    image_tokens = completion_details.get("image_tokens")

                    # Cost breakdowns
                    cost_details = usage_data.get("cost_details", {})
                    prompt_cost = cost_details.get("upstream_inference_prompt_cost")
                    completion_cost = cost_details.get("upstream_inference_completions_cost")

                    # Note: We don't forward usage events to the client here,
                    # they'll be included in the final usage event before done
                elif event["type"] == "error":
                    yield _format_sse_event("error", event["data"])
                elif event["type"] == "done":
                    # Emit final usage/meta event before done
                    duration_ms = int((time.monotonic() - started_monotonic) * 1000)
                    duration_s = max(duration_ms / 1000.0, 1e-6)

                    usage_payload = {
                        "model": final_model,  # The requested model (e.g., "openrouter/auto")
                        "actual_model": actual_model,  # The actual model selected by OpenRouter
                        "duration_ms": duration_ms,
                        "reasoning_events": reasoning_events,
                        "content_events": content_events,
                        # Token usage from OpenRouter streaming response (align with OpenRouter field names)
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        # Simple derived rate metric
                        "tokens_per_second": (
                            (completion_tokens or 0) / duration_s if completion_tokens is not None else None
                        ),
                        # Cost information (in OpenRouter credits)
                        "cost": cost,
                        "prompt_cost": prompt_cost,
                        "completion_cost": completion_cost,
                        "is_byok": is_byok,
                        # Provider and detailed token information
                        "provider": provider,
                        "cached_tokens": cached_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "audio_tokens": audio_tokens,
                        "image_tokens": image_tokens,
                        # Echo core routing knobs for observability
                        "routing": {
                            "providers": (p.routing.providers if p.routing else None),
                            "fallbacks": (p.routing.fallbacks if p.routing else None),
                            "sort": (p.routing.sort if p.routing else None),
                            "max_price": (p.routing.max_price if p.routing else None),
                            "require_parameters": (p.routing.require_parameters if p.routing else None),
                        },
                    }
                    # Emit usage log (structured) with a request correlation id
                    # Use cryptographically-strong, process-global unique id
                    request_id = f"req-{uuid.uuid4().hex}"
                    log.info(
                        "chat_usage",
                        request_id=request_id,
                        **usage_payload,
                    )
                    yield _format_sse_event("usage", usage_payload)
                    yield _format_sse_event("done", event["data"])

        except Exception as e:
            yield _format_sse_error(f"Unexpected error - {str(e)}", "unexpected_error")
            return

    # SSE vs JSON response
    if stream:
        return StreamingResponse(
            iterate(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    else:
        # Non-streaming: collect content and usage
        started_monotonic = time.monotonic()
        content_accum: list[str] = []
        # Usage fields (same as in iterate())
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        cost: float | None = None
        provider: str | None = None
        actual_model: str | None = None
        cached_tokens: int | None = None
        reasoning_tokens: int | None = None
        audio_tokens: int | None = None
        image_tokens: int | None = None
        prompt_cost: float | None = None
        completion_cost: float | None = None
        is_byok: bool | None = None

        try:
            async for event in stream_chat_completion(
                messages=messages,
                model=final_model,
                providers=(p.routing.providers if p.routing else None),
                fallbacks=(p.routing.fallbacks if p.routing else None),
                sort=(p.routing.sort if p.routing else None),
                max_price=(p.routing.max_price if p.routing else None),
                require_parameters=(p.routing.require_parameters if p.routing else None),
                reasoning=reasoning,
                temperature=(p.sampling.temperature if p.sampling else None),
                top_p=(p.sampling.top_p if p.sampling else None),
                top_k=(p.sampling.top_k if p.sampling else None),
                frequency_penalty=(p.sampling.frequency_penalty if p.sampling else None),
                presence_penalty=(p.sampling.presence_penalty if p.sampling else None),
                repetition_penalty=(p.sampling.repetition_penalty if p.sampling else None),
                min_p=(p.sampling.min_p if p.sampling else None),
                top_a=(p.sampling.top_a if p.sampling else None),
                seed=(p.sampling.seed if p.sampling else None),
                max_tokens=p.max_tokens,
                stop=p.stop,
                logit_bias=p.logit_bias,
                logprobs=p.logprobs,
                top_logprobs=p.top_logprobs,
                response_format=p.response_format,
                tools=p.tools,
                tool_choice=p.tool_choice,
            ):
                if event["type"] == "content":
                    text = event["data"].get("text")
                    if isinstance(text, str):
                        content_accum.append(text)
                elif event["type"] == "reasoning":
                    # Suppress reasoning in non-stream mode to match hide/disable behavior
                    # (we don't currently surface reasoning in non-stream responses)
                    continue
                elif event["type"] == "usage":
                    usage_data = event.get("data", {})
                    prompt_tokens = usage_data.get("prompt_tokens")
                    completion_tokens = usage_data.get("completion_tokens")
                    total_tokens = usage_data.get("total_tokens")
                    cost = usage_data.get("cost")
                    is_byok = usage_data.get("is_byok")
                    provider = usage_data.get("provider")
                    actual_model = usage_data.get("actual_model")
                    prompt_details = usage_data.get("prompt_tokens_details", {})
                    cached_tokens = prompt_details.get("cached_tokens")
                    audio_tokens = prompt_details.get("audio_tokens")
                    completion_details = usage_data.get("completion_tokens_details", {})
                    reasoning_tokens = completion_details.get("reasoning_tokens")
                    image_tokens = completion_details.get("image_tokens")
                    cost_details = usage_data.get("cost_details", {})
                    prompt_cost = cost_details.get("upstream_inference_prompt_cost")
                    completion_cost = cost_details.get("upstream_inference_completions_cost")
                elif event["type"] == "error":
                    # In non-stream mode, surface error as JSON
                    return JSONResponse(
                        {
                            "error": event["data"].get("error"),
                            "type": event["data"].get("type"),
                        },
                        status_code=502,
                    )
                # ignore reasoning/done here; we'll finalize usage below

            duration_ms = int((time.monotonic() - started_monotonic) * 1000)
            duration_s = max(duration_ms / 1000.0, 1e-6)
            usage_payload = {
                "model": final_model,
                "actual_model": actual_model,
                "duration_ms": duration_ms,
                "tokens_per_second": (
                    (completion_tokens or 0) / duration_s if completion_tokens is not None else None
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "is_byok": is_byok,
                "provider": provider,
                "cached_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
                "audio_tokens": audio_tokens,
                "image_tokens": image_tokens,
                "routing": {
                    "providers": (p.routing.providers if p.routing else None),
                    "fallbacks": (p.routing.fallbacks if p.routing else None),
                    "sort": (p.routing.sort if p.routing else None),
                    "max_price": (p.routing.max_price if p.routing else None),
                    "require_parameters": (p.routing.require_parameters if p.routing else None),
                },
            }
            request_id = f"req-{uuid.uuid4().hex}"
            log.info("chat_usage", request_id=request_id, **usage_payload)
            return JSONResponse({"content": "".join(content_accum), "usage": usage_payload})
        except Exception as e:  # pragma: no cover - safety
            return JSONResponse(
                {"error": f"Unexpected error - {str(e)}", "type": "unexpected_error"},
                status_code=500,
            )

# ---------- Caching ----------

# Expose OpenAI-compatible route alias (outside the docstring)
openai_router.add_api_route(
    "/v1/chat/completions",
    chat,
    methods=["POST"],
    response_model=None,
    summary="OpenAI-compatible chat completion endpoint",
)

# Model-capability queries add a network RTT. Cache for 1 hour.
_REASONING_CACHE_TTL = 60 * 60
_reasoning_cache: dict[str, tuple[bool, float]] = {}


async def _supports_reasoning_cached(model: str) -> bool:
    now = time.monotonic()
    hit = _reasoning_cache.get(model)
    if hit and (now - hit[1] < _REASONING_CACHE_TTL):
        return hit[0]
    value = await model_supports_reasoning(model)
    _reasoning_cache[model] = (value, now)
    return value
