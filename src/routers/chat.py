from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
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
settings = get_settings()

# ---------- Models ----------


class Message(BaseModel):
    role: Literal["user", "model"]
    content: str


class ChatRequest(BaseModel):
    history: list[Message] = Field(default_factory=list)
    message: str = Field(..., min_length=1)
    model: str | None = Field(None, description="Model ID to use for the chat")
    web_search: bool = Field(False, description="Enable web search for this request")


# ---------- Utilities ----------

SSE_HEADERS = {
    # Reduce buffering by proxies / Nginx
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    # Browsers add Connection: keep-alive automatically; safe to omit
}


def _csv_to_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    out = [v.strip() for v in value.split(",")]
    return [v for v in out if v]


def _json_loads_or_passthrough(s: str | None) -> Any | None:
    """Try json.loads; if that fails, return original string (or None)."""
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _strict_json_or_error(
    param_value: str | None, param_name: str
) -> tuple[bool, Any | None, str | None]:
    """If value looks like JSON (starts with '{' or '['), require valid JSON.
    Otherwise, pass through unchanged.
    Returns (ok, value_or_none, error_message_or_none).
    """
    if param_value is None:
        return True, None, None
    val = param_value.strip()
    if not val:
        return True, None, None
    if val[0] in "[{":
        try:
            return True, json.loads(val), None
        except json.JSONDecodeError as e:
            return False, None, f"Malformed JSON in '{param_name}': {str(e)}"
    # Not JSON-looking, passthrough as original string
    return True, param_value, None


def _format_sse_event(event_type: str, data: Any) -> str:
    """Format data as Server-Sent Events (SSE).
    If data is dict/list, JSON-encode; then emit one 'data:' per line.
    """
    payload = json.dumps(data) if isinstance(data, dict | list) else str(data)
    # SSE requires each line to start with 'data:'; split to be robust to newlines
    lines = payload.splitlines() or [""]
    buf = [f"event: {event_type}"]
    for ln in lines:
        buf.append(f"data: {ln}")
    buf.append("")  # blank line terminator
    return "\n".join(buf) + "\n"


def _format_sse_error(error_message: str, error_type: str = "error") -> str:
    """Format error as SSE event with structured data."""
    return _format_sse_event("error", {"error": error_message, "type": error_type})


def _to_openrouter_messages(history: list[Message]) -> list[OpenRouterMessage]:
    """Convert API message history to OpenRouter messages."""
    out: list[OpenRouterMessage] = []
    for m in history:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        else:
            out.append({"role": "assistant", "content": m.content})
    return out


# ---------- Route ----------


@router.post("")
async def chat(
    req: ChatRequest,
    # OpenRouter routing & cost controls:
    sort: str | None = Query(None, description="OpenRouter sort strategy (e.g. throughput, price)"),
    providers: str | None = Query(None, description="CSV list of preferred providers"),
    max_price: float | None = Query(None, ge=0, description="Maximum cost per request"),
    fallbacks: str | None = Query(None, description="CSV list of fallback models"),
    require_parameters: bool | None = Query(
        None, description="Require all providers support all parameters"
    ),
    # Model behavior parameters:
    temperature: float | None = Query(None, ge=0, le=2, description="Response randomness (0-2)"),
    top_p: float | None = Query(None, ge=0, le=1, description="Nucleus sampling (0-1)"),
    top_k: int | None = Query(None, ge=1, description="Top-k sampling"),
    frequency_penalty: float | None = Query(
        None, ge=-2, le=2, description="Reduce repetition (-2 to 2)"
    ),
    presence_penalty: float | None = Query(
        None, ge=-2, le=2, description="Encourage new topics (-2 to 2)"
    ),
    repetition_penalty: float | None = Query(
        None, ge=0, description="Alternative repetition control"
    ),
    min_p: float | None = Query(None, ge=0, le=1, description="Minimum probability threshold"),
    top_a: float | None = Query(None, ge=0, le=1, description="Alternative sampling method"),
    seed: int | None = Query(None, description="Seed for reproducible outputs"),
    # Output controls:
    max_tokens: int | None = Query(None, ge=1, description="Maximum response length"),
    stop: str | None = Query(None, description="Stop sequences (JSON array or single string)"),
    logit_bias: str | None = Query(None, description="Token bias as JSON object"),
    logprobs: bool | None = Query(None, description="Return log probabilities"),
    top_logprobs: int | None = Query(
        None, ge=0, le=20, description="Number of top logprobs to return"
    ),
    response_format: str | None = Query(None, description="Response format (JSON schema)"),
    # Function calling:
    tools: str | None = Query(None, description="Available tools as JSON array"),
    tool_choice: str | None = Query(None, description="Tool selection strategy"),
    # Reasoning controls (OpenRouter reasoning-capable models):
    reasoning_effort: Literal["low", "medium", "high"] | None = Query(
        None, description="Reasoning effort hint"
    ),
    reasoning_max_tokens: int | None = Query(None, ge=1),
    reasoning_exclude: bool | None = Query(None, description="If true, hide reasoning trace"),
    include_reasoning: bool | None = Query(
        None, description="If true, include reasoning in the response text stream"
    ),
    disable_reasoning: bool | None = Query(
        None,
        description="Force disable reasoning entirely (overrides any reasoning_* params and auto-detection)",
    ),
) -> StreamingResponse:
    """
    Streams assistant response as Server-Sent Events (SSE) with structured data.

    SSE Events:
    - event: reasoning - streaming reasoning deltas {"text": "..."}
    - event: content   - streaming content deltas   {"text": "..."}
    - event: error     - structured error           {"error": "...", "type": "..."}
    - event: done      - completion signal          {"completed": true}
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # Parse CSV parameters -> lists
    providers_list = _csv_to_list(providers)
    fallbacks_list = _csv_to_list(fallbacks)

    # Convert API messages -> OpenRouter messages
    history_messages = _to_openrouter_messages(req.history)

    # Determine model (optionally ensure :online)
    final_model = req.model or settings.model_name
    if req.web_search and not final_model.endswith(":online"):
        final_model = f"{final_model}:online"

    # Build reasoning dict (normalize conflicting flags)
    reasoning: ReasoningParams | None = None
    if reasoning_effort or (reasoning_max_tokens is not None) or (reasoning_exclude is not None):
        r: ReasoningParams = {}
        if reasoning_effort:
            r["effort"] = reasoning_effort
        if reasoning_max_tokens is not None:
            r["max_tokens"] = reasoning_max_tokens
        if reasoning_exclude is not None:
            r["exclude"] = reasoning_exclude
        reasoning = r
    else:
        # If the model supports reasoning (and caller didn't specify), opt-in with a sensible default.
        if not disable_reasoning and await model_supports_reasoning(final_model):
            reasoning = {"effort": "high"}

    # If exclude=True and include_reasoning=True, prefer exclude (don't leak)
    if reasoning and reasoning.get("exclude") and include_reasoning:
        include_reasoning = False

    # Explicit override: disable reasoning completely
    if disable_reasoning:
        reasoning = None
        include_reasoning = False

    # Parse JSON parameters (strict for JSON-looking values)
    parse_errors: list[str] = []
    # stop: allow single string OR JSON array/object. If JSON-looking, require valid JSON
    ok, parsed_stop, err = _strict_json_or_error(stop, "stop")
    if not ok and err:
        parse_errors.append(err)
    ok, parsed_logit_bias, err = _strict_json_or_error(logit_bias, "logit_bias")
    if not ok and err:
        parse_errors.append(err)
    ok, parsed_response_format, err = _strict_json_or_error(response_format, "response_format")
    if not ok and err:
        parse_errors.append(err)
    ok, parsed_tools, err = _strict_json_or_error(tools, "tools")
    if not ok and err:
        parse_errors.append(err)
    ok, parsed_tool_choice, err = _strict_json_or_error(tool_choice, "tool_choice")
    if not ok and err:
        parse_errors.append(err)

    # Add user message to conversation
    messages = history_messages + [{"role": "user", "content": req.message}]

    # If any parsing errors, emit structured SSE error and finish
    if parse_errors:

        async def iterate_error() -> AsyncIterator[str]:
            yield _format_sse_error(
                "; ".join(parse_errors),
                "bad_request",
            )
            yield _format_sse_event("done", {"completed": False})

        return StreamingResponse(
            iterate_error(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    async def iterate() -> AsyncIterator[str]:
        started_monotonic = time.monotonic()
        reasoning_events = 0
        content_events = 0
        total_reasoning_chars = 0
        total_content_chars = 0
        try:
            async for event in stream_chat_completion(
                messages=messages,
                model=final_model,
                # OpenRouter routing controls
                providers=providers_list,
                fallbacks=fallbacks_list,
                sort=sort,
                max_price=max_price,
                require_parameters=require_parameters,
                # Reasoning controls
                reasoning=reasoning,
                include_reasoning=include_reasoning,
                # Standard LLM parameters
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_a=top_a,
                seed=seed,
                max_tokens=max_tokens,
                stop=parsed_stop,
                logit_bias=parsed_logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                response_format=parsed_response_format,
                tools=parsed_tools,
                tool_choice=parsed_tool_choice,
            ):
                if event["type"] == "reasoning":
                    txt = str((event.get("data") or {}).get("text", ""))
                    total_reasoning_chars += len(txt)
                    reasoning_events += 1
                    yield _format_sse_event("reasoning", event["data"])
                elif event["type"] == "content":
                    txt = str((event.get("data") or {}).get("text", ""))
                    total_content_chars += len(txt)
                    content_events += 1
                    yield _format_sse_event("content", event["data"])
                elif event["type"] == "error":
                    yield _format_sse_event("error", event["data"])
                elif event["type"] == "done":
                    # Emit final usage/meta event before done
                    duration_ms = int((time.monotonic() - started_monotonic) * 1000)
                    usage_payload = {
                        "model": final_model,
                        "duration_ms": duration_ms,
                        "reasoning_events": reasoning_events,
                        "content_events": content_events,
                        "reasoning_chars": total_reasoning_chars,
                        "content_chars": total_content_chars,
                        # Placeholders for tokens; upstream does not include them in SSE
                        "input_tokens": None,
                        "output_tokens": None,
                        # Echo core routing knobs for observability
                        "routing": {
                            "providers": providers_list,
                            "fallbacks": fallbacks_list,
                            "sort": sort,
                            "max_price": max_price,
                            "require_parameters": require_parameters,
                        },
                    }
                    # Emit usage log (structured) with a request correlation id
                    # For simplicity, use id() of messages list as a lightweight request_id surrogate
                    request_id = f"req-{id(messages)}"
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

    return StreamingResponse(
        iterate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
