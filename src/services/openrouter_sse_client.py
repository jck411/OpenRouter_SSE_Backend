"""
Single OpenRouter SSE client for all chat completions.
Replaces PydanticAI with direct OpenRouter /chat/completions calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from app.config import get_settings

settings = get_settings()


class ReasoningParams(TypedDict, total=False):
    effort: Literal["low", "medium", "high"]
    max_tokens: int
    exclude: bool


class OpenRouterMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class SSEEvent(TypedDict):
    type: Literal["reasoning", "content", "error", "done"]
    data: dict[str, Any]


# Single OpenRouter client with proper headers
_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.base_url,
    default_headers=settings.attribution_headers(),
    timeout=settings.api_timeout,
)


async def stream_chat_completion(
    messages: list[OpenRouterMessage],
    model: str,
    *,
    # OpenRouter routing controls
    providers: list[str] | None = None,
    fallbacks: list[str] | None = None,
    sort: str | None = None,
    max_price: float | None = None,
    require_parameters: bool | None = None,
    # Reasoning controls
    reasoning: ReasoningParams | None = None,
    include_reasoning: bool | None = None,
    # Standard LLM parameters
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    min_p: float | None = None,
    top_a: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
    stop: str | list[str] | None = None,
    logit_bias: dict[str, float] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> AsyncIterator[SSEEvent]:
    """
    Stream chat completion from OpenRouter with typed frames.

    Yields SSEEvent objects:
    - type: "reasoning" → data: {"text": "thinking..."}
    - type: "content" → data: {"text": "response..."}
    - type: "error" → data: {"error": "...", "type": "..."}
    - type: "done" → data: {"completed": True}
    """
    # Build payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    # Add OpenRouter routing controls
    if providers:
        payload["providers"] = providers
    if fallbacks:
        payload["fallbacks"] = fallbacks
    if sort:
        payload["sort"] = sort
    if max_price is not None:
        payload["max_price"] = max_price
    if require_parameters is not None:
        payload["require_parameters"] = require_parameters
    if include_reasoning is not None:
        payload["include_reasoning"] = include_reasoning
    if reasoning:
        payload["reasoning"] = reasoning

    # Add standard LLM parameters
    for key, val in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("top_k", top_k),
        ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty),
        ("repetition_penalty", repetition_penalty),
        ("min_p", min_p),
        ("top_a", top_a),
        ("seed", seed),
        ("max_tokens", max_tokens),
        ("logprobs", logprobs),
        ("top_logprobs", top_logprobs),
    ):
        if val is not None:
            payload[key] = val

    # Handle complex parameters
    if stop is not None:
        payload["stop"] = stop
    if logit_bias is not None:
        payload["logit_bias"] = logit_bias
    if response_format is not None:
        payload["response_format"] = response_format
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    headers = {
        **settings.attribution_headers(),
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        async with (
            httpx.AsyncClient(timeout=settings.api_timeout) as client,
            client.stream(
                "POST",
                f"{settings.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            ) as response,
        ):
            if response.status_code != 200:
                # Try to read error body
                try:
                    body = await response.aread()
                    body_txt = body.decode("utf-8", errors="replace")
                except Exception:
                    body_txt = ""
                yield SSEEvent(
                    type="error",
                    data={
                        "error": f"Upstream returned {response.status_code}. {body_txt[:400]}",
                        "type": "upstream_http_error",
                    },
                )
                return

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Handle OpenRouter typed frames
                frame_type = frame.get("type")
                if frame_type == "reasoning":
                    delta = frame.get("delta") or ""
                    if delta:
                        yield SSEEvent(type="reasoning", data={"text": delta})
                    continue
                elif frame_type in ("message", "content"):
                    delta = frame.get("delta") or ""
                    if delta:
                        yield SSEEvent(type="content", data={"text": delta})
                    continue

                # Handle OpenAI-style choices
                if "choices" in frame:
                    choice0 = (frame.get("choices") or [{}])[0]
                    delta = choice0.get("delta", {}) or {}
                    content_delta = delta.get("content")
                    if content_delta:
                        yield SSEEvent(type="content", data={"text": content_delta})

                    # Handle reasoning in different shapes
                    reasoning_part = delta.get("reasoning")
                    if reasoning_part:
                        if isinstance(reasoning_part, str):
                            yield SSEEvent(type="reasoning", data={"text": reasoning_part})
                        elif isinstance(reasoning_part, dict):
                            if reasoning_part.get("content"):
                                yield SSEEvent(
                                    type="reasoning", data={"text": reasoning_part["content"]}
                                )
                            tokens = reasoning_part.get("tokens")
                            if isinstance(tokens, list):
                                for tok in tokens:
                                    tok_content = tok.get("content")
                                    if tok_content:
                                        yield SSEEvent(type="reasoning", data={"text": tok_content})

        yield SSEEvent(type="done", data={"completed": True})

    except asyncio.CancelledError:
        raise
    except TimeoutError:
        yield SSEEvent(type="error", data={"error": "Request timed out", "type": "timeout"})
        return
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        yield SSEEvent(
            type="error", data={"error": f"Network error - {str(e)}", "type": "network_error"}
        )
        return
    except (APITimeoutError, APIConnectionError):
        yield SSEEvent(
            type="error",
            data={"error": "API service temporarily unavailable", "type": "api_unavailable"},
        )
        return
    except APIError as e:
        error_msg = getattr(e, "message", str(e))
        yield SSEEvent(
            type="error", data={"error": f"API error - {error_msg}", "type": "api_error"}
        )
        return
    except (ConnectionError, OSError) as e:
        yield SSEEvent(
            type="error",
            data={"error": f"Connection failed - {str(e)}", "type": "connection_error"},
        )
        return
    except Exception as e:
        yield SSEEvent(
            type="error", data={"error": f"Unexpected error - {str(e)}", "type": "unexpected_error"}
        )
        return


async def simple_chat_completion(
    prompt: str,
    model: str,
    *,
    timeout: float | None = None,
) -> str:
    """
    Simple non-streaming chat completion for MCP tools.
    Returns just the text content.
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        # Apply timeout to the entire operation
        async with asyncio.timeout(timeout or settings.api_timeout):
            response = await _client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                stream=False,
            )
            # Since stream=False, response is definitely ChatCompletion
            assert hasattr(response, "choices"), "Expected ChatCompletion response"
            return response.choices[0].message.content or ""
    except TimeoutError:
        return f"Error: Operation timed out after {timeout or settings.api_timeout}s"
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return f"Error: {str(e)}"


async def get_models_list() -> list[dict[str, Any]]:
    """Get list of available models from OpenRouter."""
    try:
        resp = await _client.models.list()
        models_data: list[dict[str, Any]] = []
        for m in resp.data:
            if hasattr(m, "model_dump"):
                models_data.append(m.model_dump())
            else:
                # Convert to dict safely
                models_data.append(dict(m))
        return models_data
    except Exception:
        return []


async def model_supports_reasoning(model_id: str) -> bool:
    """Check if a model supports reasoning based on cached metadata."""
    models = await get_models_list()
    model_obj = None

    for model in models:
        if isinstance(model, dict) and model.get("id") == model_id:
            model_obj = model
            break

    if not model_obj:
        # Try base model without variant
        base = model_id.split(":")[0] if ":" in model_id else model_id
        for model in models:
            if isinstance(model, dict) and model.get("id") == base:
                model_obj = model
                break

    return _extract_reasoning_capability(model_obj or {})


def _extract_reasoning_capability(model_obj: dict[str, Any]) -> bool:
    """Best-effort detection whether a model supports reasoning."""
    # model_obj is guaranteed to be a dict (or empty dict) by the caller

    caps = model_obj.get("capabilities") or {}
    if isinstance(caps, dict) and bool(caps.get("reasoning")):
        return True

    features = model_obj.get("features")
    if isinstance(features, list) and any(
        isinstance(f, str) and f.lower() in {"reasoning", "chain_of_thought"} for f in features
    ):
        return True

    tags = model_obj.get("tags")
    if isinstance(tags, list) and any(
        isinstance(t, str) and "reasoning" in t.lower() for t in tags
    ):
        return True

    topology = model_obj.get("topology")
    if isinstance(topology, str) and topology.lower() in {"reasoning", "reasoner"}:
        return True

    for key in ("modalities", "types"):
        val = model_obj.get(key)
        if isinstance(val, list) and any(
            isinstance(v, str) and "reasoning" in v.lower() for v in val
        ):
            return True

    return False
