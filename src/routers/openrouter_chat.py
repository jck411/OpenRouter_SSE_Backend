from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.logger import log
from services.openrouter_sse_client import ReasoningParams, stream_chat_completion

from .openrouter_chat_utils import supports_reasoning_cached, to_openrouter_messages
from .openrouter_streaming import SSE_HEADERS, format_sse_error, format_sse_event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .openrouter_chat_schemas import ChatCompletionPayload


router = APIRouter(prefix="/chat", tags=["chat"])
openai_router = APIRouter(tags=["chat"])  # OpenAI-compatible alias
settings = get_settings()


@router.post("", response_model=None, summary="OpenAI-compatible chat completion endpoint")
async def chat(
    payload: ChatCompletionPayload,
    stream: bool = Query(True, description="If true, stream as SSE; else return JSON response"),
) -> Response:
    p = payload
    if not p.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    history_messages = to_openrouter_messages(p.history)

    final_model = p.model or settings.model_name
    if p.web_search and not final_model.endswith(":online"):
        final_model = f"{final_model}:online"

    reasoning: ReasoningParams | None = None
    _hide: bool | None = p.hide_reasoning
    if p.reasoning_effort or (p.reasoning_max_tokens is not None) or (_hide is not None):
        r: ReasoningParams = {}
        if p.reasoning_effort:
            r["effort"] = p.reasoning_effort
        if p.reasoning_max_tokens is not None:
            r["max_tokens"] = p.reasoning_max_tokens
        if _hide is not None:
            r["exclude"] = _hide
        reasoning = r
    else:
        if not p.disable_reasoning and await supports_reasoning_cached(final_model):
            reasoning = {"effort": "high"}
    if p.disable_reasoning:
        reasoning = None

    messages = history_messages + [{"role": "user", "content": p.message}]

    async def iterate() -> AsyncIterator[str]:
        started_monotonic = time.monotonic()
        reasoning_events = 0
        content_events = 0
        suppress_reasoning_stream = bool(p.hide_reasoning) or bool(p.disable_reasoning)

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
                etype = event.get("type")
                if etype == "reasoning":
                    if not suppress_reasoning_stream:
                        reasoning_events += 1
                        yield format_sse_event("reasoning", event["data"])
                elif etype == "content":
                    content_events += 1
                    yield format_sse_event("content", event["data"])
                elif etype == "usage":
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
                elif etype == "error":
                    yield format_sse_event("error", event["data"])
                elif etype == "done":
                    duration_ms = int((time.monotonic() - started_monotonic) * 1000)
                    duration_s = max(duration_ms / 1000.0, 1e-6)
                    usage_payload = {
                        "model": final_model,
                        "actual_model": actual_model,
                        "duration_ms": duration_ms,
                        "reasoning_events": reasoning_events,
                        "content_events": content_events,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": (
                            (completion_tokens or 0) / duration_s
                            if completion_tokens is not None
                            else None
                        ),
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
                            "require_parameters": (
                                p.routing.require_parameters if p.routing else None
                            ),
                        },
                    }
                    request_id = f"req-{uuid.uuid4().hex}"
                    log.info("chat_usage", request_id=request_id, **usage_payload)
                    yield format_sse_event("usage", usage_payload)
                    yield format_sse_event("done", event["data"])
        except Exception as e:
            yield format_sse_error(f"Unexpected error - {str(e)}", "unexpected_error")
            return

    if stream:
        return StreamingResponse(iterate(), media_type="text/event-stream", headers=SSE_HEADERS)

    # Non-streaming aggregate mode
    started_monotonic = time.monotonic()
    content_accum: list[str] = []
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
            etype = event.get("type")
            if etype == "content":
                text = event["data"].get("text")
                if isinstance(text, str):
                    content_accum.append(text)
            elif etype == "reasoning":
                continue
            elif etype == "usage":
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
            elif etype == "error":
                return JSONResponse(
                    {"error": event["data"].get("error"), "type": event["data"].get("type")},
                    status_code=502,
                )

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
    except Exception:
        return JSONResponse(
            {"error": "Unexpected error - non-stream", "type": "unexpected_error"},
            status_code=500,
        )


openai_router.add_api_route(
    "/v1/chat/completions",
    chat,
    methods=["POST"],
    response_model=None,
    summary="OpenAI-compatible chat completion endpoint",
)
