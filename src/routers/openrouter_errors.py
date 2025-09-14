from __future__ import annotations

import asyncio

import httpx
from fastapi import HTTPException
from openai import APIConnectionError, APIError, APITimeoutError


def to_http_exception(e: Exception, *, default_detail: str) -> HTTPException:
    """Translate lower-level exceptions into FastAPI HTTPException with consistent semantics.

    IMPORTANT: This will re-raise asyncio.CancelledError to let the server cancel cleanly.
    """
    if isinstance(e, HTTPException):
        return e

    if isinstance(e, asyncio.CancelledError):
        raise e

    # OpenAI SDK errors
    if isinstance(e, APITimeoutError):
        return HTTPException(status_code=504, detail="OpenRouter API timeout")
    if isinstance(e, APIConnectionError):
        return HTTPException(status_code=503, detail="Cannot connect to OpenRouter API")
    if isinstance(e, APIError):
        # Try to detect rate limiting
        body = getattr(e, "body", {})
        try:
            code = body.get("error", {}).get("code")
        except Exception:
            code = None
        if code == "rate_limit_exceeded":
            return HTTPException(
                status_code=429,
                detail="Rate limited by OpenRouter. Please reduce request frequency and try again later.",
            )
        return HTTPException(status_code=502, detail=f"OpenRouter API error: {e.message}")

    # httpx errors used for the endpoints call
    if isinstance(e, httpx.TimeoutException):
        return HTTPException(status_code=504, detail="OpenRouter API timeout")
    if isinstance(e, httpx.ConnectError):
        return HTTPException(status_code=503, detail="Cannot connect to OpenRouter API")
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 404:
            return HTTPException(status_code=404, detail=str(e))
        if status == 429:
            return HTTPException(
                status_code=429,
                detail="Rate limited by OpenRouter. Please reduce request frequency and try again later.",
            )
        return HTTPException(status_code=502, detail=f"OpenRouter API error: {status}")

    # Fallback
    return HTTPException(status_code=500, detail=default_detail)
