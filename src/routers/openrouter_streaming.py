from __future__ import annotations

import json
from typing import Any

# Default headers for SSE streams. Explicit content-type prevents buffering.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # Disable Nginx buffering
    "Content-Type": "text/event-stream; charset=utf-8",
}


def sanitize_sse_line(line: str) -> str:
    """Sanitize a single SSE line (prefix ':' lines so browsers don't drop them)."""
    return f" {line}" if line.startswith(":") else line


def format_sse_event(event_type: str, data: Any) -> str:
    """Turn data into an SSE message with event type + data lines."""
    payload = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
    lines = payload.splitlines() or [""]
    buf = [f"event: {event_type}"]
    for ln in lines:
        buf.append(f"data: {sanitize_sse_line(ln)}")
    buf.append("")
    return "\n".join(buf) + "\n"


def format_sse_error(error_message: str, error_type: str = "error") -> str:
    """Standardized SSE error event."""
    return format_sse_event("error", {"error": error_message, "type": error_type})
