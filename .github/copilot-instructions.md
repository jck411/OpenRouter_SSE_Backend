# COPILOT.md — Concise Repo Instructions

## Rule 0 — Context7
- Before writing code, dependencies, or architecture: consult **Context7 docs** (MCP server in VS Code).

## Project
- Deps & packaging: **`uv`**
- Virtualenv: **`.venv/` at repo root**; VS Code Python = `.venv/bin/python`
- Tests: **`pytest`**, files **`test_*.py`**
- Structure: **one responsibility per file**

## Transport / OpenRouter
- Client: **OpenAI SDK** with `base_url=https://openrouter.ai/api/v1` for **all chat + streaming**
- Always send headers: **`HTTP-Referer`** and **`X-Title`** (from env) on every request
- Pass-through knobs (no reinterpretation): **providers, sort, fallbacks, max_price, reasoning, include_reasoning**
- Model listing: prefer **`sdk.models.list()`**; only hit raw HTTP if SDK lacks a feature

## Tools (MCP-first)
- **All tool use goes through MCP servers** (discovery, schemas, execution)
- The app controls orchestration/retries/validation around MCP results (not the model)
- Keep MCP **thin** (no business logic); log inputs/outputs centrally

## Reasoning Models
- Use **Chat Completions** on OpenRouter
- Request thinking tokens with **`reasoning={...}`** and/or **`include_reasoning`** when supported
- Treat **OpenRouter typed frames**:
  - `type: "reasoning"` → stream to client as `event: reasoning`
  - `type: "content"` / message deltas → `event: content`
- Only use **OpenAI Responses API** when calling **OpenAI’s own base URL** AND a model **requires it** (e.g., o3/o4 variants)

## Streaming
- **Single end-to-end SSE pipeline** (server emits `reasoning` and `content` events)
- **Never buffer** full responses; flush deltas as they arrive
- Backpressure: throttle server emits if client lags; surface partials safely
- Media policy:
  - **Default SSE** for text/JSON
  - If doing **live audio/video or binary streaming**, add a **WebSocket** path (or **WebRTC** for real-time A/V)
  - Otherwise ship media via **HTTP uploads/downloads** (pre-signed URLs)

## Frontend Contract
- Listen for SSE events:
  - `event: reasoning` → append to “thinking” panel (hideable)
  - `event: content` → append to chat transcript
  - `event: usage` / `event: meta` (optional) for token/cost/finish_reason
- Do not persist “reasoning” content; treat as ephemeral UI

## Reliability & Async
- **Fail fast** with clear error types/messages
- Catch broad exceptions **only at I/O boundaries**
- Prefer **async/event-driven**; use **async I/O** for ops > ~10ms
- **Always set timeouts** (connect, read, total)
- Implement **retries with jitter** for idempotent calls
- Never swallow **`CancelledError`**; propagate cancellation

## Observability
- Log: request id, model, provider(s), pricing caps, timing, token usage, finish reason
- Redact secrets; sample payloads for large requests
- Metrics: p95 latency, token rate, SSE drop/retry counts

## Code Standards
- **Python 3.11+**
- **PEP 8**, type hints, **`mypy --strict`**
- **`ruff`** for format/lint/imports

## Security
- Secrets via env/secret manager; never commit
- Validate MCP tool inputs/outputs against declared schemas
- Enforce **model allowlist** and **price caps** at request time

## Principles
- Prefer **simple** solutions
- **Remove obsolete** code (delete the PydanticAI path)
- **No fake data** outside tests
