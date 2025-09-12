
# Copilot Instructions

## Rule 0 — Context7

* **Always check Context7 docs** (MCP server in VS Code) before coding, wiring deps, or designing architecture.

## Project

* **Deps/packaging**: `uv`
* **Venv**: `.venv/` at repo root; VS Code Python = `.venv/bin/python`
* **Tests**: `pytest`, files `test_*.py`
* **Structure**: one responsibility per file

## Transport / OpenRouter

* **Client**: OpenAI SDK (`base_url=https://openrouter.ai/api/v1`) for all chat/streaming
* **Headers**: always send `HTTP-Referer` + `X-Title` (from env)
* **Pass-through only**: `providers, sort, fallbacks, max_price, reasoning`
* **Models**: prefer `sdk.models.list()`, fall back to raw HTTP only if needed

## Tools (MCP-first)

* All tool use = **MCP servers** (discovery, schemas, execution)
* App handles orchestration/retries/validation, not the model
* MCP = **thin layer**, no business logic; log inputs/outputs centrally

## Reasoning Models

* Use **Chat Completions** on OpenRouter
* Request thinking tokens via `reasoning={...}`
* Stream typed frames:

  * `reasoning` → `event: reasoning`
  * `content` → `event: content`
* Use **OpenAI Responses API** only with OpenAI base URL if model requires (e.g., o3/o4)

## Streaming

* **One SSE pipeline**: server emits `reasoning` + `content`
* **Never buffer**, flush deltas immediately
* Handle backpressure: throttle emits if client lags, preserve partials

## Frontend Contract

* SSE events:

  * `reasoning` → “thinking” panel (ephemeral)
  * `content` → chat transcript
  * `usage` / `meta` → optional token/cost info

## Reliability & Async

* **Fail fast** with clear errors
* Catch broad exceptions only at I/O boundaries
* Prefer **async/event-driven**; use async for >10ms ops
* **Always set timeouts** (connect, read, total)
* Retries w/ jitter for idempotent calls
* Never swallow `CancelledError`

## Observability

* Log: request id, model, providers, caps, timing, tokens, finish reason
* Redact secrets; sample payloads only
* Metrics: p95 latency, token rate, SSE drops/retries

## Code Standards

* Python 3.11+
* PEP 8 + type hints, `mypy --strict`
* `ruff` for lint/format/imports

## Security

* Secrets = env/secret manager
* Validate MCP I/O vs schemas
* Enforce model allowlist + price caps

## Principles

* Prefer **simple**
* Delete obsolete paths (e.g., PydanticAI)
* **No fake data** outside tests

---
