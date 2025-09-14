# OpenRouter SSE Backend

A FastAPI backend that provides streaming chat APIs directly through OpenRouter, supporting reasoning-capable models with native SSE streaming.

## Features

- **Direct OpenRouter Integration**: Single SSE client for all chat completions
- **Reasoning Support**: Forward OpenRouter typed frames (reasoning + content)
- **MCP Server**: Model Context Protocol server for tool integration
- **Model Discovery**: Search and filter OpenRouter models
- **Parameter Pass-through**: Full OpenRouter API parameter support
- **OpenAI-compatible**: Also exposes `POST /v1/chat/completions`
- **Shared Modules**: Reusable SSE helpers, chat schemas, and utilities

## Architecture

- ✅ **End-to-end streaming** - Direct OpenRouter SSE pipeline
- ✅ **Reasoning visibility** - Stream both reasoning and content events
- ✅ **OpenRouter pass-through** - Support for providers, routing, web search
- ✅ **MCP compatibility** - FastMCP server for tool execution
- ✅ **Type safety** - Full mypy typing with structured SSE events
- ✅ **Modular routers** - Shared schemas and utilities across routes

## Quick Start

```bash
git clone <repository>
cd OpenRouter_SSE_Backend
uv sync --dev
# optional: activate if you want the interpreter in your shell
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Environment

```bash
cp .env.example .env
# Edit .env with your keys:
OPENROUTER_API_KEY=your_key_here
HTTP_REFERER=https://yourapp.com
X_TITLE=YourApp
```

### Run

```bash
# FastAPI server
uv run openrouter-server

# MCP server
uv run openrouter-mcp
```

## API Endpoints

### Chat Streaming

`POST /chat` - Stream chat completions with SSE events

Alias: `POST /v1/chat/completions` (OpenAI-compatible path)

**SSE Events:**
- `event: reasoning` - Model thinking process `{"text": "..."}`
- `event: content` - Response content `{"text": "..."}`
- `event: usage` - Final usage and routing details (tokens, cost, provider)
- `event: error` - Error information `{"error": "...", "type": "..."}`
- `event: done` - Completion signal `{"completed": true}`

**Parameters (selected):**
- Routing: `providers`, `fallbacks`, `sort`, `max_price`, `require_parameters`
- Sampling: `temperature`, `top_p`, `top_k`, `penalties`, `seed`, `max_tokens`, `stop`, `logit_bias`, `logprobs`, `top_logprobs`, `response_format`
- Tools: `tools`, `tool_choice`
- Reasoning: `reasoning_effort`, `reasoning_max_tokens`, `hide_reasoning`, `disable_reasoning`
- Web search: `web_search` (adds `:online` model suffix when true)

## Reasoning Model Support

This backend supports multiple reasoning model families with different parameter requirements:

### OpenAI Models (o1, o3 series)
- **Models**: `openai/o1-mini`, `openai/o1-preview`, `openai/o3-mini`, `openai/o3`, etc.
- **Parameters**: `reasoning_effort` (top-level parameter)
- **Values**: `"low"`, `"medium"`, `"high"`
- **Special handling**: Uses `reasoning_effort` instead of `reasoning` object
- **Example**: `{"reasoning_effort": "high"}`

### DeepSeek Models (R1 series)
- **Models**: `deepseek/deepseek-r1`, `deepseek/deepseek-r1-distill-*`, etc.
- **Parameters**: `reasoning` object + `include_reasoning` boolean
- **Special handling**: Maps `reasoning_exclude` to inverse of `include_reasoning`
- **Example**: `{"reasoning": {"effort": "high"}, "include_reasoning": true}`

### Qwen Models (Thinking variants)
- **Models**: `qwen/qwen3-next-*-thinking`, `qwen/qwq-32b-preview`, etc.
- **Parameters**: Standard `reasoning` object
- **Example**: `{"reasoning": {"effort": "high", "exclude": false}}`

### Other Reasoning Models
- **Models**: Various providers with reasoning capabilities
- **Parameters**: Detected via `supported_parameters` list
- **Common parameters**: `reasoning`, `include_reasoning`

### Model Discovery

`GET /models` - List all available models
`GET /models/search` - Search and filter models
`GET /models/{model_id}` - Get detailed model information

### Health Check

`GET /health` - Service health status

## SSE Client Examples

Curl (streamed):

```bash
curl -N -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Explain quantum computing",
    "model": "openrouter/auto"
  }' \
  http://localhost:8000/chat
```

Node/JS (stream parsing via fetch):

```js
const res = await fetch('http://localhost:8000/chat?stream=true', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hello!', model: 'openrouter/auto' })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let idx;
  while ((idx = buffer.indexOf('\n\n')) !== -1) {
    const chunk = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    // Basic SSE parse: lines starting with "event:" and "data:"
    const lines = chunk.split('\n');
    let event, data;
    for (const line of lines) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      if (line.startsWith('data:')) data = line.slice(5).trim();
    }
    if (event && data) {
      const parsed = JSON.parse(data);
      if (event === 'reasoning') console.log('Thinking:', parsed.text);
      if (event === 'content') console.log('Content:', parsed.text);
      if (event === 'usage') console.log('Usage:', parsed);
      if (event === 'done') console.log('Done');
    }
  }
}
```

## Configuration

Environment variables:

- `OPENROUTER_API_KEY` - Your OpenRouter API key
- `HTTP_REFERER` - Your application URL (required by OpenRouter)
- `X_TITLE` - Your application name (required by OpenRouter)
- `MODEL_NAME` - Default model (default: "openai/gpt-4o-mini")
- `HOST` - Server host (default: "0.0.0.0")
- `PORT` - Server port (default: 8000)
- `API_TIMEOUT` - Request timeout in seconds (default: 300)

Transport notes:
- Uses OpenAI SDK with `base_url=https://openrouter.ai/api/v1`
- Always sends `HTTP-Referer` and `X-Title` headers per OpenRouter requirements

## Technical Details

- **Transport Layer**: Direct OpenRouter HTTP/SSE client
- **AI Layer**: Single `stream_chat_completion` function for all operations
- **Event Handling**: Structured SSE with typed frames
- **Simplified**: 90% code reduction by removing PydanticAI abstraction layer
- **Shared Modules**:
  - `routers/openrouter_streaming.py` – SSE headers and event formatting
  - `routers/openrouter_chat_schemas.py` – Pydantic request models
  - `routers/openrouter_chat_utils.py` – message conversion + reasoning TTL cache
- **OpenAI-compatible alias**: `/v1/chat/completions` maps to the same handler
- **Reasoning auto-config**: If model supports reasoning and not disabled, auto-enables sensible defaults; hide via `hide_reasoning` or fully disable via `disable_reasoning`
- **Quality gates**: `uv run ruff check`, `uv run mypy --strict`, `uv run pytest -q`

## Development

```bash
# Install dev dependencies and sync lock
uv sync --dev

# Run tests
uv run pytest -q

# Lint and format
uv run ruff check src tests
uv run ruff format src tests

# Type check
uv run mypy --strict src
```

## MCP Server

The included MCP server provides a simple `llm_chat` tool:

```bash
# Run MCP server
uv run openrouter-mcp

# Use with MCP clients (VS Code, etc.)
```

## Migration from PydanticAI

This backend replaces PydanticAI with direct OpenRouter integration:

- **Removed**: `pydantic-ai` dependency
- **Single Path**: OpenRouter `/chat/completions` SSE via OpenAI SDK
- **Kept**: Same API surface and SSE event structure
- **Added**: Better reasoning support and parameter pass-through

## Testing and Monkeypatching

Tests patch the chat router module directly to simulate behaviors:
- `routers.chat.stream_chat_completion`
- `routers.chat.model_supports_reasoning`

The router re-exports a TTL-cached reasoning capability check under `model_supports_reasoning` so tests remain patchable while production benefits from caching.

Run the suite:

```bash
uv run pytest -q
```

## Nice to haves

- Centralize reasoning capability caching in the services layer, and keep a `routers.chat.model_supports_reasoning` alias for test monkeypatching.
- Add focused unit tests for the TTL cache (e.g., using time-freeze or a small TTL) to protect against regressions.
- Expand structured logging around SSE (request id, model, providers, timing, tokens, cost, finish reason) and add basic metrics.
- Document the request schema (`routers/openrouter_chat_schemas.py`) and monkeypatch surfaces for future contributors.
- Add lightweight rate limiting and backpressure handling guidance in docs for high-throughput scenarios.
- Enforce model allowlist and max price caps in configuration, surfaced clearly in logs and errors.
