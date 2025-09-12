# OpenRouter SSE Backend

A FastAPI backend that provides streaming chat APIs directly through OpenRouter, supporting reasoning-capable models with native SSE streaming.

## Features

- **Direct OpenRouter Integration**: Single SSE client for all chat completions
- **Reasoning Support**: Forward OpenRouter typed frames (reasoning + content)
- **MCP Server**: Model Context Protocol server for tool integration
- **Model Discovery**: Search and filter OpenRouter models
- **Parameter Pass-through**: Full OpenRouter API parameter support

## Architecture

- ✅ **End-to-end streaming** - Direct OpenRouter SSE pipeline
- ✅ **Reasoning visibility** - Stream both reasoning and content events
- ✅ **OpenRouter pass-through** - Support for providers, routing, web search
- ✅ **MCP compatibility** - FastMCP server for tool execution
- ✅ **Type safety** - Full mypy typing with structured SSE events

## Quick Start

```bash
git clone <repository>
cd openrouter-sse-backend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
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

**SSE Events:**
- `event: reasoning` - Model thinking process `{"text": "..."}`
- `event: content` - Response content `{"text": "..."}`
- `event: error` - Error information `{"error": "...", "type": "..."}`
- `event: done` - Completion signal `{"completed": true}`

**OpenRouter Parameters:**
- `sort` - Sort strategy (throughput, price)
- `providers` - CSV list of preferred providers
- `max_price` - Maximum cost per request
- `fallbacks` - CSV list of fallback models
- `reasoning_effort` - low, medium, high
- `reasoning_exclude` - Hide reasoning from response
- `disable_reasoning` - Force disable reasoning

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

## SSE Client Example

```javascript
const eventSource = new EventSource('/chat', {
  method: 'POST',
  body: JSON.stringify({
    message: "Explain quantum computing",
    model: "openai/o1-preview"
  })
});

eventSource.addEventListener('reasoning', (event) => {
  const data = JSON.parse(event.data);
  console.log('Thinking:', data.text);
});

eventSource.addEventListener('content', (event) => {
  const data = JSON.parse(event.data);
  console.log('Response:', data.text);
});
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

## Technical Details

- **Transport Layer**: Direct OpenRouter HTTP/SSE client
- **AI Layer**: Single `stream_chat_completion` function for all operations
- **Event Handling**: Structured SSE with typed frames
- **Simplified**: 90% code reduction by removing PydanticAI abstraction layer

## Development

```bash
# Install dev dependencies
uv pip install -e .[dev]

# Run tests
pytest

# Lint and format
ruff check src tests
ruff format src tests

# Type check
mypy src
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
