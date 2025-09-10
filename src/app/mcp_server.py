# src/app/mcp_server.py
from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from app.config import get_settings

settings = get_settings()

# Create the MCP server (stdio transport by default)
mcp = FastMCP("openrouter-mcp")


@mcp.tool()
async def llm_chat(prompt: str) -> str:
    """
    Send a prompt to the configured model and return the assistant's text output.
    Keep this tool thin: no business logic here.
    """
    try:
        # Defer to single OpenRouter path for simple completions
        from services.openrouter_sse_client import simple_chat_completion

        result = await simple_chat_completion(
            prompt=prompt,
            model=settings.model_name,
            timeout=settings.api_timeout,
        )
        return result
    except asyncio.CancelledError:
        # Re-raise cancellation to allow proper cleanup
        raise
    except Exception as e:
        # Fail fast with clear error message
        return f"Error: {str(e)}"


def main() -> None:  # pragma: no cover
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":  # allows: uv run -m app.mcp_server
    main()
