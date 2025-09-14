from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from routers.chat import openai_router, router as chat_router
from routers.openrouter_models import router as models_router

settings = get_settings()

app = FastAPI(title="OpenRouter SSE Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(openai_router)
app.include_router(models_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "ok"}


def main() -> None:  # pragma: no cover
    """Entry point for the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        factory=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
