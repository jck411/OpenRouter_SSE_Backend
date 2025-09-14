from __future__ import annotations

from fastapi import APIRouter, Query

from .openrouter_errors import to_http_exception
from .openrouter_schemas import (
    FamiliesResponse,
    ModelDetailsResponse,
    ModelListResponse,
    ModelSearchResponse,
    SortOptions,
)
from .openrouter_services import (
    get_model_details_service,
    list_families_service,
    list_models_service,
    search_models_service,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """Return every model the current key can access on OpenRouter."""
    try:
        return await list_models_service()
    except Exception as e:
        raise to_http_exception(e, default_detail="Failed to fetch models") from e


@router.get("/families", response_model=FamiliesResponse)
async def list_families() -> FamiliesResponse:
    """Return a canonical list of model families/vendors present for the current key."""
    try:
        return await list_families_service()
    except Exception as e:
        raise to_http_exception(e, default_detail="Failed to list families") from e


@router.get("/search", response_model=ModelSearchResponse)
async def search_models(
    # Search and filter parameters
    search_term: str | None = Query(None, description="Search in model name or description"),
    families: str | None = Query(
        None,
        description=(
            "Comma-separated list of model families/vendors (e.g., openai, google, anthropic). "
            "Aliases supported: gpt/chatgpt→openai, gemini→google, claude→anthropic, llama→meta, mixtral→mistral, command→cohere"
        ),
    ),
    input_modalities: str | None = Query(
        None,
        description="Comma-separated list of required input modalities (text,image,file,audio)",
    ),
    output_modalities: str | None = Query(
        None, description="Comma-separated list of required output modalities (text,image)"
    ),
    min_context_length: int | None = Query(None, ge=0, description="Minimum context length"),
    max_context_length: int | None = Query(None, ge=0, description="Maximum context length"),
    min_price: float | None = Query(None, ge=0, description="Minimum price per 1M tokens"),
    max_price: float | None = Query(None, ge=0, description="Maximum price per 1M tokens"),
    free_only: bool = Query(False, description="Show only free models"),
    supported_parameters: str | None = Query(
        None, description="Comma-separated list of required supported parameters"
    ),
    toggleable_reasoning: bool = Query(
        False,
        description="If true, only include models where reasoning can be toggled",
    ),
    # Back-compat/alias: accept 'toggable_reasoning' (single-g) from UI if used
    toggable_reasoning: bool | None = Query(
        None,
        description="Alias for toggleable_reasoning (single-g spelling)",
    ),
    # Sort parameter
    sort: SortOptions = Query(
        "newest",
        description="Sort by: newest, top_weekly, pricing_low_to_high, pricing_high_to_low, context_high_to_low, context_low_to_high, throughput_high_to_low, latency_low_to_high",
    ),
    # Pagination
    limit: int = Query(50, ge=1, le=200, description="Maximum number of models to return"),
    offset: int = Query(0, ge=0, description="Number of models to skip"),
) -> ModelSearchResponse:
    """Search and filter models with various criteria."""
    try:
        return await search_models_service(
            search_term=search_term,
            families=families,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            min_context_length=min_context_length,
            max_context_length=max_context_length,
            min_price=min_price,
            max_price=max_price,
            free_only=free_only,
            supported_parameters=supported_parameters,
            toggleable_reasoning=toggleable_reasoning,
            toggable_reasoning=toggable_reasoning,
            sort=sort,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise to_http_exception(e, default_detail="Failed to search models") from e


@router.get("/{model_id:path}", response_model=ModelDetailsResponse)
async def get_model_details(model_id: str) -> ModelDetailsResponse:
    """Get detailed information about a specific model by its ID."""
    try:
        return await get_model_details_service(model_id)
    except Exception as e:
        raise to_http_exception(e, default_detail="Failed to fetch model details") from e


# Back-compat for tests that import _is_reasoning_toggleable from this module
def _is_reasoning_toggleable(model_obj: object) -> bool:  # noqa: N802 - keep legacy name for tests
    from .openrouter_utils import is_reasoning_toggleable

    if not isinstance(model_obj, dict):
        return False
    return is_reasoning_toggleable(model_obj)
