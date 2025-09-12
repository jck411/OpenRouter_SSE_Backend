# src/routers/openrouter_models.py
import asyncio
from collections.abc import Sequence
from typing import Any, Literal, cast

import httpx
from fastapi import APIRouter, HTTPException, Query
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings

# Valid sort options for type safety and API documentation
SortOptions = Literal[
    "newest",
    "top_weekly",
    "pricing_low_to_high",
    "pricing_high_to_low",
    "context_high_to_low",
    "context_low_to_high",
    "throughput_high_to_low",
    "latency_low_to_high",
]


# Pydantic models for model search
class ModelSearchFilters(BaseModel):
    """Filters for model search functionality"""

    search_term: str | None = Field(None, description="Search in model name or description")
    input_modalities: Sequence[Literal["text", "image", "file", "audio"]] | None = Field(
        None, description="Required input modalities"
    )
    output_modalities: Sequence[Literal["text", "image"]] | None = Field(
        None, description="Required output modalities"
    )
    min_context_length: int | None = Field(None, ge=0, description="Minimum context length")
    max_context_length: int | None = Field(None, ge=0, description="Maximum context length")
    min_price: float | None = Field(None, ge=0, description="Minimum price per 1M tokens")
    max_price: float | None = Field(None, ge=0, description="Maximum price per 1M tokens")
    free_only: bool = Field(False, description="Show only free models")
    supported_parameters: Sequence[str] | None = Field(
        None, description="Required supported parameters"
    )


class ModelSearchResponse(BaseModel):
    """Response for model search endpoint"""

    models: list[dict[str, Any]]
    total_count: int
    filters_applied: dict[str, Any]
    sort_applied: str


class ModelListResponse(BaseModel):
    """Response for the main models list endpoint"""

    models: list[dict[str, Any]] = Field(description="List of available models")
    supported_parameters: dict[str, Any] = Field(description="Supported parameters by category")


class ModelDetailsResponse(BaseModel):
    """Response for detailed model information endpoint"""

    id: str = Field(description="Model ID")
    name: str = Field(description="Model display name")
    description: str | None = Field(None, description="Model description/statement")
    context_length: int | None = Field(None, description="Maximum context length")
    pricing: dict[str, Any] | None = Field(None, description="Pricing information")
    architecture: dict[str, Any] | None = Field(None, description="Model architecture details")
    supported_parameters: list[str] | None = Field(None, description="Supported parameters")
    created: int | None = Field(None, description="Model creation timestamp")
    provider_name: str | None = Field(None, description="Provider name")
    endpoints: list[dict[str, Any]] | None = Field(
        None, description="Available endpoints for this model"
    )
    provider_parameter_union: list[str] | None = Field(
        None, description="Union of parameters supported across all providers"
    )
    provider_parameter_intersection: list[str] | None = Field(
        None, description="Parameters supported by ALL providers (intersection)"
    )
    provider_parameter_details: dict[str, list[str]] | None = Field(
        None, description="Parameters supported by each provider"
    )


router = APIRouter(prefix="/models", tags=["models"])
settings = get_settings()

# Ensure attribution headers are always included
client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.base_url,  # OpenRouter endpoint
    default_headers=settings.attribution_headers(),  # required attribution headers
    timeout=settings.api_timeout,  # centralized timeout configuration
)


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    Return every model the current key can access on OpenRouter.

    The response includes model details such as:
    - Model ID, name, description
    - Context length and pricing
    - Supported parameters and capabilities
    - Provider information
    """
    try:
        resp = await client.models.list()  # GET /models
        # Ensure all models are properly converted to dict format
        models_data: list[dict[str, Any]] = []
        for m in resp.data:
            if hasattr(m, "model_dump"):
                models_data.append(m.model_dump())
            else:
                # Fallback: convert Model objects to dict using dict() constructor
                models_data.append(dict(m))

        return ModelListResponse(
            models=models_data,
            supported_parameters={
                "routing": ["sort", "providers", "max_price", "fallbacks", "require_parameters"],
                "behavior": [
                    "temperature",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                    "min_p",
                    "top_a",
                    "seed",
                ],
                "output": [
                    "max_tokens",
                    "stop",
                    "logit_bias",
                    "logprobs",
                    "top_logprobs",
                    "response_format",
                ],
                "function_calling": ["tools", "tool_choice"],
                "reasoning": [
                    "reasoning_effort",
                    "reasoning_max_tokens",
                    "reasoning_exclude",
                    "include_reasoning",
                ],
                "note": "Parameter support varies by model and provider. Use require_parameters=true to ensure compatibility.",
            },
        )
    except APITimeoutError as e:
        raise HTTPException(status_code=504, detail="OpenRouter API timeout") from e
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail="Cannot connect to OpenRouter API") from e
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter API error: {e.message}") from e
    except asyncio.CancelledError:
        # Re-raise cancellation to allow proper cleanup
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch models") from e


def _get_model_pricing(model: dict[str, Any]) -> tuple[float, float]:
    """Extract input and output pricing from model data"""
    pricing = model.get("pricing", {})
    try:
        input_price = float(pricing.get("prompt", "0"))
        output_price = float(pricing.get("completion", "0"))
    except (ValueError, TypeError):
        input_price, output_price = 0.0, 0.0
    return input_price, output_price


def _get_pricing_sort_key(model: dict[str, Any]) -> float:
    """Get pricing sort key - safe average of input and output pricing"""
    input_price, output_price = _get_model_pricing(model)
    return (input_price + output_price) / 2 if input_price > 0 or output_price > 0 else 0


def _normalize_string_list(items: Sequence[str] | None) -> list[str]:
    """Normalize string list to lowercase for case-insensitive matching"""
    if not items:
        return []
    return [item.lower().strip() for item in items if item and item.strip()]


def _model_matches_filters(model: dict[str, Any], filters: ModelSearchFilters) -> bool:
    """Check if a model matches the search filters"""

    # Search term filtering
    if filters.search_term:
        search_lower = filters.search_term.lower()
        model_id = model.get("id", "").lower()
        model_name = model.get("name", "").lower()
        model_description = model.get("description", "").lower()

        if not (
            search_lower in model_id
            or search_lower in model_name
            or search_lower in model_description
        ):
            return False

    # Input modalities filtering - case insensitive and resilient to missing fields
    if filters.input_modalities:
        architecture = model.get("architecture", {})
        model_modalities = _normalize_string_list(architecture.get("input_modalities", []))
        required_modalities = _normalize_string_list(filters.input_modalities)
        # Check if all required input modalities are supported
        for required_modality in required_modalities:
            if required_modality not in model_modalities:
                return False

    # Output modalities filtering - case insensitive and resilient to missing fields
    if filters.output_modalities:
        architecture = model.get("architecture", {})
        model_modalities = _normalize_string_list(architecture.get("output_modalities", []))
        required_modalities = _normalize_string_list(filters.output_modalities)
        # Check if all required output modalities are supported
        for required_modality in required_modalities:
            if required_modality not in model_modalities:
                return False

    # Context length filtering
    context_length = model.get("context_length")
    if context_length is not None:
        if filters.min_context_length is not None and context_length < filters.min_context_length:
            return False
        if filters.max_context_length is not None and context_length > filters.max_context_length:
            return False

    # Price filtering - use the standardized pricing logic
    avg_price = _get_pricing_sort_key(model)

    # Free only filter - both input and output must be 0
    if filters.free_only:
        input_price, output_price = _get_model_pricing(model)
        if input_price > 0 or output_price > 0:
            return False

    # Price range filtering
    if filters.min_price is not None and avg_price < filters.min_price:
        return False
    if filters.max_price is not None and avg_price > filters.max_price:
        return False

    # Supported parameters filtering - case insensitive and resilient
    if filters.supported_parameters:
        model_parameters = _normalize_string_list(model.get("supported_parameters", []))
        required_parameters = _normalize_string_list(filters.supported_parameters)
        for required_param in required_parameters:
            if required_param not in model_parameters:
                return False

    return True


def _get_model_provider_parameter_analysis(
    endpoints: list[dict[str, Any]] | None, base_supported_parameters: list[str] | None
) -> dict[str, Any]:
    """Get comprehensive parameter support analysis for a model from its endpoints.

    Args:
        endpoints: List of provider endpoint information
        base_supported_parameters: Base model-level supported parameters

    Returns:
        Dict containing union, intersection, and provider-specific parameter details
    """
    # Start with base model parameters
    base_params = set(base_supported_parameters or [])

    # If no endpoints, return base parameters
    if not endpoints:
        return {
            "union": sorted(base_params),
            "intersection": sorted(base_params),
            "provider_details": {},
        }

    # Collect parameters from all provider endpoints
    all_provider_params = []
    provider_details = {}

    for endpoint in endpoints:
        if isinstance(endpoint, dict):
            # Try different possible provider name fields
            provider_name = (
                endpoint.get("provider")
                or endpoint.get("provider_name")
                or endpoint.get("name")
                or "unknown"
            )

            # Get supported parameters for this provider
            provider_params = endpoint.get("supported_parameters", [])
            if isinstance(provider_params, list):
                provider_param_set = set(provider_params)
                # Merge with base parameters for this provider
                combined_params = base_params | provider_param_set
                all_provider_params.append(combined_params)
                provider_details[provider_name] = sorted(combined_params)

    # Calculate union and intersection
    if all_provider_params:
        union_params = set.union(*all_provider_params) if all_provider_params else base_params
        intersection_params = (
            set.intersection(*all_provider_params) if all_provider_params else base_params
        )
    else:
        # Fallback to base parameters
        union_params = base_params
        intersection_params = base_params

    return {
        "union": sorted(union_params),
        "intersection": sorted(intersection_params),
        "provider_details": provider_details,
    }


def _sort_models(models: list[dict[str, Any]], sort_by: SortOptions) -> list[dict[str, Any]]:
    """Sort models according to the specified sort strategy

    Pre-computes sort keys for performance to avoid recomputing during sort.
    Handles mixed type comparisons safely.
    """
    if not models:
        return models

    if sort_by == "newest":
        # Sort by creation date if available, fallback to 0 for consistent int comparison
        # Then by id as secondary sort for deterministic ordering
        # Cast created to int to handle providers that might send it as string
        return sorted(
            models, key=lambda m: (int(m.get("created") or 0), m.get("id", "")), reverse=True
        )

    elif sort_by == "top_weekly":
        # Sort by weekly usage/popularity if available
        return sorted(models, key=lambda m: int(m.get("weekly_usage") or 0), reverse=True)

    elif sort_by == "pricing_low_to_high":
        # Pre-compute pricing keys for performance
        models_with_keys = [(m, _get_pricing_sort_key(m)) for m in models]
        return [m for m, _ in sorted(models_with_keys, key=lambda x: x[1])]

    elif sort_by == "pricing_high_to_low":
        # Pre-compute pricing keys for performance
        models_with_keys = [(m, _get_pricing_sort_key(m)) for m in models]
        return [m for m, _ in sorted(models_with_keys, key=lambda x: x[1], reverse=True)]

    elif sort_by == "context_high_to_low":
        # Sort by context length, descending
        return sorted(models, key=lambda m: int(m.get("context_length") or 0), reverse=True)

    elif sort_by == "context_low_to_high":
        # Sort by context length, ascending
        return sorted(models, key=lambda m: int(m.get("context_length") or 0))

    elif sort_by == "throughput_high_to_low":
        # Sort by throughput if available
        return sorted(models, key=lambda m: float(m.get("throughput") or 0), reverse=True)

    elif sort_by == "latency_low_to_high":
        # Sort by latency, ascending (lower is better)
        return sorted(models, key=lambda m: float(m.get("latency") or float("inf")))

    # This should not be reachable due to SortOptions type constraint,
    # but mypy doesn't understand that all enum values are covered
    return sorted(models, key=lambda m: m.get("id", ""))  # type: ignore[unreachable]


@router.get("/search", response_model=ModelSearchResponse)
async def search_models(
    # Search and filter parameters
    search_term: str | None = Query(None, description="Search in model name or description"),
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
    # Sort parameter
    sort: SortOptions = Query(
        "newest",
        description="Sort by: newest, top_weekly, pricing_low_to_high, pricing_high_to_low, context_high_to_low, context_low_to_high, throughput_high_to_low, latency_low_to_high",
    ),
    # Pagination
    limit: int = Query(50, ge=1, le=200, description="Maximum number of models to return"),
    offset: int = Query(0, ge=0, description="Number of models to skip"),
) -> ModelSearchResponse:
    """
    Search and filter models with various criteria.

    Supports filtering by capabilities, pricing, context length, modalities,
    and sorting by different criteria similar to OpenRouter's models page.
    """
    try:
        # Fetch all models first
        resp = await client.models.list()
        all_models = [m.model_dump() if hasattr(m, "model_dump") else m for m in resp.data]
        # Ensure all models are dict format for consistent type handling
        all_models_dict = []
        for m in all_models:
            if isinstance(m, dict):
                all_models_dict.append(m)
            else:
                # Convert Model object to dict
                all_models_dict.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))

        # Parse comma-separated parameters into lists
        input_modalities_list = None
        if input_modalities:
            input_modalities_list = [
                mod.strip() for mod in input_modalities.split(",") if mod.strip()
            ]

        output_modalities_list = None
        if output_modalities:
            output_modalities_list = [
                mod.strip() for mod in output_modalities.split(",") if mod.strip()
            ]

        supported_parameters_list = None
        if supported_parameters:
            supported_parameters_list = [
                param.strip() for param in supported_parameters.split(",") if param.strip()
            ]

        # Create filters object - cast string lists to proper Literal types
        filters = ModelSearchFilters(
            search_term=search_term,
            input_modalities=cast(
                "Sequence[Literal['text', 'image', 'file', 'audio']] | None", input_modalities_list
            ),
            output_modalities=cast(
                "Sequence[Literal['text', 'image']] | None", output_modalities_list
            ),
            min_context_length=min_context_length,
            max_context_length=max_context_length,
            min_price=min_price,
            max_price=max_price,
            free_only=free_only,
            supported_parameters=supported_parameters_list,
        )

        # Apply filters
        filtered_models = [
            model for model in all_models_dict if _model_matches_filters(model, filters)
        ]

        # Sort the filtered models
        sorted_models = _sort_models(filtered_models, sort)

        # Apply pagination
        total_count = len(sorted_models)
        paginated_models = sorted_models[offset : offset + limit]

        # Build response
        filters_applied = {
            "search_term": search_term,
            "input_modalities": input_modalities_list,
            "output_modalities": output_modalities_list,
            "min_context_length": min_context_length,
            "max_context_length": max_context_length,
            "min_price": min_price,
            "max_price": max_price,
            "free_only": free_only,
            "supported_parameters": supported_parameters_list,
            "limit": limit,
            "offset": offset,
        }

        return ModelSearchResponse(
            models=paginated_models,
            total_count=total_count,
            filters_applied=filters_applied,
            sort_applied=sort,
        )

    except APITimeoutError as e:
        raise HTTPException(status_code=504, detail="OpenRouter API timeout") from e
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail="Cannot connect to OpenRouter API") from e
    except APIError as e:
        # Handle rate limiting specifically
        error_body = getattr(e, "body", {})
        if (
            isinstance(error_body, dict)
            and error_body.get("error", {}).get("code") == "rate_limit_exceeded"
        ):
            raise HTTPException(
                status_code=429,
                detail="Rate limited by OpenRouter. Please reduce request frequency and try again later.",
            ) from e
        raise HTTPException(status_code=502, detail=f"OpenRouter API error: {e.message}") from e
    except asyncio.CancelledError:
        # Re-raise cancellation to allow proper cleanup
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to search models") from e


@router.get("/{model_id:path}", response_model=ModelDetailsResponse)
async def get_model_details(model_id: str) -> ModelDetailsResponse:
    """
    Get detailed information about a specific model by its ID.

    Returns comprehensive model information including:
    - Model description and capabilities statement
    - Detailed architecture information
    - Pricing structure
    - Supported parameters
    - Provider information

    Args:
        model_id: The full model ID (e.g., "openrouter/sonoma-dusk-alpha")

    Returns:
        Detailed model information including description, pricing, capabilities, etc.
    """
    try:
        # First, get the model from the main models list using existing client
        resp = await client.models.list()
        all_models = [m.model_dump() if hasattr(m, "model_dump") else m for m in resp.data]
        # Ensure all models are dict format for consistent type handling
        all_models_dict = []
        for m in all_models:
            if isinstance(m, dict):
                all_models_dict.append(m)
            else:
                # Convert Model object to dict
                all_models_dict.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))

        # Find the specific model in the list
        target_model = None
        for model in all_models_dict:
            if model.get("id") == model_id:
                target_model = model
                break

        if not target_model:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Try to get additional endpoint information using httpx (SDK doesn't support this endpoint)
        endpoints_info = None
        try:
            if "/" in model_id:
                author, slug = model_id.split("/", 1)
                async with httpx.AsyncClient(
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        **settings.attribution_headers(),
                    },
                    timeout=settings.api_timeout,
                ) as http_client:
                    endpoints_url = f"{settings.base_url}/models/{author}/{slug}/endpoints"
                    endpoints_response = await http_client.get(endpoints_url)
                    if endpoints_response.status_code == 200:
                        endpoints_data = endpoints_response.json()
                        # Guard against different response types - could be dict or list
                        if isinstance(endpoints_data, dict):
                            # Standard response with "data" wrapper
                            raw_endpoints = endpoints_data.get("data", endpoints_data)
                        elif isinstance(endpoints_data, list):
                            # Direct list response
                            raw_endpoints = endpoints_data
                        else:
                            # Unexpected format
                            raw_endpoints = None

                        # Process the endpoints data
                        if isinstance(raw_endpoints, dict):
                            endpoints_info = [raw_endpoints]  # Wrap single dict in list
                        elif isinstance(raw_endpoints, list):
                            endpoints_info = raw_endpoints
                        else:
                            endpoints_info = None  # Invalid data type
        except Exception:
            # If endpoints call fails, continue without it - not all models may support this
            pass

        # Analyze provider parameter support
        parameter_analysis = _get_model_provider_parameter_analysis(
            endpoints_info, target_model.get("supported_parameters")
        )

        # Build the detailed response
        # Ensure required fields have safe fallbacks
        model_id_value = target_model.get("id") or model_id
        model_name_value = target_model.get("name") or model_id_value

        return ModelDetailsResponse(
            id=model_id_value,
            name=model_name_value,
            description=target_model.get("description"),
            context_length=target_model.get("context_length"),
            pricing=target_model.get("pricing"),
            architecture=target_model.get("architecture"),
            supported_parameters=target_model.get("supported_parameters"),
            created=target_model.get("created"),
            provider_name=target_model.get("provider_name"),
            endpoints=endpoints_info,
            provider_parameter_union=parameter_analysis["union"],
            provider_parameter_intersection=parameter_analysis["intersection"],
            provider_parameter_details=parameter_analysis["provider_details"],
        )
    except APITimeoutError as e:
        raise HTTPException(status_code=504, detail="OpenRouter API timeout") from e
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail="Cannot connect to OpenRouter API") from e
    except APIError as e:
        # Handle rate limiting specifically
        error_body = getattr(e, "body", {})
        if (
            isinstance(error_body, dict)
            and error_body.get("error", {}).get("code") == "rate_limit_exceeded"
        ):
            raise HTTPException(
                status_code=429,
                detail="Rate limited by OpenRouter. Please reduce request frequency and try again later.",
            ) from e
        raise HTTPException(status_code=502, detail=f"OpenRouter API error: {e.message}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="OpenRouter API timeout") from e
    except httpx.ConnectError as e:
        raise HTTPException(status_code=503, detail="Cannot connect to OpenRouter API") from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found") from e
        elif e.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limited by OpenRouter. Please reduce request frequency and try again later.",
            ) from e
        raise HTTPException(
            status_code=502, detail=f"OpenRouter API error: {e.response.status_code}"
        ) from e
    except asyncio.CancelledError:
        # Re-raise cancellation to allow proper cleanup
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch model details") from e
