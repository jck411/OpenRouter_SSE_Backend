from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    ...

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
    families: list[str] | None = Field(
        None, description="Vendor/model families to include (CSV). E.g. openai, google, anthropic"
    )
    input_modalities: list[Literal["text", "image", "file", "audio"]] | None = Field(
        None, description="Required input modalities"
    )
    output_modalities: list[Literal["text", "image"]] | None = Field(
        None, description="Required output modalities"
    )
    min_context_length: int | None = Field(None, ge=0, description="Minimum context length")
    max_context_length: int | None = Field(None, ge=0, description="Maximum context length")
    min_price: float | None = Field(None, ge=0, description="Minimum price per 1M tokens")
    max_price: float | None = Field(None, ge=0, description="Maximum price per 1M tokens")
    free_only: bool = Field(False, description="Show only free models")
    supported_parameters: list[str] | None = Field(
        None, description="Required supported parameters"
    )
    toggleable_reasoning: bool = Field(
        False,
        description=("Only include models where reasoning can be enabled/disabled via parameters"),
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


class FamiliesResponse(BaseModel):
    families: list[dict[str, Any]] = Field(
        description="List of detected model families/vendors with counts and labels"
    )


__all__ = [
    "SortOptions",
    "ModelSearchFilters",
    "ModelSearchResponse",
    "ModelListResponse",
    "ModelDetailsResponse",
    "FamiliesResponse",
]
