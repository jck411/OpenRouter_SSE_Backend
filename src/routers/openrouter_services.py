from __future__ import annotations

from typing import Any, Literal, cast

import httpx

from app.config import get_settings

from .openrouter_client import get_openrouter_client
from .openrouter_schemas import (
    FamiliesResponse,
    ModelDetailsResponse,
    ModelListResponse,
    ModelSearchFilters,
    ModelSearchResponse,
    SortOptions,
)
from .openrouter_utils import (
    FAMILY_ALIASES,
    canonical_family,
    extract_vendor_prefix,
    get_model_provider_parameter_analysis,
    label_for_family,
    model_matches_filters,
    parse_csv,
    parse_supported_parameters,
    sort_models,
)

_settings = get_settings()
_client = get_openrouter_client()


# --- Internal helpers ---------------------------------------------------------


def _to_dict(model_obj: Any) -> dict[str, Any]:
    """Normalize OpenAI Model object to a plain dict[str, Any]."""
    if isinstance(model_obj, dict):
        return model_obj
    # Handle Pydantic/BaseModel-like objects that expose model_dump()
    model_dump = getattr(model_obj, "model_dump", None)
    if callable(model_dump):
        try:
            d = model_dump()
            if isinstance(d, dict):
                return d
        except Exception:
            # If model_dump() fails, fall back to other strategies
            pass
    # Try mapping protocol
    try:
        return dict(model_obj)
    except Exception:
        # Last resort: reflect public attributes
        return {k: getattr(model_obj, k) for k in dir(model_obj) if not k.startswith("_")}


async def _fetch_all_models() -> list[dict[str, Any]]:
    resp = await _client.models.list()  # GET /models
    return [_to_dict(m) for m in resp.data]


def _supported_parameters_catalog() -> dict[str, Any]:
    """Static support matrix description for clients."""
    return {
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
        ],
        "note": "Parameter support varies by model and provider. Use require_parameters=true to ensure compatibility.",
    }


# (parsers now shared in openrouter_utils)


# --- Public services ----------------------------------------------------------


async def list_models_service() -> ModelListResponse:
    models = await _fetch_all_models()
    return ModelListResponse(
        models=models,
        supported_parameters=_supported_parameters_catalog(),
    )


async def list_families_service() -> FamiliesResponse:
    models = await _fetch_all_models()

    counts: dict[str, int] = {}
    seen_labels: dict[str, str] = {}

    for m in models:
        model_id = str(m.get("id", "")).lower()
        name = str(m.get("name", "")).lower()
        author = extract_vendor_prefix(model_id) or ""

        canonical: str | None = None
        if author:
            canonical = canonical_family(author) or author
        if not canonical:
            for kw, can in FAMILY_ALIASES.items():
                if kw in model_id or kw in name:
                    canonical = can
                    break
        if not canonical:
            continue

        counts[canonical] = counts.get(canonical, 0) + 1
        seen_labels.setdefault(canonical, label_for_family(canonical))

    families_list = [
        {"id": fam, "label": seen_labels.get(fam, fam.capitalize()), "count": counts[fam]}
        for fam in sorted(counts.keys())
    ]
    return FamiliesResponse(families=families_list)


async def search_models_service(
    *,
    search_term: str | None,
    families: str | None,
    input_modalities: str | None,
    output_modalities: str | None,
    min_context_length: int | None,
    max_context_length: int | None,
    min_price: float | None,
    max_price: float | None,
    free_only: bool,
    supported_parameters: str | None,
    toggleable_reasoning: bool,
    toggable_reasoning: bool | None,
    sort: SortOptions,
    limit: int,
    offset: int,
) -> ModelSearchResponse:
    all_models = await _fetch_all_models()

    families_list = parse_csv(families)
    input_modalities_list = parse_csv(input_modalities)
    output_modalities_list = parse_csv(output_modalities)
    supported_parameters_list = parse_supported_parameters(supported_parameters)

    toggle_flag = bool(toggleable_reasoning or toggable_reasoning)

    # Build filters object (cast for Literal types in Pydantic model)
    filters = ModelSearchFilters(
        search_term=search_term,
        families=families_list,
        input_modalities=cast(
            "list[Literal['text', 'image', 'file', 'audio']] | None", input_modalities_list
        ),
        output_modalities=cast("list[Literal['text', 'image']] | None", output_modalities_list),
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_price=min_price,
        max_price=max_price,
        free_only=free_only,
        supported_parameters=supported_parameters_list,
        toggleable_reasoning=toggle_flag,
    )

    # Filter, sort, paginate
    filtered = [m for m in all_models if model_matches_filters(m, filters)]
    sorted_models = sort_models(filtered, sort)
    total_count = len(sorted_models)
    paginated = sorted_models[offset : offset + limit]

    filters_applied = {
        "search_term": search_term,
        "families": families_list,
        "input_modalities": input_modalities_list,
        "output_modalities": output_modalities_list,
        "min_context_length": min_context_length,
        "max_context_length": max_context_length,
        "min_price": min_price,
        "max_price": max_price,
        "free_only": free_only,
        "supported_parameters": supported_parameters_list,
        "toggleable_reasoning": toggle_flag,
        "toggable_reasoning": toggable_reasoning,
        "limit": limit,
        "offset": offset,
    }

    return ModelSearchResponse(
        models=paginated,
        total_count=total_count,
        filters_applied=filters_applied,
        sort_applied=cast("str", sort),
    )


async def get_model_details_service(model_id: str) -> ModelDetailsResponse:
    all_models = await _fetch_all_models()
    target_model: dict[str, Any] | None = next(
        (m for m in all_models if m.get("id") == model_id), None
    )
    if not target_model:
        # Raise here; router-level translator will pass this through untouched
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # Try to fetch provider endpoint details via raw HTTP call (SDK missing this endpoint)
    endpoints_info: list[dict[str, Any]] | None = None
    try:
        if "/" in model_id:
            author, slug = model_id.split("/", 1)
            async with httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {_settings.openrouter_api_key}",
                    **_settings.attribution_headers(),
                },
                timeout=_settings.api_timeout,
            ) as http_client:
                endpoints_url = f"{_settings.base_url}/models/{author}/{slug}/endpoints"
                endpoints_response = await http_client.get(endpoints_url)
                endpoints_response.raise_for_status()
                endpoints_data = endpoints_response.json()

                if isinstance(endpoints_data, dict):
                    raw_endpoints = endpoints_data.get("data", endpoints_data)
                else:
                    raw_endpoints = endpoints_data

                if isinstance(raw_endpoints, dict):
                    endpoints_info = [raw_endpoints]
                elif isinstance(raw_endpoints, list):
                    endpoints_info = raw_endpoints
    except Exception:
        # Non-fatal: proceed without endpoints
        endpoints_info = None

    param_analysis = get_model_provider_parameter_analysis(
        endpoints_info, target_model.get("supported_parameters")
    )

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
        provider_parameter_union=param_analysis["union"],
        provider_parameter_intersection=param_analysis["intersection"],
        provider_parameter_details=param_analysis["provider_details"],
    )
