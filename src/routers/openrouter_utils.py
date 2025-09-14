from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only for typing to satisfy strict settings
    from collections.abc import Sequence

    from .openrouter_schemas import ModelSearchFilters, SortOptions

# Intentionally avoid importing application modules at runtime; types are available for checkers

# --- Family helpers & aliases ---

FAMILY_ALIASES: dict[str, str] = {
    # Canonical -> canonical
    "openai": "openai",
    "google": "google",
    "anthropic": "anthropic",
    "meta": "meta",
    "mistral": "mistral",
    "cohere": "cohere",
    "xai": "xai",
    "perplexity": "perplexity",
    "databricks": "databricks",
    "deepseek": "deepseek",
    "groq": "groq",
    "nvidia": "nvidia",
    "qwen": "qwen",
    # Aliases
    "gpt": "openai",
    "chatgpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "gemini": "google",
    "vertex": "google",
    "claude": "anthropic",
    "llama": "meta",
    "llama3": "meta",
    "mixtral": "mistral",
    "command": "cohere",
}


def label_for_family(canonical: str) -> str:
    """Create a friendly display label for a family."""
    return {
        "openai": "OpenAI (ChatGPT)",
        "google": "Google (Gemini)",
        "anthropic": "Anthropic (Claude)",
        "meta": "Meta (Llama)",
        "mistral": "Mistral (Mixtral)",
        "cohere": "Cohere (Command)",
        "xai": "xAI (Grok)",
        "perplexity": "Perplexity",
        "databricks": "Databricks",
        "deepseek": "DeepSeek",
        "groq": "Groq",
        "nvidia": "NVIDIA",
        "qwen": "Qwen (Alibaba)",
    }.get(canonical, canonical.capitalize())


def canonical_family(token: str) -> str | None:
    t = token.strip().lower()
    if not t:
        return None
    if t in FAMILY_ALIASES:
        return FAMILY_ALIASES[t]
    if t.startswith("meta-"):
        return "meta"
    return t


def extract_vendor_prefix(model_id: str) -> str | None:
    # Expect "author/slug"
    if not model_id or "/" not in model_id:
        return None
    author = model_id.split("/", 1)[0].strip().lower()
    return author or None


def model_in_families(model: dict[str, Any], families: set[str]) -> bool:
    """Check if a model is in requested canonical families (by vendor prefix or aliases)."""
    model_id = str(model.get("id", "")).lower()
    name = str(model.get("name", "")).lower()
    author = extract_vendor_prefix(model_id)
    requested = {f for f in (canonical_family(f) for f in families) if f}
    if not requested:
        return True

    if author and author in requested:
        return True

    for kw, canonical in FAMILY_ALIASES.items():
        if canonical in requested and (kw in model_id or kw in name):
            return True

    return False


# --- General helpers ---


def normalize_string_list(items: Sequence[str] | None) -> list[str]:
    if not items:
        return []
    return [item.lower().strip() for item in items if item and item.strip()]


def get_model_pricing(model: dict[str, Any]) -> tuple[float, float]:
    """Extract input and output pricing (per 1M tokens) from model data."""
    pricing = model.get("pricing", {})
    try:
        input_price = float(pricing.get("prompt", "0"))
        output_price = float(pricing.get("completion", "0"))
    except (ValueError, TypeError):
        input_price, output_price = 0.0, 0.0
    return input_price, output_price


def get_pricing_sort_key(model: dict[str, Any]) -> float:
    """Return safe average of input/output pricing for sorting."""
    input_price, output_price = get_model_pricing(model)
    return (input_price + output_price) / 2 if (input_price > 0 or output_price > 0) else 0.0


def is_reasoning_toggleable(model_obj: dict[str, Any]) -> bool:
    """Determine if a model's reasoning is toggleable based on metadata."""

    model_id = str(model_obj.get("id", ""))
    lower_id = model_id.lower()

    # Always-on reasoning families are NOT toggleable: OpenAI o1/o3, 'thinking'/'reasoner' variants
    if model_id.startswith("openai/") and ("o1" in lower_id or "o3" in lower_id):
        return False

    supported_params = model_obj.get("supported_parameters", [])
    if isinstance(supported_params, list):
        norm_params = [
            str(p).replace("-", "_").lower() for p in supported_params if isinstance(p, str)
        ]
        if "include_reasoning" in norm_params:
            topology = model_obj.get("topology")
            if isinstance(topology, str) and topology.lower() in {"reasoning", "reasoner"}:
                return False
            return not any(tok in lower_id for tok in ("thinking", "reasoner"))

    return False


def get_model_provider_parameter_analysis(
    endpoints: list[dict[str, Any]] | None, base_supported_parameters: list[str] | None
) -> dict[str, Any]:
    """Compute union/intersection/provider details of supported parameters from endpoints."""
    base_params = set(base_supported_parameters or [])
    if not endpoints:
        return {
            "union": sorted(base_params),
            "intersection": sorted(base_params),
            "provider_details": {},
        }

    all_provider_params: list[set[str]] = []
    provider_details: dict[str, list[str]] = {}

    for endpoint in endpoints:
        provider_name = (
            endpoint.get("provider")
            or endpoint.get("provider_name")
            or endpoint.get("name")
            or "unknown"
        )
        provider_params = endpoint.get("supported_parameters", [])
        if isinstance(provider_params, list):
            provider_param_set = set(provider_params)
            combined_params = base_params | provider_param_set
            all_provider_params.append(combined_params)
            provider_details[str(provider_name)] = sorted(combined_params)

    if all_provider_params:
        union_params = set.union(*all_provider_params) if all_provider_params else base_params
        intersection_params = (
            set.intersection(*all_provider_params) if all_provider_params else base_params
        )
    else:
        union_params = base_params
        intersection_params = base_params

    return {
        "union": sorted(union_params),
        "intersection": sorted(intersection_params),
        "provider_details": provider_details,
    }


# --- Filter & sort ---


def model_matches_filters(model: dict[str, Any], filters: ModelSearchFilters) -> bool:
    # Families
    if filters.families:
        families = normalize_string_list(filters.families)
        if families and not model_in_families(model, set(families)):
            return False

    # Search term (id, name, description)
    if filters.search_term:
        search_lower = filters.search_term.lower()
        model_id = str(model.get("id", "")).lower()
        model_name = str(model.get("name", "")).lower()
        model_description = str(model.get("description", "")).lower()
        if not (
            search_lower in model_id
            or search_lower in model_name
            or search_lower in model_description
        ):
            return False

    # Input modalities
    if filters.input_modalities:
        architecture = model.get("architecture", {})
        model_modalities = normalize_string_list(architecture.get("input_modalities", []))
        for required in normalize_string_list(filters.input_modalities):
            if required not in model_modalities:
                return False

    # Output modalities
    if filters.output_modalities:
        architecture = model.get("architecture", {})
        model_modalities = normalize_string_list(architecture.get("output_modalities", []))
        for required in normalize_string_list(filters.output_modalities):
            if required not in model_modalities:
                return False

    # Context
    context_length = model.get("context_length")
    if context_length is not None:
        if (
            filters.min_context_length is not None
            and int(context_length) < filters.min_context_length
        ):
            return False
        if (
            filters.max_context_length is not None
            and int(context_length) > filters.max_context_length
        ):
            return False

    # Pricing
    avg_price = get_pricing_sort_key(model)

    if filters.free_only:
        inp, out = get_model_pricing(model)
        if inp > 0 or out > 0:
            return False

    if filters.min_price is not None and avg_price < filters.min_price:
        return False
    if filters.max_price is not None and avg_price > filters.max_price:
        return False

    # Supported parameters
    if filters.supported_parameters:
        model_parameters = normalize_string_list(model.get("supported_parameters", []))
        for required_param in normalize_string_list(filters.supported_parameters):
            if required_param not in model_parameters:
                return False

    # Toggleable reasoning
    return not (filters.toggleable_reasoning and not is_reasoning_toggleable(model))


__all__ = [
    "FAMILY_ALIASES",
    "label_for_family",
    "canonical_family",
    "extract_vendor_prefix",
    "model_in_families",
    "normalize_string_list",
    "get_model_pricing",
    "get_pricing_sort_key",
    "is_reasoning_toggleable",
    "get_model_provider_parameter_analysis",
    "model_matches_filters",
    "sort_models",
    "parse_csv",
    "parse_supported_parameters",
]


def sort_models(models: list[dict[str, Any]], sort_by: SortOptions) -> list[dict[str, Any]]:
    if not models:
        return models

    if sort_by == "newest":
        return sorted(
            models, key=lambda m: (int(m.get("created") or 0), m.get("id", "")), reverse=True
        )

    if sort_by == "top_weekly":
        return sorted(models, key=lambda m: int(m.get("weekly_usage") or 0), reverse=True)

    if sort_by == "pricing_low_to_high":
        with_keys = [(m, get_pricing_sort_key(m)) for m in models]
        return [m for m, _ in sorted(with_keys, key=lambda x: x[1])]

    if sort_by == "pricing_high_to_low":
        with_keys = [(m, get_pricing_sort_key(m)) for m in models]
        return [m for m, _ in sorted(with_keys, key=lambda x: x[1], reverse=True)]

    if sort_by == "context_high_to_low":
        return sorted(models, key=lambda m: int(m.get("context_length") or 0), reverse=True)

    if sort_by == "context_low_to_high":
        return sorted(models, key=lambda m: int(m.get("context_length") or 0))

    if sort_by == "throughput_high_to_low":
        return sorted(models, key=lambda m: float(m.get("throughput") or 0.0), reverse=True)

    if sort_by == "latency_low_to_high":
        # Lower is better; fall back to +inf if missing to push to the end
        return sorted(models, key=lambda m: float(m.get("latency") or math.inf))

    # Fallback: deterministic id order (covers default/unknown sort_by values)
    return sorted(models, key=lambda m: m.get("id", ""))  # type: ignore[unreachable]


# --- Shared simple parsers (CSV & supported params) ---


def parse_csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated string into list[str]; returns None if value is None."""
    if value is None:
        return None
    items = [part.strip() for part in value.split(",")]
    return [x for x in items if x]


def parse_supported_parameters(raw: str | None) -> list[str] | None:
    """Parse CSV of supported params, dropping UI-only toggles like 'hide_reasoning'."""
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    def _norm(s: str) -> str:
        return s.replace("-", "_").lower()

    return [p for p in parts if _norm(p) != "hide_reasoning"]
