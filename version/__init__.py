"""Versioned model explanation dashboard package."""

from .model_explainer import (
    DEFAULT_BUY_HOLD_PROFILE,
    DEFAULT_CURRENCY,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MARKOV_CHAIN_PROFILE,
    DEFAULT_REPORT_PATH,
    compare_four_pipeline_profiles,
    compare_model_profiles,
    compare_named_model_profiles,
    load_profile,
    normalize_action_to_weights,
    parse_action_vector,
    summarize_strategy,
)
from .model_report_html import generate_dashboard_html, write_dashboard_html

__all__ = [
    "DEFAULT_CURRENCY",
    "DEFAULT_BUY_HOLD_PROFILE",
    "DEFAULT_INITIAL_CAPITAL",
    "DEFAULT_MARKOV_CHAIN_PROFILE",
    "DEFAULT_REPORT_PATH",
    "compare_four_pipeline_profiles",
    "compare_model_profiles",
    "compare_named_model_profiles",
    "generate_dashboard_html",
    "load_profile",
    "normalize_action_to_weights",
    "parse_action_vector",
    "summarize_strategy",
    "write_dashboard_html",
]
