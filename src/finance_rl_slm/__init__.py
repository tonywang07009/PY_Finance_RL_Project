"""Import-safe public surface for the SLM portfolio workflow."""

from .config import DEFAULT_CONFIG, RunConfig
from .news import label_to_score, weekly_to_daily_sentiment

__all__ = [
    "DEFAULT_CONFIG",
    "RunConfig",
    "label_to_score",
    "weekly_to_daily_sentiment",
]
