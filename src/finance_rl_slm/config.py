"""Configuration for the SLM-enabled portfolio DDPG workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RunConfig:
    """Immutable defaults shared by the script and notebook."""

    model_id: str = "ibm-granite/granite-4.1-8b"
    tickers: tuple[str, ...] = ("IBM", "NVDA", "GM", "BLK", "COST")
    start_date: str = "2010-11-18"
    end_date: str = "2025-01-01"
    train_start: str = "2011-01-01"
    train_end: str = "2015-12-31"
    valid_start: str = "2016-01-01"
    valid_end: str = "2025-12-31"
    online_start: str = "2026-01-01"
    online_end: str = "2026-06-21"
    model_path: str = "ddpg_portfolio_offline"
    slm_model_path: str = "ddpg_portfolio_slm"
    total_timesteps: int = 1_000_000
    init_wealth: float = 1.0
    wealth_norm_factor: float = 100.0
    news_max_items: int = 50
    result_picture_dir: str = "addenda/result_picture"
    result_profile_dir: str = "addenda/result_profile_comparse"
    synthetic_sentiment_dir: str = "addenda/synthetic_sentiment"

    @property
    def ticker_list(self) -> list[str]:
        return list(self.tickers)

    @property
    def rss_urls(self) -> Mapping[str, str]:
        return {
            ticker: f"http://finance.yahoo.com/rss/headline?s={ticker}"
            for ticker in self.tickers
        }


DEFAULT_CONFIG = RunConfig()
