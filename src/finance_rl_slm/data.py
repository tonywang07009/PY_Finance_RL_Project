"""Price-data loading and plotting helpers."""

from __future__ import annotations

from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT_CONFIG, RunConfig
from .paths import FINRL_ROOT, ensure_project_paths


def _load_yahoo_downloader_class():
    """Load FinRL's YahooDownloader without importing FinRL's optional top-level modules."""

    local_downloader_path = FINRL_ROOT / "finrl" / "meta" / "preprocessor" / "yahoodownloader.py"
    if local_downloader_path.is_file():
        spec = spec_from_file_location("_finance_rl_slm_yahoodownloader", local_downloader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load YahooDownloader from {local_downloader_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.YahooDownloader

    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    return YahooDownloader


def download_price_df(
    start_date: str,
    end_date: str | datetime | pd.Timestamp,
    tickers: Sequence[str] = DEFAULT_CONFIG.tickers,
) -> pd.DataFrame:
    ensure_project_paths()
    YahooDownloader = _load_yahoo_downloader_class()

    ticker_list = list(tickers)
    raw_df: pd.DataFrame = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
    ).fetch_data()

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    price_df = (
        raw_df.pivot(index="date", columns="tic", values="close")[ticker_list]
        .sort_index()
        .dropna(how="any")
    )
    return price_df


def plot_normalized_prices(
    price_df: pd.DataFrame,
    config: RunConfig = DEFAULT_CONFIG,
) -> None:
    normalized_price_df = price_df / price_df.iloc[0] # n is first day
                                                      # (n+n)/n is normalized 

    plt.figure(figsize=(12, 6))
    for tic in config.tickers:
        plt.plot(normalized_price_df.index, normalized_price_df[tic], label=tic)

    plt.legend()
    plt.title("Normalized Stock Prices (start = 1)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.tight_layout()
    plt.show()
