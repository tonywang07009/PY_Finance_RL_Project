"""Orchestration layer shared by the script and notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, RunConfig
from .data import download_price_df, plot_normalized_prices
from .evaluation import run_online_evaluation
from .news import build_weekly_market_sentiment, weekly_to_daily_sentiment
from .paths import FINRL_ROOT, PROJECT_ROOT, ensure_project_paths
from .sentiment import GraniteSentimentAnalyzer
from .training import train_offline_model


@dataclass
class PriceSplits:
    train: pd.DataFrame
    valid: pd.DataFrame


def print_runtime_context(config: RunConfig = DEFAULT_CONFIG) -> None:
    ensure_project_paths()
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("FINRL_ROOT:", FINRL_ROOT)
    print("RSS_URLS:", dict(config.rss_urls))


def build_sentiment_inputs(
    config: RunConfig = DEFAULT_CONFIG,
    analyzer: GraniteSentimentAnalyzer | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df_all_news, weekly, weekly_mkt = build_weekly_market_sentiment(
        max_items=config.news_max_items,
        config=config,
        analyzer=analyzer,
    )
    print("df_all_news head:")
    print(df_all_news.head())
    print("weekly_mkt range:", weekly_mkt.index.min(), "->", weekly_mkt.index.max())
    return df_all_news, weekly, weekly_mkt


def load_price_data(config: RunConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    price_df = download_price_df(
        config.start_date,
        config.end_date,
        tickers=config.tickers,
    )
    print("price_df range:", price_df.index.min(), "->", price_df.index.max())
    print("price_df shape:", price_df.shape)
    print(price_df.head())
    return price_df


def split_price_data(
    price_df: pd.DataFrame,
    config: RunConfig = DEFAULT_CONFIG,
) -> PriceSplits:
    price_df_train = price_df.loc[config.train_start : config.train_end].copy()
    price_df_valid = price_df.loc[config.valid_start : config.valid_end].copy()
    _validate_split(
        name="train",
        split=price_df_train,
        start=config.train_start,
        end=config.train_end,
        source=price_df,
    )
    _validate_split(
        name="valid",
        split=price_df_valid,
        start=config.valid_start,
        end=config.valid_end,
        source=price_df,
    )
    print(f"price_df_train: {price_df_train.shape}, price_df_valid: {price_df_valid.shape}")
    return PriceSplits(train=price_df_train, valid=price_df_valid)


def _validate_split(
    name: str,
    split: pd.DataFrame,
    start: str,
    end: str,
    source: pd.DataFrame,
) -> None:
    if len(split) >= 2:
        return

    if source.empty:
        available = "no available rows"
    else:
        available = f"{source.index.min()} -> {source.index.max()}"

    raise ValueError(
        f"{name} price split has {len(split)} rows, but at least 2 rows are required. "
        f"Requested {name} range: {start} -> {end}. "
        f"Available price_df range: {available}. "
        "Adjust RunConfig date fields or use the online-only workflow with an existing model."
    )


def load_online_price_data(
    online_end_str: str | None = None,
    config: RunConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    resolved_end = online_end_str or config.online_end
    price_df_online = download_price_df(
        config.online_start,
        resolved_end,
        tickers=config.tickers,
    )
    print("price_df_online range:", price_df_online.index.min(), "->", price_df_online.index.max())
    print("price_df_online shape:", price_df_online.shape)
    return price_df_online


def build_daily_sentiment(
    weekly_mkt: pd.Series,
    price_df_online: pd.DataFrame,
) -> pd.Series:
    sentiment_series = weekly_to_daily_sentiment(weekly_mkt, price_df_online)
    print("sentiment_series range:", sentiment_series.index.min(), "->", sentiment_series.index.max())
    print(sentiment_series.head())
    print(sentiment_series.tail())
    return sentiment_series


def _project_output_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_picture_path(config: RunConfig = DEFAULT_CONFIG) -> Path:
    return _project_output_path(config.result_picture_dir)


def result_profile_path(config: RunConfig = DEFAULT_CONFIG) -> Path:
    return _project_output_path(config.result_profile_dir)


def run_only_ddpg_online(config: RunConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    print_runtime_context(config)
    online_end_str = config.online_end
    price_df_online = load_online_price_data(online_end_str, config)

    return run_online_evaluation(
        price_df_online,
        sentiment_series=None,
        online_end_str=online_end_str,
        config=config,
        save_plots=True,
        plot_dir=result_picture_path(config),
        save_profile=True,
        profile_dir=result_profile_path(config),
        profile_name="only_ddpg",
        show_plots=False,
    )


def run_slm_online(
    config: RunConfig = DEFAULT_CONFIG,
    analyzer: GraniteSentimentAnalyzer | None = None,
) -> pd.DataFrame:
    print_runtime_context(config)

    df_all_news, weekly, weekly_mkt = build_sentiment_inputs(config, analyzer=analyzer)
    online_end_str = config.online_end
    price_df_online = load_online_price_data(online_end_str, config)
    sentiment_series = build_daily_sentiment(weekly_mkt, price_df_online)

    return run_online_evaluation(
        price_df_online,
        sentiment_series,
        online_end_str,
        config,
        save_plots=True,
        plot_dir=result_picture_path(config),
        save_profile=True,
        profile_dir=result_profile_path(config),
        profile_name="ddpg_slm",
        show_plots=False,
    )


def run_only_ddpg_main(config: RunConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    print_runtime_context(config)

    price_df = load_price_data(config)
    plot_normalized_prices(price_df, config)

    splits = split_price_data(price_df, config)
    train_offline_model(splits.train, splits.valid, config)

    online_end_str = config.online_end
    price_df_online = load_online_price_data(online_end_str, config)

    return run_online_evaluation(
        price_df_online,
        sentiment_series=None,
        online_end_str=online_end_str,
        config=config,
        save_plots=True,
        plot_dir=result_picture_path(config),
        save_profile=True,
        profile_dir=result_profile_path(config),
        profile_name="only_ddpg",
    )


def run_slm_main(config: RunConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    print_runtime_context(config)

    df_all_news, weekly, weekly_mkt = build_sentiment_inputs(config)
    price_df = load_price_data(config)
    plot_normalized_prices(price_df, config)

    splits = split_price_data(price_df, config)
    train_offline_model(splits.train, splits.valid, config)

    online_end_str = config.online_end
    price_df_online = load_online_price_data(online_end_str, config)
    sentiment_series = build_daily_sentiment(weekly_mkt, price_df_online)

    return run_online_evaluation(
        price_df_online,
        sentiment_series,
        online_end_str,
        config,
        save_plots=True,
        plot_dir=result_picture_path(config),
        save_profile=True,
        profile_dir=result_profile_path(config),
        profile_name="ddpg_slm",
    )


def run_main(config: RunConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    return run_slm_main(config)
