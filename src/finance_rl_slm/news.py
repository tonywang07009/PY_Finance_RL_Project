"""News fetching and sentiment aggregation utilities."""

from __future__ import annotations

from typing import Any

import feedparser
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, RunConfig
from .sentiment import GraniteSentimentAnalyzer, analyze_sentiment


def fetch_yahoo_news_texts(rss_url: str, max_items: int = 50) -> list[dict[str, Any]]:
    """Fetch Yahoo Finance RSS items as text/published dictionaries."""
    feed = feedparser.parse(rss_url)
    rows: list[dict[str, Any]] = []

    for entry in feed.entries[:max_items]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        text = f"{title}\n{summary}".strip()
        published = entry.get("published")

        if published:
            dt = pd.to_datetime(published, errors="coerce", utc=True)
            if pd.notna(dt):
                dt = dt.tz_convert(None)
        else:
            dt = pd.NaT

        rows.append({"text": text, "published": dt})

    return rows


def analyze_news_feed(
    rss_url: str,
    ticker: str,
    max_items: int = 50,
    analyzer: GraniteSentimentAnalyzer | None = None,
) -> pd.DataFrame:
    """Fetch and score one ticker RSS feed."""
    rows = fetch_yahoo_news_texts(rss_url, max_items=max_items)
    records: list[dict[str, Any]] = []

    for row in rows:
        text = row["text"]
        dt = row["published"]
        res = analyze_sentiment(text, analyzer=analyzer)
        label = res.get("label", "neutral")
        conf = float(res.get("confidence", 0.0))

        records.append(
            {
                "ticker": ticker,
                "published": dt,
                "text": text,
                "sent_label": label,
                "sent_conf": conf,
            }
        )

    df = pd.DataFrame(
        records,
        columns=["ticker", "published", "text", "sent_label", "sent_conf"],
    )
    return df.dropna(subset=["published"])


def label_to_score(label: str) -> int:
    mapping = {
        "positive": 1,
        "negative": -1,
        "neutral": 0,
        "mixed": 0,
    }
    return mapping.get(str(label).lower().strip(), 0)


def build_weekly_market_sentiment(
    max_items: int | None = None,
    config: RunConfig = DEFAULT_CONFIG,
    analyzer: GraniteSentimentAnalyzer | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build news-level, ticker-week, and market-week sentiment tables."""
    all_records = []
    item_limit = config.news_max_items if max_items is None else max_items

    for ticker, rss_url in config.rss_urls.items():
        df_t = analyze_news_feed(
            rss_url,
            ticker=ticker,
            max_items=item_limit,
            analyzer=analyzer,
        )
        all_records.append(df_t)

    if all_records:
        df_all_news = pd.concat(all_records, ignore_index=True)
    else:
        df_all_news = pd.DataFrame(
            columns=["ticker", "published", "text", "sent_label", "sent_conf"]
        )

    if df_all_news.empty:
        weekly = pd.DataFrame(columns=["ticker", "week", "sent_score"])
        weekly_mkt = pd.Series(dtype=float, name="sent_score")
        weekly_mkt.index = pd.to_datetime(weekly_mkt.index)
        return df_all_news, weekly, weekly_mkt

    df_all_news["sent_score_raw"] = (
        df_all_news["sent_label"].apply(label_to_score) * df_all_news["sent_conf"]
    )
    df_all_news["published"] = pd.to_datetime(df_all_news["published"])
    df_all_news["week"] = df_all_news["published"].dt.to_period("W-MON").dt.start_time

    weekly = (
        df_all_news.groupby(["ticker", "week"])["sent_score_raw"]
        .mean()
        .reset_index(name="sent_score")
    )

    weekly_mkt = weekly.groupby("week")["sent_score"].mean().sort_index()
    weekly_mkt.index = pd.to_datetime(weekly_mkt.index)

    return df_all_news, weekly, weekly_mkt


def weekly_to_daily_sentiment(weekly_mkt: pd.Series, price_df: pd.DataFrame) -> pd.Series:
    idx = price_df.index
    daily = pd.Series(index=idx, dtype=float)

    for week_start, score in weekly_mkt.items():
        week_start = pd.to_datetime(week_start)
        week_end = week_start + pd.Timedelta(days=6)
        mask = (idx >= week_start) & (idx <= week_end)
        daily.loc[mask] = score

    daily = daily.ffill().fillna(0.0)
    return daily.clip(-1.0, 1.0)
