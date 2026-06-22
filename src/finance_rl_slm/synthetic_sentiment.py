"""Balanced synthetic sentiment data generated from RSS text seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .config import DEFAULT_CONFIG, RunConfig
from .news import fetch_yahoo_news_texts
from .paths import PROJECT_ROOT


SENTIMENT_LABELS = ("positive", "neutral", "negative")
SENTIMENT_SCORES = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def synthetic_sentiment_filename(config: RunConfig = DEFAULT_CONFIG) -> str:
    return (
        "balanced_rss_sentiment_"
        f"{config.train_start}_{config.valid_end}.csv"
    )


def synthetic_sentiment_path(config: RunConfig = DEFAULT_CONFIG) -> Path:
    output_dir = Path(config.synthetic_sentiment_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir / synthetic_sentiment_filename(config)


# Form Yahoo RSS get ticker news world
# get the fake new data source
def fetch_rss_seed_texts(
    config: RunConfig = DEFAULT_CONFIG,
    max_items_per_ticker: int | None = None,
) -> dict[str, list[str]]:

    """Fetch a small deterministic seed corpus from current RSS feeds."""
    item_limit = max_items_per_ticker or config.news_max_items
    seed_texts: dict[str, list[str]] = {}

    for ticker, rss_url in config.rss_urls.items():

        rows = fetch_yahoo_news_texts(rss_url, max_items=item_limit)

        texts = [
            str(row.get("text", "")).strip()
            for row in rows
            if str(row.get("text", "")).strip()
        ]
        ## The add the news sed if not in texts list
        if not texts:
            texts = [f"{ticker} RSS synthetic seed"]
        seed_texts[ticker] = texts

    return seed_texts


def _seed_text_for(
    ticker: str,
    ticker_position: int,
    global_position: int,
    seed_texts: Mapping[str, Sequence[str]],
) -> str:

    texts = list(seed_texts.get(ticker, []))
    # seed_texts.get(ticker, []) ticker is avaliba just send
    # else return empty , the result will create the new list

    if not texts:
        return f"{ticker} RSS synthetic seed"
    return texts[(global_position + ticker_position) % len(texts)]

# balanced positive/neutral/negative sentiment rows.
def generate_balanced_rss_sentiment(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    tickers: Sequence[str],
    seed_texts: Mapping[str, Sequence[str]],
) -> pd.DataFrame:

    """Generate balanced positive/neutral/negative sentiment rows."""
    normalized_dates = pd.to_datetime(pd.Index(dates)).sort_values().unique()
    rows: list[dict[str, object]] = []
    global_position = 0

    for date in normalized_dates:
        for ticker_position, ticker in enumerate(tickers):

            label = SENTIMENT_LABELS[global_position % len(SENTIMENT_LABELS)]
                                     # This way maybe can used the % range(0,len(SENTIMENT_LABELS))
            rows.append(             # To enhance the len long
                {
                    "date": pd.Timestamp(date).normalize(),
                    "ticker": ticker,
                    "rss_text": _seed_text_for(
                        ticker,
                        ticker_position,
                        global_position,
                        seed_texts,
                    ),
                    "sent_label": label,
                    "sent_conf": 1.0,
                    "sent_score": SENTIMENT_SCORES[label],
                }
            )
            global_position += 1

    return pd.DataFrame(
        rows,
        columns=["date", "ticker", "rss_text", "sent_label", "sent_conf", "sent_score"],
    )

# check the emo tag is blance
def assert_balanced_sentiment(df: pd.DataFrame) -> None:

    counts = df["sent_label"].value_counts().reindex(SENTIMENT_LABELS, fill_value=0)
    if int(counts.max() - counts.min()) > 1:
        raise ValueError(f"Synthetic sentiment labels are not balanced: {counts.to_dict()}")

    # The calcuous the numeric number
    mean_score = float(pd.to_numeric(df["sent_score"], errors="coerce").mean())
    tolerance = 1.0 / max(len(df), 1) # more data , more strict

    if abs(mean_score) > tolerance:
        raise ValueError(
            "Synthetic sentiment scores are biased. "
            f"Mean score {mean_score:.6f} exceeds tolerance {tolerance:.6f}."
        )

# The build the synthetic_sentiment data
def save_balanced_rss_sentiment(
    df: pd.DataFrame,
    config: RunConfig = DEFAULT_CONFIG,
) -> Path:
    assert_balanced_sentiment(df)
    output_path = synthetic_sentiment_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

# The csv string data type transformer
def load_synthetic_sentiment(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["sent_score"] = pd.to_numeric(df["sent_score"], errors="coerce").fillna(0.0)
    return df

# The emo tabel transformer the sequene
def sentiment_frame_to_daily_market_series(
    df: pd.DataFrame,
    price_index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.Series:

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    daily = frame.groupby("date")["sent_score"].mean().sort_index()
    # only get the one group mean

    target_index = pd.to_datetime(pd.Index(price_index)).normalize()
    # The price date alaliment , ffill :call yestardy value if not news . fillna (0.0) #
    series = daily.reindex(target_index).ffill().fillna(0.0)
    series.index = pd.to_datetime(price_index)
    return series.astype(float).clip(-1.0, 1.0) # The range limted

# The rebuild news or get the cauch
def build_or_load_balanced_sentiment(
    config: RunConfig,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    seed_texts: Mapping[str, Sequence[str]] | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, Path]:

    output_path = synthetic_sentiment_path(config)

    # cauch
    if output_path.exists() and not refresh:
        df = load_synthetic_sentiment(output_path)
        assert_balanced_sentiment(df)
        return df, output_path

    if seed_texts is None:
        seed_texts = fetch_rss_seed_texts(config)

    # generate the new syn data
    df = generate_balanced_rss_sentiment(
        dates=dates,
        tickers=config.tickers,
        seed_texts=seed_texts,
    )
    saved_path = save_balanced_rss_sentiment(df, config)
    return df, saved_path


def business_dates_for_training_config(config: RunConfig = DEFAULT_CONFIG) -> pd.DatetimeIndex:
    return pd.date_range(config.train_start, config.valid_end, freq="B")
