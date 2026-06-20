#!/usr/bin/env python
# coding: utf-8

"""Offline DDPG portfolio training with online SLM sentiment evaluation."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "main" else SCRIPT_DIR
FINRL_ROOT = PROJECT_ROOT / "src" / "FinRL"

for path in (PROJECT_ROOT, FINRL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


MODEL_ID = "ibm-granite/granite-4.1-8b"
TICKERS = ['IBM', 'NVDA', 'GM', 'BLK', 'COST']
RSS_URLS = {
    ticker: f"http://finance.yahoo.com/rss/headline?s={ticker}"
    for ticker in TICKERS
}

START_DATE = "2005-01-01"
END_DATE = "2025-01-01"
TRAIN_START = "2005-01-01"
TRAIN_END = "2005-12-31"
VALID_START = "2016-01-01"
VALID_END = "2020-12-31"
ONLINE_START = "2026-01-01"
MODEL_PATH = "ddpg_portfolio_offline"
TOTAL_TIMESTEPS = 1_000_000


print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("loading model...")
granite_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def _model_device() -> torch.device:
    """Return a usable input device for the loaded Granite model."""
    if hasattr(granite_model, "device"):
        return granite_model.device
    return next(granite_model.parameters()).device


def ask_granite(prompt: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(_model_device())

    with torch.no_grad():
        outputs = granite_model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def analyze_sentiment(text: str, max_new_tokens: int = 128) -> dict[str, Any]:
    """Analyze one news item and return label/confidence JSON-like data."""
    system_prompt = (
        "You are an assistant specialized in sentiment analysis for financial news. "
        "Read the news content and determine the overall sentiment as "
        '"positive", "negative", "neutral", or "mixed". '
        "Estimate a confidence score between 0 and 1. "
        "You MUST output STRICT JSON only, with NO extra text, NO explanation, "
        "in this format only:\n"
        '{"label": "positive", "confidence": 0.83}\n'
        "If you cannot decide, choose neutral with a reasonable confidence."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Analyze this news item and output ONLY JSON.\nText:\n" + text},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(_model_device())

    with torch.no_grad():
        outputs = granite_model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip().strip("` \n")
    print("RAW OUTPUT:", repr(raw))

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        result = {"label": "neutral", "confidence": 0.0, "raw_output": raw}

    label = str(result.get("label", "neutral")).lower().strip()
    if label not in {"positive", "negative", "neutral", "mixed"}:
        label = "neutral"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "label": label,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        **({"raw_output": raw} if "raw_output" in result else {}),
    }


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


def analyze_news_feed(rss_url: str, ticker: str, max_items: int = 50) -> pd.DataFrame:
    """Fetch and score one ticker RSS feed."""
    rows = fetch_yahoo_news_texts(rss_url, max_items=max_items)
    records: list[dict[str, Any]] = []

    for row in rows:
        text = row["text"]
        dt = row["published"]
        res = analyze_sentiment(text)
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


def build_weekly_market_sentiment(max_items: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build news-level, ticker-week, and market-week sentiment tables."""
    all_records = []

    for ticker, rss_url in RSS_URLS.items():
        df_t = analyze_news_feed(rss_url, ticker=ticker, max_items=max_items)
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


def download_price_df(start_date: str, end_date: str | datetime | pd.Timestamp) -> pd.DataFrame:
    raw_df: pd.DataFrame = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=TICKERS,
    ).fetch_data()

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    price_df = (
        raw_df.pivot(index="date", columns="tic", values="close")[TICKERS]
        .sort_index()
        .dropna(how="any")
    )
    return price_df


def plot_normalized_prices(price_df: pd.DataFrame) -> None:
    normalized_price_df = price_df / price_df.iloc[0]

    plt.figure(figsize=(12, 6))
    for tic in TICKERS:
        plt.plot(normalized_price_df.index, normalized_price_df[tic], label=tic)

    plt.legend()
    plt.title("Normalized Stock Prices (start = 1)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.tight_layout()
    plt.show()


def train_offline_model(price_df_train: pd.DataFrame, price_df_valid: pd.DataFrame) -> DDPG:
    config_offline = PortfolioEnvConfig(
        use_slm=False,
        tickers=TICKERS,
        init_wealth=1.0,
        wealth_norm_factor=100.0,
    )
    env_train = GymPortfolioEnv(price_df_train, config_offline, use_slm=False)
    env_valid = GymPortfolioEnv(price_df_valid, config_offline, use_slm=False)

    obs, info = env_train.reset()
    print("obs shape after reset:", obs.shape)

    for _ in range(5):
        action = env_train.action_space.sample()
        obs, reward, terminated, truncated, info = env_train.step(action)
        print("obs shape in step:", obs.shape)
        if terminated or truncated:
            break

    print("MACD window:", list(env_train.macd_window))
    print("BB dev window:", list(env_train.bb_dev_window))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        env_train,
        action_noise=action_noise,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        device=device,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    obs, info = env_valid.reset()
    done = False
    wealth_traj = [env_valid.core_env.current_wealth]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_valid.step(action)
        wealth_traj.append(info["wealth"])
        done = terminated or truncated

    wealth_series = pd.Series(
        wealth_traj,
        index=price_df_valid.index[: len(wealth_traj)],
    )

    plt.figure(figsize=(12, 6))
    plt.plot(wealth_series.index, wealth_series.values, label="DDPG Strategy")
    plt.title("Out-of-sample Wealth Curve (Valid)")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.show()

    return model


def run_online_evaluation(
    price_df_online: pd.DataFrame,
    sentiment_series: pd.Series,
    online_end_str: str,
) -> pd.DataFrame:
    config_online = PortfolioEnvConfig(
        use_slm=True,
        tickers=TICKERS,
        init_wealth=1.0,
        wealth_norm_factor=100.0,
    )

    env_online = GymPortfolioEnv(price_df_online, config_online, use_slm=True)
    env_online.sentiment_series = sentiment_series

    model = DDPG.load(MODEL_PATH, env=env_online)

    obs, info = env_online.reset()
    print("initial obs last 2 (wealth, slm):", obs[-2:])
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_online.step(action)
        print("step obs last 2 (wealth, slm):", obs[-2:])
        if terminated or truncated:
            break

    done = False
    logs = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_online.step(action)
        done = terminated or truncated

        logs.append(
            {
                "time": env_online.core_env.price_df.index[env_online.core_env.current_step],
                "wealth": info["wealth"],
                "reward": reward,
                "drawdown": info["drawdown"],
                "action": action,
            }
        )

    df_logs = pd.DataFrame(logs)
    df_logs["time"] = pd.to_datetime(df_logs["time"])
    df_logs = df_logs.set_index("time")
    df_logs["daily_return"] = df_logs["wealth"].pct_change()

    print("df_logs range:", df_logs.index.min(), "->", df_logs.index.max())
    print(df_logs.head())

    plt.figure(figsize=(10, 4))
    plt.plot(df_logs.index, df_logs["reward"], label="Reward (Sharpe-like - DD penalty)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Reward")
    plt.title(f"Online Simulation: Step-wise Reward ({ONLINE_START} ~ {online_end_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df_logs.index, df_logs["wealth"], label="Portfolio Wealth")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.title(f"Online Simulation: Portfolio Wealth ({ONLINE_START} ~ {online_end_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df_logs.index, df_logs["daily_return"], label="Daily Portfolio Return")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.title(f"Online Simulation: Daily Portfolio Returns ({ONLINE_START} ~ {online_end_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_logs


def main() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("FINRL_ROOT:", FINRL_ROOT)
    print("RSS_URLS:", RSS_URLS)

    df_all_news, weekly, weekly_mkt = build_weekly_market_sentiment(max_items=50)
    print("df_all_news head:")
    print(df_all_news.head())
    print("weekly_mkt range:", weekly_mkt.index.min(), "->", weekly_mkt.index.max())

    price_df = download_price_df(START_DATE, END_DATE)
    print("price_df range:", price_df.index.min(), "->", price_df.index.max())
    print("price_df shape:", price_df.shape)
    print(price_df.head())

    plot_normalized_prices(price_df)

    price_df_train = price_df.loc[TRAIN_START:TRAIN_END].copy()
    price_df_valid = price_df.loc[VALID_START:VALID_END].copy()
    print(f"price_df_train: {price_df_train.shape}, price_df_valid: {price_df_valid.shape}")

    train_offline_model(price_df_train, price_df_valid)

    online_end_str = datetime.today().date().isoformat()
    price_df_online = download_price_df(ONLINE_START, online_end_str)

    print("price_df_online range:", price_df_online.index.min(), "->", price_df_online.index.max())
    print("price_df_online shape:", price_df_online.shape)

    sentiment_series = weekly_to_daily_sentiment(weekly_mkt, price_df_online)
    print("sentiment_series range:", sentiment_series.index.min(), "->", sentiment_series.index.max())
    print(sentiment_series.head())
    print(sentiment_series.tail())

    run_online_evaluation(price_df_online, sentiment_series, online_end_str)


if __name__ == "__main__":
    main()
