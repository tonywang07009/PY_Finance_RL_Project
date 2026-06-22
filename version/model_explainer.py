"""Explain portfolio model profiles in money and strategy terms."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "addenda" / "result_profile_comparse"
DEFAULT_ONLY_DDPG_PROFILE = DEFAULT_PROFILE_DIR / "only_ddpg_online_profile_2026-01-01_2026-06-21.csv"
DEFAULT_DDPG_SLM_PROFILE = DEFAULT_PROFILE_DIR / "ddpg_slm_online_profile_2026-01-01_2026-06-21.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "version" / "model_report.html"
DEFAULT_TICKERS = ("IBM", "NVDA", "GM", "BLK", "COST")
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_CURRENCY = "USD"
REQUIRED_COLUMNS = ("wealth", "reward", "drawdown", "action", "daily_return")


@dataclass(frozen=True)
class ModelStrategySummary:
    """Summary for one model profile."""

    model_name: str
    profile_path: str
    initial_capital: float
    currency: str
    start_date: str
    end_date: str
    row_count: int
    final_wealth: float
    final_investment_value: float
    profit_loss: float
    cumulative_return: float
    mean_daily_return: float
    std_daily_return: float
    average_reward: float
    max_drawdown: float
    average_turnover: float
    most_allocated_ticker: str
    average_weights: dict[str, float]
    sentiment_mean: float | None = None


def load_profile(path: str | Path) -> pd.DataFrame:
    """Load a model profile CSV and normalize index/numeric columns."""

    profile_path = Path(path)
    df = pd.read_csv(profile_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{profile_path} is missing required columns: {sorted(missing)}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    for column in ("wealth", "reward", "drawdown", "daily_return", "sentiment"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_index()


def parse_action_vector(value: Any) -> list[float]:
    """Parse one CSV action value into a flat list of floats."""

    raw_value = value
    if isinstance(value, str):
        raw_value = ast.literal_eval(value)

    array = np.asarray(raw_value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("action vector must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("action vector contains non-finite values")
    return [float(item) for item in array]


def normalize_action_to_weights(action: Iterable[float], asset_count: int | None = None) -> np.ndarray:
    """Convert a raw DDPG action vector into non-negative portfolio weights."""

    weights = np.asarray(list(action), dtype=float).reshape(-1)
    if asset_count is not None and len(weights) != asset_count:
        raise ValueError(f"Expected {asset_count} action values, received {len(weights)}.")
    if weights.size == 0:
        raise ValueError("action vector must not be empty")

    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.ones(len(weights), dtype=float) / len(weights)
    return weights / total


def action_weight_frame(profile: pd.DataFrame, tickers: Sequence[str] = DEFAULT_TICKERS) -> pd.DataFrame:
    """Convert all action rows into a date-indexed portfolio weight frame."""

    rows = [
        normalize_action_to_weights(parse_action_vector(value), asset_count=len(tickers))
        for value in profile["action"]
    ]
    return pd.DataFrame(rows, index=profile.index, columns=list(tickers))


def calculate_investment_value(wealth: float, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> float:
    """Translate normalized wealth into money value."""

    return round(float(wealth) * float(initial_capital), 2)


def calculate_profit_loss(wealth: float, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> float:
    """Calculate money profit/loss against the initial capital."""

    return round(calculate_investment_value(wealth, initial_capital) - float(initial_capital), 2)


def calculate_average_turnover(weights: pd.DataFrame) -> float:
    """Calculate average one-way turnover from normalized portfolio weights."""

    if len(weights) < 2:
        return 0.0
    turnover = weights.diff().abs().sum(axis=1).dropna() / 2.0
    return float(turnover.mean()) if not turnover.empty else 0.0


def summarize_strategy(
    profile: pd.DataFrame,
    model_name: str,
    profile_path: str | Path,
    tickers: Sequence[str] = DEFAULT_TICKERS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    currency: str = DEFAULT_CURRENCY,
) -> ModelStrategySummary:
    """Summarize capital, profit/loss, risk, and allocation behavior for one model."""

    if profile.empty:
        raise ValueError(f"{model_name} profile must not be empty")

    weights = action_weight_frame(profile, tickers)
    average_weights = weights.mean().to_dict()
    most_allocated_ticker = max(average_weights, key=average_weights.get)
    returns = profile["daily_return"].dropna()
    rewards = profile["reward"].dropna()
    drawdown = profile["drawdown"].dropna()
    wealth = profile["wealth"].dropna()
    final_wealth = float(wealth.iloc[-1])
    sentiment_mean = None
    if "sentiment" in profile.columns and not profile["sentiment"].dropna().empty:
        sentiment_mean = float(profile["sentiment"].dropna().mean())

    return ModelStrategySummary(
        model_name=model_name,
        profile_path=str(Path(profile_path)),
        initial_capital=float(initial_capital),
        currency=currency,
        start_date=str(profile.index.min().date()) if hasattr(profile.index.min(), "date") else str(profile.index.min()),
        end_date=str(profile.index.max().date()) if hasattr(profile.index.max(), "date") else str(profile.index.max()),
        row_count=int(len(profile)),
        final_wealth=final_wealth,
        final_investment_value=calculate_investment_value(final_wealth, initial_capital),
        profit_loss=calculate_profit_loss(final_wealth, initial_capital),
        cumulative_return=final_wealth - 1.0,
        mean_daily_return=float(returns.mean()) if not returns.empty else 0.0,
        std_daily_return=float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        average_reward=float(rewards.mean()) if not rewards.empty else 0.0,
        max_drawdown=float(drawdown.max()) if not drawdown.empty else 0.0,
        average_turnover=calculate_average_turnover(weights),
        most_allocated_ticker=most_allocated_ticker,
        average_weights={ticker: float(weight) for ticker, weight in average_weights.items()},
        sentiment_mean=sentiment_mean,
    )


def build_strategy_notes(summary: ModelStrategySummary) -> list[str]:
    """Build short human-readable strategy notes for the HTML dashboard."""

    sign = "profit" if summary.profit_loss >= 0.0 else "loss"
    notes = [
        f"{summary.model_name} ended with a {sign} of {summary.profit_loss:,.2f} {summary.currency}.",
        f"The largest average allocation was {summary.most_allocated_ticker}.",
        f"Average turnover was {summary.average_turnover:.2%}, showing how much the allocation changed per step.",
        f"Max drawdown was {summary.max_drawdown:.2%}, which is the largest drop from a previous peak.",
    ]
    if summary.sentiment_mean is not None:
        notes.append(f"Average SLM sentiment score during the run was {summary.sentiment_mean:.4f}.")
    return notes


def compare_model_profiles(
    only_ddpg_profile: str | Path = DEFAULT_ONLY_DDPG_PROFILE,
    ddpg_slm_profile: str | Path = DEFAULT_DDPG_SLM_PROFILE,
    tickers: Sequence[str] = DEFAULT_TICKERS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    currency: str = DEFAULT_CURRENCY,
) -> dict[str, Any]:
    """Load both profile CSVs and return side-by-side explanation data."""

    only_profile = load_profile(only_ddpg_profile)
    slm_profile = load_profile(ddpg_slm_profile)
    only_summary = summarize_strategy(
        only_profile,
        "Only-DDPG",
        only_ddpg_profile,
        tickers=tickers,
        initial_capital=initial_capital,
        currency=currency,
    )
    slm_summary = summarize_strategy(
        slm_profile,
        "DDPG+SLM",
        ddpg_slm_profile,
        tickers=tickers,
        initial_capital=initial_capital,
        currency=currency,
    )

    difference = {
        "profit_loss": round(slm_summary.profit_loss - only_summary.profit_loss, 2),
        "final_investment_value": round(
            slm_summary.final_investment_value - only_summary.final_investment_value,
            2,
        ),
        "cumulative_return": slm_summary.cumulative_return - only_summary.cumulative_return,
        "max_drawdown": slm_summary.max_drawdown - only_summary.max_drawdown,
        "mean_daily_return": slm_summary.mean_daily_return - only_summary.mean_daily_return,
    }

    return {
        "initial_capital": float(initial_capital),
        "currency": currency,
        "tickers": list(tickers),
        "models": [asdict(only_summary), asdict(slm_summary)],
        "strategy_notes": {
            only_summary.model_name: build_strategy_notes(only_summary),
            slm_summary.model_name: build_strategy_notes(slm_summary),
        },
        "difference_ddpg_slm_minus_only_ddpg": difference,
    }
