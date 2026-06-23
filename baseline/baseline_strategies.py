"""Deterministic baseline strategies for the finance RL portfolio project."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_REWARD_WINDOW = 20
DEFAULT_LAMBDA_DD = 1.0
PROFILE_COLUMNS = ("wealth", "reward", "drawdown", "action", "daily_return")


@dataclass(frozen=True)
class MarkovChainModel:
    """Transition model from a three-day up/down state to next-day up odds."""

    transition_totals: dict[tuple[int, ...], int]
    transition_successes: dict[tuple[int, ...], np.ndarray]
    unconditional_up_probability: np.ndarray
    tickers: tuple[str, ...]
    state_window: int = 3


def validate_price_frame(price_df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Return a sorted price frame with the requested ticker columns."""

    missing = set(tickers) - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df is missing ticker columns: {sorted(missing)}")

    selected = price_df.loc[:, list(tickers)].copy()
    selected.index = pd.to_datetime(selected.index)
    selected = selected.sort_index().dropna(how="any")
    if len(selected) < 2:
        raise ValueError("price_df must contain at least two rows.")
    return selected


def normalize_weights(values: Iterable[float]) -> np.ndarray:
    """Normalize non-negative values into portfolio weights."""

    weights = np.asarray(list(values), dtype=float).reshape(-1)
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(weights.size, dtype=float) / weights.size
    return weights / total


def _reward_and_drawdown_series(
    step_returns: Sequence[float],
    wealth_values: Sequence[float],
    window_size: int = DEFAULT_REWARD_WINDOW,
    lambda_dd: float = DEFAULT_LAMBDA_DD,
) -> tuple[list[float], list[float]]:
    rewards: list[float] = []
    drawdowns: list[float] = []
    returns_window: list[float] = []
    peak_wealth = 1.0

    for port_ret, wealth in zip(step_returns, wealth_values):
        returns_window.append(float(port_ret))
        if len(returns_window) < window_size:
            base_reward = float(port_ret)
        else:
            recent = np.asarray(returns_window[-window_size:], dtype=float)
            base_reward = float(recent.mean() / (recent.std() + 1e-8))

        peak_wealth = max(peak_wealth, float(wealth))
        drawdown = (peak_wealth - float(wealth)) / peak_wealth if peak_wealth > 0.0 else 0.0
        reward = float(np.clip(base_reward - lambda_dd * drawdown, -10.0, 10.0))
        rewards.append(reward)
        drawdowns.append(float(drawdown))

    return rewards, drawdowns


def _profile_frame(
    dates: pd.Index,
    wealth_values: Sequence[float],
    step_returns: Sequence[float],
    weights: Sequence[Sequence[float]],
    extra_columns: dict[str, Sequence[object]] | None = None,
) -> pd.DataFrame:
    rewards, drawdowns = _reward_and_drawdown_series(step_returns, wealth_values)
    profile = pd.DataFrame(
        {
            "wealth": list(map(float, wealth_values)),
            "reward": rewards,
            "drawdown": drawdowns,
            "action": [list(map(float, row)) for row in weights],
        },
        index=pd.to_datetime(dates),
    )
    profile["daily_return"] = profile["wealth"].pct_change()

    if extra_columns:
        for name, values in extra_columns.items():
            profile[name] = list(values)

    return profile


def build_buy_hold_profile(
    price_df_online: pd.DataFrame,
    tickers: Sequence[str],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Build an equal-dollar fixed-share Buy-and-Hold baseline profile."""

    prices = validate_price_frame(price_df_online, tickers)
    ticker_count = len(tickers)
    initial_prices = prices.iloc[0].astype(float)
    per_asset_capital = float(initial_capital) / ticker_count
    shares = (per_asset_capital / initial_prices).to_numpy(dtype=float)

    wealth_values: list[float] = []
    step_returns: list[float] = []
    weights: list[list[float]] = []
    share_rows: list[list[float]] = []
    dates = prices.index[1:]

    previous_value = float(initial_capital)
    for _, price_row in prices.iloc[1:].iterrows():
        asset_values = price_row.to_numpy(dtype=float) * shares
        total_value = float(asset_values.sum())
        wealth_values.append(total_value / float(initial_capital))
        step_returns.append(total_value / previous_value - 1.0)
        weights.append((asset_values / total_value).astype(float).tolist())
        share_rows.append(shares.astype(float).tolist())
        previous_value = total_value

    return _profile_frame(
        dates,
        wealth_values,
        step_returns,
        weights,
        extra_columns={"shares": share_rows},
    )


def _sign_state(return_rows: np.ndarray) -> tuple[int, ...]:
    return tuple((np.asarray(return_rows, dtype=float).reshape(-1) > 0.0).astype(int).tolist())


def train_markov_chain_model(
    historical_price_df: pd.DataFrame,
    tickers: Sequence[str],
    state_window: int = 3,
) -> MarkovChainModel:
    """Train a ticker-level up/down transition model from historical prices."""

    prices = validate_price_frame(historical_price_df, tickers)
    returns = prices.pct_change().dropna(how="any")
    if len(returns) <= state_window:
        raise ValueError(
            f"historical_price_df must contain more than {state_window} return rows for Markov training."
        )

    return_values = returns.to_numpy(dtype=float)
    transition_totals: dict[tuple[int, ...], int] = defaultdict(int)
    transition_successes: dict[tuple[int, ...], np.ndarray] = defaultdict(
        lambda: np.zeros(len(tickers), dtype=float)
    )

    for end_idx in range(state_window, len(return_values)):
        state = _sign_state(return_values[end_idx - state_window : end_idx])
        next_up = (return_values[end_idx] > 0.0).astype(float)
        transition_totals[state] += 1
        transition_successes[state] += next_up

    unconditional = (return_values > 0.0).mean(axis=0)
    return MarkovChainModel(
        transition_totals=dict(transition_totals),
        transition_successes=dict(transition_successes),
        unconditional_up_probability=unconditional.astype(float),
        tickers=tuple(tickers),
        state_window=state_window,
    )


def predict_markov_up_probability(model: MarkovChainModel, state: tuple[int, ...]) -> np.ndarray:
    """Return next-day up probabilities for one Markov state."""

    total = model.transition_totals.get(state, 0)
    if total > 0:
        return model.transition_successes[state] / float(total)
    return model.unconditional_up_probability.copy()


def build_markov_chain_profile(
    historical_price_df: pd.DataFrame,
    price_df_online: pd.DataFrame,
    tickers: Sequence[str],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    state_window: int = 3,
) -> pd.DataFrame:
    """Build a pure three-day Markov Chain baseline profile."""

    historical_prices = validate_price_frame(historical_price_df, tickers)
    online_prices = validate_price_frame(price_df_online, tickers)
    model = train_markov_chain_model(historical_prices, tickers, state_window=state_window)

    historical_returns = historical_prices.pct_change().dropna(how="any").to_numpy(dtype=float)
    online_returns = online_prices.pct_change().to_numpy(dtype=float)
    context_returns: list[np.ndarray] = [row.astype(float) for row in historical_returns]
    if len(context_returns) < state_window:
        raise ValueError(f"Need at least {state_window} historical return rows to seed Markov state.")

    wealth = 1.0
    wealth_values: list[float] = []
    step_returns: list[float] = []
    weights: list[list[float]] = []
    probabilities: list[list[float]] = []
    states: list[str] = []
    dates = online_prices.index[1:]

    for step_idx in range(1, len(online_prices)):
        state = _sign_state(np.asarray(context_returns[-state_window:], dtype=float))
        up_probability = predict_markov_up_probability(model, state)
        weight = normalize_weights(up_probability)
        ret_vec = np.asarray(online_returns[step_idx], dtype=float)
        port_ret = float(np.dot(weight, ret_vec))
        wealth *= 1.0 + port_ret

        wealth_values.append(float(wealth))
        step_returns.append(port_ret)
        weights.append(weight.astype(float).tolist())
        probabilities.append(up_probability.astype(float).tolist())
        states.append(json.dumps(list(state)))
        context_returns.append(ret_vec)

    return _profile_frame(
        dates,
        wealth_values,
        step_returns,
        weights,
        extra_columns={
            "predicted_up_probability": probabilities,
            "strategy_signal": states,
        },
    )


def format_vector_for_csv(value: object) -> str:
    """Serialize profile vector columns into stable JSON text."""

    if isinstance(value, str):
        return value
    return json.dumps(np.asarray(value, dtype=float).reshape(-1).tolist())


def save_baseline_profile(
    profile: pd.DataFrame,
    profile_name: str,
    online_start: str,
    online_end: str,
    profile_dir: str | Path,
) -> Path:
    """Save a baseline profile using the existing online profile naming style."""

    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{profile_name}_online_profile_{online_start}_{online_end}.csv"
    saved = profile.copy()
    for column in ("action", "shares", "predicted_up_probability"):
        if column in saved.columns:
            saved[column] = saved[column].map(format_vector_for_csv)
    saved.to_csv(output_path, index_label="time")
    return output_path
