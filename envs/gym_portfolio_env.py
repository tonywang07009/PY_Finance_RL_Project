from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset env and return initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any):
        """One environment step given an action."""
        raise NotImplementedError


@dataclass
class PortfolioEnvConfig:
    tickers: list[str]
    init_wealth: float = 1.0
    wealth_norm_factor: float = 100.0
    use_slm: bool = False


def validate_price_df(price_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Validate and return a sorted price frame for the requested tickers."""
    missing_tickers = set(tickers) - set(price_df.columns)
    if missing_tickers:
        raise ValueError(f"Missing tickers in price_df: {missing_tickers}")

    selected = price_df[tickers].sort_index()
    if len(selected) < 2:
        raise ValueError(
            "price_df must contain at least two rows for reset/step evaluation. "
            f"Received {len(selected)} rows."
        )

    return selected


class SimplePortfolioEnv(BaseEnv):
    """Core portfolio environment: price dynamics + wealth update."""

    def __init__(self, price_df: pd.DataFrame, config: PortfolioEnvConfig):
        self.config = config
        self.tickers = list(config.tickers)
        self.price_df = validate_price_df(price_df, self.tickers)
        self.num_assets = len(self.tickers)
        self.init_wealth = config.init_wealth

        self.returns = self.price_df.pct_change().fillna(0.0)
        self.max_step = len(self.price_df) - 1
        self.equal_w = np.array([1.0 / self.num_assets] * self.num_assets, dtype=float)

        self.current_step: int = 0
        self.current_wealth: float = self.init_wealth
        self.peak_wealth: float = self.init_wealth

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.current_wealth = self.init_wealth
        self.peak_wealth = self.init_wealth
        return self.price_df.iloc[self.current_step][self.tickers].values.astype(float)

    def step(self, action: np.ndarray | None = None):
        if self.current_step >= self.max_step:
            raise ValueError("Episode has ended. Please reset the environment.")

        ret_vec = self.returns.iloc[self.current_step + 1].values.astype(float)
        reward = float(np.dot(self.equal_w, ret_vec))
        self.current_wealth *= 1.0 + reward
        self.current_step += 1
        self.peak_wealth = max(self.peak_wealth, self.current_wealth)

        next_price = self.price_df.iloc[self.current_step].values.astype(float)
        done = self.current_step >= self.max_step
        info = {"current_wealth": self.current_wealth}
        return next_price, reward, done, info


def ema(series: np.ndarray, span: int) -> float:
    """Return the last EMA value for a numeric series."""
    if len(series) == 0:
        return 0.0
    alpha = 2 / (span + 1)
    ema_val = series[0]
    for x in series[1:]:
        ema_val = alpha * x + (1 - alpha) * ema_val
    return float(ema_val)


class GymPortfolioEnv(gym.Env, BaseEnv):
    """DDPG-ready Gymnasium wrapper around SimplePortfolioEnv."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        config: PortfolioEnvConfig,
        use_slm: bool = False,
    ):
        super().__init__()

        self.use_slm = use_slm
        self.config = config
        self.core_env = SimplePortfolioEnv(price_df=price_df, config=config)
        self.num_assets = self.core_env.num_assets
        self.current_sent_score = 0.0

        self.state_window_size = 20
        self.window_size = 20
        self.bb_window_size = 20
        self.macd_window_size = 20
        self.lambda_dd = 1.0

        self.state_returns_window = deque(maxlen=self.state_window_size)
        self.returns_window = deque(maxlen=self.window_size)
        self.bb_dev_window = deque(maxlen=self.bb_window_size)
        self.macd_window = deque(maxlen=self.macd_window_size)
        self.portfolio_wealth_history: list[float] = []

        base_dim = (
            self.state_window_size
            + self.macd_window_size
            + self.bb_window_size
            + 1 #wealth
        )

        self.obs_dim = base_dim + (1 if self.use_slm else 0)        

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

    def set_sentiment_series(self, sentiment_series: pd.Series) -> None:
        """Attach an optional date-indexed SLM sentiment series."""
        series = pd.Series(sentiment_series).copy()
        series.index = pd.to_datetime(series.index)
        self.sentiment_series = series.sort_index().astype(float)

    def action_to_weight(self, action: np.ndarray) -> np.ndarray:
        a = np.array(action, dtype=np.float32)
        a = np.maximum(a, 0.0)

        if a.sum() == 0:
            return np.ones(self.num_assets, dtype=float) / self.num_assets

        return a / a.sum()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)

        self.core_env.reset()
        self._reset_windows()

        observation = np.zeros(self.obs_dim, dtype=np.float32)
        
        if self.use_slm:
            observation[-2] = self._normalized_wealth()
            observation[-1] = np.float32(self.current_sent_score)
        else:
            observation[-1] = np.float32(self.current_sent_score)

        return observation, {}

    def step(self, action: np.ndarray):
        weight = self.action_to_weight(action)
        port_ret = self._portfolio_return(weight)
        self._advance_core_state(port_ret)

        state_ret_vec = self._state_returns_vector(port_ret)
        macd_vec = self._macd_vector()
        bb_vec = self._bb_vector()
        norm_wealth = self._normalized_wealth()
        self.current_sent_score = self._lookup_sentiment_score()

        if self.use_slm :
            observation = np.concatenate(
                (state_ret_vec, macd_vec, bb_vec, [norm_wealth, self.current_sent_score])
            ).astype(np.float32)
        else:
            observation = np.concatenate(
                (state_ret_vec, macd_vec, bb_vec, [norm_wealth])
            ).astype(np.float32)

        reward, drawdown = self._reward_and_drawdown(port_ret)
        terminated = self.core_env.current_step >= self.core_env.max_step
        truncated = False
        info = {
            "wealth": self.core_env.current_wealth,
            "portfolio_return": port_ret,
            "rolling_reward": reward,
            "drawdown": drawdown,
        }

        return observation, reward, terminated, truncated, info

    def _reset_windows(self) -> None:
        self.state_returns_window.clear()
        self.returns_window.clear()
        self.bb_dev_window.clear()
        self.macd_window.clear()
        self.portfolio_wealth_history = [self.core_env.current_wealth]
        self.current_sent_score = 0.0

    def _portfolio_return(self, weight: np.ndarray) -> float:
        ret_vec = self.core_env.returns.iloc[self.core_env.current_step + 1].values.astype(float)
        return float(np.dot(weight, ret_vec))

    def _advance_core_state(self, port_ret: float) -> None:
        self.core_env.current_wealth *= 1.0 + port_ret
        self.core_env.current_step += 1
        self.core_env.peak_wealth = max(
            self.core_env.peak_wealth,
            self.core_env.current_wealth,
        )

    def _state_returns_vector(self, port_ret: float) -> np.ndarray:
        self.state_returns_window.append(port_ret)
        return self._padded_window(self.state_returns_window, self.state_window_size)

    def _macd_vector(self) -> np.ndarray:
        self.portfolio_wealth_history.append(self.core_env.current_wealth)
        hist = np.array(self.portfolio_wealth_history, dtype=float)
        macd_t = ema(hist, span=12) - ema(hist, span=26)
        self.macd_window.append(macd_t)
        return self._padded_window(self.macd_window, self.macd_window_size)

    def _bb_vector(self) -> np.ndarray:
        hist = np.array(self.portfolio_wealth_history, dtype=float)
        if len(hist) >= self.bb_window_size:
            recent = hist[-self.bb_window_size :]
            ma20 = recent.mean()
            std20 = recent.std() + 1e-8
            bb_dev_t = (recent[-1] - ma20) / std20
        else:
            bb_dev_t = 0.0

        self.bb_dev_window.append(float(bb_dev_t))
        return self._padded_window(self.bb_dev_window, self.bb_window_size)

    def _normalized_wealth(self) -> np.float32:
        return np.float32(self.core_env.current_wealth / self.config.wealth_norm_factor)

    def _lookup_sentiment_score(self) -> float:
        sentiment_series = getattr(self, "sentiment_series", None)
        if not self.use_slm or sentiment_series is None:
            return 0.0

        current_date = self.core_env.returns.index[self.core_env.current_step]
        sent_val = float(sentiment_series.get(current_date, 0.0))
        return float(np.clip(sent_val, -1.0, 1.0))

    def _reward_and_drawdown(self, port_ret: float) -> tuple[float, float]:
        self.returns_window.append(port_ret)

        if len(self.returns_window) < self.window_size:
            base_reward = port_ret
        else:
            mean_reward = np.mean(self.returns_window)
            std_reward = np.std(self.returns_window) + 1e-8
            base_reward = float(mean_reward / std_reward)

        peak = self.core_env.peak_wealth
        current = self.core_env.current_wealth
        drawdown = (peak - current) / peak if peak > 0 else 0.0
        reward = base_reward - self.lambda_dd * drawdown
        return float(np.clip(reward, -10.0, 10.0)), float(drawdown)

    @staticmethod
    def _padded_window(window: deque, size: int) -> np.ndarray:
        values = list(window)
        if len(values) < size:
            values = [0.0] * (size - len(values)) + values
        return np.array(values, dtype=np.float32)
