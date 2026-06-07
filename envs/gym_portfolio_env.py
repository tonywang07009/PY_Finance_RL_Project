import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset env and return initial state"""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
        """One environment step given an action"""
        pass


@dataclass
class PortfolioEnvConfig:
    tickers: List[str]
    init_wealth: float = 1.0
    wealth_norm_factor: float = 100.0


class SimplePortfolioEnv(BaseEnv):
    """Core portfolio environment: price dynamics + wealth update."""

    def __init__(self, price_df: pd.DataFrame, config: PortfolioEnvConfig):
        missing_tickers: set[str] = set(config.tickers) - set(price_df.columns)
        if missing_tickers:
            raise ValueError(f"Missing tickers in price_df: {missing_tickers}")

        self.price_df = price_df[config.tickers].sort_index()
        self.config = config
        self.tickers = list(config.tickers)
        self.num_assets = len(self.tickers)
        self.init_wealth = config.init_wealth

        self.returns = self.price_df.pct_change().fillna(0.0 , method=None)
        self.max_step = len(self.price_df) - 1

        self.equal_w = np.array(
            [1.0 / self.num_assets] * self.num_assets, dtype=float
        )

        self.current_step: int = 0
        self.current_wealth: float = self.init_wealth
        self.peak_wealth: float = self.init_wealth  # 新增：用來算 drawdown

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.current_wealth = self.init_wealth
        self.peak_wealth = self.init_wealth
        first_price = (
            self.price_df.iloc[self.current_step][self.tickers]
            .values.astype(float)
        )
        return first_price

    def step(self, action: np.ndarray = None):
        if self.current_step >= self.max_step:
            raise ValueError("Episode has ended. Please reset the environment.")

        weight: np.ndarray = self.equal_w

        ret_vec: np.ndarray = (
            self.returns.iloc[self.current_step + 1].values.astype(float)
        )
        reward: float = float(np.dot(weight, ret_vec))
        self.current_wealth *= (1.0 + reward)
        self.current_step += 1
        self.peak_wealth = max(self.peak_wealth, self.current_wealth)

        next_price = self.price_df.iloc[self.current_step].values.astype(float)
        done = self.current_step >= self.max_step
        info = {"current_wealth": self.current_wealth}

        return next_price, reward, done, info

def ema(series: np.ndarray, span: int) -> float:
    """簡單 EMA 回傳最後一個 EMA 值。"""
    if len(series) == 0:
        return 0.0
    alpha = 2 / (span + 1)
    ema_val = series[0]
    for x in series[1:]:
        ema_val = alpha * x + (1 - alpha) * ema_val
    return ema_val


#--- for agent env 
class GymPortfolioEnv(gym.Env, BaseEnv):
    """DDPG-ready Gymnasium wrapper around SimplePortfolioEnv."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, price_df: pd.DataFrame, config: PortfolioEnvConfig):
        super().__init__()

        missing_tickers: set[str] = set(config.tickers) - set(price_df.columns)
        if missing_tickers:
            raise ValueError(f"Missing tickers in price_df: {missing_tickers}")

        self.core_env = SimplePortfolioEnv(price_df=price_df, config=config)
        self.config = config
        self.num_assets = self.core_env.num_assets

        # -------- state 用的 rolling window（過去 K 天投組報酬） --------
        self.state_window_size = 20
        self.state_returns_window = deque(maxlen=self.state_window_size)

        # -------- reward 用的 rolling window（Sharpe-like） --------
        self.window_size = 20
        self.returns_window = deque(maxlen=self.window_size)

        #--------- 懲罰用的
        self.lambda_dd = 1.0

        # -------- state: 20 日 Bollinger 偏離 window（基於 portfolio wealth） --------
        self.bb_window_size = 20
        self.bb_dev_window = deque(maxlen=self.bb_window_size)



        #-------- for macd
        self.macd_window_size = 20
        self.portfolio_wealth_history = []
        self.macd_window = deque(maxlen=self.macd_window_size)
        obs_dim = (
            self.state_window_size  
            + self.macd_window_size 
            + self.bb_window_size  
            + 1
            )
           

        # observation = [過去 K 天 portfolio returns, normalized wealth]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # action = 各資產目標權重
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

    # ---------- Action 轉成權重 ----------
    def action_to_weight(self, action: np.ndarray) -> np.ndarray:
        a = np.array(action, dtype=np.float32)
        a = np.maximum(a, 0.0)

        if a.sum() == 0:
            return np.ones(self.num_assets, dtype=float) / self.num_assets

        return a / a.sum()

    # ---------- Gym 的 reset ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)

        self.core_env.reset()
        self.state_returns_window.clear()
        self.returns_window.clear()
        self.bb_dev_window.clear()
        self.portfolio_wealth_history = [self.core_env.current_wealth]
        self.macd_window.clear()
        # 初始 state：returns 窗口全 0 + normalized wealth
        obs_dim = (
            self.state_window_size
            + self.macd_window_size
            + self.bb_window_size
            + 1
        )
        observation = np.zeros(obs_dim, dtype=np.float32)
        observation[-1] = np.float32(
            self.core_env.current_wealth / self.config.wealth_norm_factor
        )
        info = {}
        return observation, info

    def step(self, action: np.ndarray):

        # 1. action -> portfolio weight
        weight = self.action_to_weight(action)

        # 2. 計算當期 portfolio return
        ret_vec = (
            self.core_env.returns.iloc[self.core_env.current_step + 1]
            .values.astype(float)
        )
        port_ret = float(np.dot(weight, ret_vec))

        self.core_env.current_wealth *= 1.0 + port_ret
        self.core_env.current_step += 1
        self.core_env.peak_wealth = max(
            self.core_env.peak_wealth, self.core_env.current_wealth
        )

        # 3A. 更新 state returns window
        self.state_returns_window.append(port_ret)
        state_ret_list = list(self.state_returns_window)

        if len(state_ret_list) < self.state_window_size:
            pad_len = self.state_window_size - len(state_ret_list)
            state_ret_list = [0.0] * pad_len + state_ret_list

        state_ret_vec = np.array(state_ret_list, dtype=np.float32)

        # 3B. 更新 wealth history
        self.portfolio_wealth_history.append(self.core_env.current_wealth)
        hist = np.array(self.portfolio_wealth_history, dtype=float)

        # ---- MACD（wealth 的 12 / 26 EMA 差）----
        fast = ema(hist, span=12)
        slow = ema(hist, span=26)
        macd_t = fast - slow

        self.macd_window.append(macd_t)  
        macd_list = list(self.macd_window)
        
        if len(macd_list) < self.macd_window_size:
            pad_len = self.macd_window_size - len(macd_list)
            macd_list = [0.0] * pad_len + macd_list
        macd_vec = np.array(macd_list, dtype=np.float32)

        # ---- Bollinger 偏離： (W_t - MA20) / (std20 + 1e-8) ----
        if len(hist) >= self.bb_window_size:
            recent = hist[-self.bb_window_size :]
            ma20 = recent.mean()
            std20 = recent.std() + 1e-8
            bb_dev_t = (recent[-1] - ma20) / std20
        else:
            bb_dev_t = 0.0

        self.bb_dev_window.append(bb_dev_t)
        bb_list = list(self.bb_dev_window)

        if len(bb_list) < self.bb_window_size:
            pad_len = self.bb_window_size - len(bb_list)
            bb_list = [0.0] * pad_len + bb_list
        bb_vec = np.array(bb_list, dtype=np.float32)

        # 3C. wealth 特徵
        norm_wealth = np.float32(
            self.core_env.current_wealth / self.config.wealth_norm_factor
        )

        
        # observation = [過去 K 天 portfolio returns, normalized wealth]
        observation = np.concatenate(
                    (state_ret_vec, macd_vec, bb_vec, [norm_wealth])
                    ).astype(np.float32)

        # 4. reward 用的 rolling Sharpe-like
        self.returns_window.append(port_ret)

        if len(self.returns_window) < self.window_size:
            base_reward = port_ret
        else:
            mean_reward = np.mean(self.returns_window)
            std_reward = np.std(self.returns_window) + 1e-8
            sharpe_like = mean_reward / std_reward
            base_reward = float(sharpe_like)

        # 添加懲罰
        # 確保 SimplePortfolioEnv 在更新 wealth 後已更新 peak_wealth
        peak = self.core_env.peak_wealth
        current = self.core_env.current_wealth

        if peak > 0:
            drawdown = (peak - current) / peak      # 0 ~ 1 之間
        else:
            drawdown = 0.0

        dd_penalty = self.lambda_dd * drawdown
        # 最終 reward = Sharpe-like - 懲罰
        reward = base_reward - dd_penalty
        reward = float(np.clip(reward, -10.0, 10.0))

        terminated = self.core_env.current_step >= self.core_env.max_step
        truncated = False
        info = {
            "wealth": self.core_env.current_wealth,
            "portfolio_return": port_ret,
            "rolling_reward": reward,
            "drawdown": drawdown,
        }

        return observation, reward, terminated, truncated, info