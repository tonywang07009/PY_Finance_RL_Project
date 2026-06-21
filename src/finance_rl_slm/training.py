"""Offline DDPG training helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from .config import DEFAULT_CONFIG, RunConfig
from .paths import ensure_project_paths

ensure_project_paths()
from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig  # noqa: E402


def make_portfolio_config(
    config: RunConfig = DEFAULT_CONFIG,
    use_slm: bool = False,
) -> PortfolioEnvConfig:
    return PortfolioEnvConfig(
        use_slm=use_slm,
        tickers=config.ticker_list,
        init_wealth=config.init_wealth,
        wealth_norm_factor=config.wealth_norm_factor,
    )


def train_offline_model(
    price_df_train: pd.DataFrame,
    price_df_valid: pd.DataFrame,
    config: RunConfig = DEFAULT_CONFIG,
) -> Any:
    config_offline = make_portfolio_config(config, use_slm=False)
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

    model.learn(total_timesteps=config.total_timesteps)
    model.save(config.model_path)

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
