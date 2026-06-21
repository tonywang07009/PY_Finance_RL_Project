"""Online DDPG evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG

from .config import DEFAULT_CONFIG, RunConfig
from .paths import PROJECT_ROOT
from .training import GymPortfolioEnv, make_portfolio_config


class PredictModel(Protocol):
    def predict(self, obs, deterministic: bool = True): ...


@dataclass(frozen=True)
class OnlineEvaluationConfig:
    """Runtime options for online evaluation."""

    debug_steps: int = 0
    deterministic: bool = True
    profile_name: str = "ddpg_slm"
    online_start: str = DEFAULT_CONFIG.online_start
    online_end: str = DEFAULT_CONFIG.online_end
    save_plots: bool = False
    plot_dir: str | Path = "."
    save_profile: bool = False
    profile_dir: str | Path = DEFAULT_CONFIG.result_profile_dir
    show_plots: bool = True


def create_online_env(
    price_df_online: pd.DataFrame,
    sentiment_series: pd.Series | None,
    config: RunConfig = DEFAULT_CONFIG,
) -> GymPortfolioEnv:
    """Create an online env and attach optional SLM sentiment series."""
    use_slm = sentiment_series is not None
    config_online = make_portfolio_config(config, use_slm=use_slm)
    env_online = GymPortfolioEnv(price_df_online, config_online, use_slm=use_slm)
    if sentiment_series is not None:
        env_online.set_sentiment_series(sentiment_series)
    return env_online


def load_online_model(
    model_path: str | Path,
    env: GymPortfolioEnv,
):
    """Load the trained DDPG model for online evaluation."""
    resolved_path = resolve_model_path(model_path)
    return DDPG.load(str(resolved_path), env=env)


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolve a model path, accepting either the base name or `.zip` file."""
    path = Path(model_path)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
        if path.suffix != ".zip":
            candidates.append(path.with_suffix(".zip"))
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                PROJECT_ROOT / path,
            ]
        )
        if path.suffix != ".zip":
            candidates.extend(
                [
                    (Path.cwd() / path).with_suffix(".zip"),
                    (PROJECT_ROOT / path).with_suffix(".zip"),
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected = candidates[0]
    raise FileNotFoundError(
        "Trained DDPG model file was not found. "
        f"Expected model path like '{expected}' or '{expected.with_suffix('.zip')}'. "
        "Run the optional offline training cell/script first, or set RunConfig.model_path "
        "to an existing trained model."
    )


def run_debug_steps(
    model: PredictModel,
    env: GymPortfolioEnv,
    debug_steps: int,
    deterministic: bool = True,
) -> None:
    """Print a short observation trace without committing logs."""
    if debug_steps <= 0:
        return

    obs, info = env.reset()
    print("initial obs last 2 (wealth, slm):", obs[-2:])

    for _ in range(debug_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        print("step obs last 2 (wealth, slm):", obs[-2:])
        if terminated or truncated:
            break


def collect_online_logs(
    model: PredictModel,
    env: GymPortfolioEnv,
    deterministic: bool = True,
    reset_env: bool = True,
) -> pd.DataFrame:
    """Run the online env once and collect raw log rows."""
    obs, info = env.reset() if reset_env else (None, {})
    if obs is None:
        raise ValueError("collect_online_logs requires reset_env=True unless an observation is provided.")

    done = False
    logs: list[dict[str, Any]] = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        logs.append(
            {
                "time": env.core_env.price_df.index[env.core_env.current_step],
                "wealth": info["wealth"],
                "reward": reward,
                "drawdown": info["drawdown"],
                "action": action,
            }
        )

    return pd.DataFrame(logs)


def add_online_metrics(logs: pd.DataFrame) -> pd.DataFrame:
    """Add date index and daily return metrics to online logs."""
    df_logs = logs.copy()
    if df_logs.empty:
        return pd.DataFrame(columns=["wealth", "reward", "drawdown", "action", "daily_return"])

    df_logs["time"] = pd.to_datetime(df_logs["time"])
    df_logs = df_logs.set_index("time")
    df_logs["daily_return"] = df_logs["wealth"].pct_change()
    return df_logs


def date_range_tag(online_start: str, online_end: str) -> str:
    """Return the stable date tag used in output file names."""
    return f"{online_start}_{online_end}"


def profile_filename(profile_name: str, online_start: str, online_end: str) -> str:
    """Return the standard online profile CSV file name."""
    return f"{profile_name}_online_profile_{date_range_tag(online_start, online_end)}.csv"


def _format_action_for_csv(value: Any) -> str:
    arr = np.asarray(value, dtype=float).ravel()
    return json.dumps(arr.tolist())


def save_online_profile(
    df_logs: pd.DataFrame,
    profile_name: str,
    online_start: str,
    online_end: str,
    profile_dir: str | Path,
) -> Path:
    """Save online evaluation logs as a deterministic profile CSV."""
    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / profile_filename(profile_name, online_start, online_end)

    profile = df_logs.copy()
    if "action" in profile.columns:
        profile["action"] = profile["action"].map(_format_action_for_csv)

    profile.to_csv(output_path, index_label="time")
    return output_path


def _finish_plot(
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_online_logs(
    df_logs: pd.DataFrame,
    online_start: str,
    online_end_str: str,
    options: OnlineEvaluationConfig = OnlineEvaluationConfig(),
) -> None:
    """Plot reward, wealth, and daily return from online logs."""
    output_dir = Path(options.plot_dir)
    name_tag = f"{options.profile_name}_{date_range_tag(online_start, online_end_str)}"
    reward_path = output_dir / f"online_reward_{name_tag}.png" if options.save_plots else None
    wealth_path = output_dir / f"online_wealth_{name_tag}.png" if options.save_plots else None
    return_path = output_dir / f"online_daily_return_{name_tag}.png" if options.save_plots else None

    plt.figure(figsize=(10, 4))
    plt.plot(df_logs.index, df_logs["reward"], label="Reward (Sharpe-like - DD penalty)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Reward")
    plt.title(f"Online Simulation: Step-wise Reward ({online_start} ~ {online_end_str})")
    _finish_plot(reward_path, show_plot=options.show_plots)

    plt.figure(figsize=(10, 4))
    plt.plot(df_logs.index, df_logs["wealth"], label="Portfolio Wealth")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.title(f"Online Simulation: Portfolio Wealth ({online_start} ~ {online_end_str})")
    _finish_plot(wealth_path, show_plot=options.show_plots)

    plt.figure(figsize=(10, 4))
    plt.plot(df_logs.index, df_logs["daily_return"], label="Daily Portfolio Return")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.title(f"Online Simulation: Daily Portfolio Returns ({online_start} ~ {online_end_str})")
    _finish_plot(return_path, show_plot=options.show_plots)


def run_online_evaluation(
    price_df_online: pd.DataFrame,
    sentiment_series: pd.Series | None,
    online_end_str: str,
    config: RunConfig = DEFAULT_CONFIG,
    save_plots: bool = False,
    plot_dir: str | Path = ".",
    save_profile: bool = False,
    profile_dir: str | Path = DEFAULT_CONFIG.result_profile_dir,
    profile_name: str = "ddpg_slm",
    show_plots: bool = True,
    debug_steps: int = 0,
) -> pd.DataFrame:
    """High-level online evaluation wrapper used by scripts and notebooks."""
    options = OnlineEvaluationConfig(
        debug_steps=debug_steps,
        profile_name=profile_name,
        online_start=config.online_start,
        online_end=online_end_str,
        save_plots=save_plots,
        plot_dir=plot_dir,
        save_profile=save_profile,
        profile_dir=profile_dir,
        show_plots=show_plots,
    )

    env_online = create_online_env(price_df_online, sentiment_series, config)
    model = load_online_model(config.model_path, env_online)

    run_debug_steps(
        model,
        env_online,
        debug_steps=options.debug_steps,
        deterministic=options.deterministic,
    )

    logs = collect_online_logs(
        model,
        env_online,
        deterministic=options.deterministic,
        reset_env=True,
    )
    df_logs = add_online_metrics(logs)

    print("df_logs range:", df_logs.index.min(), "->", df_logs.index.max())
    print(df_logs.head())

    plot_online_logs(df_logs, config.online_start, online_end_str, options)
    if options.save_profile:
        profile_path = save_online_profile(
            df_logs,
            options.profile_name,
            options.online_start,
            options.online_end,
            options.profile_dir,
        )
        print("saved online profile:", profile_path)
    return df_logs
