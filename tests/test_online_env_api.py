from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class ConstantModel:
    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, obs, deterministic: bool = True):
        return np.ones(self.action_size, dtype=np.float32), None


def make_price_df(rows: int = 5) -> pd.DataFrame:
    tickers = ["IBM", "NVDA", "GM", "BLK", "COST"]
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            ticker: np.linspace(100.0 + i, 100.0 + i + rows - 1, rows)
            for i, ticker in enumerate(tickers)
        },
        index=index,
    )


class OnlineEnvApiTests(unittest.TestCase):
    def test_action_to_weight_clips_and_normalizes(self) -> None:
        from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig

        tickers = ["IBM", "NVDA", "GM", "BLK", "COST"]
        env = GymPortfolioEnv(make_price_df(), PortfolioEnvConfig(tickers=tickers))

        equal = env.action_to_weight(np.zeros(len(tickers), dtype=np.float32))
        self.assertTrue(np.allclose(equal, np.ones(len(tickers)) / len(tickers)))

        clipped = env.action_to_weight(np.array([-1.0, 0.0, 2.0, 2.0, 0.0], dtype=np.float32))
        self.assertTrue(np.all(clipped >= 0.0))
        self.assertAlmostEqual(float(clipped.sum()), 1.0)
        self.assertTrue(np.allclose(clipped, np.array([0.0, 0.0, 0.5, 0.5, 0.0])))

    def test_env_rejects_short_price_df(self) -> None:
        from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig

        tickers = ["IBM", "NVDA", "GM", "BLK", "COST"]
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            GymPortfolioEnv(make_price_df(rows=1), PortfolioEnvConfig(tickers=tickers))

    def test_set_sentiment_series_clips_observation_value(self) -> None:
        from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig

        tickers = ["IBM", "NVDA", "GM", "BLK", "COST"]
        price_df = make_price_df()
        env = GymPortfolioEnv(price_df, PortfolioEnvConfig(tickers=tickers, use_slm=True), use_slm=True)
        env.set_sentiment_series(pd.Series([0.0, 2.5, -2.0, 0.0, 0.0], index=price_df.index))

        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(np.ones(len(tickers), dtype=np.float32))

        self.assertEqual(obs.shape, (62,))
        self.assertEqual(obs[-1], np.float32(1.0))
        self.assertIn("wealth", info)
        self.assertIn("portfolio_return", info)
        self.assertIn("rolling_reward", info)
        self.assertIn("drawdown", info)

    def test_collect_online_logs_does_not_skip_after_debug_steps(self) -> None:
        from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig
        from finance_rl_slm.evaluation import collect_online_logs, run_debug_steps

        tickers = ["IBM", "NVDA", "GM", "BLK", "COST"]
        price_df = make_price_df(rows=6)
        env = GymPortfolioEnv(price_df, PortfolioEnvConfig(tickers=tickers))
        model = ConstantModel(len(tickers))

        run_debug_steps(model, env, debug_steps=2)
        logs = collect_online_logs(model, env, reset_env=True)

        self.assertEqual(len(logs), len(price_df) - 1)
        self.assertEqual(pd.to_datetime(logs.iloc[0]["time"]), price_df.index[1])

    def test_online_metrics_and_no_plot_write(self) -> None:
        from finance_rl_slm.evaluation import (
            OnlineEvaluationConfig,
            add_online_metrics,
            plot_online_logs,
        )

        logs = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "wealth": [1.0, 1.1],
                "reward": [0.0, 0.1],
                "drawdown": [0.0, 0.0],
                "action": [np.ones(5), np.ones(5)],
            }
        )
        df_logs = add_online_metrics(logs)
        self.assertTrue(pd.isna(df_logs.iloc[0]["daily_return"]))
        self.assertAlmostEqual(float(df_logs.iloc[1]["daily_return"]), 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            options = OnlineEvaluationConfig(save_plots=False, plot_dir=tmpdir, show_plots=False)
            plot_online_logs(df_logs, "2026-01-01", "2026-01-03", options)
            self.assertFalse((Path(tmpdir) / "online_reward_SLM.png").exists())
            self.assertFalse((Path(tmpdir) / "online_wealth_SLM.png").exists())
            self.assertFalse((Path(tmpdir) / "online_daily_return_SLM.png").exists())


if __name__ == "__main__":
    unittest.main()
