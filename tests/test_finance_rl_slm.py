from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class FinanceRlSlmTests(unittest.TestCase):
    def test_label_to_score_maps_known_and_unknown_labels(self) -> None:
        from finance_rl_slm import label_to_score

        self.assertEqual(label_to_score("positive"), 1)
        self.assertEqual(label_to_score(" negative "), -1)
        self.assertEqual(label_to_score("neutral"), 0)
        self.assertEqual(label_to_score("mixed"), 0)
        self.assertEqual(label_to_score("unexpected"), 0)

    def test_weekly_to_daily_sentiment_expands_and_clips_scores(self) -> None:
        from finance_rl_slm import weekly_to_daily_sentiment

        price_df = pd.DataFrame(
            {"IBM": [100.0, 101.0, 102.0, 103.0]},
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-13", "2026-01-20"]),
        )
        weekly_mkt = pd.Series(
            [1.5, -0.5],
            index=pd.to_datetime(["2026-01-05", "2026-01-12"]),
        )

        daily = weekly_to_daily_sentiment(weekly_mkt, price_df)

        self.assertEqual(daily.loc["2026-01-05"], 1.0)
        self.assertEqual(daily.loc["2026-01-06"], 1.0)
        self.assertEqual(daily.loc["2026-01-13"], -0.5)
        self.assertEqual(daily.loc["2026-01-20"], -0.5)

    def test_gym_portfolio_env_shape_and_slm_sentiment_injection(self) -> None:
        from envs.gym_portfolio_env import GymPortfolioEnv, PortfolioEnvConfig

        tickers = ["IBM", "NVDA", "GM", "BLK", "COST"]
        index = pd.date_range("2026-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {
                ticker: np.linspace(100.0 + i, 104.0 + i, len(index))
                for i, ticker in enumerate(tickers)
            },
            index=index,
        )
        sentiment_series = pd.Series([0.0, 0.25, -0.25, 0.5, 0.0], index=index)
        config = PortfolioEnvConfig(tickers=tickers, use_slm=True)
        env = GymPortfolioEnv(price_df, config, use_slm=True)
        env.sentiment_series = sentiment_series

        obs, info = env.reset()
        self.assertEqual(obs.shape, (62,))
        self.assertEqual(obs[-1], 0.0)

        obs, reward, terminated, truncated, info = env.step(
            np.ones(len(tickers), dtype=np.float32)
        )
        self.assertEqual(obs.shape, (62,))
        self.assertEqual(obs[-1], np.float32(0.25))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("wealth", info)

    def test_importing_package_does_not_import_transformers(self) -> None:
        code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT / "src")!r})
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(PROJECT_ROOT / "src" / "FinRL")!r})
import finance_rl_slm
print("transformers" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_data_loader_does_not_import_finrl_top_level(self) -> None:
        code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT / "src")!r})
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(PROJECT_ROOT / "src" / "FinRL")!r})
from finance_rl_slm.data import _load_yahoo_downloader_class
_load_yahoo_downloader_class()
print("finrl" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
