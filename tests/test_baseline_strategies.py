from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def make_price_frame(rows: int = 8, tickers: tuple[str, ...] = ("IBM", "NVDA", "GM")) -> pd.DataFrame:
    index = pd.date_range("2025-12-22", periods=rows, freq="B")
    return pd.DataFrame(
        {
            ticker: np.linspace(100.0 + offset * 5.0, 100.0 + offset * 5.0 + rows - 1, rows)
            for offset, ticker in enumerate(tickers)
        },
        index=index,
    )


class BaselineStrategyTests(unittest.TestCase):
    def test_buy_hold_profile_keeps_share_counts_fixed(self) -> None:
        from baseline.baseline_strategies import build_buy_hold_profile

        tickers = ("IBM", "NVDA", "GM")
        price_df = make_price_frame(rows=6, tickers=tickers)
        profile = build_buy_hold_profile(price_df, tickers, initial_capital=9000.0)

        self.assertEqual(len(profile), len(price_df) - 1)
        first_shares = np.asarray(profile.iloc[0]["shares"], dtype=float)
        last_shares = np.asarray(profile.iloc[-1]["shares"], dtype=float)
        self.assertTrue(np.allclose(first_shares, last_shares))
        self.assertTrue(np.isnan(profile.iloc[0]["daily_return"]))

        final_value = float((price_df.iloc[-1].to_numpy(dtype=float) * last_shares).sum())
        self.assertAlmostEqual(float(profile.iloc[-1]["wealth"]), final_value / 9000.0)
        self.assertAlmostEqual(sum(profile.iloc[-1]["action"]), 1.0)

    def test_markov_profile_uses_three_day_state_and_fallback(self) -> None:
        from baseline.baseline_strategies import (
            build_markov_chain_profile,
            predict_markov_up_probability,
            train_markov_chain_model,
        )

        tickers = ("IBM", "NVDA")
        historical = make_price_frame(rows=8, tickers=tickers)
        online = make_price_frame(rows=5, tickers=tickers)
        online.index = pd.date_range("2026-01-02", periods=5, freq="B")

        model = train_markov_chain_model(historical, tickers, state_window=3)
        fallback = predict_markov_up_probability(model, tuple([0] * 6))
        self.assertTrue(np.allclose(fallback, model.unconditional_up_probability))

        profile = build_markov_chain_profile(historical, online, tickers, state_window=3)
        first_state = json.loads(profile.iloc[0]["strategy_signal"])
        self.assertEqual(len(first_state), 3 * len(tickers))
        self.assertAlmostEqual(sum(profile.iloc[0]["action"]), 1.0)
        self.assertIn("predicted_up_probability", profile.columns)

    def test_save_baseline_profile_uses_existing_profile_contract(self) -> None:
        from baseline.baseline_strategies import build_buy_hold_profile, save_baseline_profile

        tickers = ("IBM", "NVDA", "GM")
        profile = build_buy_hold_profile(make_price_frame(rows=5, tickers=tickers), tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_baseline_profile(
                profile,
                "buy_hold",
                "2026-01-01",
                "2026-06-21",
                tmpdir,
            )
            saved = pd.read_csv(output)

        self.assertEqual(output.name, "buy_hold_online_profile_2026-01-01_2026-06-21.csv")
        for column in ("time", "wealth", "reward", "drawdown", "action", "daily_return"):
            self.assertIn(column, saved.columns)
        self.assertTrue(saved.loc[0, "action"].startswith("["))

    def test_run_baselines_uses_shared_data_module(self) -> None:
        from baseline import run_baselines

        tickers = tuple(run_baselines.DEFAULT_CONFIG.ticker_list)
        prices = make_price_frame(rows=40, tickers=tickers)

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "profiles"
            with patch.object(run_baselines, "download_price_df", return_value=prices) as download_mock:
                outputs = run_baselines.main(
                    [
                        "--profile-dir",
                        str(profile_dir),
                        "--plot-dir",
                        str(Path(tmpdir) / "plots"),
                        "--report-output",
                        str(Path(tmpdir) / "report.html"),
                        "--skip-report",
                    ]
                )

        download_mock.assert_called_once()
        self.assertEqual(
            Path(outputs["buy_hold_profile"]).name,
            "buy_hold_online_profile_2026-01-01_2026-06-21.csv",
        )
        self.assertEqual(
            Path(outputs["markov_chain_profile"]).name,
            "markov_chain_online_profile_2026-01-01_2026-06-21.csv",
        )
        self.assertEqual(Path(outputs["buy_hold_profile"]).parent, profile_dir)
        self.assertEqual(Path(outputs["markov_chain_profile"]).parent, profile_dir)

    def test_run_baselines_shared_data_module_error_is_user_visible(self) -> None:
        from baseline import run_baselines

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(run_baselines, "download_price_df", side_effect=RuntimeError("Yahoo unavailable")):
                with self.assertRaises(SystemExit) as context:
                    run_baselines.main(
                        [
                            "--profile-dir",
                            str(Path(tmpdir) / "profiles"),
                            "--plot-dir",
                            str(Path(tmpdir) / "plots"),
                            "--report-output",
                            str(Path(tmpdir) / "report.html"),
                            "--skip-report",
                        ]
                    )

        message = str(context.exception)
        self.assertIn("finance_rl_slm.data.download_price_df()", message)
        self.assertIn("Yahoo unavailable", message)


if __name__ == "__main__":
    unittest.main()
