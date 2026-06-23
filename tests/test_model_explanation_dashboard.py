from __future__ import annotations

import importlib
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


def make_profile(path: Path, actions: list[str], wealth: list[float], include_sentiment: bool = False) -> Path:
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "wealth": wealth,
            "reward": [0.0, 0.02, -0.01],
            "drawdown": [0.0, 0.0, 0.01],
            "action": actions,
            "daily_return": [np.nan, 0.02, -0.01],
        }
    )
    if include_sentiment:
        df["sentiment"] = [0.0, 0.25, -0.25]
    df.to_csv(path, index=False)
    return path


def make_price_frame() -> pd.DataFrame:
    tickers = ("IBM", "NVDA", "GM", "BLK", "COST")
    dates = pd.date_range("2025-12-15", periods=40, freq="B")
    return pd.DataFrame(
        {
            ticker: np.linspace(100.0 + offset * 10.0, 119.0 + offset * 10.0, len(dates))
            for offset, ticker in enumerate(tickers)
        },
        index=dates,
    )


class ModelExplanationDashboardTests(unittest.TestCase):
    def test_investment_value_and_profit_loss(self) -> None:
        from version.model_explainer import calculate_investment_value, calculate_profit_loss

        self.assertEqual(calculate_investment_value(1.05, 100000), 105000.0)
        self.assertEqual(calculate_profit_loss(1.10, 100000), 10000.0)
        self.assertEqual(calculate_profit_loss(0.95, 100000), -5000.0)

    def test_action_parsing_and_normalized_weights(self) -> None:
        from version.model_explainer import normalize_action_to_weights, parse_action_vector

        action = parse_action_vector("[2.0, -1.0, 1.0]")
        weights = normalize_action_to_weights(action, asset_count=3)

        self.assertEqual(action, [2.0, -1.0, 1.0])
        self.assertTrue(np.allclose(weights, np.array([2.0 / 3.0, 0.0, 1.0 / 3.0])))

        equal_weights = normalize_action_to_weights([-1.0, 0.0, -2.0], asset_count=3)
        self.assertTrue(np.allclose(equal_weights, np.ones(3) / 3.0))

    def test_strategy_summary_and_two_model_comparison(self) -> None:
        from version.model_explainer import compare_model_profiles, load_profile, summarize_strategy

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            only_path = make_profile(
                tmp_path / "only.csv",
                actions=["[2, 0, 1]", "[1, 1, 1]", "[0, 3, 0]"],
                wealth=[1.0, 1.05, 1.10],
            )
            slm_path = make_profile(
                tmp_path / "slm.csv",
                actions=["[1, 2, 0]", "[0, 3, 0]", "[0, 2, 1]"],
                wealth=[1.0, 1.08, 1.20],
                include_sentiment=True,
            )

            only_profile = load_profile(only_path)
            summary = summarize_strategy(
                only_profile,
                "Only-DDPG",
                only_path,
                tickers=("IBM", "NVDA", "GM"),
                initial_capital=100000,
            )
            self.assertEqual(summary.model_name, "Only-DDPG")
            self.assertEqual(summary.final_investment_value, 110000.0)
            self.assertEqual(summary.profit_loss, 10000.0)
            self.assertIn(summary.most_allocated_ticker, {"IBM", "NVDA", "GM"})

            comparison = compare_model_profiles(
                only_path,
                slm_path,
                tickers=("IBM", "NVDA", "GM"),
                initial_capital=100000,
                currency="USD",
            )
            self.assertEqual(comparison["currency"], "USD")
            self.assertEqual(len(comparison["models"]), 2)
            self.assertAlmostEqual(
                comparison["difference_ddpg_slm_minus_only_ddpg"]["profit_loss"],
                10000.0,
            )

    def test_html_output_contains_required_sections(self) -> None:
        from version.model_explainer import compare_four_pipeline_profiles, compare_model_profiles
        from version.model_report_html import generate_dashboard_html, write_dashboard_html

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            only_path = make_profile(
                tmp_path / "only.csv",
                actions=["[1, 1, 1]", "[2, 1, 0]", "[1, 0, 2]"],
                wealth=[1.0, 1.03, 1.04],
            )
            slm_path = make_profile(
                tmp_path / "slm.csv",
                actions=["[1, 1, 1]", "[0, 2, 1]", "[0, 3, 1]"],
                wealth=[1.0, 1.05, 1.08],
                include_sentiment=True,
            )
            report = compare_model_profiles(
                only_path,
                slm_path,
                tickers=("IBM", "NVDA", "GM"),
                initial_capital=100000,
                currency="USD",
            )
            html = generate_dashboard_html(report)
            self.assertIn("Only-DDPG", html)
            self.assertIn("DDPG+SLM", html)
            self.assertIn("Profit / Loss", html)
            self.assertIn("Investment Strategy Analysis", html)
            self.assertIn("USD", html)

            output = write_dashboard_html(report, tmp_path / "model_report.html")
            self.assertTrue(output.exists())
            self.assertIn("DDPG MODEL EXPLANATION DASHBOARD", output.read_text())

            buy_hold_path = make_profile(
                tmp_path / "buy_hold.csv",
                actions=["[1, 1, 1]", "[1, 1, 1]", "[1, 1, 1]"],
                wealth=[1.0, 1.01, 1.03],
            )
            markov_path = make_profile(
                tmp_path / "markov.csv",
                actions=["[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"],
                wealth=[1.0, 0.99, 1.02],
            )
            four_report = compare_four_pipeline_profiles(
                only_path,
                slm_path,
                buy_hold_path,
                markov_path,
                tickers=("IBM", "NVDA", "GM"),
                initial_capital=100000,
            )
            four_html = generate_dashboard_html(four_report)
            self.assertIn("Buy-and-Hold", four_html)
            self.assertIn("Markov Chain", four_html)

    def test_run_model_report_imports_safely_and_generates_without_server(self) -> None:
        run_model_report = importlib.import_module("version.run_model_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            action = "[1, 1, 1, 1, 1]"
            only_path = make_profile(tmp_path / "only.csv", [action, action, action], [1.0, 1.02, 1.04])
            slm_path = make_profile(
                tmp_path / "slm.csv",
                [action, action, action],
                [1.0, 1.03, 1.05],
                include_sentiment=True,
            )
            output = tmp_path / "report.html"
            buy_hold_path = make_profile(tmp_path / "buy_hold.csv", [action, action, action], [1.0, 1.01, 1.02])
            markov_path = make_profile(tmp_path / "markov.csv", [action, action, action], [1.0, 0.99, 1.01])
            with patch.object(run_model_report, "write_four_pipeline_outputs") as plots_mock:
                result = run_model_report.main(
                    [
                        "--only-ddpg-profile",
                        str(only_path),
                        "--ddpg-slm-profile",
                        str(slm_path),
                        "--buy-hold-profile",
                        str(buy_hold_path),
                        "--markov-profile",
                        str(markov_path),
                        "--output",
                        str(output),
                        "--no-serve",
                    ]
                )
            self.assertEqual(result, output)
            plots_mock.assert_called_once()
            self.assertTrue(output.exists())
            html = output.read_text()
            self.assertIn("Buy-and-Hold", html)
            self.assertIn("Markov Chain", html)

    def test_run_model_report_missing_baselines_uses_shared_data_module_error(self) -> None:
        run_model_report = importlib.import_module("version.run_model_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            action = "[1, 1, 1, 1, 1]"
            only_path = make_profile(tmp_path / "only.csv", [action, action, action], [1.0, 1.02, 1.04])
            slm_path = make_profile(
                tmp_path / "slm.csv",
                [action, action, action],
                [1.0, 1.03, 1.05],
                include_sentiment=True,
            )
            with self.assertRaises(SystemExit) as context:
                with patch("baseline.run_baselines.download_price_df", side_effect=RuntimeError("data loader failed")):
                    run_model_report.main(
                        [
                            "--only-ddpg-profile",
                            str(only_path),
                            "--ddpg-slm-profile",
                            str(slm_path),
                            "--buy-hold-profile",
                            str(tmp_path / "missing_buy_hold.csv"),
                            "--markov-profile",
                            str(tmp_path / "missing_markov.csv"),
                            "--output",
                            str(tmp_path / "report.html"),
                            "--no-serve",
                        ]
                    )

            message = str(context.exception)
            self.assertIn("finance_rl_slm.data.download_price_df()", message)
            self.assertIn("data loader failed", message)

    def test_run_model_report_generates_missing_baselines_from_data_module(self) -> None:
        run_model_report = importlib.import_module("version.run_model_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            action = "[1, 1, 1, 1, 1]"
            only_path = make_profile(tmp_path / "only.csv", [action, action, action], [1.0, 1.02, 1.04])
            slm_path = make_profile(
                tmp_path / "slm.csv",
                [action, action, action],
                [1.0, 1.03, 1.05],
                include_sentiment=True,
            )
            buy_hold_path = tmp_path / "buy_hold_online_profile_2026-01-01_2026-06-21.csv"
            markov_path = tmp_path / "markov_chain_online_profile_2026-01-01_2026-06-21.csv"
            output = tmp_path / "report.html"

            with (
                patch("baseline.run_baselines.download_price_df", return_value=make_price_frame()) as download_mock,
                patch.object(run_model_report, "write_four_pipeline_outputs") as plots_mock,
            ):
                result = run_model_report.main(
                    [
                        "--only-ddpg-profile",
                        str(only_path),
                        "--ddpg-slm-profile",
                        str(slm_path),
                        "--buy-hold-profile",
                        str(buy_hold_path),
                        "--markov-profile",
                        str(markov_path),
                        "--output",
                        str(output),
                        "--no-serve",
                    ]
                )

            self.assertEqual(result, output)
            download_mock.assert_called_once()
            plots_mock.assert_called_once()
            self.assertTrue(buy_hold_path.exists())
            self.assertTrue(markov_path.exists())
            html = output.read_text()
            self.assertIn("Buy-and-Hold", html)
            self.assertIn("Markov Chain", html)
            self.assertIn("Profit / Loss", html)

    def test_run_model_report_reuses_existing_baselines_without_yahoo_download(self) -> None:
        run_model_report = importlib.import_module("version.run_model_report")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            action = "[1, 1, 1, 1, 1]"
            only_path = make_profile(tmp_path / "only.csv", [action, action, action], [1.0, 1.02, 1.04])
            slm_path = make_profile(
                tmp_path / "slm.csv",
                [action, action, action],
                [1.0, 1.03, 1.05],
                include_sentiment=True,
            )
            buy_hold_path = make_profile(tmp_path / "buy_hold.csv", [action, action, action], [1.0, 1.01, 1.02])
            markov_path = make_profile(tmp_path / "markov.csv", [action, action, action], [1.0, 0.99, 1.01])
            output = tmp_path / "report.html"

            with (
                patch("baseline.run_baselines.download_price_df") as download_mock,
                patch.object(run_model_report, "write_four_pipeline_outputs") as plots_mock,
            ):
                run_model_report.main(
                    [
                        "--only-ddpg-profile",
                        str(only_path),
                        "--ddpg-slm-profile",
                        str(slm_path),
                        "--buy-hold-profile",
                        str(buy_hold_path),
                        "--markov-profile",
                        str(markov_path),
                        "--output",
                        str(output),
                        "--no-serve",
                    ]
                )

            download_mock.assert_not_called()
            plots_mock.assert_called_once()
            self.assertTrue(output.exists())

    def test_run_model_report_rejects_removed_include_baselines_flag(self) -> None:
        run_model_report = importlib.import_module("version.run_model_report")

        with self.assertRaises(SystemExit):
            run_model_report.build_parser().parse_args(["--include-baselines"])

    def test_version_readme_contract(self) -> None:
        readme = (PROJECT_ROOT / "version" / "README.md").read_text()
        self.assertIn("```mermaid", readme)
        self.assertIn("## 2. API Overview", readme)
        self.assertIn("calculate_profit_loss()", readme)
        self.assertIn("python version/run_model_report.py", readme)


if __name__ == "__main__":
    unittest.main()
