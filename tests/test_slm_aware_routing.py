from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class SlmAwareRoutingTests(unittest.TestCase):
    def test_balanced_synthetic_sentiment_generation(self) -> None:
        from finance_rl_slm.synthetic_sentiment import (
            assert_balanced_sentiment,
            generate_balanced_rss_sentiment,
            sentiment_frame_to_daily_market_series,
        )

        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        tickers = ["IBM", "NVDA", "GM"]
        seed_texts = {ticker: [f"{ticker} rss seed"] for ticker in tickers}
        df = generate_balanced_rss_sentiment(dates, tickers, seed_texts)

        self.assertEqual(
            list(df.columns),
            ["date", "ticker", "rss_text", "sent_label", "sent_conf", "sent_score"],
        )
        assert_balanced_sentiment(df)
        counts = df["sent_label"].value_counts()
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertAlmostEqual(float(df["sent_score"].mean()), 0.0)

        daily = sentiment_frame_to_daily_market_series(df, dates)
        self.assertEqual(len(daily), len(dates))
        self.assertTrue(np.all(daily.between(-1.0, 1.0)))

    def test_synthetic_sentiment_path_and_image_routes(self) -> None:
        from finance_rl_slm.config import DEFAULT_CONFIG
        from finance_rl_slm.synthetic_sentiment import synthetic_sentiment_path
        from finance_rl_slm.workflow import (
            comparison_picture_path,
            only_ddpg_picture_path,
            with_slm_picture_path,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = replace(
                DEFAULT_CONFIG,
                result_picture_dir=str(Path(tmpdir) / "images"),
                synthetic_sentiment_dir=str(Path(tmpdir) / "sentiment"),
            )
            self.assertEqual(only_ddpg_picture_path(cfg).name, "only_ddpg")
            self.assertEqual(with_slm_picture_path(cfg).name, "with_slm")
            self.assertEqual(comparison_picture_path(cfg).name, "comparison")
            self.assertEqual(
                synthetic_sentiment_path(cfg).name,
                "balanced_rss_sentiment_2011-01-01_2025-12-31.csv",
            )

    def test_profile_difference_plot_writes_png(self) -> None:
        from tool.compare_ddpg_profiles import plot_profile_differences

        index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        only = pd.DataFrame(
            {
                "reward": [0.0, 0.1, -0.1],
                "daily_return": [np.nan, 0.01, -0.01],
            },
            index=index,
        )
        slm = pd.DataFrame(
            {
                "reward": [0.0, 0.2, -0.05],
                "daily_return": [np.nan, 0.02, -0.005],
            },
            index=index,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = plot_profile_differences(only, slm, Path(tmpdir) / "diff.png")
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
