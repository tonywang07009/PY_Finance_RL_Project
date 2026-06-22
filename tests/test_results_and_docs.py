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


class ResultsAndDocsTests(unittest.TestCase):
    def test_output_contract_artifacts_exist(self) -> None:
        profile_dir = PROJECT_ROOT / "addenda" / "result_profile_comparse"
        picture_dir = PROJECT_ROOT / "addenda" / "result_picture"

        expected_profiles = [
            profile_dir / "only_ddpg_online_profile_2026-01-01_2026-06-21.csv",
            profile_dir / "ddpg_slm_online_profile_2026-01-01_2026-06-21.csv",
            profile_dir / "ddpg_vs_slm_comparison_2026-01-01_2026-06-21.csv",
        ]
        expected_figures = [
            picture_dir / "only_ddpg" / "online_reward_only_ddpg_2026-01-01_2026-06-21.png",
            picture_dir / "only_ddpg" / "online_wealth_only_ddpg_2026-01-01_2026-06-21.png",
            picture_dir / "only_ddpg" / "online_daily_return_only_ddpg_2026-01-01_2026-06-21.png",
            picture_dir / "with_slm" / "online_reward_ddpg_slm_2026-01-01_2026-06-21.png",
            picture_dir / "with_slm" / "online_wealth_ddpg_slm_2026-01-01_2026-06-21.png",
            picture_dir / "with_slm" / "online_daily_return_ddpg_slm_2026-01-01_2026-06-21.png",
            picture_dir / "comparison" / "reward_daily_return_difference_2026-01-01_2026-06-21.png",
            picture_dir / "comparison" / "daily_return_overlay_2026-01-01_2026-06-21.png",
            picture_dir / "comparison" / "daily_return_normal_distribution_2026-01-01_2026-06-21.png",
            picture_dir / "comparison" / "daily_return_boxplot_2026-01-01_2026-06-21.png",
        ]

        for path in expected_profiles + expected_figures:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"missing output contract artifact: {path}")

        for path in expected_profiles[:2]:
            with self.subTest(profile=path):
                df = pd.read_csv(path)
                self.assertEqual(len(df), 115)
                self.assertEqual(
                    {"wealth", "reward", "drawdown", "daily_return"}.issubset(df.columns),
                    True,
                )

    def test_save_online_profile_uses_standard_filename(self) -> None:
        from finance_rl_slm.evaluation import profile_filename, save_online_profile

        df_logs = pd.DataFrame(
            {
                "wealth": [1.0, 1.05],
                "reward": [0.0, 0.1],
                "drawdown": [0.0, 0.0],
                "action": [np.ones(5), np.ones(5) / 5],
                "daily_return": [np.nan, 0.05],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = save_online_profile(
                df_logs,
                "only_ddpg",
                "2026-01-01",
                "2026-06-21",
                tmpdir,
            )

            self.assertEqual(
                output.name,
                profile_filename("only_ddpg", "2026-01-01", "2026-06-21"),
            )
            saved = pd.read_csv(output)
            self.assertIn("time", saved.columns)
            self.assertTrue(saved.loc[0, "action"].startswith("["))

    def test_compare_ddpg_profiles_writes_metric_difference(self) -> None:
        from tool.compare_ddpg_profiles import (
            compare_profiles,
            compute_shared_ylim,
            plot_aligned_individual_daily_returns,
            plot_daily_return_overlay,
            plot_profile_boxplot,
            plot_profile_differences,
            plot_profile_distribution,
            write_comparison,
        )

        only = pd.DataFrame(
            {
                "wealth": [1.0, 1.1, 1.21],
                "reward": [0.0, 0.1, 0.1],
                "drawdown": [0.0, 0.0, 0.0],
                "daily_return": [np.nan, 0.1, 0.1],
            }
        )
        slm = pd.DataFrame(
            {
                "wealth": [1.0, 1.2, 1.32],
                "reward": [0.0, 0.2, 0.1],
                "drawdown": [0.0, 0.0, 0.02],
                "daily_return": [np.nan, 0.2, 0.1],
            }
        )

        summary = compare_profiles(only, slm)

        self.assertEqual(
            list(summary["pipeline"]),
            ["only_ddpg", "ddpg_slm", "difference_ddpg_slm_minus_only_ddpg"],
        )
        diff = summary.iloc[2]
        self.assertAlmostEqual(float(diff["final_wealth"]), 0.11)
        self.assertAlmostEqual(float(diff["max_drawdown"]), 0.02)

        y_min, y_max = compute_shared_ylim((only["daily_return"], slm["daily_return"]), center=0.0)
        self.assertLessEqual(y_min, -0.2)
        self.assertGreaterEqual(y_max, 0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_comparison(summary, Path(tmpdir) / "comparison.csv")
            self.assertTrue(output.exists())
            plot_output = plot_profile_differences(
                only.set_index(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])),
                slm.set_index(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])),
                Path(tmpdir) / "difference.png",
            )
            self.assertTrue(plot_output.exists())
            overlay_output = plot_daily_return_overlay(
                only.set_index(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])),
                slm.set_index(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])),
                Path(tmpdir) / "overlay.png",
            )
            distribution_output = plot_profile_distribution(only, slm, Path(tmpdir) / "distribution.png")
            boxplot_output = plot_profile_boxplot(only, slm, Path(tmpdir) / "boxplot.png")
            only_daily_output, slm_daily_output = plot_aligned_individual_daily_returns(
                only,
                slm,
                Path(tmpdir) / "only_daily.png",
                Path(tmpdir) / "slm_daily.png",
            )
            self.assertTrue(overlay_output.exists())
            self.assertTrue(distribution_output.exists())
            self.assertTrue(boxplot_output.exists())
            self.assertTrue(only_daily_output.exists())
            self.assertTrue(slm_daily_output.exists())

    def test_compare_ddpg_profiles_main_writes_all_plots(self) -> None:
        from tool.compare_ddpg_profiles import main as compare_main

        only = pd.DataFrame(
            {
                "wealth": [1.0, 1.1, 1.21],
                "reward": [0.0, 0.1, 0.1],
                "drawdown": [0.0, 0.0, 0.0],
                "daily_return": [np.nan, 0.1, 0.1],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        slm = pd.DataFrame(
            {
                "wealth": [1.0, 1.2, 1.32],
                "reward": [0.0, 0.2, 0.1],
                "drawdown": [0.0, 0.0, 0.02],
                "daily_return": [np.nan, 0.2, 0.1],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            only_path = tmp_path / "only.csv"
            slm_path = tmp_path / "slm.csv"
            only.to_csv(only_path, index_label="time")
            slm.to_csv(slm_path, index_label="time")

            output = tmp_path / "comparison.csv"
            diff_output = tmp_path / "difference.png"
            overlay_output = tmp_path / "overlay.png"
            only_daily_output = tmp_path / "only_daily.png"
            slm_daily_output = tmp_path / "slm_daily.png"
            distribution_output = tmp_path / "distribution.png"
            boxplot_output = tmp_path / "boxplot.png"
            args = [
                "compare_ddpg_profiles.py",
                "--only-ddpg-profile",
                str(only_path),
                "--ddpg-slm-profile",
                str(slm_path),
                "--output",
                str(output),
                "--plot-output",
                str(diff_output),
                "--daily-return-overlay-output",
                str(overlay_output),
                "--only-ddpg-daily-return-output",
                str(only_daily_output),
                "--ddpg-slm-daily-return-output",
                str(slm_daily_output),
                "--distribution-output",
                str(distribution_output),
                "--boxplot-output",
                str(boxplot_output),
            ]

            with patch.object(sys, "argv", args):
                saved_path = compare_main()

            expected_outputs = [
                saved_path,
                diff_output,
                overlay_output,
                only_daily_output,
                slm_daily_output,
                distribution_output,
                boxplot_output,
            ]
            for path in expected_outputs:
                with self.subTest(path=path):
                    self.assertTrue(Path(path).exists())
                    self.assertGreater(Path(path).stat().st_size, 0)

    def test_standardized_notebooks_have_notes_and_no_outputs(self) -> None:
        old_notebook = PROJECT_ROOT / "main" / "main_code.ipynb"
        self.assertFalse(old_notebook.exists())

        notebook_paths = [
            PROJECT_ROOT / "main" / "main_code_only_ddpg.ipynb",
            PROJECT_ROOT / "main" / "main_code_add_slm.ipynb",
        ]

        for path in notebook_paths:
            with self.subTest(path=path):
                nb = json.loads(path.read_text())
                self.assertEqual(nb["cells"][-1]["cell_type"], "markdown")
                self.assertIn("Workflow Notes", "".join(nb["cells"][-1]["source"]))
                self.assertEqual(sum(len(cell.get("outputs", [])) for cell in nb["cells"]), 0)

                for index, cell in enumerate(nb["cells"]):
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
