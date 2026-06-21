from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class MainCallSafetyDocRouterTests(unittest.TestCase):
    def test_split_price_data_reports_empty_train_and_valid_ranges(self) -> None:
        from dataclasses import replace

        from finance_rl_slm.config import DEFAULT_CONFIG
        from finance_rl_slm.workflow import split_price_data

        price_df = pd.DataFrame(
            {"IBM": [100.0, 101.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        empty_train = replace(
            DEFAULT_CONFIG,
            tickers=("IBM",),
            train_start="2005-01-01",
            train_end="2005-12-31",
            valid_start="2026-01-02",
            valid_end="2026-01-05",
        )
        with self.assertRaisesRegex(ValueError, "Requested train range: 2005-01-01 -> 2005-12-31"):
            split_price_data(price_df, empty_train)

        empty_valid = replace(
            DEFAULT_CONFIG,
            tickers=("IBM",),
            train_start="2026-01-02",
            train_end="2026-01-05",
            valid_start="2016-01-01",
            valid_end="2020-12-31",
        )
        with self.assertRaisesRegex(ValueError, "Requested valid range: 2016-01-01 -> 2020-12-31"):
            split_price_data(price_df, empty_valid)

    def test_missing_model_path_error_is_actionable(self) -> None:
        from finance_rl_slm.evaluation import resolve_model_path

        missing = PROJECT_ROOT / "does_not_exist_model_for_test"
        with self.assertRaisesRegex(FileNotFoundError, "Run the optional offline training"):
            resolve_model_path(missing)

    def test_online_workflows_do_not_call_training(self) -> None:
        from finance_rl_slm import workflow

        fake_price_df = pd.DataFrame(
            {"IBM": [100.0, 101.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        fake_logs = pd.DataFrame({"wealth": [1.0]})

        def fail_training(*args, **kwargs):
            raise AssertionError("online workflow must not call train_offline_model")

        with (
            patch.object(workflow, "load_online_price_data", return_value=fake_price_df),
            patch.object(workflow, "run_online_evaluation", return_value=fake_logs),
            patch.object(workflow, "train_offline_model", side_effect=fail_training),
        ):
            self.assertIs(workflow.run_only_ddpg_online(), fake_logs)

        with (
            patch.object(workflow, "build_sentiment_inputs", return_value=(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))),
            patch.object(workflow, "load_online_price_data", return_value=fake_price_df),
            patch.object(workflow, "build_daily_sentiment", return_value=pd.Series([0.0], index=fake_price_df.index[:1])),
            patch.object(workflow, "run_online_evaluation", return_value=fake_logs),
            patch.object(workflow, "train_offline_model", side_effect=fail_training),
        ):
            self.assertIs(workflow.run_slm_online(), fake_logs)

    def test_main_scripts_compile_and_use_online_defaults(self) -> None:
        scripts = {
            "main/main_code_only_ddpg.py": "run_only_ddpg_online",
            "main/main_code_add_slm.py": "run_slm_online",
        }

        for rel_path, expected_call in scripts.items():
            with self.subTest(script=rel_path):
                path = PROJECT_ROOT / rel_path
                source = path.read_text()
                compile(source, str(path), "exec")
                self.assertIn(expected_call, source)

    def test_main_notebooks_parse_and_guard_optional_training(self) -> None:
        notebooks = [
            PROJECT_ROOT / "main" / "main_code_only_ddpg.ipynb",
            PROJECT_ROOT / "main" / "main_code_add_slm.ipynb",
        ]

        for path in notebooks:
            with self.subTest(notebook=path):
                nb = json.loads(path.read_text())
                source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
                self.assertIn("RUN_OPTIONAL_TRAINING = False", source)
                self.assertIn("if RUN_OPTIONAL_TRAINING:", source)

                for index, cell in enumerate(nb["cells"]):
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")

    def test_folder_docs_and_root_router_links_exist(self) -> None:
        docs = [
            PROJECT_ROOT / "src" / "README.md",
            PROJECT_ROOT / "envs" / "README.md",
            PROJECT_ROOT / "addenda" / "README.md",
            PROJECT_ROOT / "modle" / "README.md",
            PROJECT_ROOT / "tests" / "README.md",
        ]
        root_readme = (PROJECT_ROOT / "README.md").read_text()

        for path in docs:
            with self.subTest(path=path):
                self.assertTrue(path.exists())
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                self.assertIn(f"]({rel})", root_readme)

    def test_modle_export_has_bootstrap_and_get_ipython_guard(self) -> None:
        path = PROJECT_ROOT / "modle" / "finrl_project.py"
        source = path.read_text()
        compile(source, str(path), "exec")
        self.assertIn("PROJECT_ROOT", source)
        self.assertIn('if "get_ipython" in globals():', source)


if __name__ == "__main__":
    unittest.main()
