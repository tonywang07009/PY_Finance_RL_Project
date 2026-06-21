#!/usr/bin/env python
# coding: utf-8

"""Pure DDPG portfolio workflow script.

This script mirrors the notebook main_code_only_ddpg.ipynb:
1. Loads price data
2. Trains or loads a DDPG model
3. Runs online evaluation without sentiment
"""

from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path
from stable_baselines3 import DDPG

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "main" else SCRIPT_DIR

for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from finance_rl_slm.config import DEFAULT_CONFIG
from finance_rl_slm.evaluation import run_online_evaluation
from finance_rl_slm.training import train_offline_model
from finance_rl_slm.workflow import (
    load_online_price_data,
    load_price_data,
    plot_normalized_prices,
    print_runtime_context,
    result_picture_path,
    result_profile_path,
    split_price_data,
)


def main() -> None:
    """Run the pure DDPG portfolio workflow."""
    # Step 1: Configure the experiment
    config = replace(
        DEFAULT_CONFIG,
        online_start="2026-01-01",
        online_end="2026-06-21",
    )
    print_runtime_context(config)

    # Step 2: Train or load DDPG model
    print("\nChoose mode:\n")
    print("1) Train new DDPG model\n")
    print("2) Load existing DDPG model\n")

    choice: int = int(input("Enter 1 or 2: ").strip())

    if choice == 1:
        # Download price data and train
        price_df = load_price_data(config)

        # Stock context check
        print("price_df range:", price_df.index.min(), "->", price_df.index.max())
        print("price_df head:\n", price_df.head())
        print("price_df tail:\n", price_df.tail())

        plot_normalized_prices(price_df, config)
        splits = split_price_data(price_df, config)
        model = train_offline_model(splits.train, splits.valid, config)

    elif choice == 2:
        # Load existing model
        model_path = PROJECT_ROOT / "ddpg_portfolio_offline"
        model = DDPG.load(str(model_path))
        print(f"Loaded existing model from {model_path}.zip\n")

    else:
        raise ValueError("Invalid choice. Please enter 1 or 2.\n")

    # Step 3: Run online evaluation without SLM
    price_df_online = load_online_price_data(config.online_end, config)

    only_ddpg_logs = run_online_evaluation(
        price_df_online,
        sentiment_series=None,
        online_end_str=config.online_end,
        config=config,
        save_plots=True,
        plot_dir=result_picture_path(config),
        save_profile=True,
        profile_dir=result_profile_path(config),
        profile_name="only_ddpg",
    )

    print("\nOnline evaluation completed.")
    print(only_ddpg_logs.head())


if __name__ == "__main__":
    main()
