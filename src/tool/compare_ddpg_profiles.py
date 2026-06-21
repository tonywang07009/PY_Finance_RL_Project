"""Compare pure DDPG and DDPG+SLM online profile CSV files."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-06-21"
DEFAULT_PROFILE_DIR = Path("addenda/result_profile_comparse")
REQUIRED_COLUMNS = ("wealth", "reward", "drawdown", "daily_return")


@dataclass(frozen=True)
class ProfileMetrics:
    pipeline: str
    mean_daily_return: float
    std_daily_return: float
    final_wealth: float
    cumulative_return: float
    max_drawdown: float
    sharpe_like_daily_return: float


def profile_filename(profile_name: str, start_date: str = DEFAULT_START, end_date: str = DEFAULT_END) -> str:
    return f"{profile_name}_online_profile_{start_date}_{end_date}.csv"


def comparison_filename(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END) -> str:
    return f"ddpg_vs_slm_comparison_{start_date}_{end_date}.csv"


def default_profile_path(
    profile_name: str,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> Path:
    return profile_dir / profile_filename(profile_name, start_date, end_date)


def load_profile(path: str | Path) -> pd.DataFrame:
    profile_path = Path(path)
    df = pd.read_csv(profile_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{profile_path} is missing required columns: {sorted(missing)}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def compute_profile_metrics(df: pd.DataFrame, pipeline: str) -> ProfileMetrics:
    returns = df["daily_return"].dropna()
    wealth = df["wealth"].dropna()
    drawdown = df["drawdown"].dropna()

    mean_daily_return = float(returns.mean()) if not returns.empty else 0.0
    std_daily_return = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    final_wealth = float(wealth.iloc[-1]) if not wealth.empty else np.nan

    if len(wealth) > 1 and float(wealth.iloc[0]) != 0.0:
        cumulative_return = float(final_wealth / float(wealth.iloc[0]) - 1.0)
    else:
        cumulative_return = 0.0

    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    sharpe_like = mean_daily_return / std_daily_return if std_daily_return > 0.0 else 0.0

    return ProfileMetrics(
        pipeline=pipeline,
        mean_daily_return=mean_daily_return,
        std_daily_return=std_daily_return,
        final_wealth=final_wealth,
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        sharpe_like_daily_return=float(sharpe_like),
    )


def compare_profiles(only_ddpg: pd.DataFrame, ddpg_slm: pd.DataFrame) -> pd.DataFrame:
    only_metrics = compute_profile_metrics(only_ddpg, "only_ddpg")
    slm_metrics = compute_profile_metrics(ddpg_slm, "ddpg_slm")

    rows = [asdict(only_metrics), asdict(slm_metrics)]
    diff = {"pipeline": "difference_ddpg_slm_minus_only_ddpg"}

    for key, value in rows[1].items():
        if key == "pipeline":
            continue
        diff[key] = float(value) - float(rows[0][key])

    rows.append(diff)
    return pd.DataFrame(rows)


def write_comparison(summary: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare online result profiles for pure DDPG and DDPG+SLM.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--only-ddpg-profile", type=Path)
    parser.add_argument("--ddpg-slm-profile", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    only_path = args.only_ddpg_profile or default_profile_path(
        "only_ddpg",
        args.start_date,
        args.end_date,
        args.profile_dir,
    )
    slm_path = args.ddpg_slm_profile or default_profile_path(
        "ddpg_slm",
        args.start_date,
        args.end_date,
        args.profile_dir,
    )
    output_path = args.output or args.profile_dir / comparison_filename(args.start_date, args.end_date)

    only_profile = load_profile(only_path)
    slm_profile = load_profile(slm_path)
    summary = compare_profiles(only_profile, slm_profile)
    saved_path = write_comparison(summary, output_path)
    print(f"saved comparison: {saved_path}")
    return saved_path


if __name__ == "__main__":
    main()
