"""Compare pure DDPG and DDPG+SLM online profile CSV files."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-06-21"
DEFAULT_PROFILE_DIR = Path("addenda/result_profile_comparse")
DEFAULT_RESULT_PICTURE_DIR = Path("addenda/result_picture")
DEFAULT_COMPARISON_PICTURE_DIR = Path("addenda/result_picture/comparison")
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


def comparison_plot_filename(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END) -> str:
    return f"reward_daily_return_difference_{start_date}_{end_date}.png"


def aligned_daily_return_plot_filename(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END) -> str:
    return f"daily_return_overlay_{start_date}_{end_date}.png"


def online_daily_return_plot_filename(
    profile_name: str,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
) -> str:
    return f"online_daily_return_{profile_name}_{start_date}_{end_date}.png"


def distribution_plot_filename(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END) -> str:
    return f"daily_return_normal_distribution_{start_date}_{end_date}.png"


def box_plot_filename(start_date: str = DEFAULT_START, end_date: str = DEFAULT_END) -> str:
    return f"daily_return_boxplot_{start_date}_{end_date}.png"


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


def _daily_returns(df: pd.DataFrame) -> pd.Series:
    if "daily_return" not in df.columns:
        raise ValueError("profile dataframe is missing required column: daily_return")
    return pd.to_numeric(df["daily_return"], errors="coerce").dropna()


def compute_shared_ylim(
    series_list: list[pd.Series] | tuple[pd.Series, ...],
    center: float = 0.0,
    pad_ratio: float = 0.08,
) -> tuple[float, float]:
    values = []
    for series in series_list:
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if not finite.empty:
            values.append(finite)

    if values:
        combined = pd.concat(values)
        max_distance = float(np.max(np.abs(combined.to_numpy(dtype=float) - center)))
    else:
        max_distance = 0.0

    if max_distance == 0.0:
        max_distance = 0.01

    padding = max_distance * pad_ratio
    limit = max_distance + padding
    return center - limit, center + limit


def plot_daily_return_overlay(
    only_ddpg: pd.DataFrame,
    ddpg_slm: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    aligned = only_ddpg[["daily_return"]].join(
        ddpg_slm[["daily_return"]],
        how="inner",
        lsuffix="_only_ddpg",
        rsuffix="_ddpg_slm",
    )
    if aligned.empty:
        raise ValueError("Cannot plot daily return overlay because the profile indexes do not overlap.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    only_returns = aligned["daily_return_only_ddpg"]
    slm_returns = aligned["daily_return_ddpg_slm"]
    y_min, y_max = compute_shared_ylim((only_returns, slm_returns), center=0.0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(aligned.index, only_returns, label="Only-DDPG daily return", linewidth=1.5)
    ax.plot(aligned.index, slm_returns, label="DDPG+SLM daily return", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return")
    ax.set_title("Daily Return Overlay with Shared Y-axis")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_single_daily_return(
    df: pd.DataFrame,
    output_path: str | Path,
    label: str,
    title: str,
    y_limits: tuple[float, float],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["daily_return"], label=label, linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_aligned_individual_daily_returns(
    only_ddpg: pd.DataFrame,
    ddpg_slm: pd.DataFrame,
    only_output_path: str | Path,
    slm_output_path: str | Path,
) -> tuple[Path, Path]:
    y_limits = compute_shared_ylim(
        (only_ddpg["daily_return"], ddpg_slm["daily_return"]),
        center=0.0,
    )
    only_path = _plot_single_daily_return(
        only_ddpg,
        only_output_path,
        label="Only-DDPG daily return",
        title="Only-DDPG Daily Portfolio Return (shared y-axis)",
        y_limits=y_limits,
    )
    slm_path = _plot_single_daily_return(
        ddpg_slm,
        slm_output_path,
        label="DDPG+SLM daily return",
        title="DDPG+SLM Daily Portfolio Return (shared y-axis)",
        y_limits=y_limits,
    )
    return only_path, slm_path


def _normal_density(x_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    safe_std = max(float(std), 1e-12)
    coefficient = 1.0 / (safe_std * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((x_values - mean) / safe_std) ** 2
    return coefficient * np.exp(exponent)


def plot_profile_distribution(
    only_ddpg: pd.DataFrame,
    ddpg_slm: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    only_returns = _daily_returns(only_ddpg)
    slm_returns = _daily_returns(ddpg_slm)
    if only_returns.empty or slm_returns.empty:
        raise ValueError("Cannot plot normal distribution because one profile has no daily_return data.")

    combined = pd.concat([only_returns, slm_returns])
    shared_mean = float(combined.mean())
    only_std = float(only_returns.std(ddof=1)) if len(only_returns) > 1 else 0.0
    slm_std = float(slm_returns.std(ddof=1)) if len(slm_returns) > 1 else 0.0
    max_std = max(only_std, slm_std, 1e-6)

    x_min = min(float(combined.min()), shared_mean - 4.0 * max_std)
    x_max = max(float(combined.max()), shared_mean + 4.0 * max_std)
    if x_min == x_max:
        x_min -= 0.01
        x_max += 0.01
    x_values = np.linspace(x_min, x_max, 400)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x_values,
        _normal_density(x_values, shared_mean, only_std),
        label=f"Only-DDPG normal curve (std={only_std:.6f})",
        linewidth=2.0,
    )
    ax.plot(
        x_values,
        _normal_density(x_values, shared_mean, slm_std),
        label=f"DDPG+SLM normal curve (std={slm_std:.6f})",
        linewidth=2.0,
    )
    ax.axvline(shared_mean, color="black", linewidth=0.9, linestyle="--", label=f"Shared mean={shared_mean:.6f}")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    ax.set_title("Daily Return Normal Distribution Comparison")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_profile_boxplot(
    only_ddpg: pd.DataFrame,
    ddpg_slm: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    only_returns = _daily_returns(only_ddpg)
    slm_returns = _daily_returns(ddpg_slm)
    if only_returns.empty or slm_returns.empty:
        raise ValueError("Cannot plot boxplot because one profile has no daily_return data.")

    y_min, y_max = compute_shared_ylim((only_returns, slm_returns), center=0.0)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [only_returns.to_numpy(dtype=float), slm_returns.to_numpy(dtype=float)],
        showmeans=True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Only-DDPG", "DDPG+SLM"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Daily return")
    ax.set_title("Daily Return Boxplot Comparison")
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_profile_differences(
    only_ddpg: pd.DataFrame,
    ddpg_slm: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    aligned = only_ddpg[["reward", "daily_return"]].join(
        ddpg_slm[["reward", "daily_return"]],
        how="inner",
        lsuffix="_only_ddpg",
        rsuffix="_ddpg_slm",
    )
    if aligned.empty:
        raise ValueError("Cannot plot profile differences because the profile indexes do not overlap.")

    reward_diff = aligned["reward_ddpg_slm"] - aligned["reward_only_ddpg"]
    return_diff = aligned["daily_return_ddpg_slm"] - aligned["daily_return_only_ddpg"]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(aligned.index, reward_diff, label="DDPG+SLM reward - Only-DDPG reward")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Reward diff")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(aligned.index, return_diff, label="DDPG+SLM daily_return - Only-DDPG daily_return")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Daily return diff")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
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
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--daily-return-overlay-output", type=Path)
    parser.add_argument("--only-ddpg-daily-return-output", type=Path)
    parser.add_argument("--ddpg-slm-daily-return-output", type=Path)
    parser.add_argument("--distribution-output", type=Path)
    parser.add_argument("--boxplot-output", type=Path)
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
    plot_output_path = args.plot_output or (
        DEFAULT_COMPARISON_PICTURE_DIR / comparison_plot_filename(args.start_date, args.end_date)
    )
    daily_return_overlay_path = args.daily_return_overlay_output or (
        DEFAULT_COMPARISON_PICTURE_DIR / aligned_daily_return_plot_filename(args.start_date, args.end_date)
    )
    only_daily_return_path = args.only_ddpg_daily_return_output or (
        DEFAULT_RESULT_PICTURE_DIR
        / "only_ddpg"
        / online_daily_return_plot_filename("only_ddpg", args.start_date, args.end_date)
    )
    slm_daily_return_path = args.ddpg_slm_daily_return_output or (
        DEFAULT_RESULT_PICTURE_DIR
        / "with_slm"
        / online_daily_return_plot_filename("ddpg_slm", args.start_date, args.end_date)
    )
    distribution_path = args.distribution_output or (
        DEFAULT_COMPARISON_PICTURE_DIR / distribution_plot_filename(args.start_date, args.end_date)
    )
    boxplot_path = args.boxplot_output or (
        DEFAULT_COMPARISON_PICTURE_DIR / box_plot_filename(args.start_date, args.end_date)
    )

    only_profile = load_profile(only_path)
    slm_profile = load_profile(slm_path)
    summary = compare_profiles(only_profile, slm_profile)
    saved_path = write_comparison(summary, output_path)
    plot_path = plot_profile_differences(only_profile, slm_profile, plot_output_path)
    overlay_path = plot_daily_return_overlay(only_profile, slm_profile, daily_return_overlay_path)
    only_daily_path, slm_daily_path = plot_aligned_individual_daily_returns(
        only_profile,
        slm_profile,
        only_daily_return_path,
        slm_daily_return_path,
    )
    distribution_plot_path = plot_profile_distribution(only_profile, slm_profile, distribution_path)
    boxplot_output_path = plot_profile_boxplot(only_profile, slm_profile, boxplot_path)
    print(f"saved comparison: {saved_path}")
    print(f"saved comparison plot: {plot_path}")
    print(f"saved daily return overlay plot: {overlay_path}")
    print(f"refreshed Only-DDPG daily return plot: {only_daily_path}")
    print(f"refreshed DDPG+SLM daily return plot: {slm_daily_path}")
    print(f"saved normal distribution plot: {distribution_plot_path}")
    print(f"saved boxplot: {boxplot_output_path}")
    return saved_path


if __name__ == "__main__":
    main()
