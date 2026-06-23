"""Generate baseline strategy profiles and optional four-pipeline reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from baseline.baseline_strategies import (  # noqa: E402
    DEFAULT_INITIAL_CAPITAL,
    build_buy_hold_profile,
    build_markov_chain_profile,
    format_vector_for_csv,
)
from finance_rl_slm.config import DEFAULT_CONFIG  # noqa: E402
from finance_rl_slm.data import download_price_df  # noqa: E402
from tool.compare_ddpg_profiles import write_four_pipeline_outputs  # noqa: E402
from version.model_explainer import DEFAULT_REPORT_PATH  # noqa: E402
from version.model_report_html import build_four_pipeline_dashboard_report  # noqa: E402


def project_output_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_baseline_profile_to_path(profile: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a baseline profile to an exact output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = profile.copy()
    for column in ("action", "shares", "predicted_up_probability"):
        if column in saved.columns:
            saved[column] = saved[column].map(format_vector_for_csv)
    saved.to_csv(path, index_label="time")
    return path


def load_baseline_price_data(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load baseline price data through the shared project data module."""

    try:
        return download_price_df(
            start_date=start_date,
            end_date=end_date,
            tickers=DEFAULT_CONFIG.tickers,
        )
    except Exception as error:
        raise SystemExit(
            "Unable to load baseline price data through "
            f"finance_rl_slm.data.download_price_df(): {error}"
        ) from None


def split_baseline_price_data(full_price_df: pd.DataFrame, start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split loaded prices into historical and online frames for baseline backtests."""

    start_timestamp = pd.Timestamp(start_date)
    historical_price_df = full_price_df.loc[full_price_df.index < start_timestamp].copy()
    online_price_df = full_price_df.loc[full_price_df.index >= start_timestamp].copy()
    if historical_price_df.empty:
        raise SystemExit(
            f"Price data has no historical rows before {start_date}; "
            "Markov Chain baseline needs pre-online data for transition counts."
        )
    if len(online_price_df) < 2:
        raise SystemExit(
            f"Price data has {len(online_price_df)} online rows from {start_date}; "
            "Buy-and-Hold and Markov Chain profiles need at least two online rows."
        )
    return historical_price_df, online_price_df


def generate_baseline_profiles(
    start_date: str,
    end_date: str,
    initial_capital: float,
    buy_hold_profile: str | Path,
    markov_profile: str | Path,
) -> dict[str, Path]:
    """Generate both baseline profiles and save them to exact paths."""

    full_price_df = load_baseline_price_data(
        start_date=DEFAULT_CONFIG.start_date,
        end_date=end_date,
    )
    historical_price_df, online_price_df = split_baseline_price_data(full_price_df, start_date)

    buy_hold_df = build_buy_hold_profile(
        online_price_df,
        DEFAULT_CONFIG.ticker_list,
        initial_capital=initial_capital,
    )
    markov_df = build_markov_chain_profile(
        historical_price_df,
        online_price_df,
        DEFAULT_CONFIG.ticker_list,
        initial_capital=initial_capital,
    )

    buy_hold_path = save_baseline_profile_to_path(buy_hold_df, buy_hold_profile)
    markov_path = save_baseline_profile_to_path(markov_df, markov_profile)
    print(f"saved Buy-and-Hold profile: {buy_hold_path}")
    print(f"saved Markov Chain profile: {markov_path}")
    return {
        "buy_hold_profile": buy_hold_path,
        "markov_chain_profile": markov_path,
    }


def ensure_baseline_profiles(
    start_date: str,
    end_date: str,
    initial_capital: float,
    buy_hold_profile: str | Path,
    markov_profile: str | Path,
) -> dict[str, Path]:
    """Generate baseline profiles only when one or both files are missing."""

    buy_hold_path = Path(buy_hold_profile)
    markov_path = Path(markov_profile)
    if buy_hold_path.is_file() and markov_path.is_file():
        return {
            "buy_hold_profile": buy_hold_path,
            "markov_chain_profile": markov_path,
        }

    return generate_baseline_profiles(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        buy_hold_profile=buy_hold_path,
        markov_profile=markov_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Buy-and-Hold and Markov Chain baselines.")
    parser.add_argument("--start-date", default=DEFAULT_CONFIG.online_start)
    parser.add_argument("--end-date", default=DEFAULT_CONFIG.online_end)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only save baseline profiles; do not refresh comparison plots or HTML.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_parser().parse_args(argv)
    profile_dir = args.profile_dir or project_output_path(DEFAULT_CONFIG.result_baseline_dir)
    plot_dir = args.plot_dir or project_output_path(Path(DEFAULT_CONFIG.result_picture_dir) / "comparison")

    buy_hold_path = profile_dir / f"buy_hold_online_profile_{args.start_date}_{args.end_date}.csv"
    markov_path = profile_dir / f"markov_chain_online_profile_{args.start_date}_{args.end_date}.csv"
    outputs = generate_baseline_profiles(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        buy_hold_profile=buy_hold_path,
        markov_profile=markov_path,
    )
    buy_hold_path = outputs["buy_hold_profile"]
    markov_path = outputs["markov_chain_profile"]

    if not args.skip_report:
        model_profile_dir = project_output_path(DEFAULT_CONFIG.result_profile_dir)
        outputs.update(
            write_four_pipeline_outputs(
                start_date=args.start_date,
                end_date=args.end_date,
                profile_dir=model_profile_dir,
                baseline_dir=profile_dir,
                comparison_dir=plot_dir,
                buy_hold_profile=buy_hold_path,
                markov_profile=markov_path,
            )
        )
        report_path = build_four_pipeline_dashboard_report(
            buy_hold_profile=buy_hold_path,
            markov_profile=markov_path,
            output_path=args.report_output,
            initial_capital=args.initial_capital,
        )
        outputs["html_report"] = report_path
        print(f"refreshed four-pipeline HTML report: {report_path}")

    return outputs


if __name__ == "__main__":
    main()
