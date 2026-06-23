"""Generate and serve the model explanation dashboard on localhost."""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.run_baselines import DEFAULT_CONFIG, ensure_baseline_profiles  # noqa: E402
from tool.compare_ddpg_profiles import write_four_pipeline_outputs  # noqa: E402
from version.model_explainer import (  # noqa: E402
    DEFAULT_BUY_HOLD_PROFILE,
    DEFAULT_CURRENCY,
    DEFAULT_DDPG_SLM_PROFILE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MARKOV_CHAIN_PROFILE,
    DEFAULT_ONLY_DDPG_PROFILE,
    DEFAULT_REPORT_PATH,
)
from version.model_report_html import build_four_pipeline_dashboard_report  # noqa: E402


def validate_profile_paths(profile_specs: list[tuple[str, Path]]) -> None:
    """Fail early with a practical command when required profile CSVs are missing."""

    missing = [(name, Path(path)) for name, path in profile_specs if not Path(path).is_file()]
    if not missing:
        return

    lines = ["Missing required profile CSV(s):"]
    lines.extend(f"- {name}: {path}" for name, path in missing)
    lines.append("")

    lines.extend(
        [
            "Generate the missing model profile CSVs first, then rerun:",
            "  rtk python version/run_model_report.py --no-serve",
        ]
    )

    raise SystemExit("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and serve the DDPG model explanation dashboard.")
    parser.add_argument("--only-ddpg-profile", type=Path, default=DEFAULT_ONLY_DDPG_PROFILE)
    parser.add_argument("--ddpg-slm-profile", type=Path, default=DEFAULT_DDPG_SLM_PROFILE)
    parser.add_argument("--buy-hold-profile", type=Path, default=DEFAULT_BUY_HOLD_PROFILE)
    parser.add_argument("--markov-profile", type=Path, default=DEFAULT_MARKOV_CHAIN_PROFILE)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--currency", default=DEFAULT_CURRENCY)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-serve", action="store_true", help="Generate HTML and exit without starting server.")
    return parser


def serve_report(output_path: Path, host: str = "127.0.0.1", port: int = 8050) -> None:
    """Serve the report directory with Python's standard HTTP server."""

    report_path = Path(output_path).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(report_path.parent))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/{report_path.name}"
    print(f"Serving model report: {url}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile_specs = [
        ("Only-DDPG", args.only_ddpg_profile),
        ("DDPG+SLM", args.ddpg_slm_profile),
    ]
    validate_profile_paths(profile_specs)
    ensure_baseline_profiles(
        start_date=DEFAULT_CONFIG.online_start,
        end_date=DEFAULT_CONFIG.online_end,
        initial_capital=args.initial_capital,
        buy_hold_profile=args.buy_hold_profile,
        markov_profile=args.markov_profile,
    )
    profile_specs.extend(
        [
            ("Buy-and-Hold", args.buy_hold_profile),
            ("Markov Chain", args.markov_profile),
        ]
    )
    validate_profile_paths(profile_specs)
    write_four_pipeline_outputs(
        start_date=DEFAULT_CONFIG.online_start,
        end_date=DEFAULT_CONFIG.online_end,
        profile_dir=PROJECT_ROOT / DEFAULT_CONFIG.result_profile_dir,
        baseline_dir=PROJECT_ROOT / DEFAULT_CONFIG.result_baseline_dir,
        comparison_dir=PROJECT_ROOT / DEFAULT_CONFIG.result_picture_dir / "comparison",
        only_ddpg_profile=args.only_ddpg_profile,
        ddpg_slm_profile=args.ddpg_slm_profile,
        buy_hold_profile=args.buy_hold_profile,
        markov_profile=args.markov_profile,
    )

    output_path = build_four_pipeline_dashboard_report(
        only_ddpg_profile=args.only_ddpg_profile,
        ddpg_slm_profile=args.ddpg_slm_profile,
        buy_hold_profile=args.buy_hold_profile,
        markov_profile=args.markov_profile,
        output_path=args.output,
        initial_capital=args.initial_capital,
        currency=args.currency,
    )
    print(f"Generated model report: {output_path}")
    if not args.no_serve:
        serve_report(output_path, host=args.host, port=args.port)
    return output_path


if __name__ == "__main__":
    main()
