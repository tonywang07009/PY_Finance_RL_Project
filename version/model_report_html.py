"""Generate a terminal-style HTML dashboard for model explanations."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .model_explainer import (
    DEFAULT_CURRENCY,
    DEFAULT_DDPG_SLM_PROFILE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_ONLY_DDPG_PROFILE,
    DEFAULT_REPORT_PATH,
    compare_model_profiles,
)


def format_money(value: float, currency: str = DEFAULT_CURRENCY) -> str:
    return f"{value:,.2f} {currency}"


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def _metric_cell(value: float, formatter=format_percent) -> str:
    css = "pos" if value >= 0.0 else "neg"
    return f'<td class="{css}">{escape(formatter(value))}</td>'


def _summary_cards(models: list[dict[str, Any]], currency: str) -> str:
    cards = []
    for model in models:
        profit_css = "pos" if model["profit_loss"] >= 0.0 else "neg"
        cards.append(
            f"""
            <section class="card">
              <div class="kicker">{escape(model["model_name"])}</div>
              <div class="big {profit_css}">{escape(format_money(model["profit_loss"], currency))}</div>
              <div class="muted">Profit / Loss</div>
              <table>
                <tr><th>Final Value</th><td>{escape(format_money(model["final_investment_value"], currency))}</td></tr>
                <tr><th>Cumulative Return</th><td>{escape(format_percent(model["cumulative_return"]))}</td></tr>
                <tr><th>Max Drawdown</th><td class="neg">{escape(format_percent(model["max_drawdown"]))}</td></tr>
              </table>
            </section>
            """
        )
    return "\n".join(cards)


def _model_table(models: list[dict[str, Any]], currency: str) -> str:
    rows = []
    for model in models:
        rows.append(
            f"""
            <tr>
              <td>{escape(model["model_name"])}</td>
              <td>{escape(format_money(model["initial_capital"], currency))}</td>
              <td>{escape(format_money(model["final_investment_value"], currency))}</td>
              {_metric_cell(model["profit_loss"], lambda value: format_money(value, currency))}
              {_metric_cell(model["cumulative_return"])}
              {_metric_cell(model["mean_daily_return"])}
              <td class="neg">{escape(format_percent(model["max_drawdown"]))}</td>
              <td>{escape(format_percent(model["average_turnover"]))}</td>
              <td>{escape(model["most_allocated_ticker"])}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def _weight_table(models: list[dict[str, Any]], tickers: list[str]) -> str:
    rows = []
    for ticker in tickers:
        cells = [f"<td>{escape(ticker)}</td>"]
        for model in models:
            weight = float(model["average_weights"].get(ticker, 0.0))
            cells.append(f"<td>{escape(format_percent(weight))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(rows)


def _strategy_notes(strategy_notes: dict[str, list[str]]) -> str:
    blocks = []
    for model_name, notes in strategy_notes.items():
        items = "".join(f"<li>{escape(note)}</li>" for note in notes)
        blocks.append(
            f"""
            <section class="panel">
              <h3>{escape(model_name)} Decision Notes</h3>
              <ul>{items}</ul>
            </section>
            """
        )
    return "\n".join(blocks)


def generate_dashboard_html(report: dict[str, Any]) -> str:
    """Return the Bloomberg-terminal style HTML string."""

    currency = str(report["currency"])
    models = list(report["models"])
    tickers = list(report["tickers"])
    difference = dict(report["difference_ddpg_slm_minus_only_ddpg"])
    diff_css = "pos" if difference["profit_loss"] >= 0.0 else "neg"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DDPG Model Explanation Dashboard</title>
  <style>
    :root {{
      --bg: #050608;
      --panel: #101418;
      --panel-2: #151b21;
      --line: #2e3a42;
      --text: #d7e0e6;
      --muted: #7f8e99;
      --accent: #f5c542;
      --pos: #36d17c;
      --neg: #ff5c5c;
      --blue: #58a6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      background: #080b0f;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 700; }}
    h1 {{ color: var(--accent); font-size: 26px; }}
    h2 {{ color: var(--blue); font-size: 18px; margin-bottom: 14px; }}
    h3 {{ color: var(--accent); font-size: 15px; margin-bottom: 10px; }}
    main {{ padding: 20px 24px 32px; }}
    .terminal-line {{
      color: var(--muted);
      margin-top: 6px;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 14px;
    }}
    .card {{ min-height: 210px; }}
    .kicker {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .big {{
      font-family: Consolas, Monaco, monospace;
      font-size: 30px;
      line-height: 1.2;
      margin-bottom: 4px;
    }}
    .muted {{ color: var(--muted); }}
    .pos {{ color: var(--pos); }}
    .neg {{ color: var(--neg); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: Consolas, Monaco, monospace;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .section {{ margin-top: 20px; }}
    .diff {{
      background: var(--panel-2);
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      margin-bottom: 18px;
      font-family: Consolas, Monaco, monospace;
    }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin-bottom: 8px; line-height: 1.45; }}
  </style>
</head>
<body>
  <header>
    <h1>DDPG MODEL EXPLANATION DASHBOARD</h1>
    <div class="terminal-line">DATA: result_profile_comparse | CAPITAL: {escape(format_money(report["initial_capital"], currency))}</div>
  </header>
  <main>
    <div class="grid">
      {_summary_cards(models, currency)}
    </div>

    <div class="diff">
      DDPG+SLM minus Only-DDPG profit/loss:
      <span class="{diff_css}">{escape(format_money(difference["profit_loss"], currency))}</span>
    </div>

    <section class="section panel">
      <h2>Capital and Profit / Loss</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Initial Capital</th>
            <th>Final Value</th>
            <th>Profit / Loss</th>
            <th>Cum Return</th>
            <th>Mean Daily Ret</th>
            <th>Max Drawdown</th>
            <th>Avg Turnover</th>
            <th>Top Ticker</th>
          </tr>
        </thead>
        <tbody>{_model_table(models, currency)}</tbody>
      </table>
    </section>

    <section class="section panel">
      <h2>Average Portfolio Weights</h2>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Only-DDPG</th>
            <th>DDPG+SLM</th>
          </tr>
        </thead>
        <tbody>{_weight_table(models, tickers)}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Investment Strategy Analysis</h2>
      <div class="grid">
        {_strategy_notes(report["strategy_notes"])}
      </div>
    </section>
  </main>
</body>
</html>
"""


def write_dashboard_html(report: dict[str, Any], output_path: str | Path = DEFAULT_REPORT_PATH) -> Path:
    """Write a dashboard HTML file and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_dashboard_html(report), encoding="utf-8")
    return path


def build_dashboard_report(
    only_ddpg_profile: str | Path = DEFAULT_ONLY_DDPG_PROFILE,
    ddpg_slm_profile: str | Path = DEFAULT_DDPG_SLM_PROFILE,
    output_path: str | Path = DEFAULT_REPORT_PATH,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    currency: str = DEFAULT_CURRENCY,
) -> Path:
    """Build explanation data from profiles and write the HTML report."""

    report = compare_model_profiles(
        only_ddpg_profile=only_ddpg_profile,
        ddpg_slm_profile=ddpg_slm_profile,
        initial_capital=initial_capital,
        currency=currency,
    )
    return write_dashboard_html(report, output_path)
