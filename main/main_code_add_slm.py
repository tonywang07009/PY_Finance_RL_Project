#!/usr/bin/env python
# coding: utf-8

"""Thin entrypoint for the SLM-enabled portfolio DDPG workflow."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "main" else SCRIPT_DIR

for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from finance_rl_slm.config import DEFAULT_CONFIG  # noqa: E402
from finance_rl_slm.workflow import run_slm_online  # noqa: E402


if __name__ == "__main__":
    run_slm_online(replace(DEFAULT_CONFIG, news_max_items=3))
