#!/usr/bin/env python
# coding: utf-8

"""Thin entrypoint for the pure price-based DDPG portfolio workflow."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "main" else SCRIPT_DIR

for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "src" / "FinRL"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from finance_rl_slm.workflow import run_only_ddpg_main  # noqa: E402


if __name__ == "__main__":
    run_only_ddpg_main()
