"""Path helpers for running from the repo root, `main/`, or notebooks."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
FINRL_ROOT = PROJECT_ROOT / "src" / "FinRL"


def ensure_project_paths(project_root: Path | None = None) -> Path:
    """Add project import roots to `sys.path` and return the project root."""
    root = (project_root or PROJECT_ROOT).resolve()
    candidates = (root / "src", root, root / "src" / "FinRL")

    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root
