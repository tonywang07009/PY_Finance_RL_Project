# Repository Guidelines

## Project Structure & Module Organization

- `main/` contains notebooks and runnable scripts for the finance RL workflow. `main/main_code_add_slm.py` is the script version that adds SLM-based sentiment scoring.
- `envs/` contains the custom Gymnasium portfolio environment, including `GymPortfolioEnv` and `PortfolioEnvConfig`.
- `modle/` contains exploratory project code and notebooks; keep the existing directory name when importing or adding files.
- `src/FinRL/` is the embedded FinRL package, with package code in `src/FinRL/finrl/`, tests in `src/FinRL/unit_tests/`, docs in `src/FinRL/docs/`, and figures in `src/FinRL/figs/`.
- `ddpg_portfolio_offline.zip` is a trained model artifact. Replace binary artifacts only when intentional.

## Build, Test, and Development Commands

- `rtk python -m pip install -r src/FinRL/requirements.txt` installs FinRL and project dependencies.
- `rtk python -m pip install -e src/FinRL` installs the local FinRL package in editable mode.
- `rtk python main/main_code_add_slm.py` runs the offline DDPG plus SLM sentiment workflow. Expect model downloads and GPU or memory requirements.
- `rtk pytest src/FinRL/unit_tests -q` runs the FinRL pytest suite.
- `rtk python -m compileall envs main modle` performs a quick syntax check for local Python modules.

## Coding Style & Naming Conventions

- Use Python style with 4-space indentation, type hints where practical, and small functions with explicit inputs.
- Follow local naming: `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants such as `TICKERS` and `START_DATE`.
- FinRL tooling includes `black`, `isort`, `mypy`, and `flake8`; `setup.cfg` sets max line length to 127.
- Do not commit generated `__pycache__/`, temporary notebooks, or regenerated model outputs without a clear reason.

## Testing Guidelines

- Add tests under `src/FinRL/unit_tests/` for FinRL package changes, using `test_*.py` names.
- For local environment changes in `envs/`, add lightweight pytest coverage when possible, especially for `reset()`, `step()`, reward calculation, action normalization, and terminal-state behavior.
- Some downloader tests rely on network data; document external-service failures.

## Commit & Pull Request Guidelines

- Existing commits use short milestone-style messages such as `the_first_version`. Prefer clearer imperative summaries, for example `add portfolio env reward test`.
- Pull requests should include purpose, changed paths, test commands, and screenshots or plots when notebook output changes.

## Security & Configuration Tips

- Keep API keys, brokerage credentials, Hugging Face tokens, and local dataset paths out of version control.
- Prefer environment variables or local ignored config files for secrets used by Alpaca, Yahoo data access, or model downloads.
