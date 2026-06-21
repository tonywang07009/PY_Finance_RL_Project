# Exploratory Model Notes

## What Is Here

- This folder contains earlier exploratory notebook code.

- The directory name is kept as `modle` to match the existing project layout.

## When To Use It

- Use this folder for learning, comparison, and historical project context.

- Do not treat it as the recommended production workflow.

- Use `main/` and `src/finance_rl_slm/` for the current DDPG and DDPG+SLM workflows.

## Important Files

- `finrl_project.ipynb`
  - Original exploratory notebook.

- `finrl_project.py`
  - Script exported from the notebook.
  - It now includes project path bootstrap and a non-Jupyter guard for `get_ipython()`.

## Common Checks

- Syntax check:

  ```bash
  python -B - <<'PY'
  from pathlib import Path
  compile(Path("modle/finrl_project.py").read_text(), "modle/finrl_project.py", "exec")
  PY
  ```

- If this script is run directly, it may download market data.
  - That is expected for exploratory code.
