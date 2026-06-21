# Source Modules

## What Is Here

- `finance_rl_slm/`
  - Project-owned workflow package.
  - Contains configuration, data loading, sentiment, training, online evaluation, and orchestration helpers.

- `tool/`
  - Small utility scripts.
  - Current main tool: DDPG vs DDPG+SLM profile comparison.

- `FinRL/`
  - Embedded FinRL dependency.
  - Treat this mostly as upstream library code.

- `SLM_emo_analizy/`
  - Reserved or legacy folder.
  - It is not part of the main runnable workflow right now.

## When To Use It

- Use `finance_rl_slm/` when changing the project workflow.

- Use `tool/` when adding analysis utilities.

- Avoid changing `FinRL/` unless the bug is inside the embedded package.

## Important Files

- `finance_rl_slm/config.py`
  - Central experiment defaults.

- `finance_rl_slm/workflow.py`
  - High-level pure DDPG and DDPG+SLM flow.

- `finance_rl_slm/evaluation.py`
  - Online evaluation, plot saving, profile CSV saving, and model loading.

- `tool/compare_ddpg_profiles.py`
  - Compares saved online profiles.

## Common Checks

- Syntax/import check:

  ```bash
  python -B -m unittest discover -s tests -p 'test_*.py' -v
  ```

- Utility script:

  ```bash
  python src/tool/compare_ddpg_profiles.py
  ```
