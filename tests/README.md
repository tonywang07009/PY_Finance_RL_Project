# Tests

## What Is Here

- This folder contains lightweight unittest coverage for the local project workflow.

- It does not own the embedded FinRL upstream test suite.

## When To Use It

- Use this folder when changing:
  - portfolio environment behavior,
  - online evaluation helpers,
  - result output contracts,
  - notebook/script call safety,
  - documentation routing.

## Important Files

- `test_finance_rl_slm.py`
  - Sentiment helpers and package import safety.

- `test_online_env_api.py`
  - Environment API behavior.

- `test_results_and_docs.py`
  - Result artifacts, notebook structure, and comparison tool behavior.

## Common Checks

- Run all local tests:

  ```bash
  python -B -m unittest discover -s tests -p 'test_*.py' -v
  ```

- Keep tests lightweight.
  - Do not require full DDPG training.
  - Avoid network-dependent assertions unless explicitly testing an online run.
