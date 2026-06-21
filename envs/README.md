# Environment API

## What Is Here

- This folder contains the custom portfolio environment used by the DDPG agent.

- Main file:
  - `gym_portfolio_env.py`

## When To Use It

- Use this folder when changing:
  - portfolio wealth update logic,
  - action-to-weight normalization,
  - observation vector design,
  - reward and drawdown behavior,
  - optional SLM sentiment injection.

## Important Files

- `PortfolioEnvConfig`
  - Holds environment settings such as tickers and initial wealth.

- `SimplePortfolioEnv`
  - Handles price data, returns, and core wealth state.

- `GymPortfolioEnv`
  - Gymnasium wrapper used by Stable-Baselines3 DDPG.

## Common Checks

- Run local environment tests:

  ```bash
  python -B -m unittest tests.test_online_env_api -v
  ```

- Important behavior to protect:
  - observation shape stays stable,
  - actions are clipped and normalized,
  - sentiment is clipped into `[-1, 1]`,
  - terminal step behavior remains predictable.
