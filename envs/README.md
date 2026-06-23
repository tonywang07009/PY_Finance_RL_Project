# Environment API

## What Is Here

- This folder contains the custom portfolio environment used by the DDPG agent.

- Main file:
  - `gym_portfolio_env.py`

- Main idea:
  - `SimplePortfolioEnv` handles core wealth movement.
  - `GymPortfolioEnv` exposes the Gymnasium API for Stable-Baselines3.

## 1. Environment Flow

```mermaid
  %% {init: {"flowchart": {"defaultRenderer": "elk"}} }%%
  %% The specy elk to daraw enegine
  flowchart TB
      PDF[__price_df__<br/>1. date by ticker table<br/>2. clean close prices]
      CFG[__PortfolioEnvConfig__<br/>1. tickers<br/>2. observation window<br/>3. reward settings]
      CORE[__SimplePortfolioEnv__<br/>1. wealth update<br/>2. portfolio return]
      SENT[__sentiment_series__<br/>1. daily score<br/>2. clipped to -1, 1]
      GYM[__GymPortfolioEnv__<br/>1. reset and step<br/>2. 61D or 62D observation]
      DDPG[__DDPG Agent__<br/>1. continuous action<br/>2. portfolio weight decision]

      PDF --> CORE
      CFG --> CORE
      CORE --> GYM
      SENT --> GYM
      GYM --> DDPG
      DDPG --> GYM
```

## 2. API Overview

| Function | Role |
|---|---|
| `BaseEnv` | Abstract base class for `reset()` and `step()` environment behavior. |
| `PortfolioEnvConfig` | Store environment settings such as tickers, initial wealth, and reward window. |
| `validate_price_df()` | Check required ticker columns, sort by date, and reject empty price data. |
| `SimplePortfolioEnv` | Hold the core price table, portfolio wealth, and return calculation. |
| `SimplePortfolioEnv.reset()` | Reset core wealth state to the first tradable step. |
| `SimplePortfolioEnv.step()` | Apply portfolio weights and update wealth using next-day returns. |
| `ema()` | Compute an exponential moving average used by technical state features. |
| `GymPortfolioEnv` | Gymnasium wrapper used by DDPG training and online evaluation. |
| `GymPortfolioEnv.__init__()` | Build action space, observation space, and optional SLM feature shape. |
| `GymPortfolioEnv.set_sentiment_series()` | Attach daily SLM sentiment scores to the environment. |
| `GymPortfolioEnv.action_to_weight()` | Clip and normalize continuous DDPG actions into portfolio weights. |
| `GymPortfolioEnv.reset()` | Reset the Gym environment and return the first observation. |
| `GymPortfolioEnv.step()` | Execute one RL step and return observation, reward, done flags, and info. |

## Common Checks

- Run environment tests:

  ```bash
  rtk python -B -m unittest tests.test_online_env_api -v
  ```

- Keep these behaviors stable:
  - Only-DDPG observation shape is 61D.
  - DDPG+SLM observation shape is 62D.
  - Sentiment value is clipped into `[-1, 1]`.
  - Actions are always clipped and normalized.
  - Empty train or validation price data raises a clear error.
