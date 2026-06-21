# The "Friendly" Patrick Bateman Model

<p align="center">
  <img src="addenda/patrick_Bateman.jpg" alt="Project portrait" width="549">
</p>

## Introduction

This project simulates a long-term portfolio management system for US stocks using
Deep Deterministic Policy Gradient (DDPG). The main focus is a **pure price-based DDPG agent**,
which learns a trading policy from historical prices and technical features.

On top of that, there is an **experimental extension** that injects a small language model (IBM Granite)
as a weekly news-sentiment feature. The goal is to explore whether coarse, low-frequency sentiment
signals can improve the behaviour of a long-horizon RL agent, but this part should be treated as
an experimental add-on, not the default recommended setup.

## 1. Quick Start

### 1.1 Recommended: Pure DDPG Pipeline

This is the **recommended default** for most users and experiments.

1. **Set up environment**

   - Create and activate a Python 3.11 environment.
   - Install FinRL and project dependencies:

     ```bash
     rtk python -m pip install -r src/FinRL/requirements.txt
     rtk python -m pip install -e src/FinRL
     ```

2. **Open the pure DDPG notebook**

   - Notebook:
     - `main/main_code_only_ddpg.ipynb`
   - Script:
     - `main/main_code_only_ddpg.py`

3. **Check the core configuration**

   - Go to `src/finance_rl_slm/config.py`.
   - Confirm:
     - `tickers`
     - `start_date`, `end_date`
     - `train_start`, `train_end`
     - `valid_start`, `valid_end`
     - `online_start = "2026-01-01"`
     - `online_end = "2026-06-21"`

4. **Run offline training and validation**

   ```bash
   rtk python main/main_code_only_ddpg.py
   ```

   - Stage 1: download historical prices with `YahooDownloader`.
   - Stage 2: build `GymPortfolioEnv` with `use_slm = False`.
   - Stage 3: train a DDPG agent and save it to `ddpg_portfolio_offline`.
   - Stage 4: run online evaluation without sentiment.

5. **Inspect pure DDPG results**

   - Figures:
     - `addenda/result_picture/online_reward_only_ddpg_2026-01-01_2026-06-21.png`
     - `addenda/result_picture/online_wealth_only_ddpg_2026-01-01_2026-06-21.png`
     - `addenda/result_picture/online_daily_return_only_ddpg_2026-01-01_2026-06-21.png`
   - Profile:
     - `addenda/result_profile_comparse/only_ddpg_online_profile_2026-01-01_2026-06-21.csv`

### 1.2 Experimental: DDPG + SLM Sentiment Extension

This pipeline reuses the same DDPG structure, but adds a weekly sentiment feature during
online evaluation.

1. **Open the SLM notebook**

   - Notebook:
     - `main/main_code_add_slm.ipynb`
   - Script:
     - `main/main_code_add_slm.py`

2. **Prepare RSS and Granite sentiment**

   - The workflow will:
     - Pull Yahoo Finance RSS for each ticker.
     - Call IBM Granite (`ibm-granite/granite-4.1-8b`) to score news sentiment.
     - Aggregate news scores into weekly market sentiment.
     - Map weekly sentiment to the daily online trading index.
   - The current script/notebook uses `news_max_items = 3` per ticker for a practical local run.

3. **Run online evaluation with SLM**

   ```bash
   rtk python main/main_code_add_slm.py
   ```

   - `GymPortfolioEnv` reads `sentiment_series[current_date]`.
   - The sentiment score is clipped to `[-1, 1]`.
   - The score is appended as the last observation feature.

4. **Inspect SLM results**

   - Figures:
     - `addenda/result_picture/online_reward_ddpg_slm_2026-01-01_2026-06-21.png`
     - `addenda/result_picture/online_wealth_ddpg_slm_2026-01-01_2026-06-21.png`
     - `addenda/result_picture/online_daily_return_ddpg_slm_2026-01-01_2026-06-21.png`
   - Profile:
     - `addenda/result_profile_comparse/ddpg_slm_online_profile_2026-01-01_2026-06-21.csv`

### 1.3 Compare Pure DDPG and DDPG + SLM

After both profile CSV files exist, run:

```bash
rtk python src/tool/compare_ddpg_profiles.py
```

The comparison CSV will be written to:

```text
addenda/result_profile_comparse/ddpg_vs_slm_comparison_2026-01-01_2026-06-21.csv
```

The script compares:

- mean daily return,
- standard deviation of daily return,
- final wealth,
- cumulative return,
- max drawdown,
- Sharpe-like daily return ratio,
- difference row: `ddpg_slm - only_ddpg`.

## 2. Source Router (Code Map)

Think of this section as an affinity diagram for the codebase.
Start from your question, then follow the path to the right module.

### 2.1 Environments and Agents

- **Question**: "How is the portfolio environment designed?"

  - Go to: `envs/gym_portfolio_env.py`
  - Main classes:
    - `PortfolioEnvConfig`
    - `SimplePortfolioEnv`
    - `GymPortfolioEnv`
  - Key ideas:
    - portfolio wealth update,
    - action normalization into portfolio weights,
    - reward = rolling Sharpe-like score minus drawdown penalty,
    - observation = return window + MACD-like wealth trend + Bollinger deviation + wealth + optional SLM score.

- **Question**: "Where is the DDPG model created?"

  - Go to: `src/finance_rl_slm/training.py`
  - Main function:
    - `train_offline_model`
  - Key settings:
    - `learning_rate = 3e-4`
    - `batch_size = 256`
    - `gamma = 0.99`
    - `tau = 0.005`
    - `NormalActionNoise`

### 2.2 Data and Sentiment

- **Question**: "Where does the price data come from?"

  - Go to: `src/finance_rl_slm/data.py`
  - Main function:
    - `download_price_df`
  - Data source:
    - `finrl.meta.preprocessor.yahoodownloader.YahooDownloader`

- **Question**: "How is news sentiment built?"

  - Go to:
    - `src/finance_rl_slm/news.py`
    - `src/finance_rl_slm/sentiment.py`
  - Main flow:
    - fetch Yahoo RSS,
    - analyze each news text with Granite,
    - map labels to numeric scores,
    - aggregate by week,
    - align weekly sentiment to daily trading dates.

### 2.3 Workflow and Results

- **Question**: "How do the notebooks stay short?"

  - Go to: `src/finance_rl_slm/workflow.py`
  - Main functions:
    - `run_only_ddpg_main`
    - `run_slm_main`
    - `load_price_data`
    - `load_online_price_data`
    - `result_picture_path`
    - `result_profile_path`

- **Question**: "Where are profile CSV files saved?"

  - Go to: `src/finance_rl_slm/evaluation.py`
  - Main functions:
    - `run_online_evaluation`
    - `save_online_profile`
    - `plot_online_logs`

- **Question**: "How do I compare the two pipelines?"

  - Go to: `src/tool/compare_ddpg_profiles.py`
  - Main functions:
    - `load_profile`
    - `compute_profile_metrics`
    - `compare_profiles`
    - `write_comparison`

## 3. Knowledge Library (Design Notes)

### 3.1 Reward Design: Sharpe-like Score + Drawdown Penalty

- The environment stores a rolling window of portfolio returns.

- It computes:
  - mean return,
  - standard deviation,
  - a Sharpe-like term: `mean / std`.

- It subtracts a drawdown penalty:
  - `drawdown = (peak_wealth - current_wealth) / peak_wealth`.

- Intuition:
  - reward smooth portfolio growth,
  - punish large drops from previous wealth peaks.

### 3.2 Technical Features in the State

- **Return window**
  - Recent portfolio returns.

- **MACD-like wealth trend**
  - Short EMA minus long EMA on portfolio wealth.
  - This measures trend in strategy wealth, not only in one stock.

- **Bollinger-style wealth deviation**
  - Measures how far current wealth is from a rolling normal band.

- **Wealth normalization**
  - Uses `wealth_norm_factor` to keep numeric scale reasonable.

### 3.3 SLM Sentiment as a Feature

- Weekly market sentiment is treated as a slow signal.

- It is not a high-frequency trading signal.

- The score is mapped into `[-1, 1]`.

- It is appended as the last observation dimension.

- This is why the SLM pipeline is experimental:
  - sentiment may help in some regimes,
  - but it may also add noise.

### 3.4 Python and Library Notes

- `Gymnasium`
  - Provides the RL environment API: `reset`, `step`, action space, observation space.

- `Stable-Baselines3`
  - Provides the DDPG implementation.
  - DDPG is useful here because portfolio weights are continuous actions.

- `pandas` and `numpy`
  - Used for price pivots, rolling windows, returns, and metrics.

- `transformers`
  - Used only in the SLM extension for Granite sentiment analysis.

## 4. Results and Comparison

### 4.1 Result Picture Folder

All generated result figures should be placed under:

```text
addenda/result_picture/
```

Existing SLM figures from the older workflow were moved into this folder.

### 4.2 Result Profile Folder

All generated profile and comparison CSV files should be placed under:

```text
addenda/result_profile_comparse/
```

The folder name keeps the current project spelling: `comparse`.

### 4.3 Profile CSV Meaning

Each online profile records one row per online environment step.

Important columns:

- `time`: trading date,
- `wealth`: portfolio wealth after the step,
- `reward`: reward from the environment,
- `drawdown`: current drawdown,
- `action`: portfolio action from the model,
- `daily_return`: percentage change in wealth.

### 4.4 Generated Result Provenance

The current generated result files use:

- Online period:
  - requested window: `2026-01-01` to `2026-06-21`,
  - available Yahoo trading rows: `2026-01-02` to `2026-06-18`,
  - profile rows: `2026-01-05` to `2026-06-18`.

- Pure DDPG:
  - loaded the existing `ddpg_portfolio_offline.zip`,
  - did not retrain the model during result generation.

- DDPG + SLM:
  - loaded the existing `ddpg_portfolio_offline.zip`,
  - used cached IBM Granite weights,
  - used Yahoo RSS with `news_max_items = 3` per ticker to keep local inference time reasonable,
  - scored 15 RSS news items in total,
  - mapped weekly sentiment to the daily online price index.

For a fuller SLM run, increase `news_max_items` in `RunConfig` or in the notebook config cell.
That will be slower because every news item needs one Granite inference call.

### 4.5 Suggested Reading Order

1. Read `main/main_code_only_ddpg.ipynb`.
2. Run or inspect the pure DDPG profile.
3. Read `main/main_code_add_slm.ipynb`.
4. Run or inspect the SLM profile.
5. Run `src/tool/compare_ddpg_profiles.py`.
6. Compare whether sentiment improved return stability or only changed risk.

## 5. Project Structure

```text
.
├── README.md
├── addenda/
│   ├── patrick_Bateman.jpg
│   ├── result_picture/
│   └── result_profile_comparse/
├── envs/
│   └── gym_portfolio_env.py
├── main/
│   ├── main_code_only_ddpg.ipynb
│   ├── main_code_only_ddpg.py
│   ├── main_code_add_slm.ipynb
│   └── main_code_add_slm.py
├── src/
│   ├── finance_rl_slm/
│   ├── tool/
│   │   └── compare_ddpg_profiles.py
│   └── FinRL/
└── tests/
```

## 6. Maintenance Notes

- Keep `.py` and `.ipynb` versions aligned.

- Do not treat the SLM extension as the default baseline.

- Do not commit regenerated model artifacts unless the experiment intentionally changed.

- If output file names change, update:
  - notebook notes,
  - README result paths,
  - `src/tool/compare_ddpg_profiles.py`,
  - related tests.
