# Result Addenda

## What Is Here

- This folder stores project images, result plots, and result CSV profiles.

- Main result folders:
  - `result_picture/`
  - `result_profile_comparse/`

## When To Use It

- Use `result_picture/` for generated wealth, reward, and daily-return figures.

- Use `result_profile_comparse/` for online evaluation profile CSV files and DDPG vs SLM comparison CSV files.

## Important Files

- `result_profile_comparse/only_ddpg_online_profile_2026-01-01_2026-06-21.csv`
  - Pure DDPG online profile.

- `result_profile_comparse/ddpg_slm_online_profile_2026-01-01_2026-06-21.csv`
  - DDPG+SLM online profile.

- `result_profile_comparse/ddpg_vs_slm_comparison_2026-01-01_2026-06-21.csv`
  - Summary comparison table.

## Common Checks

- Confirm required artifacts:

  ```bash
  python -B -m unittest tests.test_results_and_docs.ResultsAndDocsTests.test_output_contract_artifacts_exist -v
  ```

- Keep file names date-tagged when adding new experiment outputs.
