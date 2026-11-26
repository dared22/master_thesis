# Four-stock SDE-GAN baseline experiment

This toy experiment builds and checks a reproducible Neural SDE-GAN baseline on
four Oslo Børs stocks drawn directly from
[Data/processed/log_returns_interpolated.csv](../../Data/processed/log_returns_interpolated.csv)
(the canonical 10-year eligible log-return panel built by `EDA/build_dataset.py`),
and compares it against Fixed Historical Simulation,
Rolling 252-session Historical Simulation, and the zero-shot Sundial and Timestral TSFM
benchmarks.

## Stocks and data

- **Source:** `Data/processed/log_returns_interpolated.csv` (`2016-09-09` to `2026-09-08`,
  2,509 daily log returns across the 2,510-session XOSL panel).
- **Companies:** `KONGSBERG GRUPPEN` (`NO0013536151`), `FRONTLINE`
  (`CY0200352116`), `WILH, WILHELMSEN A` (`NO0010571698`), and
  `TOMRA SYSTEMS` (`NO0012470089`).
- **Preprocessing:** Canonical thesis eligibility screen, short-gap natural cubic-spline
  interpolation (`<= 5` sessions; 0 missing prices for these four stocks), and reviewed
  corporate-action corrections applied upstream in `EDA/build_dataset.py`;
  chronological **70% / 15% / 15%** train/validation/test split, log returns
  standardized with training-only mean and sample standard deviation,
  and complete 20-session windows (stride 1).

## Workflow

Run all commands from the repository root:

```sh
# 1. Prepare return splits and standardized windows from Data/processed
uv run python experiments/4stocks/scripts/prepare_data.py

# 2. Train stabilized, mean-anchored increments SDE-GAN seeds
uv run python experiments/4stocks/scripts/train.py 0 1 2

# 3. Run zero-shot TSFM benchmarks on validation (saved to results/tsfm/<forecaster>/)
uv run --extra tsfm python experiments/4stocks/scripts/run_tsfm.py --forecaster sundial
uv run --extra tsfm python experiments/4stocks/scripts/run_tsfm.py --forecaster timestral

# 4. Evaluate SDE-GAN, Fixed HS, Rolling HS (252d), and saved TSFM forecasts
uv run python experiments/4stocks/scripts/evaluate.py

# Run model and benchmark unit tests
uv run python -m unittest discover -s experiments/4stocks/model
uv run python -m unittest discover -s benchmarks
```

Use [notebooks/inspect_and_compare.ipynb](notebooks/inspect_and_compare.ipynb)
to inspect the prepared returns, compare Fixed HS, Rolling HS, TSFM, and
SDE-GAN (including VaR time-series and quantile-loss plots), and review
multivariate correlation and ACF diagnostics interactively.

## Layout

- [data/](data/): Prepared return splits, window dates, standardized window
  arrays, and `preprocessing.json`.
- [model/](model/): Equinox Neural SDE generator, Neural CDE critic, stabilized
  WGAN-GP training loop (`architecture.py`, `training.py`, `sde_gan.py`, and
  `test_sde_gan.py`).
- [scripts/](scripts/): Reproducible CLI pipeline (`prepare_data.py`,
  `train.py`, `run_tsfm.py`, `evaluate.py`).
- [notebooks/](notebooks/): Single interactive inspection and comparison
  notebook (`inspect_and_compare.ipynb`).
- [results/](results/): Saved model checkpoints (`sde_gan/seed_*/`), TSFM
  forecasts (`tsfm/<forecaster>/`), and combined evaluation tables/plots (`evaluation/`).
