# Four-stock SDE-GAN baseline experiment

This toy experiment builds and checks a reproducible Neural SDE-GAN baseline on
four Oslo Børs stocks drawn directly from
[Data/eligible/price_histories.csv](../../Data/eligible/price_histories.csv)
(the same 10-year eligible panel and short-gap interpolation pipeline used in
the main thesis analysis), and compares it against Fixed Historical Simulation,
Rolling 252-session Historical Simulation, and the zero-shot Sundial TSFM
benchmark.

## Stocks and data

- **Source:** `Data/eligible/price_histories.csv` (`2016-09-08` to `2026-09-08`,
  2,510 trading sessions → 2,509 daily returns).
- **Companies:** `KONGSBERG GRUPPEN` (`NO0013536151`), `FRONTLINE`
  (`CY0200352116`), `WILH, WILHELMSEN A` (`NO0010571698`), and
  `TOMRA SYSTEMS` (`NO0012470089`).
- **Preprocessing:** Thesis eligibility screen, natural cubic-spline
  interpolation for internal gaps up to 5 sessions (matching
  `EDA/02_sector_dependence.ipynb`; 0 missing prices for these four stocks),
  chronological **70% / 15% / 15%** train/validation/test split, `log1p`
  returns standardized with training-only mean and sample standard deviation,
  and complete 20-session windows (stride 1).

## Workflow

Run all commands from the repository root:

```sh
# 1. Prepare return splits and standardized windows from Data/eligible
uv run python experiments/4stocks/scripts/prepare_data.py

# 2. Train stabilized, mean-anchored increments SDE-GAN seeds
uv run python experiments/4stocks/scripts/train.py 0 1 2

# 3. Run the zero-shot Sundial TSFM benchmark on validation (saved to results/tsfm/)
uv run --extra tsfm python experiments/4stocks/scripts/run_tsfm.py

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
  forecasts (`tsfm/`), and combined evaluation tables/plots (`evaluation/`).
