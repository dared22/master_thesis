# Four-stock SDE-GAN experiment

This experiment fits an SDE-GAN to 20-session windows of four stocks and
compares the generated returns with observed returns and historical simulation.

## Notebooks

Open the notebooks with the repository's Python environment. The first two
notebooks produce the files in [data/](data/). The remaining notebooks use
those files or synthetic data.

| Notebook | Purpose |
| --- | --- |
| [Adjusted price histories](notebooks/adjusted_price_histories.ipynb) | Extract the four stocks from the source export and write the chronological return splits. |
| [Prepare windows and train the SDE-GAN](notebooks/prepare_windows_and_train_sde_gan.ipynb) | Build standardized windows and scalers. Set `MODE` to inspect saved runs, run a smoke check, or train. |
| [Inspect the increment SDE-GAN](notebooks/inspect_increment_sde_gan.ipynb) | Compare solver steps, sample paths, and check volatility and autocorrelation at a fixed checkpoint. |
| [Compare with historical simulation](notebooks/compare_sde_gan_with_historical_simulation.ipynb) | Compare the fitted models and historical simulation on validation returns. |
| [Signature diagnostics](notebooks/signature_diagnostics.ipynb) | Check which path distortions the signature features detect. |
| [Synthetic IID returns benchmark](notebooks/synthetic_iid_returns_benchmark.ipynb) | Compare the generator variants on independent daily returns. Set `MODE = "inspect"` to use saved models. |

## Code and results

The [model/](model/) directory contains the SDE-GAN implementation and its
tests. The [scripts/](scripts/) directory contains the three ablation entry
points. Use these commands from the repository root.

```sh
uv run python experiments/4stocks/scripts/train_increment_ablation.py gp5 0
uv run python experiments/4stocks/scripts/evaluate_increment_ablations.py
uv run python experiments/4stocks/scripts/check_monte_carlo_risk.py
```

The training command handles one variant and seed at a time. The evaluation
uses the saved runs for both variants (`gp5` and `absolute_acf_1`) and seeds
0–2. The Monte Carlo check uses the sample files produced by the evaluation.

- [results/sde_gan/](results/sde_gan/) holds models fitted to stock returns.
  `direct_returns_baseline/` is the original model and `increments/` is the
  increment baseline. Each seed directory contains the selected model, with
  intermediate snapshots in `checkpoints/` where available.
- [results/increment_ablations/](results/increment_ablations/) holds the
  across-seed metrics, plots, samples, and Monte Carlo risk checks. Paths
  in `selected_checkpoints.csv` are relative to the experiment directory.
- [results/signature_diagnostics/](results/signature_diagnostics/) and
  [results/synthetic_iid/](results/synthetic_iid/) hold the two diagnostics.

Notebook smoke runs write scratch output to `.cache/4stocks/smoke/` at the
repository root. Use this command from the repository root to run the model
tests.

```sh
uv run python -m unittest discover -s experiments/4stocks/model
```
