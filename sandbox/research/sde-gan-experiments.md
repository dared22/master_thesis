# Four-stock SDE-GAN: experiment notes

Updated November 16, 2026. This note summarizes the experiments in
[4stocks_experiment](../4stocks_experiment/) and the current setup.
**Current candidate: increments.** Excessive signed-return persistence is
largely gone across three seeds; volatility clustering and tails still differ.

## Experiments so far

The sequence below records what changed and what we learned. Model folders
are relative to `sandbox/4stocks_experiment/results/`.

| Experiment | Change or purpose | Main finding | Saved results |
| --- | --- | --- | --- |
| Original baseline; seeds 0, 1, 2 | Generate daily return levels directly. | Excessive persistence; seed-0 signed ACF error 0.619. | `sde_gan/seed_*` |
| Temporal penalty 10; seeds 0, 1, 2 | Match training signed log-return ACF at lags 1–5. | Improved, but not fixed; seed-0 error 0.348. | `sde_gan/temporal_penalty/` |
| Temporal penalty 30; seed 0 | Increase the same penalty. | Later ACF improvement came with collapsed volatility. MMD selected step 200. | `sde_gan/temporal_penalty_30/` |
| Signature diagnostic | Test depths 2 and 3 on halved-volatility and sorted-path controls. | Detected known distortions; diagnostic success did not guarantee training success. | `signature_diagnostics/` |
| Signature loss; seed 0 | Add depth-2 signature weight 1 to temporal weight 10. | Better feature and cross-stock fit; signed ACF error still 0.344. | `sde_gan/signature_loss_depth2/` |
| Synthetic benchmark; seed 0 | Compare outputs on Gaussian returns with independent days and same-day stock correlation 0.5. | Increments reduced maximum absolute signed ACF from 0.706 to 0.024. Mean and cross-stock errors remained. | `synthetic_iid/{returns,increments}/` |
| Real-data increments; seeds 0, 1, 2 | Take daily differences of cumulative outputs; disable extra penalties. | Repeatable ACF improvement; seed-0 error 0.045. Weak volatility clustering remains. | `sde_gan/increments/` |
| GP 5; seeds 0, 1, 2 | Reduce critic gradient penalty to 5. | Mixed larger-sample risk-score gains; worse coverage and mean bias. | `sde_gan/increments_gp5/` |
| Absolute ACF 1; seeds 0, 1, 2 | Add absolute log-return ACF loss with weight 1. | Only a small clustering improvement; no risk-score improvement over the baseline overall. | `sde_gan/increments_absolute_acf_1/` |
| Historical comparison | Compare against complete training windows; portfolio VaR/ES at 95% and 99%. | Mixed validation risk results; test superiority not established. | [Comparison notebook](../4stocks_experiment/compare_historical_sde_gan.ipynb) |
| Numerical checks | Known additive SDE, gradients, sampling, save/load, and 50-update smoke runs. | Passed during implementation. Refining step 0.25 to 0.125 gave small inspected differences, not a general convergence proof. | [Tests](../4stocks_experiment/test_sde_gan.py) |

Real-data ACF errors above are seed-0 RMSE against training, across lags 1–5,
from the
[inspection table](../4stocks_experiment/results/sde_gan/increments/inspection/step_1500/model_comparison.csv).

## Current data and model

Inputs remain daily returns. The increments generator takes differences
internally; do not difference the CSV inputs again.

- **Stocks:** KONGSBERG GRUPPEN, FRONTLINE, WILH Wilhelmsen B, TOMRA SYSTEMS.
- **Data:** 1,679 / 709 / 798 complete training/validation/test windows, split
  chronologically. Test data remain reserved.
- **Preprocessing:** `log1p`, then per-stock training mean and sample standard
  deviation; shape `(20, 4)`, stride 1. Reject incomplete windows without
  filling gaps, clipping returns, or removing missing calendar rows.
- **Generator:** latent size 8, initial noise 4, Brownian size 4; width-32,
  depth-2 tanh networks. Output `"increments"`: 21 cumulative values become 20
  returns. Independent noise can still produce dependent returns.
- **Critic and solver:** neural CDE with hidden size 16; Heun solver with
  generator and critic step 0.25. Solver refinement checks use 0.125.
- **Training:** 2,000 generator updates, five critic updates per generator
  update, batch size 8, Adam learning rates `1e-4`, moments `(0, 0.9)`.
  WGAN gradient penalty stays at 10. Temporal and signature weights are zero.
- **Selection:** lowest validation Gaussian-kernel MMD², checked every 100
  updates; retain all validation checkpoints. Lower MMD doesn't guarantee
  that every individual statistic matches.
- **Evaluation:** 1,000 paths, sampling seed 100; 20-session portfolio with
  daily rebalancing to 25% per stock and simple-return compounding.

Use `sandbox/.venv`. Set `generator_output="increments"` explicitly; the class
default remains `"returns"` for old-checkpoint compatibility.

## Saved checkpoints and notebook settings

All three increments fits are complete. Their root run folders contain the
per-seed MMD-selected models:

| Seed | Selected generator update | Validation MMD² |
| --- | ---: | ---: |
| 0 | 1500 | 0.006802 |
| 1 | 1000 | 0.010242 |
| 2 | 800 | 0.007183 |

`results/sde_gan/increments/seed_1/` loads step 1000; its
`checkpoints/step_1500/` subfolder loads step 1500. Don't mix these policies.

| Notebook | Current purpose/settings |
| --- | --- |
| [Preprocessing](../4stocks_experiment/preprocessing.ipynb) | Increments, zero extra penalties; saved preset `MODE="prepare"`, `TRAINING_SEEDS=[0]`. |
| [Inspection](../4stocks_experiment/inspect_sde_gan.ipynb) | Seed 0, step 1500; output under `increments/inspection/step_1500/`. |
| [Historical comparison](../4stocks_experiment/compare_historical_sde_gan.ipynb) | Seeds 0, 1, 2 at **fixed step 1500**, not their individual selected steps; `EVALUATE_TEST=False`. |
| [Synthetic benchmark](../4stocks_experiment/synthetic_iid_benchmark.ipynb) | Compare `"returns"` and `"increments"`; save separately. |
| [Signature diagnostics](../4stocks_experiment/signature_diagnostics.ipynb) | Detect known distortions using training data; no retraining. |

## Interpretation and next decision

Remaining issues: low absolute-return ACF, narrow window-volatility
distributions, and varying means and portfolio tails across seeds.

The earlier 1,000-path, per-seed MMD-selected comparison found lower 95%
validation quantile loss than historical simulation for all three seeds, and
lower 99% loss for two
of three. Calibration remains uneven. This is preliminary validation evidence,
not a final test result; it also differs from the notebook's fixed-step review.

Freeze checkpoint selection and evaluation settings before the final test.
Validation already influenced selection; overlapping windows and repeated
training seeds do not provide independent market datasets.

Configuration and model behavior are defined in
[training.py](../4stocks_experiment/training.py),
[architecture.py](../4stocks_experiment/architecture.py), and
[sde_gan.py](../4stocks_experiment/sde_gan.py). The historical benchmark
is implemented in [historical.py](../../benchmarks/historical.py).

The [follow-up ablation report](sde-gan-ablation-results.md) records the GP-5
and absolute-ACF experiments, including the larger-sample risk check. Keep the
GP-10 increments setup as the current reference.
