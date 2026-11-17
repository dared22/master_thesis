# Increments ablations: gradient penalty and absolute-return ACF

Completed November 16, 2026. All six full fits finished successfully.
**Keep the current GP-10 increments model as the baseline.** GP 5 is a mixed
challenger: its larger-sample risk scores improve, but coverage worsens.
Absolute-return ACF weight 1 gives little benefit.

## Protocol

Each new setting uses seeds 0, 1, and 2, with 2,000 generator updates per seed.
Architecture, data, optimizer, batch size, solver step 0.25, and validation
sampling stay unchanged. Each seed uses its lowest-validation-MMD checkpoint,
rather than fixing every run at step 1500. Test returns are not loaded.

| Setting | Critic gradient penalty | Signed ACF penalty | Absolute ACF penalty | Signature loss |
| --- | ---: | ---: | ---: | ---: |
| Existing increments | 10 | 0 | 0 | 0 |
| GP 5 | 5 | 0 | 0 | 0 |
| Absolute ACF 1 | 10 | 0 | 1 | 0 |

The new loss matches training absolute **log-return** correlations at lags
1–5, using minibatch mean squared error. It restores original centering with
`training_mean / training_std`; positive scaling cancels from correlations.
Targets use complete training windows only. Weight 1 is a pilot, not a tuned
optimum. Displayed ACF diagnostics use simple returns, matching earlier plots.

All six runs share the same code and data hashes. Initial validation MMD
matches the corresponding baseline seed exactly. All 26 regression tests and
a 50-update stock-data smoke run passed. The existing baseline smoke run also
reproduced its previous numerical results.

## Risk results

Quantile loss is averaged across three fitted seeds; lower is better. Scores
use the notebook's scaling by 100. Historical simulation always uses all 1,679
training blocks. Portfolio losses cover 20 sessions with daily rebalancing.

| Setting | 95% loss, 1,000 paths | 99% loss, 1,000 paths | 95% loss, 10,000 paths | 99% loss, 10,000 paths |
| --- | ---: | ---: | ---: | ---: |
| Historical simulation | 0.6367 | 0.1699 | 0.6367 | 0.1699 |
| Existing increments | **0.6096** | **0.1597** | 0.6247 | 0.1657 |
| GP 5 | 0.6324 | 0.1644 | **0.6189** | **0.1560** |
| Absolute ACF 1 | 0.6213 | 0.1633 | 0.6270 | 0.1673 |

The 1,000-path review favors the unchanged baseline. A nested 10,000-path
cross-check, with the same sampling seed 100, reverses the ranking for GP 5:
its average losses are 0.9% lower at 95% and 5.8% lower at 99%. GP 5 improves
two of three paired seeds at each level in that larger-sample check.

Coverage remains a problem. These are average validation breach rates with
10,000 generated paths:

| Setting | 95% VaR breaches; target 5% | 99% VaR breaches; target 1% |
| --- | ---: | ---: |
| Historical simulation | 2.12% | 0.00% |
| Existing increments | 5.74% | 1.36% |
| GP 5 | 8.93% | 1.97% |
| Absolute ACF 1 | 5.78% | 1.41% |

Averages hide large seed differences. At 95%, GP-5 breach rates are 13.96%,
5.92%, and 6.91%. Its seed-0 quantile loss is worse than the baseline at 95%.
Thus lower average score does not establish reliable calibration across fits.

## Distribution fit

The following diagnostics average the original 1,000-path samples across seeds.
Errors compare with the retained training windows; lower is better.

| Setting | Signed ACF RMSE | Absolute ACF RMSE | Cross-stock correlation RMSE | Daily mean error in training-std units |
| --- | ---: | ---: | ---: | ---: |
| Existing increments | 0.0477 | 0.1208 | 0.1955 | 0.0995 |
| GP 5 | 0.0464 | 0.1183 | 0.0932 | 0.1323 |
| Absolute ACF 1 | 0.0499 | 0.1167 | 0.1977 | 0.0982 |

GP 5 improves cross-stock correlation fit, but increases mean-return error.
Its average generated portfolio return rises from the baseline's 2.98% to
6.20% per 20 sessions in these samples. Reducing the critic penalty therefore
does more than change volatility.

Absolute ACF 1 reduces the absolute-return ACF error by only about 3.3%.
Large-move clustering remains much weaker than observed, and the distribution
of window volatility remains narrow. This pilot does not justify enabling the
loss by default; it does not rule out other weights or estimators.

## Decision and limits

Retain GP 10 with both ACF penalties and signature loss at zero as the stable
reference. Keep GP 5 as a challenger for further investigation of mean bias and
calibration. Neither experiment provides a clear improvement across all goals.

Solver refinement to 0.125 preserves this broad conclusion. The largest
observed 95% VaR change is about 0.30 percentage points for absolute-ACF seed 0;
GP 5 changes by at most 0.12 points. Numerical differences are not zero.

These are exploratory validation results after checkpoint selection. The 709
validation windows overlap; three training seeds are not three independent
market datasets. The larger sample reduces simulation noise, not uncertainty
about market behavior. Freeze the method and selection rule before evaluating
the reserved test split.

## Files and reproduction

The saved models and analysis files let you review every seed separately.

- [GP-5 checkpoints](../4stocks_experiment/results/sde_gan/increments_gp5/)
  and [absolute-ACF checkpoints](../4stocks_experiment/results/sde_gan/increments_absolute_acf_1/).
- [Selected checkpoints](../4stocks_experiment/results/sde_gan/increment_ablations/selected_checkpoints.csv),
  [distribution metrics](../4stocks_experiment/results/sde_gan/increment_ablations/run_metrics.csv),
  and [solver sensitivity](../4stocks_experiment/results/sde_gan/increment_ablations/solver_risk_sensitivity.csv).
- [Risk scores: 1,000 paths](../4stocks_experiment/results/sde_gan/increment_ablations/validation_risk_scores.csv)
  and [risk scores: 10,000 paths](../4stocks_experiment/results/sde_gan/increment_ablations/validation_risk_scores_10000_paths.csv).
- [ACF comparison](../4stocks_experiment/results/sde_gan/increment_ablations/return_acf.png)
  and [window-volatility CDFs](../4stocks_experiment/results/sde_gan/increment_ablations/window_volatility_cdf.png).
- [Risk-score comparison with 10,000 paths](../4stocks_experiment/results/sde_gan/increment_ablations/validation_quantile_loss_10000_paths.png).

Use `sandbox/.venv`. The
[training runner](../4stocks_experiment/run_increment_ablation.py) takes
`gp5` or `absolute_acf_1` and a seed, and skips completed matching runs.
The [evaluation script](../4stocks_experiment/evaluate_increment_ablations.py)
reproduces the 1,000-path review. The saved
[Monte Carlo check](../4stocks_experiment/results/sde_gan/increment_ablations/monte_carlo_check.py)
reproduces the 10,000-path check. Restart notebook kernels before loading
models with the newly added absolute-ACF configuration field.
