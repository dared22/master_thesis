# Copula and GARCH–copula portfolio risk benchmark specification

This document specifies the static and GARCH-filtered multivariate copula
portfolio risk benchmark in `master_thesis/benchmarks/`. It mirrors the
[historical simulation specification](historical-simulation-spec.md) and
[TSFM benchmark specification](tsfm-benchmark-spec.md) so the same dated return
CSVs, portfolio weights, horizons, risk definitions, and evaluation dates can be
compared directly against historical simulation, TSFMs, and the multivariate
Neural SDE.

Unlike the 1D TSFM benchmark, this module simulates full multivariate
`(num_samples, horizon_sessions, n_assets)` joint return paths. It supports both
`daily` and `buy_and_hold` rebalancing and exposes simulated simple-return paths
for cross-stock correlation, volatility-clustering, and joint tail-loss
diagnostics.

## Location and public interface

| Path, relative to `master_thesis/` | Responsibility |
| --- | --- |
| `benchmarks/_common.py` | Shared input validation, portfolio compounding, empirical VaR/ES, and coverage helpers. |
| `benchmarks/copula.py` | `CopulaConfig`, `CopulaBenchmark`, `CopulaSummary`, `CopulaBacktest`, univariate GARCH(1,1) / static marginals, and Gaussian / Student-*t* copula estimation and simulation. |
| `benchmarks/__init__.py` | Re-exports `CopulaConfig`, `CopulaBenchmark`, `CopulaSummary`, and `CopulaBacktest`. |
| `benchmarks/test_copula.py` | Unit tests covering MLE recovery, tail dependence, multi-day variance recursion, strict causality, missing data, and 4-stock CSV integration. |
| `benchmarks/docs/copula-benchmark-spec.md` | This reference specification. |

The module uses only NumPy, SciPy, and pandas on the CPU. No method writes files.

```python
from benchmarks import CopulaBenchmark, CopulaConfig

config = CopulaConfig(
    horizon_sessions=20,
    rebalance='daily',
    confidence_levels=(0.95, 0.99),
    copula='t',
    marginal='garch_t',
    pit='empirical',
    estimation='fixed',
    num_samples=10_000,
    seed=0,
)
benchmark = CopulaBenchmark(config).fit(
    'experiments/4stocks/data/train_returns.csv'
)
summary = benchmark.summarize()
validation = benchmark.backtest(
    'experiments/4stocks/data/validation_returns.csv'
)
```

`sample_paths(num_samples=None, seed=None)` returns simulated simple-return paths
of shape `(num_samples, horizon_sessions, n_assets)` from the fitted state so
multivariate diagnostics (signed/absolute ACF, pairwise correlation, lower-tail
co-exceedance) can be evaluated alongside the Neural SDE.

## Mathematical formulation

### Return space

Inputs follow the shared contract in [historical-simulation-spec.md](historical-simulation-spec.md)
(dated unique increasing ISO sessions, simple or log returns, long-only weights
summing to 1 within `1e-10`). Internally, all marginals and copulas model
**daily log returns** $r_{i,t}$. When `return_type="log"`, validated raw log
returns are retained directly without round-tripping through `expm1`/`log1p`, so
extreme finite negative log returns (for example `-40`) are preserved rather
than underflowing to `-1` in simple-return space. When `return_type="simple"`,
$r_{i,t} = \log(1 + R_{i,t})$ and any $R_{i,t} \le -1$ triggers
`estimate_status="invalid_log_return_history"` (or raises `ValueError` in
`summarize()`). Simulated log-return paths are converted back to simple returns
via $R_{i,t}^{(m)} = \operatorname{expm1}(r_{i,t}^{(m)}) > -1$ before portfolio
compounding.

### Marginals (`marginal`)

Each asset $i \in \{1, \dots, d\}$ is fitted separately on complete daily
log-return rows:

1. **`static_empirical`**: Empirical distribution of observed daily log returns
   (i.i.d. across time; requires `pit="empirical"`).
2. **`static_t`**: Static standardized Student-*t* marginal
   $r_{i,t} = \mu_i + \sigma_i z_{i,t}$, $z_{i,t} \sim t_{\text{std}}(\nu_i)$
   ($\nu_i \in [2.05, 200]$), fitted by MLE.
3. **`garch_normal`**: Univariate GARCH(1,1) with standard normal innovations:
   $$r_{i,t} = \mu_i + \varepsilon_{i,t}, \qquad \varepsilon_{i,t} = \sigma_{i,t} z_{i,t}, \qquad \sigma_{i,t}^2 = \omega_i + \alpha_i \varepsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2,$$
   with $\omega_i > 0$, $\alpha_i > 0$, $\beta_i > 0$, and $\alpha_i + \beta_i < 0.9995$,
   fitted by MLE in unit-variance coordinates and rescaled analytically.
4. **`garch_t`** (default): Univariate GARCH(1,1) with standardized Student-*t*
   innovations $z_{i,t} \sim t_{\text{std}}(\nu_i)$ ($\mathbb{E}[z] = 0$,
   $\operatorname{Var}(z) = 1$, $\nu_i \in [2.05, 200]$).

Constant (zero-variance) series are handled cleanly as degenerate marginals with
zero innovation variance.

### Probability integral transform (`pit`)

Fitted innovations $\hat{z}_{i,t}$ (or log returns for `static_empirical`) are
mapped to uniform pseudo-observations $u_{i,t} \in (0, 1)$:

- **`empirical`** (default, semi-parametric / FHS-copula):
  $u_{i,t} = \operatorname{rank}(\hat{z}_{i,t}) / (T + 1)$ using average ranks;
  simulation inverts simulated uniforms via linear empirical quantiles of
  $\hat{z}_{i,1:T}$.
- **`parametric`** (Inference Functions for Margins):
  $u_{i,t} = F_{z,i}(\hat{z}_{i,t})$ using the fitted normal or standardized
  Student-*t* CDF; simulation inverts simulated uniforms via $F_{z,i}^{-1}(u)$,
  allowing parametric tail extrapolation beyond the historical maximum residual.

### Multivariate copula (`copula`)

Degenerate (constant) assets are excluded from copula correlation and degrees-of-freedom
estimation and embedded as independent unit-diagonal entries in $R$.

- **`gaussian`**: Normal-score sample correlation matrix $R = \operatorname{Corr}(\Phi^{-1}(U))$,
  projected to the Frobenius-nearest positive-definite correlation matrix
  ($\lambda_{\min} \ge 10^{-6}$) via Higham's (2002) alternating projections
  with Dykstra's correction.
- **`t`** (default): Elliptical correlation matrix estimated via Kendall's $\tau$
  inversion $R_{ij} = \sin(\frac{\pi}{2}\hat{\tau}_{ij})$ (projected via Higham's
  nearest-PD correlation algorithm), followed by 1D bounded MLE of the copula
  degrees of freedom $\nu_C \in [2.05, 200]$ under the exact multivariate
  Student-*t* copula log-likelihood over the non-degenerate assets.

### Multi-day path simulation ($H$ sessions)

For each Monte Carlo path $m = 1, \dots, M$ and step $h = 1, \dots, H$:

1. Draw $\mathbf{u}_h^{(m)} \sim C_{\hat{\theta}}$ (independent across $h$,
   dependent across assets $1, \dots, d$).
2. Invert each marginal to obtain $z_{i,h}^{(m)} = F_{z,i}^{-1}(u_{i,h}^{(m)})$.
3. For static marginals, set $r_{i,h}^{(m)} = z_{i,h}^{(m)}$ (`static_empirical`)
   or $r_{i,h}^{(m)} = \hat{\mu}_i + \hat{\sigma}_i z_{i,h}^{(m)}$ (`static_t`).
4. For GARCH marginals, initialize $(\sigma_{i,1}^{(m)})^2 = \hat{\sigma}_{i,t_0+1}^2$
   from the causal GARCH filter at `forecast_origin` $t_0$, set
   $\varepsilon_{i,h}^{(m)} = \sigma_{i,h}^{(m)} z_{i,h}^{(m)}$,
   $r_{i,h}^{(m)} = \hat{\mu}_i + \varepsilon_{i,h}^{(m)}$, and update recursively:
   $$(\sigma_{i,h+1}^{(m)})^2 = \hat{\omega}_i + \hat{\alpha}_i (\varepsilon_{i,h}^{(m)})^2 + \hat{\beta}_i (\sigma_{i,h}^{(m)})^2.$$
5. Convert $R_{i,h}^{(m)} = \operatorname{expm1}(r_{i,h}^{(m)})$ and aggregate
   with `_portfolio_returns`, `_empirical_var`, and `_empirical_es`.

## Causality, estimation modes, and missing data

`backtest(evaluation_returns, *, warmup_returns=None)` requires evaluation dates
strictly after fitted history (and strictly after `warmup_returns` when
supplied) and never uses any observation at or after `target_start` to compute a
forecast.

| `estimation` | Parameter fitting | GARCH variance state |
| --- | --- | --- |
| `fixed` | Fitted once on calibration history (restricted to trailing `lookback_sessions` if set). | Filtered causally forward day by day through trailing incomplete fit rows, any intervening `warmup_returns`, and evaluation sessions strictly before `target_start`. Static marginals reuse the same forecast across all targets. |
| `expanding` | Refitted every `refit_every` targets on all history (including `warmup_returns`) strictly before `target_start`. | Initialized from the latest refit, filtered across any trailing incomplete rows in the refit window, and filtered causally across intervening sessions strictly before `target_start`. |
| `rolling` | Refitted every `refit_every` targets on the trailing `lookback_sessions` raw rows strictly before `target_start`. | Same causal filtering across trailing incomplete rows and between refits. |

- Missing rows inside a fit window are excluded from parameter estimation without
  imputation. Any trailing incomplete rows after the last complete fit row still
  advance the GARCH variance state causally so refits and between-refit steps
  treat partial sessions consistently.
- On any filtered session where asset $i$ is missing (`NaN`), the causal GARCH
  filter propagates the conditional expectation
  $\sigma_{i,t+1}^2 = \hat{\omega}_i + (\hat{\alpha}_i + \hat{\beta}_i)\sigma_{i,t}^2$
  (using a scale-relative variance floor `1e-12 * sample_var`) without inventing
  returns, and records the count in `n_missing_filter_steps`.
- Realized target blocks with any missing return are retained with
  `realization_status="missing_returns"` and `breach = pd.NA`.

## Configuration

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `assets` | `None` | All asset columns, or an explicit nonempty tuple of distinct names. |
| `weights` | `None` | Equal weights over selected assets, or mapping matching selected assets. |
| `return_type` | `"simple"` | `"simple"` or `"log"`. |
| `horizon_sessions` | `1` | Positive integer number of forecast sessions $H$. |
| `rebalance` | `"daily"` | `"daily"` or `"buy_and_hold"`. |
| `confidence_levels` | `(0.95, 0.99)` | Distinct finite confidence levels in $(0, 1)$. |
| `copula` | `"t"` | `"gaussian"` or `"t"`. |
| `marginal` | `"garch_t"` | `"static_empirical"`, `"static_t"`, `"garch_normal"`, or `"garch_t"`. |
| `pit` | `"empirical"` | `"empirical"` (semi-parametric) or `"parametric"` (IFM). Must be `"empirical"` for `static_empirical`. |
| `estimation` | `"fixed"` | `"fixed"`, `"expanding"`, or `"rolling"`. |
| `lookback_sessions` | `None` | Raw session count. Required for `"rolling"`, forbidden for `"expanding"`, optional for `"fixed"`. Must be $\ge$ `min_observations`. |
| `refit_every` | `1` | Positive integer target stride between parameter refits in `"expanding"` / `"rolling"` mode. |
| `min_observations` | `60` | Minimum complete daily observations required to fit ($\ge 10$). |
| `num_samples` | `10000` | Monte Carlo simulated paths per origin ($\ge 2$). |
| `seed` | `0` | Nonnegative integer base seed (`seed + target_index` per origin in `backtest()`). |

## Thesis comparison role

This benchmark provides four cleanly separated baselines using one unified API:

1. **Static Gaussian / Student-*t* copula** (`marginal="static_empirical"` or
   `"static_t"`, `estimation="fixed"`): unconditional cross-sectional dependence
   benchmarks comparable directly to Fixed Historical Simulation and an
   unconditional Neural SDE.
2. **GARCH–Gaussian / GARCH–*t* copula** (`marginal="garch_t"`,
   `copula="gaussian"` or `"t"`): conditional volatility-clustering + joint
   dependence benchmarks comparable to Rolling Historical Simulation, TSFMs, and
   a conditional Neural SDE.
