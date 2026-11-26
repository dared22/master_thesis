# TSFM portfolio risk benchmark specification

This document specifies the zero-shot Time Series Foundation Model (TSFM)
portfolio risk benchmark in `master_thesis/benchmarks/`. It mirrors the
[historical simulation specification](historical-simulation-spec.md) so the
same return CSVs, portfolio weights, horizons, risk definitions, and
evaluation dates can be compared directly.

The benchmark constructs a causal 1D daily-rebalanced portfolio series from
dated asset returns, draws sample trajectories from a probabilistic TSFM
(default: `thuml/sundial-base-128m`), converts simulated paths to fractional
horizon losses, and estimates value-at-risk (VaR) and expected shortfall (ES).

## Location and public interface

| Path, relative to `master_thesis/` | Responsibility |
| --- | --- |
| `benchmarks/_common.py` | Shared pure NumPy/pandas input validation, portfolio compounding, empirical VaR/ES, and coverage helpers. |
| `benchmarks/tsfm.py` | `SampleForecaster` and `QuantileForecaster` protocols, lazy-loaded `SundialForecaster` and `TimestralForecaster`, `TSFMConfig`, and `TSFMForecastBenchmark`. |
| `benchmarks/__init__.py` | Re-exports benchmark classes without importing `torch`, `transformers`, or `timestral`. |
| `benchmarks/test_tsfm.py` | Offline unit tests using deterministic mock forecasters (no PyTorch or network required). |
| `benchmarks/docs/tsfm-benchmark-spec.md` | This reference specification. |

Importing `benchmarks`, `benchmarks.historical`, or `benchmarks.tsfm`, or
calling `fit()`, must never import `torch`, `transformers`, or `timestral` and
must never write files. Deep-learning dependencies are imported lazily only when
`SundialForecaster.sample()` or `TimestralForecaster.predict_quantiles()` executes.

```python
from benchmarks import TSFMConfig, TSFMForecastBenchmark

config = TSFMConfig(
    horizon_sessions=20,
    target='log_price',
    context_sessions=252,
    num_samples=1000,
    confidence_levels=(0.95, 0.99),
    device='cpu',
)
benchmark = TSFMForecastBenchmark(config).fit(
    'experiments/4stocks/data/train_returns.csv'
)
summary = benchmark.summarize()
validation = benchmark.backtest(
    'experiments/4stocks/data/validation_returns.csv'
)
```

## Input contract and rebalancing

The input contract is identical to
[historical-simulation-spec.md](historical-simulation-spec.md): dated ISO
calendar rows, unique increasing dates, float64 simple (default) or log
returns, long-only fully invested weights summing to 1 within `1e-10` without
silent normalization, and strict preservation of raw calendar rows.

This module supports `rebalance="daily"` only. Under daily rebalancing,
daily portfolio returns $r_{p,t} = \sum_i w_i r_{i,t}$ define a single 1D
cumulative portfolio wealth index $V_t = \prod_{u \le t} (1 + r_{p,u})$ whose
endpoint ratio $V_{t_0 + H} / V_{t_0} - 1$ equals the compounded $H$-session
portfolio return identically for any $H \ge 1$. Buy-and-hold weights drift
across sessions and cannot be reduced to a single 1D series without modelling
joint multi-asset paths.

## Target representations and missing-context policy

For a forecast origin at the end of a causal history window (strictly before
`target_start`), daily portfolio returns are computed only on **complete**
rows where every selected asset has a finite return (matching the
all-selected-assets rule of `HistoricalSimulation`, including zero-weighted
selected assets).

The benchmark selects the last `context_sessions` complete portfolio rows:

- If fewer than `context_sessions` complete rows exist strictly before
  `target_start`, `summarize()` raises a `ValueError` and `backtest()` retains
  the target row with `estimate_status="insufficient_history"`.
- Incomplete rows inside the calendar span from the first selected complete
  row through `forecast_origin` are skipped without inventing zero returns or
  interpolating prices. The raw span (`history_start`, `history_end`,
  `n_history_sessions`), `n_context_sessions`, and
  `n_excluded_context_sessions` are recorded on every row.
- Realized target blocks in `backtest()` are never altered or filled; any
  incomplete realized block is retained with
  `realization_status="missing_returns"` and `breach = pd.NA`.

Two target representations are supported over the same `L = context_sessions`
complete daily returns $r_{p,1}, \dots, r_{p,L}$:

1. **`target="log_price"` (default):**
   Builds `L + 1` cumulative log-wealth points anchored at $y_L = 0$ at the
   last complete session:
   $$y_k = -\sum_{u=k+1}^{L} \log(1 + r_{p,u}) \quad (k = 0, \dots, L-1), \qquad y_L = 0.$$
   Using `L + 1` levels ensures all `L` returns appear in the first
   differences $y_k - y_{k-1} = \log(1 + r_{p,k})$ and keeps `log_price` and
   `return` modes aligned on the exact same observations. If any complete
   portfolio return in the window equals $-1$, $\log(1 + r_p)$ is undefined
   and the origin is marked `estimate_status="invalid_log_price_context"` (or
   raises in `summarize()`). Simulated $H$-step trajectories
   $\hat{y}_{1:H}^{(m)}$ map to $H$-session portfolio returns via
   $$R_p^{(m)} = \operatorname{expm1}\!\left(\hat{y}_H^{(m)}\right) > -1.$$
2. **`target="return"`:**
   Passes the `L` complete daily portfolio returns directly. Simulated daily
   returns $\hat{r}_{1:H}^{(m)}$ are compounded as
   $\prod_{h=1}^{H} (1 + \hat{r}_h^{(m)}) - 1$. Under limited liability for
   an unleveraged long-only portfolio, any simulated path that hits
   $\hat{r}_h^{(m)} \le -1$ on any step is floored at total loss
   ($R_p^{(m)} = -1$, $L^{(m)} = 1$) rather than compounding negative wealth
   back to positive values. The count of floored paths (`n_clamped_paths`) is
   reported on every forecast row and summed in metadata.

Nonfinite forecaster outputs raise `FloatingPointError` in both modes.

## Quantile-head forecasters (Timestral)

`TimestralForecaster` implements `QuantileForecaster.predict_quantiles()` around
Mistral AI's `mistralai/Timestral-28M` (21 non-crossing marginal quantiles on
`{0.01, 0.05, 0.10, ..., 0.90, 0.95, 0.99}` per step, `clip_nonnegative=False`,
`symmetric_averaging=True`, `float32`).

- **Exact terminal VaR:** In `target="log_price"` mode, $x \mapsto -\operatorname{expm1}(x)$
  is strictly decreasing, so the $(1 - c)$-quantile of terminal cumulative log-wealth
  $Y_H$ maps directly to $H$-session VaR without interpolation:
  $$\widehat{\operatorname{VaR}}_c = -\operatorname{expm1}\!\left(q_{1-c}(Y_H)\right).$$
  Any confidence level $c$ whose $1 - c$ is not on the forecaster's exact grid
  raises `ValueError`.
- **Target restriction:** Marginal daily quantiles cannot be compounded across
  $H > 1$ sessions without a temporal copula. Quantile forecasters therefore
  raise `ValueError` for `target="return"` when `horizon_sessions > 1`;
  `target="return"` is permitted only at `horizon_sessions == 1` (with limited-liability
  clamping at `-1.0`).
- **Expected shortfall (`es`):** Because the quantile head provides no grid
  points below `0.01`, ES is not identifiable from the model output and is
  recorded as `NaN` (`n_scenarios = 0`, `tail_mass_observations = NaN`,
  `small_tail_sample = pd.NA`). `TSFMSummary.quantiles` stores the full terminal
  quantile table (`quantile_level`, `portfolio_return`, `loss`), while
  `TSFMSummary.scenarios` is empty.

## Configuration

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `assets` | `None` | All asset columns, or an explicit nonempty tuple of distinct names. |
| `weights` | `None` | Equal weights over selected assets, or mapping matching selected assets. |
| `return_type` | `"simple"` | `"simple"` or `"log"`. |
| `target` | `"log_price"` | `"log_price"` or `"return"`. |
| `horizon_sessions` | `1` | Positive integer number of forecast sessions $H$. |
| `rebalance` | `"daily"` | `"daily"` only. |
| `confidence_levels` | `(0.95, 0.99)` | Distinct finite confidence levels strictly in $(0, 1)$. For `forecaster="timestral"`, every $1 - c$ must lie on the 21-level grid. |
| `context_sessions` | `252` | Integer $\ge 2$ complete daily returns in the causal context window. |
| `num_samples` | `1000` | Integer $\ge 2$ simulated trajectories per forecast origin (sample forecasters only). |
| `target_stride` | `1` | Positive integer stride across available evaluation target starts. |
| `batch_size` | `1` | Positive integer number of origins batched per forecaster call. |
| `seed` | `0` | Nonnegative integer base seed (`seed + target_index` per origin/batch; sample forecasters only). |
| `forecaster` | `"sundial"` | `"sundial"` (sample-path) or `"timestral"` (quantile-head). |
| `model_id` | `None` | Hugging Face model identifier; resolves to `"thuml/sundial-base-128m"` for `"sundial"` and `"mistralai/Timestral-28M"` for `"timestral"`. |
| `device` | `"cpu"` | `"cpu"`, `"mps"`, `"cuda"`, or `"auto"` (`"auto"` chooses `"cuda"` else `"cpu"`, skipping Apple MPS where Sundial's flow-matching sampler produced nonfinite outputs on M2). |
| `dtype` | `"float32"` | `"float32"`, `"float16"`, or `"bfloat16"` (`"float32"` only for `"timestral"`). |
| `sample_chunk_size` | `None` | Optional positive integer chunk size over `num_samples` to cap peak RAM (Sundial only). |
| `revin` | `None` (`False` for `"log_price"`, `True` for `"return"`) | Whether to enable Sundial's context-level re-normalization. Must be `False` when `target="log_price"` to avoid mean-reversion bias on trending cumulative log-wealth series. |
| `revision` | `None` | Optional pinned Hugging Face commit hash for reproducibility. |

Changing `batch_size` or `sample_chunk_size` alters the PyTorch RNG stream;
keep them fixed when reproducing exact sample draws.

## VaR, ES, and output schemas

Empirical VaR and fractional-tail ES use the exact shared implementations
(`_empirical_var`, `_empirical_es`) from `benchmarks/_common.py` with
positive-loss convention $L^{(m)} = -R_p^{(m)}$ and `small_tail_sample`
flagged whenever `num_samples * (1 - confidence) < 20`.

- `TSFMSummary.risk` contains one row per confidence level with `method="tsfm"`,
  `forecaster`, `target`, risk metrics, context counts, `n_clamped_paths`,
  `forecast_origin`, and simulated mean/std return.
- `TSFMSummary.scenarios` contains one row per simulated sample path
  (`sample_index`, `portfolio_return`, `loss`).
- `TSFMBacktest.forecasts` and `TSFMBacktest.coverage` match the
  `HistoricalSimulationBacktest` column structure and breach semantics
  (`realized_loss > var`), plus `forecaster`, `target`, `n_context_sessions`,
  `n_excluded_context_sessions`, and `n_clamped_paths`.

## Thesis comparison caveats

1. **Conditional vs. unconditional comparison:** A TSFM conditions on the
   recent `context_sessions` window ending at each `forecast_origin`, making
   it a conditional rolling forecaster. Compare it directly against rolling or
   weighted historical simulation in addition to fixed-window benchmarks.
2. **Release-date provenance:** `thuml/sundial-base-128m` was released in
   February 2025 (`arXiv:2502.00816`), so evaluation windows after February 2025
   are strictly post-release. `mistralai/Timestral-28M` was released in October
   2026 (`v1.0.0`), after the September 2026 end of the 4stocks dataset; its
   release date is recorded in `metadata()` for provenance. Never use test data
   for tuning `target`, `context_sessions`, or `num_samples`.
3. **1D portfolio scope:** Forecasting the portfolio series directly avoids
   assuming cross-asset independence or perfect correlation, but does not
   generate multivariate cross-stock paths for pairwise correlation or joint
   tail-loss diagnostics.
