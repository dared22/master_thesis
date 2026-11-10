# Historical simulation module specification

This document specifies a planned historical simulation benchmark for the
thesis. It is not an implemented module. You can reuse the same interface
for the four-stock experiment and other explicitly selected asset sets.

The module takes dated daily returns and a portfolio configuration. It
produces historical portfolio scenarios, value-at-risk (VaR), expected
shortfall (ES), and optional chronological backtest results. Its estimates
use observed returns directly and have no dependency on a neural model.

## Location and public interface

Place the shared module in `master_thesis/benchmarks/`, outside `sandbox/` and
experiments. Use NumPy and pandas on the CPU. Historical simulation requires
no neural network training or GPU.

| Planned path, relative to `master_thesis/` | Responsibility |
| --- | --- |
| `benchmarks/__init__.py` | Export `HistoricalSimulation` and `HistoricalSimulationConfig`. |
| `benchmarks/historical.py` | Configuration, input checks, historical scenarios, risk estimates, and chronological evaluation. |
| `benchmarks/test_historical.py` | Numerical examples, missing-session behavior, and checks against future information. |
| `benchmarks/docs/historical-simulation-spec.md` | This reference specification. |

Keep the first implementation in one substantive Python file. Add files
when an implemented benchmark creates a concrete reason to share logic.
The public interface accepts return tables for any supported number of
assets; it must not import from `4stocks_experiment` or hard-code stock
names, dates, sector definitions, or OSEBX membership.

Use this planned interface from the `master_thesis/` working directory:

```python
# Planned interface; the benchmarks package is not implemented yet.
import json
from pathlib import Path
from benchmarks import HistoricalSimulation, HistoricalSimulationConfig

config = HistoricalSimulationConfig(
    horizon_sessions=20,
    confidence_levels=(0.95, 0.99),
    estimation="fixed",
    rebalance="daily",
)
benchmark = HistoricalSimulation(config).fit(
    "sandbox/4stocks_experiment/data/train_returns.csv"
)
summary = benchmark.summarize()
validation = benchmark.backtest(
    "sandbox/4stocks_experiment/data/validation_returns.csv"
)

output = Path("sandbox/4stocks_experiment/results/historical_simulation")
output.mkdir(parents=True, exist_ok=True)
summary.risk.to_csv(output / "risk_summary.csv", index=False)
summary.scenarios.to_csv(output / "historical_scenarios.csv", index=False)
validation.forecasts.to_csv(output / "validation_forecasts.csv", index=False)
validation.coverage.to_csv(output / "validation_coverage.csv", index=False)
(output / "summary_metadata.json").write_text(
    json.dumps(summary.metadata, indent=2, allow_nan=False) + "\n"
)
(output / "validation_metadata.json").write_text(
    json.dumps(validation.metadata, indent=2, allow_nan=False) + "\n"
)
```

`fit` accepts a CSV path or a pandas DataFrame with the same columns and
returns the fitted object. `summarize` returns a result with `risk` and
`scenarios` DataFrames and a JSON-compatible `metadata` dictionary.
`backtest` returns `forecasts` and `coverage` DataFrames plus `metadata`.
The notebook saves those results and metadata in its experiment directory.
Importing or fitting the module must not write files.

## Input contract

Use the same unstandardized return CSVs as the current stock experiment.
Each row represents one market session and each asset column contains a
return for that session, calculated on the corrected session calendar.
Use `float64` for internal returns and risk calculations.

```csv
Date,Stock_A,Stock_B
2024-01-02,0.01,-0.02
2024-01-03,0.00,0.005
2024-01-04,,0.003
2024-01-05,-0.01,0.004
```

The contract has the following requirements:

- Require a `Date` column containing unique, increasing ISO calendar dates
  and at least one selected asset column. Reject duplicate or unordered
  dates instead of choosing a row or averaging observations.
- Interpret simple returns as fractions by default: `0.01` means 1%.
  Require numeric asset values, with empty cells allowed as missing.
  Reject infinite values and nonmissing simple returns below -1. A return
  of -1 represents a total loss and is valid.
- Retain genuine zero returns. Preserve missing rows while constructing
  scenarios; do not fill, interpolate, or join returns across them.
- Allow explicit `return_type="log"`; convert finite log returns with
  `expm1` before portfolio calculations. Reject nonfinite conversion results.
  Accept no standardized returns or implicit percentage-point conversion.
  The caller must declare the return convention correctly.
- Require compatible adjustment and currency conventions across assets.
  The caller supplies that economic consistency; column values alone cannot
  establish it.
- Require all genuine market sessions to remain present, including rows
  with missing returns. The module does not infer an exchange calendar from
  weekdays or detect a missing session when the entire date row is absent.

Select assets at fit time and retain their names and order throughout the
run. Match portfolio weights by column name. Evaluation data must contain
every selected asset; extra, unselected asset columns may be ignored.

`fit` stores validated daily returns and resolves the asset selection and
weights. Refitting replaces that state. Calling `summarize` or `backtest`
before fitting raises a clear error. Keep the first version deterministic;
exact empirical scenarios require no random seed or bootstrap sampling.

## Configuration

Expose the settings needed for the thesis comparisons, with their effective
values included in every result's metadata.

| Field | Default | Meaning and constraints |
| --- | --- | --- |
| `assets` | `None` | Use all asset columns, or an explicit nonempty tuple of distinct column names. |
| `weights` | `None` | Equal weights over selected assets, or a mapping from every selected asset name to its portfolio weight. |
| `return_type` | `"simple"` | `"simple"` or `"log"`; describes the input, not the output. |
| `horizon_sessions` | `1` | Positive integer number of market sessions in a scenario. |
| `rebalance` | `"daily"` | `"daily"` or `"buy_and_hold"`, as defined below. |
| `confidence_levels` | `(0.95, 0.99)` | Distinct finite confidence levels strictly between zero and one. |
| `estimation` | `"fixed"` | `"fixed"`, `"expanding"`, or `"rolling"` for chronological evaluation. |
| `lookback_sessions` | `None` | Raw session-row count. Required for rolling estimates; optional for fixed estimates; absent for expanding estimates. |
| `min_scenarios` | `100` | Minimum eligible historical blocks for publishing an estimate; integer at least two. |

Use long-only, fully invested portfolios in this version. Require finite,
nonnegative weights summing to one within a documented absolute tolerance
of `1e-10`. Do not silently normalize supplied weights, drop missing assets,
or redistribute their weights. Validate weight keys against selected assets.
Reject a positive lookback shorter than the horizon. For a valid lookback,
the available complete blocks can still be too few for `min_scenarios`.

These defaults are starting conventions, not evidence that 100 scenarios
provide reliable tail estimation. Confidence levels and portfolio design
belong to the experiment protocol. Tune eligible parameters on development
data, then freeze the configuration for the final test comparison.

## Historical scenarios and portfolio returns

Construct historical scenarios from intact, date-aligned vectors. Every
eligible scenario has equal probability; recency weighting is not part of
this first benchmark.

For horizon `H`, slide an H-row window through the selected history with
stride one. Apply the lookback to raw session rows before filtering missing
values. Exclude a block if any selected asset return is missing anywhere in
that block. Record both candidate and excluded block counts. Never construct
blocks by first dropping incomplete dates.

At `H=1`, this is ordinary historical simulation using observed joint daily
returns. At `H=20`, it uses historical 20-session blocks and preserves the
observed temporal order and stock co-movement inside each block. Overlapping
blocks share observations and are not independent samples.

Let `r[d, i]` be the simple return of asset `i` on day `d`, and let `w[i]`
be its configured weight at the start of a scenario. Define portfolio
returns exactly as follows:

```text
Daily rebalancing:
R = product_d(1 + sum_i(w[i] * r[d, i])) - 1

Buy and hold over the scenario:
R = sum_i(w[i] * (product_d(1 + r[d, i]) - 1))

Fractional portfolio loss:
L = -R
```

The two rules agree for a one-session horizon and can differ for longer
horizons. This version excludes transaction costs, short positions,
leverage, and time-varying portfolio weights. It does not approximate
multi-session risk by multiplying daily VaR by the square root of time.

## VaR and expected shortfall

Use a positive-loss convention: a portfolio return of -0.03 is a loss of
0.03. Report fractional horizon losses. Leave negative VaR or ES values
unchanged when the historical distribution implies gains even in the
specified tail. Monetary risk equals fractional risk times a separately
specified positive portfolio value.

For `n` eligible scenarios, sort their losses as
`L[1] <= ... <= L[n]`. At confidence `c`, use the inverse empirical CDF:

```text
VaR_c = L[ceil(n * c)]        # One-based indexing
```

This is the smallest loss whose empirical cumulative probability reaches
`c`. Fix this convention for the thesis rather than exposing a quantile
interpolation option. In particular, it is not the default linearly
interpolated percentile in some numerical libraries.

Define ES as the mean of the worst `1-c` probability mass of the empirical
loss distribution. Let `D[1] >= ... >= D[n]` be losses in descending order,
`m = n * (1-c)`, `k = floor(m)`, and `f = m-k`. Then

```text
ES_c = (sum(D[1:k]) + f * D[k+1]) / m
```

Here `sum(D[1:k])` includes exactly the first `k` losses and is zero when
`k=0`; the fractional term is zero when `f=0`. Handle floating-point values
close to an integer consistently. This convention includes fractional
boundary mass and remains defined when fewer than one full observation
occupies the target tail. Such a tiny tail is a limitation, not added data.

Do not replace this definition with the unqualified mean of every loss
greater than or equal to VaR: ties can include too much probability mass.
The distinction follows the quantile-integral definition of ES for discrete
distributions; see the primary reference below.

As a numerical acceptance example, losses `[-0.02, 0.01, 0.03, 0.10]` at
confidence `0.625` have VaR `0.03` and ES
`(0.10 + 0.5 * 0.03) / 1.5 = 0.0766666667`.

## Fixed summaries and chronological evaluation

Separate a single calibration summary from a sequence of estimates made
through time. They use the same historical simulation calculation.

`summarize()` builds scenarios from the fitted history, restricted to the
last `lookback_sessions` raw rows when configured, and returns one risk row
per confidence level. It raises an informative error if fewer than
`min_scenarios` complete blocks remain. The message includes the selected
assets, horizon, candidate count, excluded count, and eligible count.

`backtest(evaluation_returns)` requires evaluation dates strictly after the
last fitted-history date. It processes the combined historical and
evaluation session calendar in order. For a target starting on session `t`,
the available information ends on the previous session. Both returns and
scenario endpoints used for estimation must precede `t`.

| Estimation mode | Historical information used for a target starting at `t` |
| --- | --- |
| `fixed` | Only the original fitted history, with an optional fixed lookback applied to it. Estimates stay fixed through evaluation. |
| `expanding` | Original history plus evaluation rows strictly before `t`; no finite lookback. |
| `rolling` | The last configured number of raw session rows strictly before `t`. |

The realized comparison uses the actual H-session block starting at `t` and
the same weights and rebalancing rule as the historical scenarios. Include
only target starts with H calendar rows available in the evaluation input;
report the trailing excluded count. Do not use future realized returns to
construct the estimate or choose its parameters.

If a realized block contains missing returns, retain its forecast row and
mark the realization unavailable. If an estimation window has too few
scenarios, retain the target row with unavailable risk estimates and an
`insufficient_history` status. Do not silently remove difficult dates.

The input must include the intervening observed sessions. For example,
rolling evaluation on the test split needs the preceding validation history
as well as training history. The caller can concatenate the chronological
training and validation tables and pass that DataFrame to `fit` when the
experiment protocol permits updating. A fixed comparison with a generator
trained on training data alone must use that same training history alone.

For H greater than one, stride-one realized target blocks overlap. Report
their coverage descriptively; do not treat the resulting breach indicators
as independent Bernoulli observations.

## Output tables and provenance

Make the returned tables directly usable for comparisons across experiments.
Store numbers as fractions without rounding; format percentages in the
notebook or thesis table.

`summary.risk` contains one row per confidence level with these fields:

- `method="historical_simulation"`, confidence, horizon, and rebalance rule.
- Raw history start and end dates, session count, candidate block count,
  excluded incomplete block count, and eligible scenario count.
- Portfolio scenario mean return and sample standard deviation (`ddof=1`).
- `var`, `es`, and `tail_mass_observations = n_scenarios * (1-confidence)`.
- `small_tail_sample`, true when tail mass represents fewer than 20
  observations. This is a transparent descriptive flag, not an adequacy test.
- `overlapping_scenarios`, true for stride-one horizons greater than one.

`summary.scenarios` contains one row per eligible block: its original start
and end dates, the compounded portfolio return, and its loss. Preserve these
identifiers so an estimate can be traced back to the source returns.

`backtest.forecasts` contains one row per eligible target start and
confidence level. Include `forecast_origin`, `target_start`, `target_end`,
the actual estimation-history bounds, eligible scenario count, VaR, ES,
realized portfolio return and loss, and separate `estimate_status` and
`realization_status` fields. Define a breach as `realized_loss > var`;
equality is not a breach. Unavailable comparisons have a missing breach
value rather than `False`.

`backtest.coverage` contains the confidence level, horizon, number of target
starts, available estimates, valid realized comparisons, breach count,
breach rate, and nominal tail probability `1-confidence`. Formal VaR
coverage tests and ES backtests are outside the first implementation.
An average of realized VaR breaches is not an ES calibration test.

Both results' metadata dictionaries record the complete resolved
configuration, ordered assets and weights, input return convention,
fractional loss units, date bounds, source paths when available, a
fingerprint of the validated input tables, package versions, missing-block
policy, quantile convention, and ES convention. Record actual counts, not
only requested settings. Save metadata as JSON beside exported CSVs.

## Scope and comparison rules

Keep the implemented method limited to unweighted empirical historical
simulation. The configuration varies its portfolio, horizon, calibration
history, and evaluation protocol; it does not switch to a different model.

The same CSV interface supports any fixed selected asset universe with
adequate shared coverage. A changing OSEBX membership or time-varying index
portfolio requires dated membership and weights beyond a return-only CSV;
construct that portfolio outside this version or design an explicit future
extension. Never select surviving assets or reallocate weights implicitly.

For the four-stock SDE-GAN comparison, use the raw training CSV with
`return_type="simple"`, the same four assets and weights, the same horizon,
the same portfolio aggregation, and fixed estimation. Convert generated
SDE-GAN outputs to unstandardized simple returns before applying the same
portfolio and risk definitions in the comparison notebook. Historical
simulation must never read the generator's outputs.

Keep the empirical quantile and fractional-tail ES calculations as small
functions in `historical.py`, with their conventions documented and tested.
They can become shared risk-calculation functions when a second benchmark
is implemented; no generic benchmark base class or plugin system is needed
for this first version.

## Acceptance checks

Verify numerical and causal behavior with small known examples before
running the existing CSVs.

1. Reproduce the VaR/ES example above, including fractional tail mass,
   repeated loss values, confidence levels with less than one tail
   observation, all-zero returns, and gain-only scenarios. Check `ES >= VaR`
   within numerical tolerance.
2. Match weighted portfolio returns against hand calculations. Check that
   daily rebalancing and buy-and-hold agree for H=1 and differ correctly
   for a known H=2 example.
3. Show that an internal missing session invalidates every block crossing
   it. Retain zeros and verify that dropping incomplete rows would produce
   a different, rejected block construction.
4. Reject invalid dates, weights, asset names, return values, horizons,
   confidence levels, incompatible configuration, and premature evaluation
   dates. Match weights correctly when CSV columns are reordered.
5. Confirm that simple-return and equivalent log-return inputs give the
   same scenario and risk results.
6. Change evaluation observations at or after a target start and confirm
   that its forecast does not change. Verify that later expanding and
   rolling forecasts incorporate only observations available by then.
   Fixed estimates must remain unchanged throughout evaluation.
7. Compare CSV and equivalent DataFrame inputs. On the existing four-stock
   CSVs, verify reported scenario counts by independent session-window
   enumeration, for H=1 and H=20. No missing rows may disappear before that
   enumeration.

## References and related material

This specification implements the historical-simulation row of the existing
[portfolio risk benchmark overview](../../sandbox/research/portfolio-risk-benchmarks.md).
The quantile and discrete-tail definitions are grounded in
[Acerbi and Tasche, On the coherence of Expected Shortfall, definitions 2.2 and 2.6 and section 3](https://arxiv.org/pdf/cond-mat/0104295).
That paper uses a tail-probability notation; this specification consistently
uses confidence `c`, with tail probability `1-c`.

The module interface, default configurations, data checks, and diagnostic
threshold above are proposed thesis implementation choices, not empirical
findings or adequacy guarantees from that reference.
