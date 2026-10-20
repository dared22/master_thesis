# Reproducing the experiment results

Run both notebooks from fresh kernels using the existing project environment.
From the repository root, execute:

```bash
sandbox/.venv/bin/python sandbox/rerun_experiments.py
```

To rerun one notebook, pass its path:

```bash
sandbox/.venv/bin/python sandbox/rerun_experiments.py sandbox/likelihood_calc.ipynb
sandbox/.venv/bin/python sandbox/rerun_experiments.py sandbox/sde_test_data.ipynb
```

The [runner](../rerun_experiments.py) uses the Python interpreter that launches
it, instead of silently selecting a kernel registered for another project. It
clears old outputs, executes cells in order, saves outputs as execution
progresses, and stops on a cell error. A successful run sets
`metadata.experiment_execution.all_cells_completed` in the notebook.

The neural notebook reruns the original fit, all three data sweeps, the ten
candidate-configuration repeats, and five independent repeats of all three
sweeps. The new repeated sweeps add 85 fits. The original update budgets,
seeds, data splits, and observation grids are retained. It can take
substantially longer than the analytic notebook.

Section 15 adds six extended training runs: three configurations and two
data/training seed pairs, each continued to 6,000 updates. It retains nested
checkpoints at 1,500, 3,000, and 6,000 updates. The first checkpoint is checked
against the corresponding result in section 14. This isolates the effect of
the optimization budget while preserving each run's training sequence.

## Exported tables and provenance

The final cell of each notebook calls the
[artifact exporter](../experiment_artifacts.py). It writes full-precision CSV
tables and a `results.json` file to:

- [Analytic diffusion results](../experiment_results/likelihood_calc/results.json).
- [Neural SDE results](../experiment_results/sde_test_data/results.json).

Each JSON file contains the result tables, execution time, Python version,
installed package versions, random seeds, experiment settings, and source
hashes. Notebook hashes cover cell types and sources, excluding outputs and
execution metadata. These records identify the environment used for a result;
they do not guarantee bitwise agreement across platforms or library versions.

The analytic exports distinguish Brownian endpoint and increment estimates,
oscillator corrected totals, pooled increments, and endpoint estimates under
continuous and discrete Euler variances. Noise diagnostics record the exact
diffusion denominator used in each standardization.

The neural exports include the initial fit, endpoint summaries, observation
density, duration, trajectory count, and every candidate repeat. They also
include `repeated_sweeps.csv`, containing one row per new fit, and
`repeated_sweep_summary.csv`, containing the median and 25th/75th percentiles
for each configuration. Parameter errors are percentages; drift error uses a
uniform 512-point grid over one period. The initial fit uses the final model,
while the sweeps and candidate repeats use their best validation checkpoint.

Section 14 of the notebook controls the repeated sweeps. Each repeat uses a
new 50-path training pool and a new training seed, with seeds recorded in
the raw rows and JSON metadata. Configurations within a repeat share that
pool and training seed. Validation data and the validation sample stay fixed.
The plotted bands describe variation across fits conditional on this
validation set; they are not confidence intervals.

`convergence_budgets.csv` records current and best-validation model errors at
each retained budget. `convergence_history.csv` records validation loss every
100 updates, including step zero. The best-validation loss is nonincreasing
by construction, so its monotonicity alone does not establish convergence.
Inspect current validation loss and coefficient errors as well.

## Interpreting the rerun

A clean rerun resolves code/output mismatches. The five repeated sweeps give
an initial estimate of variability across training datasets and training
seeds. They do not establish optimizer convergence or a minimum sufficient
dataset size. See the remaining
[experiment points to resolve](experiment-points-to-resolve.md) before making
those stronger claims.
