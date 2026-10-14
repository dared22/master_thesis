# How much SDE training data is enough?

Use a two-dimensional learning curve: training trajectory count on the x-axis,
observations per training trajectory on the y-axis, and estimation error as the
color. A heatmap with an accuracy contour makes the acceptable data region
visible in one plot. This note inspects `sde_test_data.ipynb` and proposes an
experiment; no data-sufficiency sweep has been run.

## What the notebook currently does

The implemented experiment differs from some of its explanatory text:

- It generates 100 independent paths, each with 15,000 saved positions over
  `T=500`, with initial position zero and observation interval about 0.0333.
- It uses 40 paths for training and 60 for validation through `[:40]` and
  `[40:]`. Varying the global `n_trajectories` alone does not vary the training
  count once it exceeds 40.
- It learns a periodic drift through `sin(x)` and `cos(x)` and a constant
  positive diffusion through `raw_sigma`. The declared diffusion MLP is unused.
- Its increment loss is an Euler Gaussian likelihood approximation, with terms
  independent of the fitted model omitted. Its current plot measures optimizer
  progress for one dataset size, not accuracy versus dataset size.

Saved outputs report direct estimates `A=1.0057`, `sigma=1.0916`, and neural
summaries `A=1.0528`, `sigma=1.0927`. These describe one realization, not a
minimum sufficient sample size. Projecting the network onto the sine basis can
also conceal errors in the learned drift's shape.

## Decide what observations per trajectory means

These are distinct experimental questions:

| Design | What stays fixed | What increasing observations changes | Question |
| --- | --- | --- | --- |
| Fixed horizon | `T=500` | Smaller sampling interval | How densely must you observe each path? |
| Fixed interval | Observation interval `h` | Longer duration `T=(m-1)h` | How long must you observe each path? |
| Fixed increment budget | `n*(m-1)` | Allocation between independent paths and observations | Where do you spend a recording budget? |

For a first experiment, keep `T=500` fixed and vary both training path count
`n` and saved observations `m`. Do not assume that equal `n*(m-1)` means equal
information: dense measurements on the same path remain dependent.

There is a statistical reason to expect different behavior for drift and
diffusion. In classical stationary diffusion estimation, drift variance scales
with observation duration, while diffusion benefits more directly from dense
sampling. At a fixed duration, taking arbitrarily dense observations of one
path does not by itself make drift estimation consistent. These results are
established under assumptions that do not directly cover this nonstationary
real-line periodic process; use them as design guidance, not a sample-size
guarantee. See [Comte, Genon-Catalot, and Rozenholc (2007)](https://arxiv.org/abs/0708.4165),
especially the introduction and Sections 3–4 of the
[authors' full paper](https://helios2.mi.parisdescartes.fr/~comte/driftvol.pdf).
Multiple independent paths at a fixed horizon are explicitly studied in
[Drift estimation with denoising diffusion models (2026)](https://arxiv.org/html/2602.17830v1),
including sensitivity experiments for trajectory count and sampling interval.

## Choose a comparable score

Because you know the generating coefficients, coefficient recovery is a more
direct primary score than the increment likelihood. On a fixed uniform phase
grid `x_grid` spanning one period `[-pi, pi]`, define

```text
E_b = sqrt(mean((b_hat(x_grid) - b_true(x_grid))**2))
      / sqrt(mean(b_true(x_grid)**2))
E_s = abs(s_hat - 1.1) / 1.1
C   = max(E_b / epsilon_b, E_s / epsilon_s)
```

For example, predeclare `epsilon_b=0.05` and `epsilon_s=0.02` if 5% relative
drift RMS error and 2% diffusion error meet your intended use. These are example
requirements, not statistically justified universal cutoffs. Color the heatmap
by `C` and draw the `C=1` contour: both accuracy requirements pass exactly when
`C <= 1`. A full-period metric gives equal weight to rarely visited phases;
alternatively, use a fixed large test pool's phase distribution if accuracy
where the process spends time is your objective. Keep that weighting identical
for every cell.

Other usable metrics are held-out NLL, relative errors in direct estimates of
`A` and `sigma`, or distributional discrepancies at fixed prediction times.
The endpoint distribution alone can miss local coefficient errors. Keep any
held-out NLL on the same untouched test paths and the same fixed observation
interval for every fit. Changing the evaluation interval changes both the
prediction task and the drift sensitivity; even restoring the omitted Gaussian
normalizing terms does not resolve that comparison problem.

An optional nonnegative score is the average Gaussian transition KL over the
same phase grid, using a fixed reference interval `h_ref`:

```text
K = mean(log(s_hat / sigma)
         + (sigma**2 + h_ref*(b_true - b_hat)**2)/(2*s_hat**2)
         - 0.5)
```

This compares the true and fitted *Euler Gaussian approximations*; it is not
the exact finite-time SDE transition KL. Smaller `h_ref` downweights drift
errors, so the coefficient score is easier to interpret for this notebook.

## Run a controlled sweep

The following protocol separates data effects from numerical and optimizer
effects:

1. Start with training counts `[5, 10, 20, 40]` and observation counts
   `[501, 1001, 3001, 7501, 15000]`. Keep the other 60 existing paths untouched
   for evaluation. If selecting training steps or checkpoints, split validation
   paths from a separate test set instead of using test scores for selection.
2. Generate a fine master trajectory pool once per independent data replicate.
   Subset paths and subsample saved times to create each training dataset. Use
   deterministic, approximately evenly spaced integer indices including both
   endpoints. Compute actual intervals from the selected timestamps, which can
   be slightly nonuniform. Never linearly interpolate Brownian paths to make
   observations, and never shuffle individual increments to emulate independent
   paths.
3. Keep the fine numerical integration grid unchanged when changing observation
   counts. The current notebook derives `solver_ts` from `n_observations`, so
   rerunning it for each count changes the simulator too. Diffrax distinguishes
   saved output times from numerical integration; see its official
   [SaveAt documentation](https://docs.kidger.site/diffrax/api/saveat/).
4. Reinitialize both network and optimizer for every configuration. Give each
   fit enough updates to converge, using the same stopping rule and a generous
   maximum budget. A fixed 1,500-update limit can create an optimizer plateau
   that looks like a data plateau. Report budget and convergence diagnostics.
5. Repeat across at least five independently regenerated training pools, then
   increase repeats around the apparent threshold. Pair the nested subsets
   within each replicate to make comparisons less noisy. Changing network seeds
   or reselecting subsets of one pool measures conditional variability; it does
   not replace independent data replicates.
6. Fit the notebook's inexpensive direct `A` and `sigma` estimator alongside
   the neural model. If the direct estimator succeeds while the neural model
   struggles, more data may not be the immediate remedy.
7. Store one row per fit with `n_train`, `m`, duration, spacing, data seed,
   network seed, drift error, diffusion error, fixed-test NLL, optimizer updates,
   and runtime. Summarize each grid cell by median error and uncertainty.

Define enough data before examining the results: an accuracy tolerance with
acceptable repeat-to-repeat uncertainty, or a prespecified small improvement
when doubling the budget. A flat curve can reflect finite-interval bias,
insufficient phase coverage, solver error, or incomplete optimization; it does
not automatically establish sufficiency.

## Put both data axes in one plot

All these alternatives fit in one figure:

| Plot | Encoding | Best use |
| --- | --- | --- |
| Heatmap plus contours | x: training paths; y: observations; color: error | Recommended overview and sufficient-data boundary |
| Several learning curves | x: paths; y: error; one line per observation count | Few observation settings, with uncertainty bands |
| 3D surface | x: paths; y: observations; height/color: error | Interactive exploration; static perspective hides differences |
| Budget trade-off plot | x: total increments; y: error; color/marker: allocation | Comparing acquisition strategies at the same recording cost |

For the recommended heatmap, use logarithmic data axes and a sequential color
scale. Show cell values or retain the measured cells beneath interpolated
contours so interpolation does not imply untested accuracy. Overlay constant
`n*(m-1)` budget curves if acquisition cost matters, and mark cells whose upper
uncertainty estimate fails the tolerance. If you prefer a single coefficient
instead of the combined score, color by relative drift RMS and overlay a
diffusion-tolerance contour.
