# NSDE accuracy and data requirements

To estimate how much data your neural stochastic differential equation (NSDE)
needs, plot drift and diffusion errors separately against training dataset
size. Repeat the experiment and define acceptable accuracy before selecting a
minimum dataset size. Training loss alone cannot answer this question.

This note explains the metrics, a simple starting experiment, and options for
exploring trajectory count, duration, and observation spacing.

## Recommended metrics

Because you know the true coefficients in your synthetic experiment, measure
coefficient recovery directly. Your current model in
[`sde_test_data.ipynb`](../sde_test_data.ipynb) learns periodic drift and
constant positive diffusion through `raw_sigma`. The unused diffusion MLP has
been removed.

| Metric | What it measures | Role |
| --- | --- | --- |
| Relative drift RMSE | Error in the learned drift function | Primary |
| Relative diffusion error | Error in the constant noise amplitude | Primary |
| Held-out negative log-likelihood (NLL) | Fit to unseen increments | Supporting |
| Rollout statistics | Agreement in simulated behavior | Final validation |

On a fixed grid covering one spatial period, define

$$
E_b = \sqrt{\frac{\sum_j [\hat b(x_j)-b(x_j)]^2}{\sum_j b(x_j)^2}}.
$$

For constant diffusion amplitude, define

$$
E_\sigma = \frac{|\hat\sigma-\sigma|}{\sigma}.
$$

These normalizations require a nonzero reference drift norm and positive
reference diffusion. An error of 0.05 means 5% relative error under the stated
metric. Keep the evaluation grid identical across fits. A uniform grid measures
recovery across the whole period; weighting by a fixed test-state distribution
instead emphasizes frequently visited regions. State which objective you use.

If you later learn state-dependent diffusion, use a grid-based error for either
the amplitude `sigma(x)` or the variance coefficient `a(x)=sigma(x)^2`. State
which coefficient you evaluate.

## Why plot coefficient errors separately

Your likelihood uses the small-interval approximation

$$
\Delta X\mid X=x\approx
\mathcal N\bigl(b_\theta(x)\Delta t,\sigma_\theta(x)^2\Delta t\bigr).
$$

The likelihood couples drift and diffusion: the diffusion estimate determines
how strongly drift residuals are penalized. Splitting likelihood terms into a
“drift loss” and a “diffusion loss” does not give independent accuracy measures.
Plot the two coefficient errors separately instead.

Keep training and validation loss curves to diagnose optimization. Evaluate
held-out NLL on the same independent test trajectories and observation interval
for every model. At short intervals, NLL can be relatively insensitive to drift
error, so use it alongside coefficient errors.

For rollout validation, compare distributions or statistics such as variance,
autocorrelation, and displacement at fixed times. Independently simulated sample
paths need not match point by point, even for the correct model.

## The easiest initial experiment

Start by varying only the number of independent training trajectories. Keep
trajectory duration and observation spacing fixed.

1. Choose about six trajectory counts, roughly doubling each time.
2. Generate independent training pools and keep validation and test trajectories
   separate from training.
3. Within each pool, use nested trajectory subsets to compare dataset sizes.
4. Reinitialize the model and optimizer for every fit. Keep architecture and
   optimization settings consistent, allowing enough updates for convergence.
5. Repeat each size across 5–10 independent datasets and training seeds as an
   initial experiment. Add repeats near the apparent accuracy threshold.
6. Evaluate coefficient errors on the fixed grid and NLL on the fixed test set.
7. Plot median drift and diffusion errors against distinct training increments,
   with bands for variation across repeats and horizontal tolerance lines.

For `M` trajectories with `K` observations each,

$$
N_{\mathrm{increments}}=M(K-1).
$$

Use a logarithmic horizontal axis. Label uncertainty bands explicitly: an
interquartile range describes variation across fits, not a confidence interval
for the median. Changing initialization on one dataset does not measure the
variability from collecting new data.

Reusing increments in minibatches does not increase dataset size. Error versus
optimizer updates measures training progress; error versus distinct training
increments measures data requirements. Use validation data for checkpoint
selection and test data for final scoring.

## Defining enough data

Choose tolerances based on your intended use before inspecting the curves. One
illustrative requirement is drift error below 5% and diffusion error below 2%
in at least 90% of repeated fits. These are example requirements, not universal
accuracy standards.

A combined score can summarize both tolerances:

$$
C=\max\left(\frac{E_b}{0.05},\frac{E_\sigma}{0.02}\right).
$$

A fit passes both requirements exactly when `C <= 1`. Estimate the fraction of
fits that pass at each dataset size. Five to ten repeats give an initial
picture; estimating a 90% success probability reliably requires more repeats
and an uncertainty interval around the observed success fraction.

Report the smallest tested size that meets your requirement, together with its
uncertainty and acquisition settings. Keep the separate error plots to identify
which coefficient determines the data requirement.

A flattening curve alone does not establish adequate accuracy. It can reflect
insufficient optimization, model limitations, poor state coverage, or bias in
the finite-interval likelihood approximation.

## Why increment count is not the whole answer

The local increment approximation is

$$
\Delta X\approx b(X)\Delta t+sigma(X)\sqrt{\Delta t}\,Z,
\qquad Z\sim\mathcal N(0,1).
$$

As spacing decreases, the drift contribution shrinks faster than the random
contribution. Denser observations over a fixed time span can therefore help
diffusion estimation much more than drift estimation. More observation time
and broader state coverage can be especially valuable for drift recovery.

Classical ergodic diffusion estimation distinguishes rates associated with
observation duration and observation count. See
[Cheng, Hufnagel, and Masuda, Estimation of ergodic square-root diffusion under high-frequency sampling](https://arxiv.org/html/2103.15457).
This paper concerns a specific parametric diffusion: it supports the design
distinction, but does not provide a sample-complexity guarantee for your NSDE.

Your experiment options are:

| Change | Keep fixed | Question |
| --- | --- | --- |
| More independent trajectories | Duration and spacing | How much additional trajectory data do you need? |
| Longer trajectories | Observation spacing | Does more time improve drift and coverage? |
| Denser observations | Trajectory count and duration | Does finer sampling improve diffusion and reduce approximation bias? |
| Trajectory count and spacing | Duration | Which combination meets the accuracy requirement? |
| Allocation at equal increment count | Total recording budget | Which observed acquisition strategy performs best? |

Report trajectory count, duration, and spacing alongside increment count.
Equal increment counts need not contain equal information. Keep numerical
integration resolution fixed when changing saved observation times so that
simulator changes do not confound the comparison.

## Using the existing experiment helper

[`train_sde.py`](../train_sde.py) provides `train_nsde`, which initializes a
fresh model and optimizer for each fit, samples observed increments, and
returns the best validation checkpoint, losses, and dataset counts. The
notebook computes coefficient errors on a fixed grid and runs the data sweeps
and candidate repeats. The helper does not fit a direct parametric baseline
or estimate uncertainty across independently generated datasets.

The update budget is fixed and does not guarantee convergence. Validation
loss selects the checkpoint, including the untrained initial model as a
candidate. A selected step of zero means no evaluated trained checkpoint
improved on the initial validation loss.

See [reproducing the experiment results](reproducing-experiments.md) to rerun
the notebooks and inspect their exported tables. A minimum sufficient dataset
size still requires repeated configurations and checks of coefficient
accuracy and convergence.
