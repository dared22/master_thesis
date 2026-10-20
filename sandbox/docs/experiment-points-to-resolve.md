# Experiment points to resolve

The remaining items require additional experiments or scientific analysis.
Notebook syntax errors, mixed duration variables, stale outputs, inaccurate
model descriptions, and ambiguous noise standardization have been corrected.
The thesis uses the rerun results, including the direct comparison between
continuous and Euler oscillator endpoint variances.

See [reproducing the experiment results](reproducing-experiments.md) for the
execution procedure and exported tables from
[`likelihood_calc.ipynb`](../likelihood_calc.ipynb) and
[`sde_test_data.ipynb`](../sde_test_data.ipynb).

## Resolve remaining optimization sensitivity

The notebook now supplements the original single-run sweeps with five repeats
using new training pools and training seeds. Medians and 25th/75th percentiles
describe their combined variability, conditional on the fixed validation set.
Five repeats provide an initial assessment of repeatability. Section 15 now
compares nested budgets of 1,500, 3,000, and 6,000 updates for three
configurations and two seed pairs. Their 1,500-update results agree with the
corresponding repeated-sweep fits.

The checks do not establish universal convergence. The shorter-duration case
shows substantial drift-error changes despite small validation-loss changes,
and both coarse-observation checks still select the initial model after
6,000 updates. Current iterates also differ from the best-validation models.

- Investigate whether a smaller or decaying learning rate stabilizes drift
  recovery, and define an acceptable stability criterion before declaring
  an update budget sufficient.
- Increase the number of repeats if stable tail estimates or precise
  uncertainty estimates are needed. The existing interquartile bands are
  descriptive, rather than confidence intervals.
- Compare analytic estimators across repeated datasets with a stated
  observation budget before claiming a general accuracy ranking.

The thesis reports the observed diffusion trends and nonmonotonic drift
results. It does not infer that additional trajectories are intrinsically
unhelpful for drift recovery.

## Diagnose the cause of coarse-sampling failure

With 500 observations over `T = 500`, the original sweep selects the untrained
model.
The selected diffusion remains at its initial value of 0.9. Large observation
intervals can invalidate the local Gaussian approximation, but the current
experiment does not separate that effect from optimization and checkpoint
selection.

Two extended runs retain that initial checkpoint through 6,000 updates, so
increasing the budget alone has not resolved their selected-model failure.
Compare a more accurate transition model or alternative optimization settings.
A direct parametric
fit can help distinguish neural optimization from likelihood approximation.
The thesis records the selected checkpoint without asserting a single cause.

## Establish a reliable minimum dataset size

The candidate configuration has ten repeats with specified drift and diffusion
error thresholds. These repeats vary the training data and training
randomness together and reuse one validation set. Their observed pass rate is
a descriptive result, rather than a precise population success probability.

Before claiming a minimum sufficient dataset size, define the required
success probability, compare several configurations with enough repeats, and
quantify uncertainty in their success rates. Separate data and optimization
seeds if their individual effects matter. Use an independent final test set
for predictive performance claims after validation-based selection.

## Evaluate rollout generalization and density accuracy

The original and learned ensembles use the same Brownian keys, and the
original ensemble includes training trajectories. This paired design is now
stated in the notebook and thesis, but it does not provide a purely held-out
distributional evaluation.

For a stronger generalization claim, generate new evaluation ensembles and
state whether their forcing is paired or independent. Add uncertainty
estimates and a quantitative distributional discrepancy. Assess the numerical
Fokker--Planck solver's spatial and temporal convergence before claiming
accuracy of its density curves. The thesis currently reports endpoint
summaries without asserting equality of laws or complete temporal recovery.
