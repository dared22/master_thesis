# SDE-GAN specification

This specification defines the first joint generator for the four stock return
series. It describes planned implementation, not an implemented or trained
model. All new notebooks, code, data, and results belong in
`sandbox/4stocks_experiment/`.

The objective is to generate distributions of 20-session, four-stock return
paths and compare their behavior with observed data and copula benchmarks.
The first model generates unconditional scenarios from the training period's
mixture of market conditions. Date-specific risk forecasting requires a
separate conditioning and evaluation design.

## File responsibilities

Keep preprocessing in a notebook and model computation in three Python files.
Use the existing JAX, Equinox, Diffrax, and Optax environment. The notebook
uses the `SDEGAN` class as its model interface.

| Path within this directory | Responsibility |
| --- | --- |
| `adjusted_price_histories.ipynb` | Existing price audit, calendar correction, return calculation, and chronological CSV split. |
| `preprocessing.ipynb` (planned) | Load the CSVs, transform and standardize returns, extract windows, train the imported model, and inspect results. |
| `architecture.py` (planned) | Equinox generator and critic definitions, vector fields, and their forward numerical solves. |
| `training.py` (planned) | `SDEGANConfig`, losses, optimizer updates, fixed validation comparison, and checkpoint selection. |
| `sde_gan.py` (planned) | Public `SDEGAN` class: fit, sample, save, and load; re-export `SDEGANConfig`. |
| `data/` | Existing split CSVs; prepared arrays and preprocessing metadata. |
| `results/sde_gan/seed_<seed>/` | Model checkpoints, training history, and run settings. |

The model interface accepts prepared arrays. The notebook owns CSV parsing,
calendar rules, stock labels, dates, scaling, plots, and portfolio evaluation.
Importing these files must not start training or write files.

`architecture.py` has no imports from the other two files. Its constructors
receive dimensions and initialization keys directly. `training.py` imports
the architectures and defines the configuration containing the settings in
this specification. `sde_gan.py` imports both. This direction avoids circular
imports and keeps the configuration defined in one place.

Keep losses beside their optimizer updates in `training.py`. Keep generation
in the generator's forward solve; `SDEGAN.sample` batches that computation.
Serialization belongs with the public class. These operations do not need
additional files for the first implementation.

## Notebook-to-model data contract

Prepare the inputs in the following order so that gaps, volatility levels, and
the chronological split remain intact.

1. Load the three split CSVs with their dates. Preserve the same stock order:
   KONGSBERG GRUPPEN, FRONTLINE, WILH Wilhelmsen B, TOMRA SYSTEMS.
2. Convert each simple return `r` to log return `log1p(r)`. Reject any
   nonmissing simple return less than or equal to -1. Preserve missing values
   and genuine zero returns.
3. Estimate one mean and sample standard deviation per stock from the
   nonmissing training rows, before extracting overlapping windows. Require
   positive finite standard deviations.
4. Apply those training means and standard deviations to all three splits.
   Use `x = (log_return - training_mean) / training_std`. Apply no per-window
   scaling or clipping of extreme returns.
5. Extract windows of 20 consecutive rows on the corrected market-session
   calendar. Keep a window only when all 80 return entries are finite. Build
   windows separately inside each split, before removing incomplete rows.
6. Keep each window as a separate sequence. Save its start and end dates as
   metadata. Overlapping windows may be used in training; their overlap does
   not create independent trajectories.

The module receives `float32` arrays with shapes `(n_train, 20, 4)` and
`(n_validation, 20, 4)`. It must reject empty, incorrectly shaped, or nonfinite
inputs. Test data is not an argument to the training function.

Use observation times `0, 1, ..., 19`, with one time unit equal to one market
session. The first output is a random return observation, not an artificial
zero return. Keep calendar dates outside the model.

## Generator

Use one shared hidden SDE for all four stocks. These dimensions and numerical
settings are pilot choices to validate, not claims about optimal architecture.

Let `z(t)` have eight components and let `W(t)` be four-dimensional standard
Brownian motion. Define the Stratonovich SDE and its output by

```text
xi ~ Normal(0, I_4)
z(0) = initial_network(xi)
dz(t) = drift_network(t / 19, z(t)) dt
      + diffusion_network(t / 19, z(t)) o dW(t)
x(t) = readout(z(t))
```

Here `o dW` denotes Stratonovich integration. `x(t)` contains the four
standardized log returns. The random initial state is learned jointly with
the dynamics.

| Component | Input | Hidden layers | Output |
| --- | --- | --- | --- |
| Initial network | 4 Gaussian noise values | 32, 32 | 8 latent values |
| Drift network | Normalized time and 8 latent values | 32, 32 | 8 drift values |
| Diffusion network | Normalized time and 8 latent values | 32, 32 | 32 values reshaped to an 8-by-4 matrix |
| Readout | 8 latent values | None | Linear map to 4 returns |

Use `tanh` hidden activations. Apply a final `tanh` to the drift and diffusion
outputs; leave the initial network and readout outputs linear. The diffusion
matrix can have signed entries; covariance is formed implicitly through the
matrix times its transpose. Do not constrain the four stock outputs to be
independent or initialize diffusion identically to zero.

Use Diffrax's Stratonovich Heun solver with a constant step of 0.25 sessions
and save only at the 20 observation times. Use independent initial and
Brownian randomness for each generated window. Keep the same Brownian path
throughout each numerical solve. Use direct differentiation through the
discretized solve for this short-path pilot.

## Critic

Use a deterministic neural CDE that maps a complete window to one unrestricted
real score. Larger scores mean that the critic favors observed data.

Construct a piecewise linear control path with five channels:
`X(t) = [t / 19, x_1(t), ..., x_4(t)]`. Apply identical interpolation to real
and generated observations. Give the critic only the 20 sampled generator
outputs, so both inputs have the same temporal resolution.

The CDE has 16 hidden components and the following construction:

```text
h(0) = critic_initial(x(0))
dh(t) = critic_vector_field(h(t)) dX(t)
D(x) = critic_readout(h(19))
```

Use two 32-unit `tanh` hidden layers for both `critic_initial` and
`critic_vector_field`. The initial network maps 4 values to 16; the vector
field maps 16 values to 80, reshaped to a 16-by-5 matrix. Bound the vector
field output with `tanh`. Use a linear scalar readout without a sigmoid.
Solve the interpolated CDE with constant-step Heun at 0.25 sessions.

## Training objective and update rules

Use a Wasserstein GAN with a gradient penalty on the complete discretized
critic. This is an explicit pilot choice: Kidger also describes a carefully
constrained critic that avoids the penalty. The penalty costs extra
derivatives, so its gradients must pass the numerical checks below.

For observed windows `x`, generated windows `g`, and independent scalars
`alpha ~ Uniform(0, 1)`, minimize

```text
x_hat = alpha * x + (1 - alpha) * stop_gradient(g)
penalty = mean((norm(gradient_x_hat D(x_hat)) - 1)**2)
critic_loss = mean(D(stop_gradient(g))) - mean(D(x)) + 10 * penalty
generator_loss = -mean(D(g))
```

Use one `alpha` per window, shared across its times and stocks. The gradient
norm is the Euclidean norm over all 20-by-4 return values; the time channel
is fixed. Compute the penalty per window before taking the batch mean.
Differentiate it through the entire critic, including its initial network.

For each generator update, take five critic updates. Sample training windows
uniformly with replacement and use fresh generator randomness for every
update. Detach generator outputs during critic updates. During generator
updates, keep critic parameters fixed but differentiate its score through
the generated path. Neither player updates the other player's optimizer state.

Use the following initial training settings:

| Setting | Pilot value |
| --- | --- |
| Batch size | 8 windows, targeting the M2 Air with 8 GB RAM |
| Optimizer | Separate Adam optimizers for generator and critic |
| Learning rates | 0.0001 for both |
| Adam moments | `b1=0.0`, `b2=0.9` |
| Gradient penalty coefficient | 10 |
| Generator updates | 2,000 after a 50-update smoke run |
| Critic updates per generator update | 5 |
| Validation frequency | Every 100 generator updates, including step 0 |
| Initial repeated-run seeds | 0, 1, 2 |

Log generator loss, critic loss, gradient penalty, gradient norms, validation
score, and elapsed time. Stop with a clear error on nonfinite losses,
gradients, solver outputs, or parameters. A completed update budget does not
establish convergence.

Use CPU execution as the initial hardware target; the current environment
does not have the JAX Metal plugin installed. Before committing to 2,000
updates, measure time per complete generator update cycle, including its
five critic updates, after compilation has finished. Monitor process memory
and system memory pressure during the 50-update smoke run. Training memory
includes solver intermediates, derivative calculations, and compilation;
array storage alone is not a measure of peak memory. A full-run time or
memory requirement has not yet been measured.

Generate validation and requested sample paths in batches of eight, then
combine the completed paths on the host. Keep random keys tied to path
indices so batching does not change samples for a given seed. This avoids
requiring a single solve for all 256 validation or 1,000 requested paths.

## Validation and checkpoint selection

Choose checkpoints using a fixed comparison of generated and validation paths,
not the changing critic score alone.

Flatten each standardized window to 80 coordinates. Compute the biased
squared maximum mean discrepancy (MMD) using a Gaussian kernel
`k(a, b) = exp(-sum((a - b)**2) / (2 * bandwidth**2))`:

```text
MMD2 = mean(k(real, real)) + mean(k(fake, fake))
     - 2 * mean(k(real, fake))
```

Before training, fix the bandwidth to the median positive Euclidean distance
among distinct pairs in a seeded sample of up to 256 training windows.
Fail clearly if no positive distance exists. Save the bandwidth in the run
metadata; validation and test data must not determine it.

At every validation point, use the same seeded subset of up to 256 validation
windows and 256 generated windows from a fixed evaluation seed. Evaluate
without parameter updates. Save the generator with the lowest validation
MMD2, keeping the earlier checkpoint on a tie. This is a descriptive model
selection score, not a hypothesis test, especially with overlapping windows.

The notebook also checks marginal quantiles and volatility, Pearson and rank
correlations, autocorrelation of absolute returns, and joint 5% downside
frequencies. A good MMD score alone does not establish calibrated risk.
Freeze model choices before using the test split.

## Public interface and saved outputs

Keep the interface small. The following names and calls are planned, not
currently runnable APIs. The notebook imports from `sde_gan.py` only.

```python
from sde_gan import SDEGAN, SDEGANConfig

model = SDEGAN(SDEGANConfig())
history = model.fit(train_windows, validation_windows, seed=0)
generated = model.sample(n_paths=1000, seed=100)
# generated.shape == (1000, 20, 4), in standardized log-return units

model.save("results/sde_gan/seed_0")
restored = SDEGAN.load("results/sde_gan/seed_0")
```

`SDEGANConfig` records the architecture and training settings defined above.
Its definition lives in `training.py` and is re-exported by `sde_gan.py`.
`SDEGAN` is a normal Python class that owns the configuration and fitted
generator. The generator and critic remain Equinox modules; the wrapper is
not itself differentiated or passed into compiled optimizer steps.

`fit` initializes fresh generator and critic parameters and optimizer states,
then calls the training function in `training.py`. That function accepts the
initialized architectures, prepared arrays, configuration, and seed. It
returns the best generator, its step and validation score, the fixed kernel
bandwidth, the final generator and critic, and the logged history. It does
not write files. `fit` stores the best generator and run metadata on the
wrapper, retains the returned training result for inspection, and returns
the history. Calling `fit` again starts a fresh fit.

`sample` generates separate paths from the stored best generator. Sampling
or saving before fitting or loading must raise a clear error. Each sampling
call uses its explicit seed without changing hidden random state.

`save` writes `generator.eqx`, `config.json`, `run.json`, and `history.csv`
inside the requested directory. The run metadata includes the fit seed,
selected step, validation score, kernel bandwidth, and package versions.
`load` is a class method that reconstructs the architecture and restores
the selected generator, settings, and history for sampling. These are
inference checkpoints; restoring optimizer states for continued training
is outside this first implementation.

The notebook must also save stock order, training scaling statistics, split
date boundaries, and window length and stride alongside its prepared data.
A restored generator must reproduce samples with the same seed and
configuration in the same software environment.

Sampling returns standardized log returns. In the notebook, recover original
log returns with `generated * training_std + training_mean`, then recover
simple returns with `expm1`. Compute portfolio returns from the simple stock
returns. Adding or weighting individual log returns does not give the exact
simple return of a rebalanced portfolio.

## Acceptance checks before stock training

The implementation must pass checks of its numerical behavior and training
connections before an expensive fit.

1. Confirm that sampling returns finite arrays with shape `(n_paths, 20, 4)`.
   Equal seeds reproduce samples; different seeds produce different paths.
2. Verify the stochastic solve against a constant-coefficient SDE with known
   mean and covariance, allowing Monte Carlo uncertainty.
3. Verify finite, nonzero generator gradients through the sampled path and
   finite critic gradients through the full gradient penalty.
4. Verify that critic updates leave generator parameters fixed and generator
   updates leave critic parameters fixed.
5. Run 50 updates on synthetic correlated data. Confirm finite training and
   validation results and successful serialization and restoration.
6. Before interpreting fitted stock results, compare ensemble statistics at
   solver steps 0.25 and 0.125 using shared Brownian realizations. Investigate
   changes that exceed Monte Carlo uncertainty.

Model success is judged by held-out path and risk comparisons with the
benchmarks, not by assertions that neural SDEs must outperform them. A
continuous-output generator also does not explicitly model the point mass at
exactly zero observed returns; retain those observations and report this
limitation if ties materially affect results.

## Method references

The SDE generator and neural CDE critic follow the construction discussed in
[Kidger's thesis, sections 4.2 and 4.3.1](https://arxiv.org/html/2202.02435#Ch4.S3.SS1).
The full-critic penalty and its differentiation tradeoffs are discussed in
[section 4.4.3.3](https://arxiv.org/html/2202.02435#Ch4.S4.SS3.SSS3).
Network dimensions, training budgets, and the validation rule above are this
experiment's proposed choices.

The [authors' SDE-GAN example](https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py)
is a reference for the generator, critic, and sampling structure. Its
framework and critic regularization differ from this JAX pilot specification.
