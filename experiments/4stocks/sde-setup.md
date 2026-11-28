# Four-stock Neural SDE-GAN setup

This document explains how the four-stock Neural SDE-GAN baseline in
`experiments/4stocks/` works from end to end: what inputs go in, how the
generator and critic operate, how training and checkpoint selection work, why
specific penalties are turned on or off, and how synthetic paths become
portfolio risk estimates.

## Big picture

The model is an **unconditional synthetic market simulator** trained to
generate realistic **20-trading-day return windows** for four Oslo Børs stocks
moving together: `KONGSBERG GRUPPEN`, `FRONTLINE`, `WILH, WILHELMSEN A`, and
`TOMRA SYSTEMS`.

It is trained as a Wasserstein GAN with gradient penalty (WGAN-GP), which pairs
two continuous-time neural networks in an adversarial game:

- **Generator (Neural SDE):** Starts from random noise and evolves a latent
  stochastic process forward across 20 trading sessions to produce synthetic
  `(20, 4)` return windows.
- **Critic (Neural CDE):** Reads a complete `(20, 4)` return window smoothly
  from session 0 to session 19 and outputs a single scalar score measuring how
  realistic the window looks.

> **Note:** This setup is **unconditional**. At generation time, it does not
> condition on recent prices, realized volatility, volume, or macro indicators.
> Instead, it learns the unconditional distribution of 20-session return blocks
> from the training split and draws fresh synthetic 20-session scenarios from
> random noise (analogous to Fixed Historical Simulation, except the generator
> can synthesize new paths).

## Inputs and features

Data preparation lives in [scripts/prepare_data.py](scripts/prepare_data.py),
with saved split statistics in
[data/preprocessing.json](data/preprocessing.json).

| Input or feature | Where it enters | Why it is used |
| --- | --- | --- |
| **Standardized daily log returns** `(20, 4)` | Real windows fed to the critic; target output of the generator | Only daily returns enter the model (no raw prices, volume, macro series, or sector labels). Log returns $r_t = \log P^*_t - \log P^*_{t-1}$ are unbounded on $(-\infty, \infty)$, additive over time (matching SDEs), and guarantee positive prices when mapped back via $e^r - 1$. |
| **Training-only z-scoring** $\tilde{r} = (r - \mu_{\text{train}}) / \sigma_{\text{train}}$ | Applied per stock before windowing | Places all four stocks on a comparable numerical scale (mean $\approx 0$, std $\approx 1$) so a more volatile stock such as Frontline ($\sigma \approx 3.2\%$/day) does not dominate Kongsberg ($\sigma \approx 1.9\%$/day) inside `tanh` layers. Computing $\mu$ and $\sigma$ on the training split only prevents leakage from validation and test. |
| **20-session sliding windows (stride 1)** | Chronological 70% / 15% / 15% split: 1,737 train, 357 validation, 358 test windows | Matches the 20-session portfolio risk horizon (about one trading month). Only complete, finite windows are kept. |
| **Normalized time** $t / T \in [0, 1]$ | Appended internally inside both generator (`t / 20`) and critic (`t / 19`) | Lets the SDE represent time-varying drift and diffusion within a window and gives the critic an explicit clock along the path. |
| **Initial noise** $v \sim \mathcal{N}(0, I_4)$ | Input to the generator at $t = 0$ | Randomizes the starting latent state of each 20-session window. |
| **4D Brownian motion** $W_t$ | Drives the generator across $t \in [0, 20]$ | Supplies continuous random shocks across the 20 sessions (`VirtualBrownianTree`, tolerance `0.0625`). |

## Generator: Neural SDE

The generator is implemented in [model/architecture.py](model/architecture.py).
Rather than evolving the four stock returns directly, it evolves an
**8-dimensional latent state** $z_t \in \mathbb{R}^8$:

1. **Random initial state ($t = 0$):** Draw $v \sim \mathcal{N}(0, I_4)$ and
   map it through an MLP (`initial`) to obtain $z_0 \in \mathbb{R}^8$. Extra
   latent dimensions allow the SDE to carry hidden factors (such as shared
   market drivers or latent volatility states) beyond the four observed stocks.
2. **Stratonovich SDE evolution:**
   $$dz_t = f_\theta\!\left(\frac{t}{20},\, z_t\right) dt + g_\theta\!\left(\frac{t}{20},\, z_t\right) \circ dW_t$$
   - Both `drift` ($f_\theta \in \mathbb{R}^8$) and `diffusion`
     ($g_\theta \in \mathbb{R}^{8 \times 4}$) take the concatenated vector
     `[t / 20, z_t]` (size 9).
   - Each MLP has width `32`, depth `2`, `tanh` hidden activations, and a
     **final `tanh` activation**. Bounding drift and diffusion to $[-1, 1]$
     keeps numerical integration stable and prevents blow-ups.
   - Even though the four Brownian channels in $dW_t$ are independent,
     multiplying them by the state-dependent $8 \times 4$ `diffusion` matrix
     mixes the shocks to produce cross-stock correlation and state-dependent
     volatility.
   - The SDE is integrated with `diffrax.Heun()` (consistent with Stratonovich
     SDEs) using fixed step size `dt = 0.25` (four substeps per trading day)
     and `DirectAdjoint()` backpropagation.
3. **The `"increments"` readout:** In the baseline configuration
   ([scripts/train.py](scripts/train.py)), `generator_output="increments"`:
   - The solver integrates from $t = 0$ to $t = 20$, saving **21** latent
     states $z_0, z_1, \dots, z_{20}$.
   - A bias-free linear `readout` maps each $z_t \in \mathbb{R}^8$ to
     $y_t \in \mathbb{R}^4$, which acts as a standardized cumulative log-price
     path.
   - Taking day-to-day differences $\hat{r}_t = y_t - y_{t-1}$ for
     $t = 1, \dots, 20$ produces the `(20, 4)` window of daily standardized
     log returns.

Modeling cumulative paths and taking daily differences matches standard
financial SDEs (where log-prices evolve continuously and returns are
increments) and avoids the artificial day-to-day persistence that occurs when a
continuous SDE outputs daily return levels directly.

## Critic: Neural CDE

The critic is implemented in [model/architecture.py](model/architecture.py) as a
**Neural Controlled Differential Equation (Neural CDE)**, which behaves like a
continuous-time recurrent network reading the full 20-day trajectory:

1. **Continuous path construction:** Prepend normalized time $t / 19$ to the
   `(20, 4)` return window and linearly interpolate between days to build a
   5-channel continuous control path $X(t) \in \mathbb{R}^5$.
2. **Hidden state initialization:** Map day 0's four returns through `initial`
   MLP to obtain $h_0 \in \mathbb{R}^{16}$.
3. **Controlled evolution:** Integrate
   $$dh_t = F_\phi(h_t)\, dX_t$$
   from $t = 0$ to $t = 19$ with `diffrax.Heun()` at `dt = 0.25`, where
   $F_\phi(h_t) \in \mathbb{R}^{16 \times 5}$ is a width-`32`, depth-`2` `tanh`
   MLP (with final `tanh`).
4. **Scalar readout:** Map the final hidden state $h_{19} \in \mathbb{R}^{16}$
   through a linear layer to an unrestricted scalar score (higher means the
   critic scores the window as more realistic).

## Training objective and stabilizers

Training logic is defined in [model/training.py](model/training.py) and
configured for the baseline in [scripts/train.py](scripts/train.py). Each seed
runs for **2,000 generator updates** with `batch_size = 64`:

1. **Critic updates (5 steps per generator step):** Minimize the WGAN-GP
   objective:
   $$\mathcal{L}_D = \mathbb{E}[D(\text{fake})] - \mathbb{E}[D(\text{real})] + 10 \cdot \mathbb{E}\!\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right]$$
   where $\hat{x} = \alpha x_{\text{real}} + (1 - \alpha) x_{\text{fake}}$ is a
   random convex combination of real and detached fake windows. The gradient
   penalty (`gradient_penalty_weight = 10.0`) enforces the 1-Lipschitz
   constraint so critic gradients remain well behaved.
2. **Generator update (1 step):** Minimize the adversarial loss plus an
   unbiased mean-anchoring penalty:
   $$\mathcal{L}_G = -\mathbb{E}[D(\text{fake})] + 10 \cdot \text{MeanPenalty}(\text{fake}, \mu_{\text{target}})$$
   - **Why mean anchoring (`mean_penalty_weight = 10.0`)?** Without it,
     different seeds drifted in their average daily return; small daily mean
     offsets compound over 20 sessions and noticeably shift portfolio VaR.
   - **Why the cross-path estimator?** A naive squared batch mean
     $(\bar{x} - \mu)^2$ has expectation $\text{bias}^2 + \text{variance}/B$,
     which subtly rewards shrinking variance. Using cross-path products removes
     the variance term, giving an unbiased estimate of squared mean bias.
3. **Training stabilizers:**
   - **Adam** with learning rate `1e-4`, `adam_b1 = 0.0`, and `adam_b2 = 0.9`
     (standard WGAN-GP momentum settings).
   - **Cosine learning-rate decay** (`lr_schedule = "cosine"`) decaying down to
     10% of the initial rate.
   - **Global gradient norm clipping** (`grad_clip_norm = 10.0`) to curb
     occasional SDE/CDE gradient spikes.
   - **Exponential moving average (EMA)** of generator weights
     (`ema_decay = 0.99`, with warmup $1 - 1 / (\text{step} + 1)$). The
     smoothed EMA generator is what gets validated and saved.

## Why ACF and path-signature penalties are off in the baseline

The training code in [model/training.py](model/training.py) also implements
optional signed-return ACF (`temporal_penalty_weight`), absolute-return ACF
(`absolute_temporal_penalty_weight`), and depth-2 lead-lag path signature
(`signature_loss_weight`) losses. All three are set to `0.0` in the current
baseline based on earlier ablations documented in
[../research/sde-gan-experiments.md](../research/sde-gan-experiments.md) and
[../research/sde-gan-ablation-results.md](../research/sde-gan-ablation-results.md):

| Earlier experiment | Signed ACF RMSE (seed 0) | Key takeaway |
| --- | ---: | --- |
| Original `"returns"` generator (no penalties) | `0.619` | Severe day-to-day persistence because continuous SDE states change slowly. |
| `"returns"` + signed ACF penalty (`weight = 10`) | `0.348` | Cut error nearly in half, but signed returns were still far too persistent. |
| `"returns"` + signed ACF penalty (`weight = 30`) | *(collapsed)* | Lowered ACF late in training only by collapsing volatility; MMD picked step 200. |
| `"returns"` + depth-2 signature (`1`) + signed ACF (`10`) | `0.344` | Improved signature and cross-stock fit, but signed ACF stayed stuck near `0.344`. |
| Switch to `"increments"` (ACF & signature weights = `0`) | **`0.045`** | Fixed the persistence problem structurally without extra ACF penalties. |

On the current 10-year panel
([results/evaluation/distribution_metrics.csv](results/evaluation/distribution_metrics.csv)),
the `"increments"` baseline with ACF and signature losses disabled achieves
signed ACF RMSE between `0.038` and `0.083` across seeds 0–2.

After `"increments"` resolved signed ACF, **absolute-return ACF** (volatility
clustering) remained too weak (RMSE around `0.12`–`0.15`). A follow-up pilot
adding `absolute_temporal_penalty_weight = 1.0` on top of `"increments"`
reduced absolute-ACF RMSE by only ~3.3% (`0.1208` to `0.1167`) while slightly
worsening validation 95% and 99% quantile loss, so it was kept at `0.0`.

> **Note:** Two caveats apply to these ablations:
> `absolute_temporal_penalty_weight = 1.0` was a pilot weight rather than a
> full sweep, and depth-2 `signature_loss` was only tested on the older
> `"returns"` generator, not re-tested on top of the stabilized `"increments"`
> baseline.

## Checkpoint selection and portfolio risk evaluation

Model selection happens during training in [model/training.py](model/training.py)
without touching the test split, and final risk scoring runs in
[scripts/evaluate.py](scripts/evaluate.py):

1. **Validation checkpoint selection:** Every `100` generator steps, the EMA
   generator samples `1,000` synthetic paths (using fixed
   `evaluation_seed = 100`) and compares them against all `357` validation
   windows using **multi-scale Gaussian-kernel $\text{MMD}^2$** (averaged over
   bandwidth multipliers `0.5`, `1.0`, and `2.0` times the training median
   pairwise distance). The checkpoint with the lowest validation $\text{MMD}^2$
   is saved as the selected model via [model/sde_gan.py](model/sde_gan.py).
2. **Sampling and unit conversion:** At evaluation time, each selected seed
   generates `10,000` synthetic `(20, 4)` standardized log-return windows,
   un-standardizes them ($r = \tilde{r} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$),
   and converts daily log returns to simple returns ($R = e^r - 1$).
3. **Portfolio risk metrics:** Simple returns are combined into an
   equal-weight (25% per stock), daily-rebalanced 20-session portfolio:
   $$R_{\text{port}} = \prod_{t=1}^{20}\left(1 + \frac{1}{4}\sum_{i=1}^{4} R_{i,t}\right) - 1$$
   Empirical 95% and 99% Value-at-Risk (VaR) and Expected Shortfall (ES) are
   computed on the simulated loss distribution $-R_{\text{port}}$ and scored
   against realized validation (or reserved test) losses alongside Fixed
   Historical Simulation, Rolling 252-session Historical Simulation, and TSFM
   benchmarks.
4. **Solver sensitivity check:** Each seed is also re-sampled with a finer
   step size (`dt = 0.125` vs. `0.25`) on 2,000 paths to verify that numerical
   discretization error does not materially distort VaR or ES.

## End-to-end flow summary

```text
Random noise (4D initial noise + 4D Brownian motion)
   │
   ▼
Generator Neural SDE (8D hidden state; drift & diffusion MLPs take [t/20, z_t]; Heun dt=0.25)
   │  linear readout (8 → 4) at t = 0..20, then day-to-day differences
   ▼
Fake window: (20, 4) standardized log returns ◄─────────┐
   │                                                    │ compared with real (20, 4) windows
   ▼                                                    │
Critic Neural CDE (16D hidden state; reads [t/19, r_t]; Heun dt=0.25) → scalar score
   │
   ▼
Train with WGAN-GP (GP=10) + unbiased Mean Penalty (10) + Cosine LR + Grad Clip (10) + EMA (0.99)
   │  select lowest multi-scale validation MMD² checkpoint
   ▼
Sample 10,000 paths → un-standardize → expm1 → 20-session portfolio VaR & ES
```
