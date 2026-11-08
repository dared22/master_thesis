# Source notes: neural SDEs and copulas for OSEBX research

These notes check the two linked sources and identify a feasible connection
between them. The proposed thesis uses a neural SDE to generate equity-return
paths and copulas to measure and benchmark cross-stock dependence. It does not
assume that the available OSEBX panel has the intraday depth of the published
stock experiment.

## What Kidger et al. did

[Kidger et al. (2021), *Neural SDEs as Infinite-Dimensional GANs*](https://arxiv.org/html/2102.03657v2)
train a continuous-time **generative** model for distributions of whole paths.
Its generator maps random initial state and Brownian motion through a latent SDE
whose drift and diffusion are neural networks. A neural controlled differential
equation (CDE) reads each real or generated path and acts as a discriminator.
The adversarial objective compares distributions of paths through a Wasserstein
GAN-style critic; it does not directly optimize next-day return accuracy,
portfolio performance, or value-at-risk. The paper's broad approximation claim
concerns the idealized infinite-data/global-optimum setting; it does not
guarantee successful finite-sample training or generalization for equities.
[Source: method and qualifications](https://arxiv.org/html/2102.03657v2#S3).

The paper evaluates a synthetic time-dependent Ornstein–Uhlenbeck process,
Google/Alphabet limit-order data, Beijing air quality, and neural-network
weight trajectories. Its stock study uses about a year of 2018–2019 LOBSTER
bid/ask observations, averaging 605,054 observations per day. After
downsampling, it uses roughly one-minute, 100-observation windows of
**midpoint and log spread**, totaling about 14.6 million data points. The
comparators are a latent ODE and a continuous-time flow process; the tests
assess discrimination of real versus synthetic paths, prediction using a
model trained on synthetic data, and signature-based maximum mean discrepancy.
The paper reports better scores on all three stock metrics for its neural SDE.
These are **sample-quality** results on a particular dataset, not evidence of
profitable forecasts or calibrated market risk.
[Source: stock experiment](https://arxiv.org/html/2102.03657v2#S4.SS2);
[source: stock implementation details](https://arxiv.org/html/2102.03657v2#Sx1.SS3).

The authors report practical difficulty with GAN training. Their setup used
gradient penalty for the critic, 5 critic updates per generator update, and
stochastic weight averaging. The published experiment used a single GPU and
could take up to a week. A [maintained `torchsde` example](https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py)
reproduces the **synthetic Ornstein–Uhlenbeck example**, rather than the stock
benchmark. Its current solver and training details differ from the paper, so
it is a starting implementation, not a literal stock-replication package.
[Source: training and compute details](https://arxiv.org/html/2102.03657v2#A1).

## What the copula article establishes

[Chang's introductory article](https://bochang.me/blog/posts/copula/)
explains how a joint distribution of continuously distributed returns can be
written as

\[F(x_1,\ldots,x_d)=C(F_1(x_1),\ldots,F_d(x_d)).\]

Here each \(F_i\) is a marginal distribution and \(C\) captures dependence.
The decomposition is Sklar's theorem, also stated and used in
[Demarta and McNeil's original t-copula research](https://www.planchet.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf),
Section 2.2. A Gaussian copula has zero **asymptotic** tail-dependence
coefficient for correlation below one; a finite-degree-of-freedom t copula
has positive upper and lower tail dependence. Zero asymptotic tail dependence
does **not** mean zero co-movement at a finite loss threshold.
[Source: Demarta and McNeil, Section 3.2](https://www.planchet.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf).

**Correction to the blog:** Its Section 5 prints rank-correlation integrals
without the required scale and offset. For continuous margins and copula
\(C\), the standard population formulas are

\[\rho_S=12\int_{[0,1]^2}uv\,dC(u,v)-3,\qquad
  \tau=4\int_{[0,1]^2}C(u,v)\,dC(u,v)-1.\]

The omitted terms can be checked immediately under independence
\(C(u,v)=uv\): both correct correlations equal zero. The blog's two
expressions instead yield \(-11/4\) and \(1/4\), respectively.
[Source: article's equations](https://bochang.me/blog/posts/copula/#5-rank-correlations);
[source: Demarta and McNeil, Section 3.1, for Kendall's tau](https://www.planchet.net/EXT/ISFA/1226.nsf/0/303eb11b4d617b79c1257b0800744575/$FILE/t%20copula%20demarta%20mcneil.pdf).
For Spearman's rho, the factor 12 follows directly because the transformed
uniform margins each have mean \(1/2\) and variance \(1/12\).

## Feasible connection to OSEBX data

The supplied file contains **daily** stock observations, with one bid and ask
per stock-day. The published one-minute microstructure experiment therefore
cannot be replicated literally. The
closest adaptation is a modest-dimensional panel of **daily adjusted
log-return paths** for consistently observed OSEBX stocks or sectors. The
model can generate scenarios. Compare generated versus observed marginal
returns, volatility clustering, cross-stock correlations, rank dependence,
joint downside-event frequencies, and out-of-sample portfolio loss quantiles.
These checks address whether paths are useful for risk analysis; any claim of
forecasting skill needs a separate, strictly chronological evaluation.

The cleanest copula integration is an **evaluation and benchmark layer**:
fit Gaussian and t copulas to training-period rank-transformed returns,
combine them with fitted univariate margins, and compare their joint-loss
scenarios with neural SDE scenarios on untouched later dates. Estimate the
same finite-threshold lower-tail co-exceedance rates from actual and simulated
returns. A neural SDE driven by Brownian motion does not *require* a separate
copula to generate a joint distribution; its multi-time-step outputs already
have one. Imposing a non-Gaussian copula on its driving noise would change the
stochastic model and is a larger, less direct research project.

For time-varying covariance risk, [Engle's dynamic conditional correlation
(DCC) model](https://doi.org/10.1198/073500102288618487) is another
established benchmark: it pairs univariate GARCH volatility estimates with a
parsimonious model for changing correlations. It answers a related, but
narrower, dependence question than full scenario generation. Start with a
historical-resampling baseline and the Gaussian and t copulas; add DCC if the
thesis explicitly studies whether changing dependence matters. A fair
comparison must hold portfolio weights, forecast horizon, and training dates
fixed across methods.

If the output is a one-day portfolio value-at-risk forecast, evaluate both
exception frequency and exception clustering. Those are distinct components
of [Christoffersen's conditional-coverage test](https://users.ssc.wisc.edu/~bhansen/718/Christoffersen1998.pdf).
If the output includes expected shortfall, a raw count of VaR exceptions does
not assess loss severity. [Acerbi and Székely (2014)](https://www.msci.com/resources/research/articles/2014/Research_Insight_Backtesting_Expecting_Shortfall_December_2014.pdf)
give nonparametric ES backtests. For ranking **pairs** of VaR and ES forecasts,
[Fissler and Ziegel (2016)](https://arxiv.org/abs/1503.08123) derive
jointly consistent scoring functions, under stated conditions. These tests
have limited power with few exceptions, so report event counts and
uncertainty alongside any p-values.

For any proposed comparison, use dates and securities that were known at the
time, construct features only from information available before the predicted
window, and evaluate on later dates. Interpret lower-tail estimates with care:
if a test set has \(n\) days, a 1% marginal threshold supplies only about
\(0.01n\) events for each stock on average, before conditioning on joint
events. Strong statements about asymptotic tail coefficients are unlikely to
be justified by a short daily panel.

## Audit of the available data

The following counts come from one of the three CSV exports referenced by the
local notebooks, not from the article. `data_comparison.ipynb` reports that
the exports contain the same rows in different orders; they must not be
concatenated. The file has 1,876,617 daily records from January 2, 1980,
through September 8, 2026. The `IN_OSEBX == 1` panel has 489,482 rows across
7,730 trading dates from January 2, 1996, onward. It contains 300 ISINs and
a median of 60 member rows per date. The membership panel is unbalanced:
`SecurityId` and `CompanyId` are missing on 13,078 member rows.

In member rows, `lnDeltaP` agrees with the consecutive log change in
`AdjustedPrice` for more than 99.9% of comparable same-security observations.
This is a numerical check, not a verified vendor definition. The largest
absolute `lnDeltaP` is 8.92, so corporate actions, stale quotes, and data
errors need review before tail-risk estimation. The alleged common index
return `lnDeltaOSEBX` differs across member rows on 3,199 dates, mainly before
2019. Sums of `OSEBXmktshare_prevmnth` across member rows have a median of
about 0.82 per day over the whole membership period, but roughly 1.0 from
2021 onward. Investigate these fields before treating them as a verified
index return or complete portfolio weights. `Equity`, `Debt`, and `Earnings`
are missing on approximately 13%, 22%, and 22% of member rows, respectively;
their publication dates also need verification before use as predictors.

These checks favor a small, transparently constructed sector or stock
portfolio for the first experiment. Use historical membership and lagged
positions, document any exclusions, and compare the constructed market
series with an independently verified OSEBX index before making index-level
claims.

## Thesis claim the sources can support

One bounded research question is: **Does a neural SDE generate daily OSEBX
equity-return scenarios whose dependence and portfolio downside risk are
better calibrated than simple Gaussian- and t-copula benchmarks?** The paper
supports the neural-SDE path-generation method; Demarta and McNeil support
the dependence benchmark. Whether the answer is yes is an empirical question
for the available panel. A narrower fallback is to compare the two copula
benchmarks with a low-dimensional neural SDE on sector portfolios, reducing
missing-data and training burdens while preserving the risk question.

Three possible thesis directions follow from the distinction between scenario
generation and prediction:

- **Path realism and downside dependence (recommended):** Fit a multivariate
  neural SDE to a small set of daily sector returns. Use copulas to benchmark
  and diagnose rank dependence, joint losses, and portfolio loss tails. This
  is closest to the paper's generative objective.
- **Conditional portfolio risk:** Add lagged market state to forecast
  next-day VaR and ES, and compare with historical simulation and a
  GARCH-based copula. This asks a forecasting question and therefore needs
  chronological retraining or explicitly conditional generation; the
  paper's unconditional generator alone is insufficient.
- **Regime-dependent co-movement:** Study how cross-sector dependence and
  downside diversification change across market periods, comparing a neural
  SDE with static or time-varying copulas. This requires careful regime
  definitions and sufficient tail observations in each period.
