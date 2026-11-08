# Source checks for theory citations

Checked on 2026-11-08 against the written theory in
`thesis/sections/02-theory/section.tex`. These notes record source locations,
not independent validation of the notebook results. Firecrawl failed with a
DNS error; the source checks used the available web reader instead.

## Ordinary differential equations

### Analysis background and Lipschitz continuity

[Springer's record for the 1980 first edition](https://link.springer.com/book/10.1007/978-1-4757-3971-8)
lists Kenneth A. Ross as the sole author. The supplied reference has therefore
been corrected to omit Jorge M. López. In the
[first-edition front matter](https://unina2.on-line.it/sebina/repository/catalogazione/documenti/Ross%20-%20Elementary%20analysis%20the%20theory%20of%20calculus.pdf),
Jorge Lopez appears in the acknowledgements.

The [text of a later printing of the first edition](https://studylib.net/doc/28035876/elementary-analysis--the-theory-of-calculus---kenneth-a-ross)
was checked for Section 17 (continuity) and the proof of Theorem 19.6.
That proof obtains a linear bound on function differences from a bounded
derivative using the mean value theorem. The preface states that pagination
changed from the twelfth printing, so the thesis uses section and theorem
locators instead of page numbers. Ross supports the analysis background and
bounded-derivative explanation; the ODE reference supports the state-dependent
local Lipschitz condition and uniqueness application. At the author's request,
the entire Lipschitz discussion has been removed from the thesis, including
the explanation, inequality, proof discussion, and worked example. The checks
above are retained as source notes only. Ross remains in the bibliography
source for future analysis references but is not currently cited in the text.

### ODE reference

The thesis author selected `kohler2014elementary`, Kohler and Johnson,
*Elementary Differential Equations with Boundary Value Problems* (2014),
as the source for standard ODE material. The thesis now uses that
supplied reference for ODE definitions, existence and uniqueness, the explicit
Euler update, convergence, and numerical stability. No page, theorem, or
section numbers from another book have been transferred to this reference.
The bibliography uses the edition, year, publisher, and ISBN supplied by the
thesis author; passages in that edition were not independently read.
The following Teschl checks document the earlier verification of the claims;
Teschl is no longer cited in the thesis or included in its bibliography source.

[Teschl, *Ordinary Differential Equations and Dynamical Systems*](https://radon.mat.univie.ac.at/~gerald/ftp/book-ode/ode.pdf)
provides the precise support for the current local existence paragraph:
Section 2.2, Theorem 2.2, assumes continuity and local Lipschitz continuity
in the state uniformly in time and concludes local existence and uniqueness.
Section 2.6, Theorem 2.17, gives extension under a linear-growth bound.
The [author's book page](https://radon.mat.univie.ac.at/~gerald/ftp/book-ode/)
identifies the published book as AMS, Graduate Studies in Mathematics 140,
Providence, 2012. The online PDF is an authorized preliminary version, so use
section and theorem numbers rather than assuming identical printed pages.

## Stochastic processes and observed paths

[Pishro-Nik, Section 10.1.0, Basic Concepts](https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php)
defines a stochastic process as time-indexed random variables and a sample
path as one realization. It uses stock prices as an example, distinguishes
continuous and discrete time, and explains sampling a continuous-time process
at discrete times. Its continuous-time counting-process example also shows
that continuous time does not imply continuous paths.
The [textbook homepage](https://www.probabilitycourse.com/) supplies its
recommended citation: Hossein Pishro-Nik, *Introduction to Probability,
Statistics, and Random Processes*, Kappa Research LLC, 2014. These passages
support the new stochastic-process subsection and the financial framing in
the theory introduction.

Brownian motion has been moved into that subsection, before the SDE definition.
The existing SDE interpretation and integral form retain Higham's citation.
Särkkä and Solin, Section 3.2, printed pp. 33–34, explains drift, stochastic
forcing, and the fact that different noise realizations produce different
solution paths. The observation equation assumes direct observations without
measurement noise, consistently with the later likelihood derivation.

## Stochastic calculus, likelihoods, and oscillator calculations

[Särkkä and Solin, *Applied Stochastic Differential Equations*](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf)
has a recommended citation on its title page: Cambridge University Press,
2019. Checked passages:

- Section 2.6, Algorithm 2.6 and the following paragraph, printed pp. 16–17:
  explicit Euler and global order one.
- Section 8.4, printed pp. 143–144: numerical instability of explicit methods
  and the step-size restriction for stiff equations.
- Section 4.2, Theorem 4.2: Itô formula.
- Section 4.5, printed pp. 54–55: strong/pathwise uniqueness and sufficient
  Lipschitz and linear-growth conditions.
- Section 5.4, printed pp. 67–68: Markov property and transition densities.
- Section 6.1, equations (6.2)–(6.7), printed pp. 77–79: Gaussian linear SDE
  solutions, means, covariance integrals, and transition densities. These
  support deriving the oscillator formulas; the thesis's particular
  trigonometric expressions are specializations, not quoted formulas.
- Section 11.4, equations (11.28)–(11.31), printed pp. 243–244: Euler Gaussian
  transition approximation, product likelihood, and negative log-likelihood.

Do not cite this book as evidence for numerical values from the notebooks.
The oscillator energy statement can cite the Itô formula followed by the
thesis's own calculation with energy
`E = (V^2 + omega_0^2 X^2)/2`.

## Brownian motion and Euler–Maruyama

[Higham (2001), *An Algorithmic Introduction to Numerical Simulation of
Stochastic Differential Equations*](https://epubs.siam.org/doi/10.1137/S0036144500378302)
was checked in the full publisher HTML. Section 2 defines Brownian motion;
Section 3 explains Itô integration; Section 4 gives the integral SDE,
nowhere differentiability of Brownian paths, and Euler–Maruyama in equation
(4.3). Section 5 treats strong order one-half and weak order one under
appropriate assumptions. Section 8 gives the stochastic chain rule. Retain
qualifications on convergence rather than making unconditional claims.

## Neural differential equations and sensitivities

[Chen et al. (2018), *Neural Ordinary Differential Equations*](https://papers.nips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)
supports the neural vector-field definition and residual/Euler connection
in Section 1. Section 2, equation (4), gives the continuous adjoint; Appendix
B provides derivations. The paper uses a row-vector convention, equivalent
to the transposed Jacobian form in the thesis.

[Kidger, *On Neural Differential Equations*](https://arxiv.org/pdf/2202.02435)
provides a broad account. Section 2.1.1, Theorem 2.1, assumes a uniform/global
Lipschitz bound; it does not directly state the current local-Lipschitz plus
growth formulation. Section 4.1 introduces neural SDEs. Section 5.1.2,
Theorem 5.2, gives the adjoint. Section 5.1.2.3 explains why differentiating
the numerical solver and numerically solving continuous adjoints can give
different gradients. The cover dates the thesis to Trinity 2021; the checked
arXiv posting is February 2022, consistent with the existing citation key.

## Statistical estimation and training

The [NIST/SEMATECH e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/)
provides the checked statistical definitions cited as `NISTHandbook`:

- [Section 1.3.6.5.2](https://itl.nist.gov/div898/handbook/eda/section3/eda3652.htm):
  maximum likelihood chooses parameters that maximize the sample likelihood.
- [Section 1.3.6.5.1](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3651.htm):
  method-of-moments estimation matches sample moments to model parameters.
- [Section 1.3.6.6.1](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm):
  the normal density with mean and scale parameters.

[Kingma and Ba, *Adam: A Method for Stochastic Optimization*](https://arxiv.org/pdf/1412.6980)
identifies itself as an ICLR 2015 conference paper. Section 2, Algorithm 1,
defines updates based on stochastic gradients and their moment estimates.
It supports the citation to Adam; the minibatch size and learning rate are
choices made in the thesis experiments.

The [SciPy fitting documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html)
confirms maximum likelihood as the default and `floc` as the fixed-location
argument. The existing `SciPy` citation credits the package, not the thesis's
numerical endpoint estimate.

## Thomas' Calculus

These sources do not verify the supplied 2003 Thomas' Calculus edition.
The thesis author explicitly authorized using `thomas2003thomas` for standard
Calculus 1 and Calculus 2 material without further passage checks. On that
basis, the thesis cites it for the fundamental theorem of calculus. ODE
definitions and Euler's method now cite the selected Kohler and Johnson book.
The [2003 book record](https://books.google.com/books?id=fctkQgAACAAJ)
matches the supplied authors, title, year, and publisher. Catalogues distinguish
the standard and early-transcendentals versions; no ISBN, edition number, or
page locator has been guessed. These citations rely on the author's assurance,
not an independent reading of this edition. Stochastic calculus and neural
differential equations retain their specialist sources. Sources for a method should accompany its first relevant
definition or statement; original experimental settings and results should
retain notebook or internal cross-references.
