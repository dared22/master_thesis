"""Static and GARCH-filtered Gaussian and Student-t copula portfolio benchmarks.

Models multivariate daily log returns, simulates joint H-session paths, converts
them to simple returns via expm1, and computes empirical portfolio VaR and ES
using the shared conventions in benchmarks._common.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats

from ._common import (
    ES_INTEGER_TOLERANCE,
    WEIGHT_TOLERANCE,
    ReturnInput,
    _bounds,
    _coverage_table,
    _dated_table,
    _empirical_es,
    _empirical_var,
    _load,
    _portfolio_returns,
    _resolve_weights,
    _tail_mass,
    _validate_assets,
    _validate_choice,
    _validate_confidence_levels,
    _validate_int,
    _validate_weights,
)

DF_MIN = 2.05
DF_MAX = 200.0
PERSISTENCE_MAX = 0.9995
PROB_CLIP = 1e-12
MIN_EIGENVALUE = 1e-6
RELATIVE_VAR_FLOOR = 1e-12
DEGENERATE_VAR_FLOOR = 1e-30
HIGHAM_MAX_ITER = 200
HIGHAM_TOL = 1e-10


@dataclass(frozen=True)
class CopulaConfig:
    """Configuration for static and GARCH-filtered multivariate copula risk models.

    Daily returns are modeled internally in log-return space and converted back
    to simple returns with expm1 before portfolio compounding. Weights are never
    silently normalized.
    """

    assets: tuple[str, ...] | None = None
    weights: Mapping[str, float] | None = None
    return_type: str = 'simple'
    horizon_sessions: int = 1
    rebalance: str = 'daily'
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    copula: str = 't'
    marginal: str = 'garch_t'
    pit: str = 'empirical'
    estimation: str = 'fixed'
    lookback_sessions: int | None = None
    refit_every: int = 1
    min_observations: int = 60
    num_samples: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        for name, choices in (
            ('return_type', ('simple', 'log')),
            ('rebalance', ('daily', 'buy_and_hold')),
            ('copula', ('gaussian', 't')),
            ('marginal', ('static_empirical', 'static_t', 'garch_normal', 'garch_t')),
            ('pit', ('empirical', 'parametric')),
            ('estimation', ('fixed', 'expanding', 'rolling')),
        ):
            _validate_choice(name, getattr(self, name), choices)
        if self.marginal == 'static_empirical' and self.pit != 'empirical':
            raise ValueError("marginal='static_empirical' requires pit='empirical'")
        for name, minimum, allow_none in (
            ('horizon_sessions', 1, False),
            ('lookback_sessions', 2, True),
            ('refit_every', 1, False),
            ('min_observations', 10, False),
            ('num_samples', 2, False),
            ('seed', 0, False),
        ):
            object.__setattr__(
                self, name, _validate_int(name, getattr(self, name), minimum, allow_none=allow_none)
            )
        if self.estimation == 'rolling' and self.lookback_sessions is None:
            raise ValueError('rolling estimation requires lookback_sessions')
        if self.estimation == 'expanding' and self.lookback_sessions is not None:
            raise ValueError('expanding estimation requires lookback_sessions=None')
        if self.lookback_sessions is not None and self.lookback_sessions < self.min_observations:
            raise ValueError('lookback_sessions must be at least min_observations')
        _validate_assets(self.assets)
        object.__setattr__(
            self, 'confidence_levels', _validate_confidence_levels(self.confidence_levels)
        )
        object.__setattr__(self, 'weights', _validate_weights(self.weights, self.assets))


@dataclass(frozen=True)
class _FittedMarginal:
    """Fitted univariate marginal parameters and filtered innovation state."""

    kind: str
    mu: float
    omega: float | None
    alpha: float | None
    beta: float | None
    scale: float
    df: float | None
    unconditional_var: float
    last_var: float
    last_eps: float
    var_floor: float
    sorted_innovations: np.ndarray
    degenerate: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'mu': float(self.mu),
            'omega': None if self.omega is None else float(self.omega),
            'alpha': None if self.alpha is None else float(self.alpha),
            'beta': None if self.beta is None else float(self.beta),
            'scale': float(self.scale),
            'df': None if self.df is None else float(self.df),
            'unconditional_var': float(self.unconditional_var),
            'degenerate': bool(self.degenerate),
        }


@dataclass(frozen=True)
class _FittedCopulaModel:
    """Complete fitted marginal + copula state for simulation."""

    marginals: tuple[_FittedMarginal, ...]
    copula_kind: str
    pit_kind: str
    corr: np.ndarray
    cholesky: np.ndarray
    copula_df: float | None
    n_fit_observations: int

    def to_metadata(self, assets: tuple[str, ...]) -> dict[str, Any]:
        return {
            'copula': self.copula_kind,
            'pit': self.pit_kind,
            'copula_df': None if self.copula_df is None else float(self.copula_df),
            'copula_corr': [[float(v) for v in row] for row in self.corr],
            'n_fit_observations': int(self.n_fit_observations),
            'marginals': {
                asset: marginal.to_metadata()
                for asset, marginal in zip(assets, self.marginals)
            },
        }


@dataclass(frozen=True)
class CopulaSummary:
    """Single-origin copula risk summary, simulated scenarios, joint paths, and provenance."""

    risk: pd.DataFrame
    scenarios: pd.DataFrame
    metadata: dict[str, Any]
    paths: np.ndarray


@dataclass(frozen=True)
class CopulaBacktest:
    """Chronological copula forecasts and descriptive coverage; no independence tests."""

    forecasts: pd.DataFrame
    coverage: pd.DataFrame
    metadata: dict[str, Any]


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _std_t_cdf(z: np.ndarray, df: float) -> np.ndarray:
    scale = np.sqrt((df - 2.0) / df)
    return special.stdtr(df, z / scale)


def _std_t_ppf(u: np.ndarray, df: float) -> np.ndarray:
    scale = np.sqrt((df - 2.0) / df)
    return scale * special.stdtrit(df, u)


def _garch_variances(
    eps2: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_var: float,
    var_floor: float = RELATIVE_VAR_FLOOR,
) -> np.ndarray:
    n = len(eps2)
    variances = np.empty(n, dtype=np.float64)
    floor = max(float(var_floor), np.finfo(np.float64).tiny)
    current = max(float(init_var), floor)
    for idx in range(n):
        variances[idx] = current
        current = omega + alpha * eps2[idx] + beta * current
        if current < floor:
            current = floor
    return variances


def _fit_garch_univariate(series: np.ndarray, kind: str) -> tuple[_FittedMarginal, np.ndarray]:
    """Fit GARCH(1,1) with normal or standardized Student-t innovations by MLE."""
    mean_r = float(np.mean(series))
    centered = series - mean_r
    sample_var = float(np.var(centered, ddof=1)) if len(series) > 1 else 0.0
    if sample_var <= DEGENERATE_VAR_FLOOR:
        zeros = np.zeros_like(series, dtype=np.float64)
        marginal = _FittedMarginal(
            kind=kind,
            mu=mean_r,
            omega=0.0,
            alpha=0.0,
            beta=0.0,
            scale=0.0,
            df=None,
            unconditional_var=0.0,
            last_var=0.0,
            last_eps=0.0,
            var_floor=0.0,
            sorted_innovations=zeros,
            degenerate=True,
        )
        return marginal, zeros

    std_r = float(np.sqrt(sample_var))
    y = centered / std_r
    init_var = 1.0
    is_t = kind == 'garch_t'

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float | None]:
        mu_s = float(theta[0])
        omega_s = float(np.exp(theta[1]))
        persistence = PERSISTENCE_MAX * _sigmoid(float(theta[2]))
        arch_share = _sigmoid(float(theta[3]))
        alpha = persistence * arch_share
        beta = persistence * (1.0 - arch_share)
        df = float(DF_MIN + np.exp(theta[4])) if is_t else None
        return mu_s, omega_s, alpha, beta, df

    def neg_log_lik(theta: np.ndarray) -> float:
        mu_s, omega_s, alpha, beta, df = unpack(theta)
        eps = y - mu_s
        eps2 = eps * eps
        sigma2 = _garch_variances(eps2, omega_s, alpha, beta, init_var, RELATIVE_VAR_FLOOR)
        if not np.isfinite(sigma2).all() or np.any(sigma2 <= 0.0):
            return 1e12
        if is_t:
            assert df is not None
            const = (
                special.gammaln(0.5 * (df + 1.0))
                - special.gammaln(0.5 * df)
                - 0.5 * np.log(np.pi * (df - 2.0))
            )
            ll = const - 0.5 * np.log(sigma2) - 0.5 * (df + 1.0) * np.log1p(eps2 / ((df - 2.0) * sigma2))
        else:
            ll = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + eps2 / sigma2)
        total = float(np.sum(ll))
        return -total if np.isfinite(total) else 1e12

    bounds: list[tuple[float, float]] = [
        (-5.0, 5.0),
        (float(np.log(1e-7)), float(np.log(10.0))),
        (-12.0, 12.0),
        (-12.0, 12.0),
    ]
    starts = [
        [0.0, np.log(0.05), _logit(0.95 / PERSISTENCE_MAX), _logit(0.07 / 0.95)],
        [0.0, np.log(0.10), _logit(0.90 / PERSISTENCE_MAX), _logit(0.12 / 0.90)],
    ]
    if is_t:
        bounds.append((float(np.log(1e-4)), float(np.log(DF_MAX - DF_MIN))))
        starts = [s + [float(np.log(8.0 - DF_MIN))] for s in starts]

    best_x = np.asarray(starts[0], dtype=np.float64)
    best_val = neg_log_lik(best_x)
    for start in starts:
        x0 = np.asarray(start, dtype=np.float64)
        res = optimize.minimize(
            neg_log_lik,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-10},
        )
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = float(res.fun)
            best_x = np.asarray(res.x, dtype=np.float64)

    mu_s, omega_s, alpha, beta, df = unpack(best_x)
    mu = mean_r + std_r * mu_s
    omega = sample_var * omega_s
    raw_var_floor = max(RELATIVE_VAR_FLOOR * sample_var, float(np.finfo(np.float64).tiny))
    eps = series - mu
    sigma2 = _garch_variances(eps * eps, omega, alpha, beta, sample_var, raw_var_floor)
    innovations = eps / np.sqrt(sigma2)
    uncond_var = omega / max(1.0 - alpha - beta, 1e-8)

    marginal = _FittedMarginal(
        kind=kind,
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        scale=std_r,
        df=df,
        unconditional_var=uncond_var,
        last_var=float(sigma2[-1]),
        last_eps=float(eps[-1]),
        var_floor=raw_var_floor,
        sorted_innovations=np.sort(innovations),
        degenerate=False,
    )
    return marginal, innovations


def _fit_static_univariate(series: np.ndarray, kind: str) -> tuple[_FittedMarginal, np.ndarray]:
    """Fit a static empirical or univariate Student-t marginal to daily log returns."""
    mean_r = float(np.mean(series))
    sample_var = float(np.var(series, ddof=1)) if len(series) > 1 else 0.0
    if sample_var <= DEGENERATE_VAR_FLOOR:
        zeros = np.zeros_like(series, dtype=np.float64)
        marginal = _FittedMarginal(
            kind=kind,
            mu=mean_r,
            omega=None,
            alpha=None,
            beta=None,
            scale=0.0,
            df=None,
            unconditional_var=0.0,
            last_var=0.0,
            last_eps=0.0,
            var_floor=0.0,
            sorted_innovations=zeros,
            degenerate=True,
        )
        return marginal, zeros

    std_r = float(np.sqrt(sample_var))
    raw_var_floor = max(RELATIVE_VAR_FLOOR * sample_var, float(np.finfo(np.float64).tiny))
    if kind == 'static_empirical':
        marginal = _FittedMarginal(
            kind=kind,
            mu=mean_r,
            omega=None,
            alpha=None,
            beta=None,
            scale=std_r,
            df=None,
            unconditional_var=sample_var,
            last_var=sample_var,
            last_eps=float(series[-1] - mean_r),
            var_floor=raw_var_floor,
            sorted_innovations=np.sort(series.astype(np.float64, copy=True)),
            degenerate=False,
        )
        return marginal, series.astype(np.float64, copy=True)

    # Fit standardized Student-t location, variance scale, and df by MLE on scaled series.
    y = (series - mean_r) / std_r

    def neg_log_lik(theta: np.ndarray) -> float:
        loc_s = float(theta[0])
        sigma_s = float(np.exp(theta[1]))
        df = float(DF_MIN + np.exp(theta[2]))
        z = (y - loc_s) / sigma_s
        const = (
            special.gammaln(0.5 * (df + 1.0))
            - special.gammaln(0.5 * df)
            - 0.5 * np.log(np.pi * (df - 2.0))
        )
        ll = const - np.log(sigma_s) - 0.5 * (df + 1.0) * np.log1p((z * z) / (df - 2.0))
        total = float(np.sum(ll))
        return -total if np.isfinite(total) else 1e12

    bounds = [
        (-5.0, 5.0),
        (float(np.log(1e-4)), float(np.log(10.0))),
        (float(np.log(1e-4)), float(np.log(DF_MAX - DF_MIN))),
    ]
    x0 = np.array([0.0, 0.0, np.log(8.0 - DF_MIN)], dtype=np.float64)
    res = optimize.minimize(neg_log_lik, x0, method='L-BFGS-B', bounds=bounds)
    theta = res.x if np.isfinite(res.fun) else x0
    mu = mean_r + std_r * float(theta[0])
    sigma = std_r * float(np.exp(theta[1]))
    df = float(DF_MIN + np.exp(theta[2]))
    innovations = (series - mu) / sigma
    marginal = _FittedMarginal(
        kind=kind,
        mu=mu,
        omega=None,
        alpha=None,
        beta=None,
        scale=sigma,
        df=df,
        unconditional_var=sigma * sigma,
        last_var=sigma * sigma,
        last_eps=float(series[-1] - mu),
        var_floor=raw_var_floor,
        sorted_innovations=np.sort(innovations),
        degenerate=False,
    )
    return marginal, innovations


def _compute_pit(innovations: np.ndarray, marginal: _FittedMarginal, pit_kind: str) -> np.ndarray:
    """Transform fitted residuals or returns into uniform pseudo-observations on (0, 1)."""
    if marginal.degenerate:
        return np.full(len(innovations), 0.5, dtype=np.float64)
    if pit_kind == 'empirical':
        ranks = stats.rankdata(innovations, method='average')
        u = ranks / (len(innovations) + 1.0)
    elif marginal.kind == 'garch_normal':
        u = special.ndtr(innovations)
    else:
        assert marginal.df is not None
        u = _std_t_cdf(innovations, marginal.df)
    return np.clip(u, PROB_CLIP, 1.0 - PROB_CLIP)


def _sorted_linear_quantile(sorted_values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Exact linear-interpolation quantile on an already-sorted 1D array."""
    n = len(sorted_values)
    if n == 1:
        return np.full_like(probs, sorted_values[0], dtype=np.float64)
    pos = probs * (n - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _invert_marginal(u: np.ndarray, marginal: _FittedMarginal, pit_kind: str) -> np.ndarray:
    """Map uniform draws in (0, 1) to innovations (or raw log returns for static_empirical)."""
    if marginal.degenerate:
        return np.zeros_like(u, dtype=np.float64)
    clipped = np.clip(u, PROB_CLIP, 1.0 - PROB_CLIP)
    if pit_kind == 'empirical' or marginal.kind == 'static_empirical':
        return _sorted_linear_quantile(marginal.sorted_innovations, clipped)
    if marginal.kind == 'garch_normal':
        return special.ndtri(clipped)
    assert marginal.df is not None
    return _std_t_ppf(clipped, marginal.df)


def _nearest_pd_correlation(
    matrix: np.ndarray,
    min_eigenvalue: float = MIN_EIGENVALUE,
    max_iter: int = HIGHAM_MAX_ITER,
    tol: float = HIGHAM_TOL,
) -> np.ndarray:
    """Compute the Frobenius-nearest PD correlation matrix via Higham (2002) with Dykstra correction."""
    sym = 0.5 * (matrix + matrix.T)
    sym = np.where(np.isfinite(sym), sym, 0.0)
    np.fill_diagonal(sym, 1.0)
    if np.linalg.eigvalsh(sym)[0] >= min_eigenvalue:
        return sym

    y = sym.copy()
    ds = np.zeros_like(sym)
    for _ in range(max_iter):
        r = y - ds
        eigvals, eigvecs = np.linalg.eigh(r)
        x = (eigvecs * np.maximum(eigvals, min_eigenvalue)) @ eigvecs.T
        x = 0.5 * (x + x.T)
        ds = x - r
        y_next = x.copy()
        np.fill_diagonal(y_next, 1.0)
        diff = float(np.linalg.norm(y_next - y, ord='fro'))
        scale = max(1.0, float(np.linalg.norm(y_next, ord='fro')))
        y = y_next
        if diff <= tol * scale:
            break

    corr = 0.5 * (y + y.T)
    np.fill_diagonal(corr, 1.0)
    eigvals_check = np.linalg.eigvalsh(corr)
    if eigvals_check[0] <= 0.0:
        shrink = min(0.999, 1.0 / (1.0 + abs(eigvals_check[0]) + min_eigenvalue))
        corr = shrink * corr + (1.0 - shrink) * np.eye(corr.shape[0])
        np.fill_diagonal(corr, 1.0)
    return corr


def _kendall_correlation_matrix(u: np.ndarray) -> np.ndarray:
    """Estimate elliptical copula correlation matrix via R_ij = sin(pi/2 * tau_ij)."""
    d = u.shape[1]
    corr = np.eye(d, dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            tau = stats.kendalltau(u[:, i], u[:, j]).statistic
            val = float(np.sin(0.5 * np.pi * tau)) if np.isfinite(tau) else 0.0
            corr[i, j] = corr[j, i] = val
    return _nearest_pd_correlation(corr)


def _t_copula_neg_log_lik(df: float, u: np.ndarray, chol: np.ndarray, log_det: float) -> float:
    """Negative log-likelihood of a multivariate Student-t copula for scalar df."""
    d = u.shape[1]
    x = special.stdtrit(df, u)
    if not np.isfinite(x).all():
        return 1e12
    solved = scipy.linalg.solve_triangular(chol, x.T, lower=True)
    quad = np.sum(solved * solved, axis=0)
    const = (
        special.gammaln(0.5 * (df + d))
        + (d - 1) * special.gammaln(0.5 * df)
        - d * special.gammaln(0.5 * (df + 1.0))
        - 0.5 * log_det
    )
    ll = (
        const
        - 0.5 * (df + d) * np.log1p(quad / df)
        + 0.5 * (df + 1.0) * np.sum(np.log1p((x * x) / df), axis=1)
    )
    total = float(np.sum(ll))
    return -total if np.isfinite(total) else 1e12


def _fit_copula_parameters(
    u: np.ndarray,
    copula_kind: str,
    non_degenerate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Fit Gaussian or Student-t copula parameters on non-degenerate pseudo-observations."""
    d = u.shape[1]
    col_active = np.ptp(u, axis=0) > 1e-12
    if non_degenerate_mask is not None:
        col_active &= np.asarray(non_degenerate_mask, dtype=bool)
    active_idx = np.flatnonzero(col_active)
    if len(active_idx) < 2:
        eye = np.eye(d, dtype=np.float64)
        return eye, eye, (None if copula_kind == 'gaussian' else float(DF_MAX))

    u_active = u[:, active_idx]
    corr = np.eye(d, dtype=np.float64)
    if copula_kind == 'gaussian':
        z = special.ndtri(u_active)
        sub = _nearest_pd_correlation(np.corrcoef(z, rowvar=False))
        corr[np.ix_(active_idx, active_idx)] = sub
        chol = np.linalg.cholesky(corr)
        return corr, chol, None

    sub = _kendall_correlation_matrix(u_active)
    sub_chol = np.linalg.cholesky(sub)
    log_det = float(2.0 * np.sum(np.log(np.diag(sub_chol))))

    def obj_log(log_excess: float) -> float:
        return _t_copula_neg_log_lik(float(DF_MIN + np.exp(log_excess)), u_active, sub_chol, log_det)

    res = optimize.minimize_scalar(
        obj_log,
        bounds=(float(np.log(1e-3)), float(np.log(DF_MAX - DF_MIN))),
        method='bounded',
        options={'xatol': 1e-4},
    )
    best_df = float(DF_MIN + np.exp(res.x)) if np.isfinite(res.fun) else 10.0
    if _t_copula_neg_log_lik(DF_MAX, u_active, sub_chol, log_det) < res.fun:
        best_df = float(DF_MAX)
    corr[np.ix_(active_idx, active_idx)] = sub
    chol = np.linalg.cholesky(corr)
    return corr, chol, best_df


def _fit_joint_model(
    log_returns: np.ndarray, config: CopulaConfig
) -> _FittedCopulaModel:
    """Fit univariate marginals and multivariate copula to complete log-return rows."""
    n_obs, n_assets = log_returns.shape
    marginals: list[_FittedMarginal] = []
    u_cols: list[np.ndarray] = []
    is_garch = config.marginal.startswith('garch_')
    for col in range(n_assets):
        series = log_returns[:, col]
        if is_garch:
            marginal, innovations = _fit_garch_univariate(series, config.marginal)
        else:
            marginal, innovations = _fit_static_univariate(series, config.marginal)
        marginals.append(marginal)
        u_cols.append(_compute_pit(innovations, marginal, config.pit))
    u = np.column_stack(u_cols)
    non_degenerate_mask = np.array([not m.degenerate for m in marginals], dtype=bool)
    corr, chol, copula_df = _fit_copula_parameters(
        u, config.copula, non_degenerate_mask=non_degenerate_mask
    )
    return _FittedCopulaModel(
        marginals=tuple(marginals),
        copula_kind=config.copula,
        pit_kind=config.pit,
        corr=corr,
        cholesky=chol,
        copula_df=copula_df,
        n_fit_observations=n_obs,
    )


def _advance_garch_state(
    model: _FittedCopulaModel,
    log_returns: np.ndarray,
    start_vars: np.ndarray,
    start_eps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Filter GARCH variances causally across subsequent observed sessions.

    On sessions with a missing return for asset i, propagate the expected
    conditional variance omega + (alpha + beta) * sigma^2 without inventing returns.
    """
    next_vars = start_vars.astype(np.float64, copy=True)
    last_eps = start_eps.astype(np.float64, copy=True)
    missing_steps = 0
    for row in log_returns:
        for i, m in enumerate(model.marginals):
            if m.degenerate or m.omega is None:
                continue
            assert m.alpha is not None and m.beta is not None
            val = row[i]
            if np.isnan(val):
                missing_steps += 1
                next_vars[i] = max(m.omega + (m.alpha + m.beta) * next_vars[i], m.var_floor)
                last_eps[i] = 0.0
            else:
                eps = float(val - m.mu)
                next_vars[i] = max(m.omega + m.alpha * (eps * eps) + m.beta * next_vars[i], m.var_floor)
                last_eps[i] = eps
    return next_vars, last_eps, missing_steps


def _next_step_variances(model: _FittedCopulaModel) -> np.ndarray:
    """Compute one-step-ahead conditional variance sigma_{t0+1}^2 from fitted GARCH state."""
    out = np.empty(len(model.marginals), dtype=np.float64)
    for i, m in enumerate(model.marginals):
        if m.degenerate or m.omega is None:
            out[i] = m.last_var
        else:
            assert m.alpha is not None and m.beta is not None
            out[i] = max(m.omega + m.alpha * (m.last_eps ** 2) + m.beta * m.last_var, m.var_floor)
    return out


def _simulate_paths(
    model: _FittedCopulaModel,
    horizon: int,
    num_samples: int,
    seed: int,
    initial_next_var: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate (num_samples, horizon, n_assets) simple return paths."""
    rng = np.random.default_rng(seed)
    n_assets = len(model.marginals)
    # Draw joint uniform shocks of shape (num_samples, horizon, n_assets).
    normals = rng.standard_normal((num_samples, horizon, n_assets))
    correlated = normals @ model.cholesky.T
    if model.copula_kind == 'gaussian':
        u = special.ndtr(correlated)
    else:
        assert model.copula_df is not None
        chi2 = rng.chisquare(model.copula_df, size=(num_samples, horizon, 1))
        t_draws = correlated / np.sqrt(chi2 / model.copula_df)
        u = special.stdtr(model.copula_df, t_draws)

    # Invert marginals into innovations z (or log returns directly for static_empirical).
    z = np.empty_like(u)
    for i, m in enumerate(model.marginals):
        z[:, :, i] = _invert_marginal(u[:, :, i], m, model.pit_kind)

    log_paths = np.empty_like(z)
    is_garch = model.marginals[0].kind.startswith('garch_')
    if not is_garch:
        for i, m in enumerate(model.marginals):
            if m.degenerate:
                log_paths[:, :, i] = m.mu
            elif m.kind == 'static_empirical':
                log_paths[:, :, i] = z[:, :, i]
            else:
                log_paths[:, :, i] = m.mu + m.scale * z[:, :, i]
    else:
        start_var = _next_step_variances(model) if initial_next_var is None else initial_next_var
        sigma2 = np.broadcast_to(start_var[None, :], (num_samples, n_assets)).copy()
        mu = np.array([m.mu for m in model.marginals], dtype=np.float64)
        omega = np.array([0.0 if m.omega is None else m.omega for m in model.marginals], dtype=np.float64)
        alpha = np.array([0.0 if m.alpha is None else m.alpha for m in model.marginals], dtype=np.float64)
        beta = np.array([0.0 if m.beta is None else m.beta for m in model.marginals], dtype=np.float64)
        degenerate = np.array([m.degenerate for m in model.marginals], dtype=bool)
        for step in range(horizon):
            eps = np.sqrt(np.maximum(sigma2, 0.0)) * z[:, step, :]
            eps[:, degenerate] = 0.0
            log_paths[:, step, :] = mu + eps
            sigma2 = omega + alpha * (eps * eps) + beta * sigma2
            sigma2[:, degenerate] = 0.0

    with np.errstate(over='raise', invalid='raise'):
        simple_paths = np.expm1(log_paths)
    if not np.isfinite(simple_paths).all():
        raise FloatingPointError('Copula simulation produced nonfinite simple returns')
    return simple_paths


def _load_simple_and_log(
    returns: ReturnInput,
    assets: tuple[str, ...] | None,
    return_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate returns and return aligned (simple_df, log_df, info) without log->simple->log loss."""
    validated, info = _load(returns, assets, return_type, convert_log_to_simple=False)
    arr = validated.to_numpy(dtype=np.float64)
    if return_type == 'log':
        with np.errstate(over='ignore', invalid='ignore'):
            simple_arr = np.expm1(arr)
        if np.isinf(simple_arr).any():
            raise ValueError('Log return conversion produced nonfinite simple returns')
        simple_df = pd.DataFrame(simple_arr, index=validated.index, columns=validated.columns)
        return simple_df, validated, info
    with np.errstate(divide='ignore', invalid='ignore'):
        log_arr = np.where(np.isnan(arr), np.nan, np.where(arr <= -1.0, -np.inf, np.log1p(arr)))
    log_df = pd.DataFrame(log_arr, index=validated.index, columns=validated.columns)
    return validated, log_df, info


def _fit_with_trailing_state(
    complete_log: np.ndarray,
    trailing_log: np.ndarray,
    config: CopulaConfig,
) -> tuple[_FittedCopulaModel, np.ndarray | None, np.ndarray | None, int]:
    """Fit joint model on complete rows and advance GARCH state across trailing partial rows."""
    model = _fit_joint_model(complete_log, config)
    if not config.marginal.startswith('garch_'):
        return model, None, None, 0
    next_var = _next_step_variances(model)
    last_eps = np.array([m.last_eps for m in model.marginals], dtype=np.float64)
    trailing_miss = 0
    if len(trailing_log):
        next_var, last_eps, trailing_miss = _advance_garch_state(
            model, trailing_log, next_var, last_eps
        )
    return model, next_var, last_eps, trailing_miss


class CopulaBenchmark:
    """Fit static or GARCH-filtered Gaussian/Student-t copulas and estimate VaR/ES.

    No method writes files. Backtest calls are strictly causal and never mutate
    fitted state.
    """

    def __init__(self, config: CopulaConfig | None = None) -> None:
        self.config = CopulaConfig() if config is None else config
        self._history: pd.DataFrame | None = None
        self._log_history: pd.DataFrame | None = None
        self._assets: tuple[str, ...] = ()
        self._weights = np.empty(0, dtype=np.float64)
        self._fit_info: dict[str, Any] = {}
        self._base_cache: tuple[_FittedCopulaModel, np.ndarray | None, dict[str, Any], int] | None = None

    def fit(self, returns: ReturnInput) -> 'CopulaBenchmark':
        """Validate returns, store calibration history, and fit parameters lazily."""
        history, log_history, info = _load_simple_and_log(
            returns, self.config.assets, self.config.return_type
        )
        if history.empty:
            raise ValueError('Fitted history must contain at least one session')
        assets = tuple(history.columns)
        weights = _resolve_weights(assets, self.config.weights)
        self._history, self._log_history, self._fit_info = history, log_history, info
        self._assets, self._weights = assets, weights
        self._base_cache = None
        return self

    def _require_history(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._history is None or self._log_history is None:
            raise RuntimeError('Fit CopulaBenchmark before summarizing, sampling, or backtesting')
        return self._history, self._log_history

    def _prepare_fit_window(
        self, log_history: pd.DataFrame
    ) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any], str]:
        """Extract complete log-return rows and trailing incomplete rows from the active window."""
        window = (
            log_history.iloc[-self.config.lookback_sessions:]
            if self.config.lookback_sessions is not None
            else log_history
        )
        raw = window.to_numpy(dtype=np.float64)
        complete_mask = ~np.isnan(raw).any(axis=1)
        complete_indices = np.flatnonzero(complete_mask)
        complete = raw[complete_indices]
        trailing = (
            raw[int(complete_indices[-1]) + 1:]
            if len(complete_indices) and int(complete_indices[-1]) < len(raw) - 1
            else np.empty((0, raw.shape[1]), dtype=np.float64)
        )
        counts = {
            **_bounds(window),
            'n_fit_observations': int(len(complete)),
            'n_excluded_fit_sessions': int(len(window) - len(complete)),
        }
        if len(complete) < self.config.min_observations:
            return None, trailing, counts, 'insufficient_history'
        if not np.isfinite(complete).all():
            return None, trailing, counts, 'invalid_log_return_history'
        if len(trailing) and np.any(~np.isnan(trailing) & ~np.isfinite(trailing)):
            return None, trailing, counts, 'invalid_log_return_history'
        return complete, trailing, counts, 'ok'

    def _get_base_fit(self) -> tuple[_FittedCopulaModel, np.ndarray | None, dict[str, Any], int]:
        _, log_history = self._require_history()
        if self._base_cache is None:
            complete_log, trailing_log, counts, status = self._prepare_fit_window(log_history)
            if complete_log is None:
                raise ValueError(
                    f'Unavailable fit history ({status}): assets={self._assets}, '
                    f'min_observations={self.config.min_observations}, {counts}'
                )
            model, next_var, _, trailing_miss = _fit_with_trailing_state(
                complete_log, trailing_log, self.config
            )
            self._base_cache = (model, next_var, counts, trailing_miss)
        return self._base_cache

    def sample_paths(
        self, num_samples: int | None = None, seed: int | None = None
    ) -> np.ndarray:
        """Simulate (num_samples, horizon_sessions, n_assets) simple returns from fitted state."""
        model, next_var, _, _ = self._get_base_fit()
        n = self.config.num_samples if num_samples is None else _validate_int('num_samples', num_samples, 2)
        s = self.config.seed if seed is None else _validate_int('seed', seed, 0)
        assert n is not None and s is not None
        return _simulate_paths(model, self.config.horizon_sessions, n, s, initial_next_var=next_var)

    def _risk_rows(self, losses: np.ndarray | None) -> list[dict[str, Any]]:
        available = losses is not None
        n = len(losses) if available else 0
        sorted_losses = np.sort(losses) if available else np.empty(0, dtype=np.float64)
        records = []
        for confidence in self.config.confidence_levels:
            mass = _tail_mass(n, confidence) if available else 0.0
            records.append({
                'method': 'copula',
                'copula': self.config.copula,
                'marginal': self.config.marginal,
                'pit': self.config.pit,
                'confidence': confidence,
                'horizon_sessions': self.config.horizon_sessions,
                'rebalance': self.config.rebalance,
                'var': _empirical_var(sorted_losses, confidence) if available else np.nan,
                'es': _empirical_es(sorted_losses, confidence) if available else np.nan,
                'tail_mass_observations': mass,
                'small_tail_sample': mass < 20,
                'overlapping_scenarios': False,
                'n_scenarios': n,
            })
        return records

    def _metadata(
        self,
        counts: dict[str, Any],
        fitted_model: _FittedCopulaModel | None = None,
        evaluation: dict[str, Any] | None = None,
        warmup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = {
            name: getattr(self.config, name)
            for name in self.config.__dataclass_fields__
            if name != 'weights'
        }
        config['assets'] = list(self._assets)
        config['confidence_levels'] = list(self.config.confidence_levels)
        config['weights'] = dict(zip(self._assets, self._weights.tolist()))
        meta: dict[str, Any] = {
            'method': 'copula',
            'configuration': config,
            'input_return_type': self.config.return_type,
            'modeling_return_space': 'log',
            'loss_units': 'fractional',
            'inputs': {
                'fit': dict(self._fit_info),
                **({'warmup': warmup} if warmup is not None else {}),
                **({'evaluation': evaluation} if evaluation is not None else {}),
            },
            'counts': counts,
            'package_versions': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'scipy': scipy.__version__,
            },
            'missing_data_policy': (
                'fit marginals and copula on complete sessions only without filling; '
                'for GARCH causal filtering across trailing or subsequent missing sessions, '
                'propagate expected conditional variance omega + (alpha + beta) * sigma^2'
            ),
            'correlation_projection': 'Higham (2002) Frobenius-nearest PD correlation matrix',
            'quantile_convention': 'inverse empirical CDF over Monte Carlo scenarios; no interpolation',
            'es_convention': 'mean of worst 1-confidence mass with fractional boundary',
            'es_integer_mass_tolerance_per_scenario': ES_INTEGER_TOLERANCE,
            'weight_sum_absolute_tolerance': WEIGHT_TOLERANCE,
            'overlapping_scenarios': False,
            'overlapping_targets': self.config.horizon_sessions > 1,
        }
        if fitted_model is not None:
            meta['fitted_parameters'] = fitted_model.to_metadata(self._assets)
        return meta

    def summarize(self) -> CopulaSummary:
        """Fit on calibration history, simulate H-session paths, and compute VaR/ES."""
        history, _ = self._require_history()
        model, next_var, counts, trailing_miss = self._get_base_fit()
        paths = _simulate_paths(
            model,
            self.config.horizon_sessions,
            self.config.num_samples,
            self.config.seed,
            initial_next_var=next_var,
        )
        port_returns = _portfolio_returns(paths, self._weights, self.config.rebalance)
        losses = -port_returns
        with np.errstate(over='raise', invalid='raise'):
            mean = float(np.mean(port_returns))
            std = float(np.std(port_returns, ddof=1))
        if not np.isfinite([mean, std]).all():
            raise FloatingPointError('Portfolio scenario statistics are nonfinite')
        scenarios = pd.DataFrame({
            'sample_index': np.arange(len(port_returns), dtype=int),
            'portfolio_return': port_returns,
            'loss': losses,
        })
        records = [
            {
                **risk,
                **counts,
                'forecast_origin': history.index[-1],
                'portfolio_mean_return': mean,
                'portfolio_std_return': std,
            }
            for risk in self._risk_rows(losses)
        ]
        risk = _dated_table(
            records, list(records[0]), ('history_start', 'history_end', 'forecast_origin')
        )
        summary_counts = {
            **counts,
            'n_scenarios': len(port_returns),
            'n_missing_filter_steps': trailing_miss,
        }
        return CopulaSummary(
            risk=risk,
            scenarios=scenarios,
            metadata=self._metadata(summary_counts, fitted_model=model),
            paths=paths,
        )

    def backtest(
        self,
        evaluation_returns: ReturnInput,
        *,
        warmup_returns: ReturnInput | None = None,
    ) -> CopulaBacktest:
        """Forecast each complete-length evaluation target strictly causally.

        In estimation='fixed' mode, parameters are estimated once on fitted
        history; static marginals reuse the same forecast across targets while
        GARCH marginals filter conditional variances causally through any
        intervening warmup_returns and evaluation sessions strictly before
        target_start.
        """
        history, log_history = self._require_history()
        evaluation, eval_log, eval_info = _load_simple_and_log(
            evaluation_returns, self._assets, self.config.return_type
        )
        warmup_info: dict[str, Any] | None = None
        warmup_log: pd.DataFrame | None = None
        if warmup_returns is not None:
            warmup_simple, warmup_log, warmup_info = _load_simple_and_log(
                warmup_returns, self._assets, self.config.return_type
            )
            if len(warmup_simple) and warmup_simple.index[0] <= history.index[-1]:
                raise ValueError('Warmup dates must be strictly after fitted history')
            if len(warmup_simple) and len(evaluation) and evaluation.index[0] <= warmup_simple.index[-1]:
                raise ValueError('Evaluation dates must be strictly after warmup history')
        if len(evaluation) and evaluation.index[0] <= history.index[-1]:
            raise ValueError('Evaluation dates must be strictly after fitted history')

        parts = [log_history]
        if warmup_log is not None and len(warmup_log):
            parts.append(warmup_log)
        parts.append(eval_log)
        combined_log = pd.concat(parts)
        eval_offset = len(log_history) + (len(warmup_log) if warmup_log is not None else 0)

        horizon = self.config.horizon_sessions
        n_targets = max(0, len(evaluation) - horizon + 1)
        is_garch = self.config.marginal.startswith('garch_')

        records: list[dict[str, Any]] = []
        n_refits = 0
        total_missing_filter_steps = 0

        # Prepare fixed-parameter state if requested.
        fixed_model: _FittedCopulaModel | None = None
        fixed_counts: dict[str, Any] = {}
        fixed_status = 'ok'
        fixed_static_risks: list[dict[str, Any]] | None = None
        garch_next_var: np.ndarray | None = None
        garch_last_eps: np.ndarray | None = None
        garch_filter_invalid = False
        last_fit_position = eval_offset
        cached_model: _FittedCopulaModel | None = None
        cached_counts: dict[str, Any] = {}

        if self.config.estimation == 'fixed':
            complete_log, trailing_log, fixed_counts, fixed_status = self._prepare_fit_window(log_history)
            if complete_log is not None:
                fixed_model, garch_next_var, garch_last_eps, trailing_miss = _fit_with_trailing_state(
                    complete_log, trailing_log, self.config
                )
                n_refits += 1
                total_missing_filter_steps += trailing_miss
                if not is_garch:
                    paths = _simulate_paths(
                        fixed_model, horizon, self.config.num_samples, self.config.seed
                    )
                    losses = -_portfolio_returns(paths, self._weights, self.config.rebalance)
                    fixed_static_risks = self._risk_rows(losses)
                elif warmup_log is not None and len(warmup_log):
                    warm_arr = warmup_log.to_numpy(dtype=np.float64)
                    if np.any(~np.isnan(warm_arr) & ~np.isfinite(warm_arr)):
                        garch_filter_invalid = True
                    else:
                        assert garch_next_var is not None and garch_last_eps is not None
                        garch_next_var, garch_last_eps, warm_miss = _advance_garch_state(
                            fixed_model, warm_arr, garch_next_var, garch_last_eps
                        )
                        total_missing_filter_steps += warm_miss

        for target_idx in range(n_targets):
            position = eval_offset + target_idx
            status = 'ok'
            risks: list[dict[str, Any]] | None = None
            counts: dict[str, Any]

            if self.config.estimation == 'fixed':
                counts = dict(fixed_counts)
                if fixed_model is None:
                    status = fixed_status
                elif not is_garch:
                    risks = fixed_static_risks
                else:
                    if target_idx > 0 and not garch_filter_invalid:
                        prev_log = eval_log.iloc[target_idx - 1:target_idx].to_numpy(dtype=np.float64)
                        if np.any(~np.isnan(prev_log) & ~np.isfinite(prev_log)):
                            garch_filter_invalid = True
                        else:
                            assert garch_next_var is not None and garch_last_eps is not None
                            garch_next_var, garch_last_eps, miss = _advance_garch_state(
                                fixed_model, prev_log, garch_next_var, garch_last_eps
                            )
                            total_missing_filter_steps += miss
                    if garch_filter_invalid:
                        status = 'invalid_log_return_history'
                    else:
                        assert garch_next_var is not None
                        paths = _simulate_paths(
                            fixed_model,
                            horizon,
                            self.config.num_samples,
                            self.config.seed + target_idx,
                            initial_next_var=garch_next_var,
                        )
                        losses = -_portfolio_returns(paths, self._weights, self.config.rebalance)
                        risks = self._risk_rows(losses)
            else:
                # Expanding or rolling mode.
                should_refit = (cached_model is None) or (target_idx % self.config.refit_every == 0)
                if should_refit:
                    available_log = combined_log.iloc[:position]
                    complete_log, trailing_log, counts, status = self._prepare_fit_window(available_log)
                    if complete_log is not None:
                        cached_model, garch_next_var, garch_last_eps, trailing_miss = (
                            _fit_with_trailing_state(complete_log, trailing_log, self.config)
                        )
                        cached_counts = dict(counts)
                        last_fit_position = position
                        n_refits += 1
                        total_missing_filter_steps += trailing_miss
                        garch_filter_invalid = False
                    else:
                        cached_model = None
                        cached_counts = dict(counts)
                else:
                    counts = dict(cached_counts)
                    if cached_model is None:
                        status = 'insufficient_history'
                    elif is_garch and not garch_filter_invalid:
                        step_log = combined_log.iloc[last_fit_position:position].to_numpy(dtype=np.float64)
                        if len(step_log):
                            if np.any(~np.isnan(step_log) & ~np.isfinite(step_log)):
                                garch_filter_invalid = True
                            else:
                                assert garch_next_var is not None and garch_last_eps is not None
                                garch_next_var, garch_last_eps, miss = _advance_garch_state(
                                    cached_model, step_log, garch_next_var, garch_last_eps
                                )
                                total_missing_filter_steps += miss
                            last_fit_position = position

                if cached_model is not None and status == 'ok':
                    if is_garch and garch_filter_invalid:
                        status = 'invalid_log_return_history'
                    else:
                        paths = _simulate_paths(
                            cached_model,
                            horizon,
                            self.config.num_samples,
                            self.config.seed + target_idx,
                            initial_next_var=garch_next_var if is_garch else None,
                        )
                        losses = -_portfolio_returns(paths, self._weights, self.config.rebalance)
                        risks = self._risk_rows(losses)

            if risks is None:
                risks = self._risk_rows(None)

            block = evaluation.iloc[target_idx:target_idx + horizon].to_numpy()
            realized_ok = not np.isnan(block).any()
            realized = (
                float(_portfolio_returns(block[None], self._weights, self.config.rebalance)[0])
                if realized_ok
                else np.nan
            )
            estimate_ok = status == 'ok'
            for risk in risks:
                records.append({
                    **risk,
                    **counts,
                    'forecast_origin': combined_log.index[position - 1],
                    'target_start': evaluation.index[target_idx],
                    'target_end': evaluation.index[target_idx + horizon - 1],
                    'realized_return': realized,
                    'realized_loss': -realized,
                    'estimate_status': status,
                    'realization_status': 'ok' if realized_ok else 'missing_returns',
                    'breach': bool(-realized > risk['var']) if estimate_ok and realized_ok else pd.NA,
                })

        columns = [
            'method', 'copula', 'marginal', 'pit', 'confidence', 'horizon_sessions', 'rebalance',
            'var', 'es', 'tail_mass_observations', 'small_tail_sample', 'overlapping_scenarios',
            'n_scenarios', 'history_start', 'history_end', 'n_history_sessions',
            'n_fit_observations', 'n_excluded_fit_sessions', 'forecast_origin', 'target_start',
            'target_end', 'realized_return', 'realized_loss', 'estimate_status',
            'realization_status', 'breach',
        ]
        forecasts = _dated_table(
            records,
            columns,
            ('history_start', 'history_end', 'forecast_origin', 'target_start', 'target_end'),
        )
        forecasts['breach'] = forecasts.breach.astype('boolean')
        coverage = _coverage_table(forecasts, self.config.confidence_levels, horizon, n_targets)
        first_conf = forecasts.loc[forecasts.confidence.eq(self.config.confidence_levels[0])]
        counts_out = {
            'n_targets': n_targets,
            'n_trailing_excluded': len(evaluation) - n_targets,
            'n_forecast_rows': len(forecasts),
            'n_refits': n_refits,
            'n_missing_filter_steps': total_missing_filter_steps,
            'n_insufficient_history_targets': int(
                first_conf['estimate_status'].eq('insufficient_history').sum()
            ),
            'n_invalid_log_return_targets': int(
                first_conf['estimate_status'].eq('invalid_log_return_history').sum()
            ),
            'n_missing_realization_targets': int(
                first_conf['realization_status'].eq('missing_returns').sum()
            ),
        }
        return CopulaBacktest(
            forecasts=forecasts,
            coverage=coverage,
            metadata=self._metadata(
                counts_out,
                fitted_model=fixed_model if self.config.estimation == 'fixed' else None,
                evaluation=eval_info,
                warmup=warmup_info,
            ),
        )
