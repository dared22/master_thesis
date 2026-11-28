"""Run uv run python -m unittest benchmarks.test_copula from the repository root."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats

from benchmarks import CopulaBenchmark, CopulaConfig
from benchmarks.copula import (
    _fit_copula_parameters,
    _fit_garch_univariate,
    _nearest_pd_correlation,
    _next_step_variances,
    _std_t_cdf,
    _std_t_ppf,
)


def make_frame(values: np.ndarray, start: str = '2024-01-01') -> pd.DataFrame:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    cols = [f'S{i}' for i in range(array.shape[1])]
    df = pd.DataFrame(array, columns=cols)
    df.insert(0, 'Date', pd.date_range(start, periods=len(df)))
    return df


def simulate_synthetic_garch_t(
    n: int,
    mu: float = 0.0005,
    omega: float = 1e-5,
    alpha: float = 0.09,
    beta: float = 0.87,
    df: float = 6.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = np.sqrt((df - 2.0) / df)
    z = scale * rng.standard_t(df, size=n)
    sigma2 = np.empty(n, dtype=np.float64)
    eps = np.empty(n, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    return mu + eps


def simulate_synthetic_t_copula(
    n: int, rho: float, df: float, seed: int = 7
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corr = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    chol = np.linalg.cholesky(corr)
    normals = rng.standard_normal((n, 2)) @ chol.T
    chi2 = rng.chisquare(df, size=(n, 1))
    t_draws = normals / np.sqrt(chi2 / df)
    return stats.t.cdf(t_draws, df=df)


class TestMathAndRecovery(unittest.TestCase):
    def test_standardized_t_roundtrip_and_unit_variance(self) -> None:
        df = 5.0
        grid = np.linspace(0.01, 0.99, 99)
        z = _std_t_ppf(grid, df)
        np.testing.assert_allclose(_std_t_cdf(z, df), grid, atol=1e-12)
        rng = np.random.default_rng(0)
        draws = _std_t_ppf(rng.uniform(1e-6, 1 - 1e-6, size=200_000), df)
        self.assertAlmostEqual(float(np.var(draws)), 1.0, delta=0.03)

    def test_garch_t_mle_recovery(self) -> None:
        series = simulate_synthetic_garch_t(
            2500, mu=0.0005, omega=1e-5, alpha=0.09, beta=0.87, df=6.0, seed=123
        )
        marginal, innovations = _fit_garch_univariate(series, 'garch_t')
        self.assertFalse(marginal.degenerate)
        assert marginal.alpha is not None and marginal.beta is not None and marginal.df is not None
        self.assertAlmostEqual(marginal.alpha, 0.09, delta=0.035)
        self.assertAlmostEqual(marginal.beta, 0.87, delta=0.045)
        self.assertAlmostEqual(marginal.df, 6.0, delta=1.5)
        self.assertAlmostEqual(float(np.std(innovations, ddof=1)), 1.0, delta=0.08)

    def test_copula_recovery_and_tail_dependence(self) -> None:
        u = simulate_synthetic_t_copula(3000, rho=0.65, df=4.0, seed=11)
        corr_g, _, df_g = _fit_copula_parameters(u, 'gaussian')
        corr_t, _, df_t = _fit_copula_parameters(u, 't')
        self.assertIsNone(df_g)
        assert df_t is not None
        self.assertAlmostEqual(corr_g[0, 1], 0.65, delta=0.05)
        self.assertAlmostEqual(corr_t[0, 1], 0.65, delta=0.05)
        self.assertAlmostEqual(df_t, 4.0, delta=1.2)

        # Verify Student-t copula produces stronger joint 1% lower-tail co-exceedance than Gaussian.
        log_rets = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)) * 0.015
        simple = np.expm1(log_rets)
        df_in = make_frame(simple)
        g_paths = CopulaBenchmark(CopulaConfig(
            copula='gaussian', marginal='static_empirical', pit='empirical',
            horizon_sessions=1, num_samples=40_000, seed=0,
        )).fit(df_in).sample_paths()[:, 0, :]
        t_paths = CopulaBenchmark(CopulaConfig(
            copula='t', marginal='static_empirical', pit='empirical',
            horizon_sessions=1, num_samples=40_000, seed=0,
        )).fit(df_in).sample_paths()[:, 0, :]
        q = 0.01
        g_co = np.mean((g_paths[:, 0] <= np.quantile(g_paths[:, 0], q))
                       & (g_paths[:, 1] <= np.quantile(g_paths[:, 1], q))) / q
        t_co = np.mean((t_paths[:, 0] <= np.quantile(t_paths[:, 0], q))
                       & (t_paths[:, 1] <= np.quantile(t_paths[:, 1], q))) / q
        self.assertGreater(t_co, g_co + 0.05)

    def test_garch_responds_to_volatility_shock_unlike_static(self) -> None:
        rng = np.random.default_rng(5)
        r1 = simulate_synthetic_garch_t(600, alpha=0.12, beta=0.84, df=7.0, seed=5)
        r2 = 0.6 * r1 + 0.8 * simulate_synthetic_garch_t(600, alpha=0.12, beta=0.84, df=7.0, seed=6)
        base = np.expm1(np.column_stack([r1, r2]))
        calm = base.copy()
        calm[-1] = [0.0005, 0.0005]
        shock = base.copy()
        shock[-1] = [-0.12, -0.12]

        cfg_garch = CopulaConfig(
            copula='t', marginal='garch_t', pit='empirical',
            horizon_sessions=20, num_samples=5000, seed=0,
        )
        var_calm = CopulaBenchmark(cfg_garch).fit(make_frame(calm)).summarize().risk['var'].to_numpy()
        var_shock = CopulaBenchmark(cfg_garch).fit(make_frame(shock)).summarize().risk['var'].to_numpy()
        self.assertTrue(np.all(var_shock > var_calm * 1.2))

        cfg_static = CopulaConfig(
            copula='t', marginal='static_empirical', pit='empirical',
            horizon_sessions=20, num_samples=5000, seed=0,
        )
        s_calm = CopulaBenchmark(cfg_static).fit(make_frame(calm)).summarize().risk['var'].to_numpy()
        s_shock = CopulaBenchmark(cfg_static).fit(make_frame(shock)).summarize().risk['var'].to_numpy()
        self.assertLess(np.max(np.abs(s_shock - s_calm)), np.min(var_shock - var_calm) * 0.25)

    def test_degenerate_constant_asset_and_single_asset(self) -> None:
        rng = np.random.default_rng(9)
        active = np.expm1(rng.normal(0.0, 0.015, size=120))
        flat = np.zeros(120, dtype=float)
        two_asset = make_frame(np.column_stack([active, flat]))
        for marginal, pit in (
            ('static_empirical', 'empirical'),
            ('static_t', 'parametric'),
            ('garch_normal', 'parametric'),
            ('garch_t', 'empirical'),
        ):
            for copula in ('gaussian', 't'):
                res = CopulaBenchmark(CopulaConfig(
                    copula=copula, marginal=marginal, pit=pit,
                    horizon_sessions=5, num_samples=400, seed=1,
                )).fit(two_asset).summarize()
                self.assertTrue(np.isfinite(res.risk[['var', 'es']].to_numpy()).all())
                self.assertTrue(np.all(res.risk['es'] + 1e-12 >= res.risk['var']))
                np.testing.assert_allclose(res.paths[:, :, 1], 0.0, atol=1e-15)
                json.dumps(res.metadata, allow_nan=False)

        one_asset = make_frame(active)
        res1 = CopulaBenchmark(CopulaConfig(
            copula='t', marginal='garch_t', horizon_sessions=5, num_samples=300, seed=2,
        )).fit(one_asset).summarize()
        self.assertEqual(res1.paths.shape, (300, 5, 1))
        self.assertTrue(np.isfinite(res1.risk[['var', 'es']].to_numpy()).all())

    def test_log_simple_equivalence_and_rebalance_modes(self) -> None:
        rng = np.random.default_rng(14)
        log_mat = rng.normal(0.0003, 0.02, size=(120, 2))
        simple_mat = np.expm1(log_mat)
        df_simple = make_frame(simple_mat)
        df_log = make_frame(log_mat)

        cfg_s = CopulaConfig(
            return_type='simple', horizon_sessions=10, num_samples=1000, seed=3,
            marginal='garch_t', copula='t', pit='parametric',
        )
        cfg_l = replace(cfg_s, return_type='log')
        sum_s = CopulaBenchmark(cfg_s).fit(df_simple).summarize()
        sum_l = CopulaBenchmark(cfg_l).fit(df_log).summarize()
        # return_type='log' avoids expm1->log1p rounding; 1-ULP input diffs propagate through L-BFGS-B to ~1e-7.
        np.testing.assert_allclose(sum_s.paths, sum_l.paths, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            sum_s.risk[['var', 'es']], sum_l.risk[['var', 'es']], rtol=1e-5, atol=1e-6
        )

        # Daily rebalancing and buy-and-hold agree at H=1 and can differ at H=20.
        h1_d = CopulaBenchmark(replace(cfg_s, horizon_sessions=1, rebalance='daily')).fit(df_simple).summarize()
        h1_bh = CopulaBenchmark(replace(cfg_s, horizon_sessions=1, rebalance='buy_and_hold')).fit(df_simple).summarize()
        np.testing.assert_allclose(h1_d.scenarios.portfolio_return, h1_bh.scenarios.portfolio_return, atol=1e-14)
        h20_d = CopulaBenchmark(replace(cfg_s, horizon_sessions=20, rebalance='daily')).fit(df_simple).summarize()
        h20_bh = CopulaBenchmark(replace(cfg_s, horizon_sessions=20, rebalance='buy_and_hold')).fit(df_simple).summarize()
        self.assertFalse(np.allclose(h20_d.scenarios.portfolio_return, h20_bh.scenarios.portfolio_return))

    def test_constant_asset_does_not_alter_t_copula_df(self) -> None:
        u = simulate_synthetic_t_copula(1500, rho=0.65, df=4.0, seed=19)
        log_rets = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)) * 0.015
        simple2 = np.expm1(log_rets)
        simple3 = np.column_stack([simple2, np.zeros(len(simple2), dtype=float)])

        cfg2 = CopulaConfig(
            copula='t', marginal='static_empirical', pit='empirical',
            horizon_sessions=1, num_samples=200, seed=0,
        )
        cfg3 = CopulaConfig(
            copula='t', marginal='static_empirical', pit='empirical',
            weights={'S0': 0.5, 'S1': 0.5, 'S2': 0.0},
            horizon_sessions=1, num_samples=200, seed=0,
        )
        fp2 = CopulaBenchmark(cfg2).fit(make_frame(simple2)).summarize().metadata['fitted_parameters']
        fp3 = CopulaBenchmark(cfg3).fit(make_frame(simple3)).summarize().metadata['fitted_parameters']
        self.assertAlmostEqual(fp2['copula_df'], fp3['copula_df'], places=10)
        np.testing.assert_allclose(
            np.asarray(fp2['copula_corr']), np.asarray(fp3['copula_corr'])[:2, :2], atol=1e-12
        )

    def test_garch_scale_equivariance_for_low_volatility_series(self) -> None:
        series = simulate_synthetic_garch_t(600, mu=0.0, omega=1e-5, alpha=0.10, beta=0.85, df=6.0, seed=77)
        scale_factor = 1e-5
        m_base, _ = _fit_garch_univariate(series, 'garch_t')
        m_tiny, _ = _fit_garch_univariate(series * scale_factor, 'garch_t')
        self.assertFalse(m_tiny.degenerate)
        assert m_base.omega is not None and m_base.alpha is not None and m_base.beta is not None
        assert m_tiny.omega is not None and m_tiny.alpha is not None and m_tiny.beta is not None
        vol_base = np.sqrt(m_base.omega + m_base.alpha * m_base.last_eps ** 2 + m_base.beta * m_base.last_var)
        vol_tiny = np.sqrt(m_tiny.omega + m_tiny.alpha * m_tiny.last_eps ** 2 + m_tiny.beta * m_tiny.last_var)
        self.assertAlmostEqual(vol_tiny / scale_factor, vol_base, delta=vol_base * 1e-3)

    def test_extreme_negative_log_return_preserved_without_underflow(self) -> None:
        rng = np.random.default_rng(29)
        log_mat = rng.normal(0.0, 0.015, size=(90, 2))
        log_mat[10, 0] = -40.0  # expm1(-40.0) == -1.0 in float64, but log return is finite.
        df_log = make_frame(log_mat)
        summ = CopulaBenchmark(CopulaConfig(
            return_type='log', copula='t', marginal='garch_t',
            horizon_sessions=1, min_observations=30, num_samples=300, seed=0,
        )).fit(df_log).summarize()
        self.assertTrue(np.isfinite(summ.risk[['var', 'es']].to_numpy()).all())
        self.assertEqual(summ.metadata['counts']['n_fit_observations'], 90)

        # In simple-return mode, -1.0 still triggers invalid_log_return_history.
        df_simple = make_frame(np.expm1(log_mat))
        with self.assertRaisesRegex(ValueError, 'invalid_log_return_history'):
            CopulaBenchmark(CopulaConfig(
                return_type='simple', copula='t', marginal='garch_t',
                horizon_sessions=1, min_observations=30, num_samples=300, seed=0,
            )).fit(df_simple).summarize()

    def test_higham_nearest_pd_correlation_beats_spectral_clipping(self) -> None:
        # Classic Higham (2002) asymmetric non-PSD correlation matrix where diagonal rescaling is suboptimal.
        raw = np.array([
            [1.0, 0.95, 0.0],
            [0.95, 1.0, 0.95],
            [0.0, 0.95, 1.0],
        ], dtype=np.float64)
        higham = _nearest_pd_correlation(raw)
        np.testing.assert_allclose(higham, higham.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(higham), 1.0, atol=1e-12)
        self.assertGreater(float(np.linalg.eigvalsh(higham)[0]), 0.0)

        # Compare against one-shot spectral clipping + diagonal rescaling.
        eigvals, eigvecs = np.linalg.eigh(raw)
        clipped = (eigvecs * np.maximum(eigvals, 1e-6)) @ eigvecs.T
        diag = np.sqrt(np.diag(clipped))
        spectral = clipped / np.outer(diag, diag)
        np.fill_diagonal(spectral, 1.0)
        self.assertLess(
            float(np.linalg.norm(higham - raw, ord='fro')),
            float(np.linalg.norm(spectral - raw, ord='fro')) - 1e-3,
        )


class TestValidationAndLifecycle(unittest.TestCase):
    def test_invalid_configuration(self) -> None:
        invalid = [
            {'copula': 'clayton'},
            {'marginal': 'egarch'},
            {'pit': 'rank'},
            {'marginal': 'static_empirical', 'pit': 'parametric'},
            {'return_type': 'percent'},
            {'rebalance': 'weekly'},
            {'estimation': 'adaptive'},
            {'estimation': 'rolling'},
            {'estimation': 'expanding', 'lookback_sessions': 100},
            {'estimation': 'rolling', 'lookback_sessions': 30, 'min_observations': 60},
            {'horizon_sessions': 0},
            {'refit_every': 0},
            {'min_observations': 5},
            {'num_samples': 1},
            {'seed': -1},
            {'seed': True},
            {'weights': {'S0': 0.8}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CopulaConfig(**kwargs)

    def test_lifecycle_csv_parity_and_no_writes(self) -> None:
        benchmark = CopulaBenchmark(CopulaConfig(min_observations=20, num_samples=200))
        for op in (benchmark.summarize, benchmark.sample_paths, lambda: benchmark.backtest(make_frame(np.zeros((5, 2))))):
            with self.assertRaises(RuntimeError):
                op()
        rng = np.random.default_rng(21)
        source = make_frame(np.expm1(rng.normal(0.0, 0.01, size=(80, 2))))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'returns.csv'
            source.to_csv(path, index=False)
            before = path.read_bytes()
            a = CopulaBenchmark(CopulaConfig(min_observations=20, num_samples=300, seed=4)).fit(path).summarize()
            b = CopulaBenchmark(CopulaConfig(min_observations=20, num_samples=300, seed=4)).fit(
                pd.read_csv(path)
            ).summarize()
            assert_frame_equal(a.risk, b.risk)
            assert_frame_equal(a.scenarios, b.scenarios)
            np.testing.assert_array_equal(a.paths, b.paths)
            self.assertEqual(a.metadata['inputs']['fit']['fingerprint'],
                             b.metadata['inputs']['fit']['fingerprint'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            json.dumps(a.metadata, allow_nan=False)


class TestChronologyAndCausality(unittest.TestCase):
    def _sample_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(31)
        full = np.expm1(rng.normal(0.0, 0.015, size=(120, 2)))
        df = make_frame(full, start='2024-01-01')
        return df.iloc[:100].reset_index(drop=True), df.iloc[100:].reset_index(drop=True)

    def test_strict_future_invariance_across_modes(self) -> None:
        history, evaluation = self._sample_data()
        configs = [
            CopulaConfig(
                copula='t', marginal='static_empirical', estimation='fixed',
                horizon_sessions=5, min_observations=30, num_samples=400, seed=10,
            ),
            CopulaConfig(
                copula='t', marginal='garch_t', estimation='fixed',
                horizon_sessions=5, min_observations=30, num_samples=400, seed=10,
            ),
            CopulaConfig(
                copula='gaussian', marginal='garch_normal', estimation='expanding',
                refit_every=4, horizon_sessions=5, min_observations=30, num_samples=300, seed=10,
            ),
            CopulaConfig(
                copula='t', marginal='garch_t', estimation='rolling',
                lookback_sessions=80, refit_every=5, horizon_sessions=5,
                min_observations=30, num_samples=300, seed=10,
            ),
        ]
        cols = ['var', 'es', 'n_fit_observations', 'history_start', 'history_end', 'estimate_status']
        for cfg in configs:
            with self.subTest(marginal=cfg.marginal, estimation=cfg.estimation):
                bm = CopulaBenchmark(cfg).fit(history)
                orig = bm.backtest(evaluation)
                json.dumps(orig.metadata, allow_nan=False)
                # Static + fixed produces identical VaR across all targets and matches summarize().
                if cfg.marginal.startswith('static_') and cfg.estimation == 'fixed':
                    self.assertEqual(orig.forecasts['var'].nunique(), len(cfg.confidence_levels))
                    summ = bm.summarize()
                    np.testing.assert_allclose(
                        orig.forecasts.iloc[:2][['var', 'es']], summ.risk[['var', 'es']]
                    )
                # Mutate evaluation returns at or after target k; target k forecast must be invariant.
                for target in (0, 3, 8):
                    mutated = evaluation.copy()
                    mutated.loc[target:, ['S0', 'S1']] = -0.35
                    alt = bm.backtest(mutated)
                    sl = slice(target * 2, target * 2 + 2)
                    assert_frame_equal(orig.forecasts.iloc[sl][cols], alt.forecasts.iloc[sl][cols])

    def test_missing_data_total_loss_and_short_evaluation(self) -> None:
        history, evaluation = self._sample_data()
        eval_missing = evaluation.copy()
        eval_missing.loc[1, 'S0'] = np.nan
        bm = CopulaBenchmark(CopulaConfig(
            copula='t', marginal='garch_t', estimation='fixed',
            horizon_sessions=3, min_observations=30, num_samples=300, seed=0,
        )).fit(history)
        res = bm.backtest(eval_missing)
        self.assertGreater(res.metadata['counts']['n_missing_filter_steps'], 0)
        # Realized targets overlapping row 1 are marked missing_returns with NA breach.
        t0 = res.forecasts.loc[res.forecasts.confidence.eq(0.95)]
        self.assertEqual(t0.realization_status.iloc[0], 'missing_returns')
        self.assertTrue(pd.isna(t0.breach.iloc[0]))
        self.assertEqual(t0.estimate_status.iloc[2], 'ok')

        # Total loss (-1.0) invalidates subsequent GARCH log-return filtering cleanly.
        eval_wipe = evaluation.copy()
        eval_wipe.loc[0, 'S0'] = -1.0
        wipe_res = bm.backtest(eval_wipe)
        w95 = wipe_res.forecasts.loc[wipe_res.forecasts.confidence.eq(0.95)]
        self.assertEqual(w95.estimate_status.iloc[0], 'ok')
        self.assertEqual(w95.estimate_status.iloc[1], 'invalid_log_return_history')
        self.assertTrue(pd.isna(w95.breach.iloc[1]))

        # Short / empty evaluation preserves schema.
        for short_df in (evaluation.iloc[:2], evaluation.iloc[:0]):
            short_res = bm.backtest(short_df)
            self.assertTrue(short_res.forecasts.empty)
            self.assertEqual(short_res.forecasts.columns.tolist(), res.forecasts.columns.tolist())
            self.assertEqual(str(short_res.forecasts.breach.dtype), 'boolean')
            self.assertTrue(short_res.coverage.n_targets.eq(0).all())
            json.dumps(short_res.metadata, allow_nan=False)

        # Insufficient fit history retains target rows with insufficient_history status.
        tiny_bm = CopulaBenchmark(CopulaConfig(
            copula='t', marginal='garch_t', estimation='fixed',
            horizon_sessions=2, min_observations=500, num_samples=200,
        )).fit(history)
        with self.assertRaisesRegex(ValueError, 'insufficient_history'):
            tiny_bm.summarize()
        tiny_bt = tiny_bm.backtest(evaluation.iloc[:5])
        self.assertTrue(tiny_bt.forecasts.estimate_status.eq('insufficient_history').all())
        self.assertTrue(tiny_bt.forecasts['var'].isna().all())

    def test_trailing_incomplete_session_updates_garch_on_refit(self) -> None:
        r0 = simulate_synthetic_garch_t(120, alpha=0.14, beta=0.82, df=7.0, seed=41)
        r1 = simulate_synthetic_garch_t(120, alpha=0.10, beta=0.85, df=7.0, seed=42)
        full = make_frame(np.expm1(np.column_stack([r0, r1])), start='2024-01-01')
        history = full.iloc[:100].reset_index(drop=True)
        eval_calm = full.iloc[100:106].reset_index(drop=True)
        eval_calm.loc[0, 'S0'] = 0.001
        eval_calm.loc[0, 'S1'] = np.nan
        eval_shock = eval_calm.copy()
        eval_shock.loc[0, 'S0'] = 0.20

        # refit_every=1 forces a refit right after the partial row (target_idx=1).
        cfg = CopulaConfig(
            copula='t', marginal='garch_t', pit='empirical', estimation='expanding',
            refit_every=1, horizon_sessions=1, min_observations=30, num_samples=2000, seed=5,
        )
        bt_calm = CopulaBenchmark(cfg).fit(history).backtest(eval_calm)
        bt_shock = CopulaBenchmark(cfg).fit(history).backtest(eval_shock)
        var_calm = bt_calm.forecasts.loc[bt_calm.forecasts.confidence.eq(0.95), 'var'].iloc[1]
        var_shock = bt_shock.forecasts.loc[bt_shock.forecasts.confidence.eq(0.95), 'var'].iloc[1]
        self.assertGreater(var_shock, var_calm * 1.5)
        self.assertGreater(bt_shock.metadata['counts']['n_missing_filter_steps'], 0)

    def test_warmup_advances_garch_state_without_refitting_parameters(self) -> None:
        r0 = simulate_synthetic_garch_t(150, alpha=0.12, beta=0.84, df=7.0, seed=51)
        r1 = simulate_synthetic_garch_t(150, alpha=0.10, beta=0.86, df=7.0, seed=52)
        full = make_frame(np.expm1(np.column_stack([r0, r1])), start='2024-01-01')
        train = full.iloc[:100].reset_index(drop=True)
        val = full.iloc[100:120].reset_index(drop=True)
        test = full.iloc[120:130].reset_index(drop=True)

        cfg = CopulaConfig(
            copula='t', marginal='garch_t', pit='empirical', estimation='fixed',
            horizon_sessions=1, min_observations=30, num_samples=1000, seed=0,
        )
        bm = CopulaBenchmark(cfg).fit(train)
        bt_val = bm.backtest(val)
        bt_test = bm.backtest(test, warmup_returns=val)
        # Fitted parameters remain strictly training-only (100 obs, identical params).
        self.assertEqual(bt_test.metadata['fitted_parameters']['n_fit_observations'], 100)
        self.assertEqual(bt_val.metadata['fitted_parameters'], bt_test.metadata['fitted_parameters'])
        self.assertIn('warmup', bt_test.metadata['inputs'])
        # Compare target 0 of test against a continuous backtest over concat(val, test) with aligned seed.
        bt_full = CopulaBenchmark(replace(cfg, seed=0)).fit(train).backtest(
            pd.concat([val, test], ignore_index=True)
        )
        bt_test_aligned = CopulaBenchmark(replace(cfg, seed=len(val))).fit(train).backtest(
            test, warmup_returns=val
        )
        full_test_slice = bt_full.forecasts.loc[bt_full.forecasts.target_start >= test.Date.iloc[0]].reset_index(drop=True)
        assert_frame_equal(
            full_test_slice[['var', 'es', 'forecast_origin', 'target_start']],
            bt_test_aligned.forecasts[['var', 'es', 'forecast_origin', 'target_start']],
        )
        # Out-of-order warmup dates are rejected.
        with self.assertRaises(ValueError):
            bm.backtest(test, warmup_returns=train.iloc[-5:])
        with self.assertRaises(ValueError):
            bm.backtest(val, warmup_returns=test)


class TestFourStockCSVIntegration(unittest.TestCase):
    def test_four_stock_static_and_garch_copulas(self) -> None:
        directory = Path(__file__).resolve().parents[1] / 'experiments/4stocks/data'
        train_path = directory / 'train_returns.csv'
        val_path = directory / 'validation_returns.csv'
        for copula, marginal in (
            ('gaussian', 'static_empirical'),
            ('t', 'static_t'),
            ('gaussian', 'garch_normal'),
            ('t', 'garch_t'),
        ):
            with self.subTest(copula=copula, marginal=marginal):
                cfg = CopulaConfig(
                    horizon_sessions=20,
                    rebalance='daily',
                    confidence_levels=(0.95, 0.99),
                    copula=copula,
                    marginal=marginal,
                    pit='empirical',
                    estimation='fixed',
                    num_samples=1000,
                    seed=0,
                )
                bm = CopulaBenchmark(cfg).fit(train_path)
                summ = bm.summarize()
                self.assertEqual(summ.paths.shape, (1000, 20, 4))
                self.assertTrue(np.isfinite(summ.risk[['var', 'es']].to_numpy()).all())
                self.assertTrue(np.all(summ.risk['es'] + 1e-12 >= summ.risk['var']))
                bt = bm.backtest(val_path)
                self.assertEqual(len(bt.forecasts), 714)
                self.assertEqual(bt.coverage.n_comparisons.tolist(), [357, 357])
                self.assertEqual(bt.metadata['counts']['n_trailing_excluded'], 19)
                json.dumps(summ.metadata, allow_nan=False)
                json.dumps(bt.metadata, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
