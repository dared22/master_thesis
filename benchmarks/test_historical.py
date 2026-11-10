"""Run from the repository root: sandbox/.venv/bin/python -m unittest benchmarks.test_historical."""

from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from benchmarks import HistoricalSimulation, HistoricalSimulationConfig
from benchmarks.historical import _empirical_es, _empirical_var


def frame(values: object, start: str = '2024-01-01') -> pd.DataFrame:
    """Small observed calendars; consecutive calendar days are intentional."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    result = pd.DataFrame(array, columns=list('ABC')[:array.shape[1]])
    result.insert(0, 'Date', pd.date_range(start, periods=len(result)))
    return result


def model(returns: pd.DataFrame, **kwargs: object) -> HistoricalSimulation:
    return HistoricalSimulation(HistoricalSimulationConfig(min_scenarios=2, **kwargs)).fit(returns)


class TestRiskAndPortfolios(unittest.TestCase):
    def test_fractional_tail_and_ties(self) -> None:
        losses = np.array([-.02, .01, .03, .10])
        self.assertEqual(_empirical_var(losses, .625), .03)
        self.assertAlmostEqual(_empirical_es(losses, .625), .07666666666666667)
        ties = np.array([-.02, .01, .03, .03, .10])
        self.assertEqual(_empirical_var(ties, .6), .03)
        self.assertAlmostEqual(_empirical_es(ties, .6), .065)
        for confidence in (.01, .5, .75, .99, np.nextafter(1., 0.)):
            self.assertGreaterEqual(_empirical_es(losses, confidence) + 1e-14,
                                    _empirical_var(losses, confidence))
        self.assertEqual(_empirical_es(losses, .99), .1)
        self.assertAlmostEqual(_empirical_es(np.arange(100.) / 100, .8), .895)
        # Exact empirical CDF steps and immediately higher confidences differ.
        self.assertEqual(_empirical_var(np.arange(100.), .07), 6)
        self.assertEqual(_empirical_var(np.arange(100.), np.nextafter(.07, 1)), 7)

    def test_zero_gains_and_total_loss(self) -> None:
        for values in ([0, 0, 0, 0], [.01, .02, .03, .04]):
            risk = model(frame(values), confidence_levels=(.75,)).summarize().risk.iloc[0]
            self.assertLessEqual(risk['var'], 0)
            self.assertLessEqual(risk.es, 0)
            self.assertGreaterEqual(risk.es, risk['var'])
        result = model(frame([-1, 0, .1])).summarize()
        self.assertEqual(result.scenarios.loss.iloc[0], 1)

    def test_portfolio_rules_and_named_weights(self) -> None:
        source = frame([[.1, 0], [0, .1], [.02, -.03]])
        daily = model(source, horizon_sessions=2).summarize()
        hold = model(source, horizon_sessions=2, rebalance='buy_and_hold').summarize()
        self.assertAlmostEqual(daily.scenarios.portfolio_return.iloc[0], .1025)
        self.assertAlmostEqual(hold.scenarios.portfolio_return.iloc[0], .10)
        weights = {'A': .75, 'B': .25}
        a = model(source, weights=weights).summarize()
        b = model(source[['Date', 'B', 'A']], weights=weights).summarize()
        np.testing.assert_allclose(a.scenarios.portfolio_return, [.075, .025, .0075])
        np.testing.assert_allclose(a.scenarios.portfolio_return, b.scenarios.portfolio_return)
        hold_one = model(source, weights=weights, rebalance='buy_and_hold').summarize()
        np.testing.assert_allclose(a.scenarios.portfolio_return, hold_one.scenarios.portfolio_return)
        self.assertAlmostEqual(a.risk.portfolio_std_return.iloc[0], np.std([.075, .025, .0075], ddof=1))

    def test_missing_blocks_zeros_and_raw_lookback(self) -> None:
        source = frame([[0, 0], [.1, 0], [np.nan, 0], [0, .1], [0, 0], [.02, 0]])
        result = model(source, horizon_sessions=2).summarize()
        self.assertEqual(result.risk.n_candidate_blocks.iloc[0], 5)
        self.assertEqual(result.risk.n_excluded_blocks.iloc[0], 2)
        self.assertEqual(len(result.scenarios), 3)
        self.assertEqual(result.scenarios.scenario_start.tolist(), source.Date.iloc[[0, 3, 4]].tolist())
        self.assertEqual(len(model(source.dropna(), horizon_sessions=2).summarize().scenarios), 4)
        zero_weight = model(source, horizon_sessions=2, weights={'A': 0, 'B': 1}).summarize()
        self.assertEqual(len(zero_weight.scenarios), 3)
        with self.assertRaisesRegex(ValueError, 'n_candidate_blocks.*3.*n_excluded_blocks.*1'):
            HistoricalSimulation(HistoricalSimulationConfig(
                horizon_sessions=2, lookback_sessions=4, min_scenarios=3)).fit(source).summarize()

    def test_log_equivalence_and_overflow(self) -> None:
        source = frame([[.1, -.02], [0, np.nan], [-.05, .01], [.03, 0]])
        logs = source.copy()
        logs[['A', 'B']] = np.log1p(source[['A', 'B']])
        a = model(source).summarize()
        b = model(logs, return_type='log').summarize()
        np.testing.assert_allclose(a.scenarios.portfolio_return, b.scenarios.portfolio_return)
        np.testing.assert_allclose(a.risk[['var', 'es']], b.risk[['var', 'es']])
        with self.assertRaisesRegex(ValueError, 'conversion'):
            model(frame([1000, 0]), return_type='log')
        with self.assertRaises(FloatingPointError):
            model(frame([1e300, 1e300, 1e300]), horizon_sessions=2).summarize()
        with self.assertRaises(FloatingPointError):
            model(frame([1e300, 0, 1e300])).summarize()


class TestValidationAndState(unittest.TestCase):
    def test_invalid_configuration(self) -> None:
        invalid = [
            {'assets': ()}, {'assets': ('A', 'A')}, {'assets': ('Date',)},
            {'return_type': 'percent'}, {'rebalance': 'weekly'}, {'estimation': 'other'},
            {'horizon_sessions': 0}, {'horizon_sessions': True}, {'horizon_sessions': 1.5},
            {'min_scenarios': 1}, {'min_scenarios': False},
            {'confidence_levels': ()}, {'confidence_levels': (.95, .95)},
            {'confidence_levels': (0.,)}, {'confidence_levels': (1.,)},
            {'confidence_levels': (np.nan,)}, {'confidence_levels': (True,)},
            {'lookback_sessions': 0}, {'lookback_sessions': True},
            {'lookback_sessions': 1, 'horizon_sessions': 2},
            {'estimation': 'rolling'}, {'estimation': 'expanding', 'lookback_sessions': 3},
            {'weights': {'A': -.1, 'B': 1.1}}, {'weights': {'A': np.inf}},
            {'weights': {'A': .9}}, {'weights': {'A': True}},
            {'assets': ('A',), 'weights': {'B': 1.}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HistoricalSimulationConfig(**kwargs)
        config = HistoricalSimulationConfig(weights={'A': 1 + 5e-11})
        result = HistoricalSimulation(config).fit(frame([.1, .2] * 50)).summarize()
        self.assertEqual(result.metadata['configuration']['weights']['A'], 1 + 5e-11)

    def test_invalid_inputs_and_selection(self) -> None:
        source = frame([[0, .1], [.1, .2], [.2, .3]])
        invalid = [source.drop(columns='Date'), source.iloc[[1, 0, 2]],
                   pd.concat([source.iloc[:1], source]), source.iloc[:0],
                   source.assign(Date=['2024-1-01', '2024-01-02', '2024-01-03']),
                   source.assign(Date=['2024-02-30', '2024-03-01', '2024-03-02']),
                   source.assign(Date=[None] * 3), source.assign(Date=[1, 2, 3]),
                   source.assign(Date=pd.date_range('2024-01-01', periods=3, tz='UTC')),
                   source.assign(Date=pd.date_range('2024-01-01 01:00', periods=3)),
                   source.assign(A=[-.01, -1.01, 0]), source.assign(A=[0, np.inf, 0]),
                   source.assign(A=[0, -np.inf, 0]), source.assign(A=['oops', 0, 0]),
                   source.assign(A=[True, False, True]), source.assign(A=[1j, 0j, 0j]),
                   source.assign(A=pd.date_range('2024-01-01', periods=3)),
                   source.assign(A=pd.to_timedelta([1, 2, 3], unit='D')),
                   source.rename(columns={'B': 'A'})]
        for values in invalid:
            with self.subTest(values=str(values)), self.assertRaises(ValueError):
                model(values)
        with self.assertRaisesRegex(ValueError, 'Missing selected'):
            model(source, assets=('C',))
        with self.assertRaisesRegex(ValueError, 'Weight keys'):
            model(source, weights={'A': 1})
        result = model(source.assign(extra='ignored'), assets=('B', 'A')).summarize()
        self.assertEqual(result.metadata['configuration']['assets'], ['B', 'A'])
        self.assertEqual(len(model(source.assign(Date=[date(2024, 1, i) for i in (1, 2, 3)])).summarize().scenarios), 3)

    def test_lifecycle_and_provenance(self) -> None:
        benchmark = HistoricalSimulation(HistoricalSimulationConfig(min_scenarios=2))
        for operation in (benchmark.summarize, lambda: benchmark.backtest(frame([0]))):
            with self.assertRaises(RuntimeError):
                operation()
        source = frame([0, .1, .2])
        weights = {'A': 1.}
        config = replace(benchmark.config, weights=weights)
        weights['A'] = 0
        benchmark = HistoricalSimulation(config).fit(source)
        original = benchmark.summarize()
        source.loc[0, 'A'] = .9
        assert_frame_equal(original.scenarios, benchmark.summarize().scenarios)
        benchmark.backtest(frame([0, .1], '2024-02-01'))
        assert_frame_equal(original.risk, benchmark.summarize().risk)
        original.scenarios.loc[0, 'loss'] = 999
        self.assertEqual(benchmark.summarize().scenarios.loss.iloc[0], 0)
        self.assertIs(benchmark.fit(frame([0, 0])), benchmark)
        self.assertEqual(len(benchmark.summarize().scenarios), 2)
        benchmark.fit(frame([np.nan]))
        with self.assertRaisesRegex(ValueError, 'assets=.*horizon=.*n_scenarios.*0'):
            benchmark.summarize()

    def test_csv_dataframe_fingerprints_and_no_writes(self) -> None:
        source = frame([[0, .1], [np.nan, .02], [.2, 0]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'returns.csv'
            source.to_csv(path, index=False)
            before = path.read_bytes()
            a = model(path).summarize()
            b = model(pd.read_csv(path)).summarize()
            assert_frame_equal(a.scenarios, b.scenarios)
            assert_frame_equal(a.risk, b.risk)
            self.assertEqual(a.metadata['inputs']['fit']['fingerprint'],
                             b.metadata['inputs']['fit']['fingerprint'])
            self.assertEqual(a.metadata['inputs']['fit']['source'], str(path.resolve()))
            self.assertIsNone(b.metadata['inputs']['fit']['source'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            path.write_text('Date,A,A\n2024-01-01,0,0\n')
            with self.assertRaisesRegex(ValueError, 'unique'):
                model(path)
        json.dumps(a.metadata, allow_nan=False)
        changed = model(source.assign(A=[0, np.nan, .3])).summarize()
        self.assertNotEqual(a.metadata['inputs']['fit']['fingerprint'],
                            changed.metadata['inputs']['fit']['fingerprint'])


class TestChronology(unittest.TestCase):
    def test_breaches_and_realized_compounding(self) -> None:
        result = model(frame([0, -.1, -.2, -.3]), confidence_levels=(.5,)).backtest(
            frame([-.2, -.1, 0], '2024-02-01'))
        self.assertEqual(result.forecasts.breach.tolist(), [True, False, False])
        self.assertEqual(result.coverage.n_breaches.iloc[0], 1)
        self.assertAlmostEqual(result.coverage.breach_rate.iloc[0], 1 / 3)
        for rule, expected in (('daily', .1025), ('buy_and_hold', .1)):
            result = model(frame([[0, 0]] * 3), horizon_sessions=2,
                           rebalance=rule).backtest(frame([[.1, 0], [0, .1]], '2024-02-01'))
            np.testing.assert_allclose(result.forecasts.realized_return, expected)

    def test_modes_and_future_invariance(self) -> None:
        history = frame([-.1, .02, -.03, .04])
        evaluation = frame([-.5, .3, -.2, .1], '2024-01-05')
        for estimation in ('fixed', 'expanding', 'rolling'):
            kwargs = {'estimation': estimation, 'confidence_levels': (.5, .95)}
            if estimation == 'rolling':
                kwargs['lookback_sessions'] = 3
            benchmark = model(history, **kwargs)
            result = benchmark.backtest(evaluation)
            forecasts = result.forecasts
            for target in range(len(evaluation)):
                changed = evaluation.copy()
                changed.loc[target:, 'A'] = .8
                alternate = benchmark.backtest(changed).forecasts
                columns = ['var', 'es', 'n_scenarios', 'history_start', 'history_end']
                assert_frame_equal(forecasts.iloc[target * 2:target * 2 + 2][columns],
                                   alternate.iloc[target * 2:target * 2 + 2][columns])
                available = history if estimation == 'fixed' else pd.concat([history, evaluation.iloc[:target]])
                expected = model(available, **kwargs).summarize().risk
                np.testing.assert_allclose(forecasts.iloc[target * 2:target * 2 + 2][['var', 'es']],
                                           expected[['var', 'es']])
            self.assertEqual(forecasts.forecast_origin.iloc[0], history.Date.iloc[-1])
            if estimation == 'fixed':
                self.assertEqual(forecasts['var'].nunique(), 2)
                self.assertTrue(forecasts.history_end.eq(history.Date.iloc[-1]).all())
            else:
                self.assertNotEqual(forecasts['var'].iloc[1], forecasts['var'].iloc[3])
            json.dumps(result.metadata, allow_nan=False)

    def test_missing_realizations_and_insufficient_histories(self) -> None:
        history = frame([np.nan, 0])
        evaluation = frame([0, np.nan, 0, 0, 0], '2024-01-03')
        result = model(history, estimation='expanding', confidence_levels=(.5,)).backtest(evaluation)
        rows = result.forecasts
        self.assertEqual(rows.estimate_status.tolist(), ['insufficient_history', 'ok', 'ok', 'ok', 'ok'])
        self.assertEqual(rows.realization_status.tolist(), ['ok', 'missing_returns', 'ok', 'ok', 'ok'])
        self.assertTrue(pd.isna(rows.breach.iloc[0]))
        self.assertTrue(pd.isna(rows.breach.iloc[1]))
        self.assertFalse(rows.breach.iloc[2])  # Equal loss and VaR is not a breach.
        self.assertEqual(rows.realized_return.iloc[0], 0)
        coverage = result.coverage.iloc[0]
        self.assertEqual((coverage.n_targets, coverage.n_estimates, coverage.n_comparisons), (5, 4, 3))
        self.assertEqual(coverage.breach_rate, 0)
        fixed = model(history).backtest(evaluation)
        self.assertTrue(fixed.forecasts['var'].isna().all())
        self.assertTrue(fixed.coverage.breach_rate.isna().all())
        self.assertEqual(fixed.metadata['counts']['n_insufficient_history_targets'], 5)

    def test_multisession_future_invariance_and_raw_rolling_bounds(self) -> None:
        history = frame([0, np.nan, .02, .03, -.01])
        evaluation = frame([.05, .1, -.2, .3], '2024-01-06')
        for mode in ('fixed', 'expanding', 'rolling'):
            kwargs = {'estimation': mode, 'horizon_sessions': 2}
            if mode == 'rolling':
                kwargs['lookback_sessions'] = 4
            benchmark = model(history, **kwargs)
            original = benchmark.backtest(evaluation).forecasts
            for target in range(3):
                changed = evaluation.copy()
                changed.loc[target:, 'A'] = -.9
                other = benchmark.backtest(changed).forecasts
                fields = ['var', 'es', 'n_scenarios', 'n_candidate_blocks',
                          'n_excluded_blocks', 'history_start', 'history_end']
                assert_frame_equal(original.iloc[2 * target:2 * target + 2][fields],
                                   other.iloc[2 * target:2 * target + 2][fields])
            if mode == 'rolling':
                self.assertEqual(original.n_history_sessions.tolist(), [4] * 6)
                self.assertEqual(original.history_start.iloc[0], history.Date.iloc[1])
                self.assertEqual(original.n_excluded_blocks.iloc[0], 1)
                self.assertEqual(original.history_start.iloc[2], history.Date.iloc[2])
                self.assertEqual(original.n_excluded_blocks.iloc[2], 0)

    def test_overlapping_targets_boundaries_and_short_inputs(self) -> None:
        history = frame([0, .1, -.1, .2])
        evaluation = frame([.1, np.nan, -.1, .1], '2024-01-05')
        benchmark = model(history, horizon_sessions=2, estimation='rolling', lookback_sessions=3)
        result = benchmark.backtest(evaluation)
        self.assertEqual(result.metadata['counts']['n_trailing_excluded'], 1)
        self.assertEqual(len(result.forecasts), 6)
        self.assertEqual(result.coverage.n_comparisons.tolist(), [0, 0])
        # First two target blocks are incomplete; last has insufficient history.
        self.assertEqual(result.forecasts.realization_status.iloc[::2].tolist(),
                         ['missing_returns', 'missing_returns', 'ok'])
        self.assertEqual(result.forecasts.n_candidate_blocks.tolist(), [2] * 6)
        for values in (evaluation.iloc[:1], evaluation.iloc[:0]):
            short = benchmark.backtest(values)
            self.assertTrue(short.forecasts.empty)
            self.assertEqual(short.forecasts.columns.tolist(), result.forecasts.columns.tolist())
            self.assertEqual(str(short.forecasts.breach.dtype), 'boolean')
            self.assertTrue(short.coverage.n_targets.eq(0).all())
            self.assertTrue(short.coverage.breach_rate.isna().all())
            json.dumps(short.metadata, allow_nan=False)
        for invalid in (history.iloc[-1:], pd.concat([history.iloc[-1:], evaluation]),
                        evaluation.drop(columns='A')):
            with self.assertRaises(ValueError):
                benchmark.backtest(invalid)


class TestExistingCSVs(unittest.TestCase):
    def test_independent_window_counts_and_fixed_validation(self) -> None:
        directory = Path(__file__).resolve().parents[1] / 'sandbox/4stocks_experiment/data'
        expected = {'train': (3811, 3208, 1679), 'validation': (817, 802, 709),
                    'test': (817, 817, 798)}
        for split, (n_rows, h1, h20) in expected.items():
            path = directory / f'{split}_returns.csv'
            source = pd.read_csv(path)
            self.assertEqual(len(source), n_rows)
            values = source.drop(columns='Date').to_numpy()
            for horizon, n_eligible in ((1, h1), (20, h20)):
                eligible = [i for i in range(len(source) - horizon + 1)
                            if np.isfinite(values[i:i + horizon]).all()]
                self.assertEqual(len(eligible), n_eligible)
                result = HistoricalSimulation(HistoricalSimulationConfig(
                    horizon_sessions=horizon)).fit(path).summarize()
                self.assertEqual(len(result.scenarios), n_eligible)
                self.assertEqual(result.risk.n_candidate_blocks.iloc[0], n_rows - horizon + 1)
                self.assertEqual(result.risk.n_excluded_blocks.iloc[0], n_rows - horizon + 1 - n_eligible)
                returns = [np.prod(1 + values[i:i + horizon].mean(axis=1)) - 1 for i in eligible]
                np.testing.assert_allclose(result.scenarios.portfolio_return, returns, atol=1e-14)
        benchmark = HistoricalSimulation(HistoricalSimulationConfig(horizon_sessions=20)).fit(
            directory / 'train_returns.csv')
        with patch.object(benchmark, '_scenarios', wraps=benchmark._scenarios) as build:
            result = benchmark.backtest(directory / 'validation_returns.csv')
            self.assertEqual(build.call_count, 1)
        self.assertEqual(len(result.forecasts), 1596)
        self.assertEqual(result.coverage.n_targets.tolist(), [798, 798])
        self.assertEqual(result.coverage.n_estimates.tolist(), [798, 798])
        self.assertEqual(result.coverage.n_comparisons.tolist(), [709, 709])
        self.assertEqual(result.metadata['counts']['n_trailing_excluded'], 19)
        self.assertEqual(result.metadata['counts']['n_missing_realization_targets'], 89)
        self.assertEqual(benchmark.summarize().risk.small_tail_sample.tolist(), [False, True])
        json.dumps(result.metadata, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
