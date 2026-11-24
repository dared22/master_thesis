"""Offline unit tests for benchmarks.tsfm (no torch/transformers required).

Run: uv run python -m unittest benchmarks.test_tsfm
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import benchmarks
from benchmarks import (
    HistoricalSimulation,
    HistoricalSimulationConfig,
    SundialForecaster,
    TSFMConfig,
    TSFMForecastBenchmark,
)
from benchmarks._common import _empirical_es, _empirical_var


def frame(values: object, start: str = '2024-01-01') -> pd.DataFrame:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    result = pd.DataFrame(array, columns=list('ABC')[:array.shape[1]])
    result.insert(0, 'Date', pd.date_range(start, periods=len(result)))
    return result


class ConstantForecaster:
    """Deterministic mock forecaster returning fixed trajectories per batch row."""

    name = 'mock_constant'

    def __init__(self, trajectories: np.ndarray) -> None:
        # shape (num_samples, horizon)
        self.trajectories = np.asarray(trajectories, dtype=np.float64)
        self.calls: list[dict[str, object]] = []

    def metadata(self) -> dict[str, object]:
        return {'forecaster': self.name, 'package_versions': {}}

    def sample(self, contexts: np.ndarray, horizon: int, num_samples: int, seed: int) -> np.ndarray:
        self.calls.append({
            'contexts': contexts.copy(),
            'horizon': horizon,
            'num_samples': num_samples,
            'seed': seed,
        })
        batch = contexts.shape[0]
        return np.broadcast_to(self.trajectories[None, :, :], (batch, num_samples, horizon)).copy()


class ContextEchoForecaster:
    """Deterministic mock forecaster whose samples depend on context mean and seed."""

    name = 'mock_echo'

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def metadata(self) -> dict[str, object]:
        return {'forecaster': self.name, 'package_versions': {}}

    def sample(self, contexts: np.ndarray, horizon: int, num_samples: int, seed: int) -> np.ndarray:
        self.calls.append({
            'contexts': contexts.copy(),
            'horizon': horizon,
            'num_samples': num_samples,
            'seed': seed,
        })
        batch = contexts.shape[0]
        offsets = np.linspace(-0.05, 0.05, num_samples, dtype=np.float64)
        base = contexts.mean(axis=1)[:, None, None]
        steps = np.arange(1, horizon + 1, dtype=np.float64)[None, None, :] / horizon
        return (base + offsets[None, :, None]) * steps


class TestLazyImportAndValidation(unittest.TestCase):
    def test_lazy_import_does_not_load_torch(self) -> None:
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        forecaster = SundialForecaster()
        self.assertEqual(forecaster.name, 'sundial')
        self.assertEqual(forecaster.metadata()['model_id'], 'thuml/sundial-base-128m')
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        self.assertTrue(hasattr(benchmarks, 'TSFMForecastBenchmark'))

    def test_invalid_configuration(self) -> None:
        invalid = [
            {'assets': ()}, {'assets': ('A', 'A')}, {'assets': ('Date',)},
            {'return_type': 'percent'}, {'target': 'price'}, {'rebalance': 'buy_and_hold'},
            {'device': 'tpu'}, {'dtype': 'float64'}, {'model_id': ''},
            {'revin': 'yes'}, {'revision': ''},
            {'horizon_sessions': 0}, {'horizon_sessions': True},
            {'context_sessions': 1}, {'context_sessions': False},
            {'num_samples': 1}, {'num_samples': True},
            {'target_stride': 0}, {'batch_size': 0}, {'seed': -1},
            {'sample_chunk_size': 0},
            {'confidence_levels': ()}, {'confidence_levels': (0.95, 0.95)},
            {'confidence_levels': (0.0,)}, {'confidence_levels': (1.0,)},
            {'weights': {'A': -0.1, 'B': 1.1}}, {'weights': {'A': 0.9}},
            {'assets': ('A',), 'weights': {'B': 1.0}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TSFMConfig(**kwargs)
        with self.assertRaises(ValueError):
            SundialForecaster(model_id='')
        with self.assertRaises(ValueError):
            SundialForecaster(device='tpu')
        with self.assertRaises(ValueError):
            SundialForecaster(sample_chunk_size=0)


class TestContextAndTargetMath(unittest.TestCase):
    def test_log_price_window_uses_all_L_returns_and_matches_return_mode(self) -> None:
        daily = np.array([0.05, -0.02, 0.03, -0.01])
        source = frame(daily)
        sample_returns = np.array([
            [0.02, -0.01],
            [-0.03, -0.02],
            [0.01, 0.04],
            [-0.05, 0.00],
        ])
        sample_log_prices = np.cumsum(np.log1p(sample_returns), axis=1)

        ret_mock = ConstantForecaster(sample_returns)
        lp_mock = ConstantForecaster(sample_log_prices)

        ret_bench = TSFMForecastBenchmark(
            TSFMConfig(target='return', horizon_sessions=2, context_sessions=4,
                       num_samples=4, confidence_levels=(0.5, 0.75)),
            forecaster=ret_mock,
        ).fit(source)
        lp_bench = TSFMForecastBenchmark(
            TSFMConfig(target='log_price', horizon_sessions=2, context_sessions=4,
                       num_samples=4, confidence_levels=(0.5, 0.75)),
            forecaster=lp_mock,
        ).fit(source)

        ret_sum = ret_bench.summarize()
        lp_sum = lp_bench.summarize()

        # Verify context lengths: L for return, L + 1 ending at 0 for log_price.
        ret_ctx = ret_mock.calls[0]['contexts'][0]
        lp_ctx = lp_mock.calls[0]['contexts'][0]
        self.assertEqual(len(ret_ctx), 4)
        self.assertEqual(len(lp_ctx), 5)
        self.assertEqual(lp_ctx[-1], 0.0)
        np.testing.assert_allclose(np.diff(lp_ctx), np.log1p(daily), atol=1e-15)
        np.testing.assert_allclose(ret_ctx, daily, atol=1e-15)

        # Verify identical terminal returns, VaR, and ES on equivalent trajectories.
        np.testing.assert_allclose(lp_sum.scenarios.portfolio_return,
                                   ret_sum.scenarios.portfolio_return, atol=1e-14)
        np.testing.assert_allclose(lp_sum.risk[['var', 'es']],
                                   ret_sum.risk[['var', 'es']], atol=1e-14)
        self.assertEqual(lp_sum.metadata['context_vector_length'], 5)
        self.assertEqual(ret_sum.metadata['context_vector_length'], 4)

    def test_limited_liability_clamping_and_nonfinite_rejection(self) -> None:
        source = frame([0.01, -0.01, 0.02])
        # Path 0: two -120% steps (would bounce positive without limited liability)
        # Path 1: hits -100% on step 1
        # Path 2: normal path (+10%, -5% -> +4.5%)
        # Path 3: normal loss (-20%, -10% -> -28%)
        paths = np.array([
            [-1.2, -1.2],
            [-1.0, 0.5],
            [0.10, -0.05],
            [-0.20, -0.10],
        ])
        mock = ConstantForecaster(paths)
        summary = TSFMForecastBenchmark(
            TSFMConfig(target='return', horizon_sessions=2, context_sessions=3,
                       num_samples=4, confidence_levels=(0.5, 0.75)),
            forecaster=mock,
        ).fit(source).summarize()
        expected_returns = np.array([-1.0, -1.0, 0.045, -0.28])
        np.testing.assert_allclose(summary.scenarios.portfolio_return, expected_returns, atol=1e-14)
        self.assertEqual(summary.risk.n_clamped_paths.tolist(), [2, 2])
        self.assertEqual(summary.metadata['counts']['n_clamped_paths'], 2)
        losses = np.sort(-expected_returns)
        self.assertAlmostEqual(summary.risk['var'].iloc[0], _empirical_var(losses, 0.5))
        self.assertAlmostEqual(summary.risk['es'].iloc[1], _empirical_es(losses, 0.75))

        # Nonfinite output must raise FloatingPointError.
        bad = ConstantForecaster(np.array([[np.nan, 0.0], [0.0, 0.0]]))
        with self.assertRaises(FloatingPointError):
            TSFMForecastBenchmark(
                TSFMConfig(target='return', horizon_sessions=2, context_sessions=3, num_samples=2),
                forecaster=bad,
            ).fit(source).summarize()

        # Overflow in expm1 must raise FloatingPointError.
        huge = ConstantForecaster(np.array([[1000.0], [0.0]]))
        with self.assertRaises(FloatingPointError):
            TSFMForecastBenchmark(
                TSFMConfig(target='log_price', horizon_sessions=1, context_sessions=3, num_samples=2),
                forecaster=huge,
            ).fit(source).summarize()

        # Wrong shape from forecaster must raise ValueError.
        wrong_shape = ConstantForecaster(np.zeros((3, 1)))
        with self.assertRaisesRegex(ValueError, 'shape'):
            TSFMForecastBenchmark(
                TSFMConfig(target='return', horizon_sessions=2, context_sessions=3, num_samples=2),
                forecaster=wrong_shape,
            ).fit(source).summarize()

    def test_log_input_equivalence_and_named_weights(self) -> None:
        source = frame([[0.10, -0.02], [0.0, 0.04], [-0.03, 0.01]])
        logs = source.copy()
        logs[['A', 'B']] = np.log1p(source[['A', 'B']])
        weights = {'A': 0.75, 'B': 0.25}
        a = TSFMForecastBenchmark(
            TSFMConfig(weights=weights, context_sessions=3, num_samples=4),
            forecaster=ContextEchoForecaster(),
        ).fit(source).summarize()
        b = TSFMForecastBenchmark(
            TSFMConfig(weights=weights, return_type='log', context_sessions=3, num_samples=4),
            forecaster=ContextEchoForecaster(),
        ).fit(logs[['Date', 'B', 'A']]).summarize()
        np.testing.assert_allclose(a.scenarios.portfolio_return, b.scenarios.portfolio_return, atol=1e-14)
        np.testing.assert_allclose(a.risk[['var', 'es']], b.risk[['var', 'es']], atol=1e-14)


class TestMissingContextAndCausality(unittest.TestCase):
    def test_missing_context_skipping_and_total_loss_context(self) -> None:
        source = frame([
            [0.01, 0.02],
            [np.nan, 0.03],  # incomplete: skipped
            [0.02, -0.01],
            [0.00, np.nan],  # incomplete: skipped
            [-0.02, 0.04],
        ])
        mock = ContextEchoForecaster()
        summary = TSFMForecastBenchmark(
            TSFMConfig(target='return', context_sessions=3, num_samples=4,
                       weights={'A': 0.0, 'B': 1.0}),
            forecaster=mock,
        ).fit(source).summarize()
        # Even with weight A=0, rows where A is NaN must still be excluded.
        np.testing.assert_allclose(mock.calls[0]['contexts'][0], [0.02, -0.01, 0.04])
        row = summary.risk.iloc[0]
        self.assertEqual(row.history_start, source.Date.iloc[0])
        self.assertEqual(row.history_end, source.Date.iloc[4])
        self.assertEqual(row.n_history_sessions, 5)
        self.assertEqual(row.n_context_sessions, 3)
        self.assertEqual(row.n_excluded_context_sessions, 2)

        # Insufficient complete rows raises in summarize.
        with self.assertRaisesRegex(ValueError, 'insufficient_history'):
            TSFMForecastBenchmark(
                TSFMConfig(context_sessions=4, num_samples=4),
                forecaster=ContextEchoForecaster(),
            ).fit(source).summarize()

        # Total loss (-1.0) in log_price context marks invalid_log_price_context.
        total_loss = frame([-1.0, 0.01, 0.02])
        with self.assertRaisesRegex(ValueError, 'invalid_log_price_context'):
            TSFMForecastBenchmark(
                TSFMConfig(target='log_price', context_sessions=3, num_samples=4),
                forecaster=ContextEchoForecaster(),
            ).fit(total_loss).summarize()

    def test_future_invariance_batching_stride_and_realized_matching(self) -> None:
        history = frame([0.01, -0.02, np.nan, 0.03, -0.01, 0.02])
        evaluation = frame([-0.04, 0.02, np.nan, -0.01, 0.03, -0.02], '2024-01-07')
        config = TSFMConfig(
            target='log_price', horizon_sessions=2, context_sessions=4,
            num_samples=6, confidence_levels=(0.5, 0.95), batch_size=2, seed=7,
        )
        benchmark = TSFMForecastBenchmark(config, forecaster=ContextEchoForecaster()).fit(history)
        baseline = benchmark.backtest(evaluation)

        # Future invariance: mutating evaluation at or after target_start must not alter its forecast.
        n_targets = len(evaluation) - 2 + 1
        self.assertEqual(baseline.coverage.n_targets.tolist(), [n_targets, n_targets])
        for target in range(n_targets):
            changed = evaluation.copy()
            changed.loc[target:, 'A'] = 0.5
            alt = benchmark.backtest(changed).forecasts
            cols = ['var', 'es', 'n_context_sessions', 'n_excluded_context_sessions',
                    'history_start', 'history_end']
            assert_frame_equal(
                baseline.forecasts.iloc[2 * target:2 * target + 2][cols],
                alt.iloc[2 * target:2 * target + 2][cols],
            )

        # Realized returns and missing_returns flags match HistoricalSimulation identically.
        hs = HistoricalSimulation(HistoricalSimulationConfig(
            horizon_sessions=2, confidence_levels=(0.5, 0.95), min_scenarios=2,
        )).fit(history).backtest(evaluation)
        assert_frame_equal(
            baseline.forecasts[['forecast_origin', 'target_start', 'target_end',
                                'realized_return', 'realized_loss', 'realization_status']],
            hs.forecasts[['forecast_origin', 'target_start', 'target_end',
                          'realized_return', 'realized_loss', 'realization_status']],
        )

        # Stride=2 selects targets 0, 2, 4.
        strided = TSFMForecastBenchmark(
            replace(config, target_stride=2, batch_size=1),
            forecaster=ContextEchoForecaster(),
        ).fit(history).backtest(evaluation)
        self.assertEqual(strided.forecasts.target_start.iloc[::2].tolist(),
                         evaluation.Date.iloc[[0, 2, 4]].tolist())
        self.assertEqual(strided.metadata['counts']['n_targets'], 3)
        self.assertEqual(strided.metadata['counts']['n_trailing_excluded'], 1)
        json.dumps(strided.metadata, allow_nan=False)

    def test_insufficient_history_and_short_evaluation_backtest(self) -> None:
        history = frame([np.nan, 0.01])
        evaluation = frame([0.02, -0.01, 0.03], '2024-01-03')
        benchmark = TSFMForecastBenchmark(
            TSFMConfig(context_sessions=2, num_samples=4, confidence_levels=(0.5,)),
            forecaster=ContextEchoForecaster(),
        ).fit(history)
        result = benchmark.backtest(evaluation)
        self.assertEqual(result.forecasts.estimate_status.tolist(),
                         ['insufficient_history', 'ok', 'ok'])
        self.assertTrue(pd.isna(result.forecasts.breach.iloc[0]))
        self.assertEqual(result.coverage.n_estimates.iloc[0], 2)
        self.assertEqual(result.coverage.n_comparisons.iloc[0], 2)

        # Empty / too-short evaluation tables preserve schema and dtypes.
        short_bench = TSFMForecastBenchmark(
            TSFMConfig(horizon_sessions=3, context_sessions=2, num_samples=4),
            forecaster=ContextEchoForecaster(),
        ).fit(frame([0.01, 0.02]))
        for short_eval in (evaluation.iloc[:2], evaluation.iloc[:0]):
            empty = short_bench.backtest(short_eval)
            self.assertTrue(empty.forecasts.empty)
            self.assertEqual(str(empty.forecasts.breach.dtype), 'boolean')
            self.assertTrue(empty.coverage.n_targets.eq(0).all())
            self.assertTrue(empty.coverage.breach_rate.isna().all())
            json.dumps(empty.metadata, allow_nan=False)

        # Unfitted error, premature evaluation date, immutability, and no file writes.
        unfitted = TSFMForecastBenchmark(TSFMConfig(context_sessions=2, num_samples=4),
                                         forecaster=ContextEchoForecaster())
        with self.assertRaises(RuntimeError):
            unfitted.summarize()
        with self.assertRaises(RuntimeError):
            unfitted.backtest(evaluation)
        with self.assertRaises(ValueError):
            benchmark.backtest(history.iloc[-1:])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'returns.csv'
            history.to_csv(path, index=False)
            TSFMForecastBenchmark(
                TSFMConfig(context_sessions=2, num_samples=4),
                forecaster=ContextEchoForecaster(),
            ).fit(path).backtest(evaluation)
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
