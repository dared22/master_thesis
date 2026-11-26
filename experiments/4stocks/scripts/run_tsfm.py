"""Run zero-shot TSFM portfolio risk benchmarks (Sundial / Timestral) on 4stocks splits.

Requires the optional TSFM extra:
    uv run --extra tsfm python experiments/4stocks/scripts/run_tsfm.py --forecaster sundial
    uv run --extra tsfm python experiments/4stocks/scripts/run_tsfm.py --forecaster timestral
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.tsfm import (
    DEFAULT_MODEL_IDS, SundialForecaster, TSFMConfig, TSFMForecastBenchmark, TimestralForecaster,
)

CONFIDENCE_LEVELS = (0.95, 0.99)


class _ProgressBase:
    """Shared tqdm progress-bar base for forecaster wrappers."""

    def __init__(self, inner: object, split: str, total_targets: int) -> None:
        self._inner = inner
        self.name = inner.name
        self._bar = tqdm(
            total=total_targets,
            desc=f'TSFM ({self.name}) {split}',
            unit='origin',
            dynamic_ncols=True,
        )

    def metadata(self) -> dict:
        return self._inner.metadata()

    def close(self) -> None:
        self._bar.close()


class _SampleProgressForecaster(_ProgressBase):
    def sample(self, contexts: np.ndarray, horizon: int, num_samples: int, seed: int) -> np.ndarray:
        out = self._inner.sample(contexts, horizon, num_samples, seed)
        self._bar.update(len(contexts))
        return out


class _QuantileProgressForecaster(_ProgressBase):
    def predict_quantiles(self, contexts: np.ndarray, horizon: int) -> tuple[np.ndarray, tuple[float, ...]]:
        out = self._inner.predict_quantiles(contexts, horizon)
        self._bar.update(len(contexts))
        return out


def build_forecaster(config: TSFMConfig) -> SundialForecaster | TimestralForecaster:
    """Instantiate the lazy TSFM forecaster matching config.forecaster."""
    assert config.model_id is not None
    if config.forecaster == 'timestral':
        return TimestralForecaster(
            model_id=config.model_id,
            device=config.device,
            dtype=config.dtype,
            revision=config.revision,
        )
    assert config.revin is not None
    return SundialForecaster(
        model_id=config.model_id,
        device=config.device,
        dtype=config.dtype,
        sample_chunk_size=config.sample_chunk_size,
        revin=config.revin,
        revision=config.revision,
    )


def wrap_progress(
    inner: SundialForecaster | TimestralForecaster,
    split: str,
    total_targets: int,
) -> _SampleProgressForecaster | _QuantileProgressForecaster:
    if isinstance(inner, TimestralForecaster):
        return _QuantileProgressForecaster(inner, split, total_targets)
    return _SampleProgressForecaster(inner, split, total_targets)


def pinball_loss_pct(realized: np.ndarray, var: np.ndarray, confidence: float) -> float:
    """Mean quantile (pinball) loss scaled by 100."""
    residual = realized - var
    return float(100.0 * np.mean((confidence - (residual < 0)) * residual))


def run_split(
    split: str,
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: TSFMConfig,
    preprocessing: dict,
    output_dir: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Run causal walk-forward TSFM backtest for one split and save CSV + metadata."""
    forecasts_path = output_dir / f'{split}_forecasts.csv'
    coverage_path = output_dir / f'{split}_coverage.csv'
    meta_path = output_dir / f'{split}_metadata.json'

    protocol = {
        'split': split,
        'preprocessing': preprocessing,
        'target': config.target,
        'horizon_sessions': config.horizon_sessions,
        'context_sessions': config.context_sessions,
        'num_samples': config.num_samples,
        'target_stride': config.target_stride,
        'batch_size': config.batch_size,
        'sample_chunk_size': config.sample_chunk_size,
        'seed': config.seed,
        'model_id': config.model_id,
        'revin': config.revin,
        'revision': config.revision,
        'history_sha256': hashlib.sha256(
            history_df.to_csv(index=False).encode('utf-8')).hexdigest(),
        'eval_sha256': hashlib.sha256(
            eval_df.to_csv(index=False).encode('utf-8')).hexdigest(),
    }
    if not force and forecasts_path.is_file() and meta_path.is_file():
        previous = json.loads(meta_path.read_text())
        if previous.get('protocol') == protocol:
            logging.info('Already completed matching TSFM %s run: %s', split, forecasts_path)
            return pd.read_csv(forecasts_path, parse_dates=['forecast_origin', 'target_start', 'target_end'])

    output_dir.mkdir(parents=True, exist_ok=True)
    n_available = max(0, len(eval_df) - config.horizon_sessions + 1)
    n_targets = len(range(0, n_available, config.target_stride))
    logging.info('START TSFM (%s) %s: model=%s target=%s revin=%s context=%d samples=%d stride=%d batch=%d origins=%d device=%s',
                 config.forecaster, split, config.model_id, config.target, config.revin, config.context_sessions,
                 config.num_samples, config.target_stride, config.batch_size, n_targets, config.device)
    started = perf_counter()
    inner_forecaster = build_forecaster(config)
    forecaster = wrap_progress(inner_forecaster, split, n_targets)
    try:
        benchmark = TSFMForecastBenchmark(config, forecaster=forecaster).fit(history_df)
        result = benchmark.backtest(eval_df)
    finally:
        forecaster.close()
    elapsed = perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_csv(forecasts_path, index=False)
    result.coverage.to_csv(coverage_path, index=False)
    meta_path.write_text(json.dumps({
        'protocol': protocol,
        'elapsed_seconds': elapsed,
        'benchmark_metadata': result.metadata,
    }, indent=2, allow_nan=False) + '\n')

    logging.info('DONE TSFM (%s) %s in %.1fs (%d forecast rows)',
                 config.forecaster, split, elapsed, len(result.forecasts))
    for confidence in CONFIDENCE_LEVELS:
        sub = result.forecasts.loc[
            result.forecasts.confidence.eq(confidence)
            & result.forecasts.estimate_status.eq('ok')
            & result.forecasts.realization_status.eq('ok')
        ]
        ql = pinball_loss_pct(sub['realized_loss'].to_numpy(), sub['var'].to_numpy(), confidence)
        br = 100.0 * float(sub['breach'].mean())
        mean_es = float(sub['es'].mean())
        es_text = f'{100.0 * mean_es:.2f}%' if np.isfinite(mean_es) else 'n/a'
        logging.info('  %s %.0f%%: targets=%d mean_VaR=%.2f%% mean_ES=%s breach=%.2f%% QL=%.4f',
                     split, 100 * confidence, len(sub),
                     100.0 * float(sub['var'].mean()), es_text, br, ql)
    return result.forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecaster', choices=tuple(DEFAULT_MODEL_IDS), default='sundial')
    parser.add_argument('--target', choices=('log_price', 'return'), default='log_price')
    parser.add_argument('--context-sessions', type=int, default=252)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--target-stride', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sample-chunk-size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-id', default=None,
                        help='Hugging Face repo id (defaults to the selected forecaster default).')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda', 'auto'), default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Also run causal walk-forward TSFM backtest on the reserved test split.')
    parser.add_argument('--force', action='store_true', help='Re-run even if matching output exists.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    preprocessing = json.loads((DIRECTORY / 'data/preprocessing.json').read_text())
    stocks = tuple(preprocessing['stock_order'])
    horizon = int(preprocessing['window_length'])

    config = TSFMConfig(
        assets=stocks,
        weights={stock: 1.0 / len(stocks) for stock in stocks},
        return_type='simple',
        target=args.target,
        horizon_sessions=horizon,
        rebalance='daily',
        confidence_levels=CONFIDENCE_LEVELS,
        context_sessions=args.context_sessions,
        num_samples=args.num_samples,
        target_stride=args.target_stride,
        batch_size=args.batch_size,
        sample_chunk_size=args.sample_chunk_size,
        seed=args.seed,
        forecaster=args.forecaster,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )

    train_df = pd.read_csv(DIRECTORY / 'data/train_returns.csv', parse_dates=['Date'])
    val_df = pd.read_csv(DIRECTORY / 'data/validation_returns.csv', parse_dates=['Date'])
    output_dir = DIRECTORY / 'results/tsfm' / args.forecaster

    run_split('validation', train_df, val_df, config, preprocessing, output_dir, force=args.force)
    if args.evaluate_test:
        test_df = pd.read_csv(DIRECTORY / 'data/test_returns.csv', parse_dates=['Date'])
        causal_history = pd.concat([train_df, val_df], ignore_index=True)
        run_split('test', causal_history, test_df, config, preprocessing, output_dir, force=args.force)


if __name__ == '__main__':
    main()
