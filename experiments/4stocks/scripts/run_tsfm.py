"""Run the zero-shot Sundial TSFM portfolio risk benchmark on 4stocks splits.

Requires the optional TSFM extra:
    uv run --extra tsfm python experiments/4stocks/scripts/run_tsfm.py
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
    SUNDIAL_DEFAULT_MODEL_ID, SundialForecaster, TSFMConfig, TSFMForecastBenchmark,
)

CONFIDENCE_LEVELS = (0.95, 0.99)


class _ProgressForecaster:
    """Thin tqdm progress-bar wrapper around a SampleForecaster."""

    def __init__(self, inner: object, split: str, total_targets: int) -> None:
        self._inner = inner
        self.name = inner.name
        self._bar = tqdm(total=total_targets, desc=f'TSFM {split}', unit='origin', dynamic_ncols=True)

    def metadata(self) -> dict:
        return self._inner.metadata()

    def sample(self, contexts: np.ndarray, horizon: int, num_samples: int, seed: int) -> np.ndarray:
        out = self._inner.sample(contexts, horizon, num_samples, seed)
        self._bar.update(len(contexts))
        return out

    def close(self) -> None:
        self._bar.close()


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
    logging.info('START TSFM %s: model=%s target=%s context=%d samples=%d stride=%d batch=%d origins=%d device=%s',
                 split, config.model_id, config.target, config.context_sessions,
                 config.num_samples, config.target_stride, config.batch_size, n_targets, config.device)
    started = perf_counter()
    inner_forecaster = SundialForecaster(
        model_id=config.model_id, device=config.device, dtype=config.dtype,
        sample_chunk_size=config.sample_chunk_size, revin=config.revin, revision=config.revision,
    )
    forecaster = _ProgressForecaster(inner_forecaster, split, n_targets)
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

    logging.info('DONE TSFM %s in %.1fs (%d forecast rows)', split, elapsed, len(result.forecasts))
    for confidence in CONFIDENCE_LEVELS:
        sub = result.forecasts.loc[
            result.forecasts.confidence.eq(confidence)
            & result.forecasts.estimate_status.eq('ok')
            & result.forecasts.realization_status.eq('ok')
        ]
        ql = pinball_loss_pct(sub['realized_loss'].to_numpy(), sub['var'].to_numpy(), confidence)
        br = 100.0 * float(sub['breach'].mean())
        logging.info('  %s %.0f%%: targets=%d mean_VaR=%.2f%% mean_ES=%.2f%% breach=%.2f%% QL=%.4f',
                     split, 100 * confidence, len(sub),
                     100.0 * float(sub['var'].mean()), 100.0 * float(sub['es'].mean()), br, ql)
    return result.forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('log_price', 'return'), default='log_price')
    parser.add_argument('--context-sessions', type=int, default=252)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--target-stride', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sample-chunk-size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-id', default=SUNDIAL_DEFAULT_MODEL_ID)
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
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )

    train_df = pd.read_csv(DIRECTORY / 'data/train_returns.csv', parse_dates=['Date'])
    val_df = pd.read_csv(DIRECTORY / 'data/validation_returns.csv', parse_dates=['Date'])
    output_dir = DIRECTORY / 'results/tsfm'

    run_split('validation', train_df, val_df, config, preprocessing, output_dir, force=args.force)
    if args.evaluate_test:
        test_df = pd.read_csv(DIRECTORY / 'data/test_returns.csv', parse_dates=['Date'])
        causal_history = pd.concat([train_df, val_df], ignore_index=True)
        run_split('test', causal_history, test_df, config, preprocessing, output_dir, force=args.force)


if __name__ == '__main__':
    main()
