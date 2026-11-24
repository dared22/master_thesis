"""Evaluate trained SDE-GAN seeds against HistoricalSimulation (fixed & rolling) and TSFM."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = DIRECTORY.parents[1]
sys.path[:0] = [str(DIRECTORY / 'model'), str(ROOT)]

from benchmarks.historical import (
    HistoricalSimulation, HistoricalSimulationConfig, _empirical_es, _empirical_var,
)
from sde_gan import SDEGAN

CONFIDENCE_LEVELS = (0.95, 0.99)
ROLLING_LOOKBACK_SESSIONS = 252
SUNDIAL_RELEASE_DATE = pd.Timestamp('2025-02-01')


def acf(paths: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Pooled within-path autocorrelation at lags 1–5 without crossing path boundaries."""
    values = np.abs(paths) if absolute else paths
    n_stocks = values.shape[2]
    return np.array([
        [np.corrcoef(values[:, :-lag, j].ravel(), values[:, lag:, j].ravel())[0, 1]
         for j in range(n_stocks)]
        for lag in range(1, 6)
    ])


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Compound an equal-weight, daily-rebalanced portfolio."""
    return np.prod(1.0 + paths.mean(axis=2), axis=1) - 1.0


def score_forecast_arrays(
    realized_losses: np.ndarray,
    var: np.ndarray | float,
    es: np.ndarray | float,
    confidence: float,
) -> dict[str, float | int]:
    """Score constant or time-varying VaR/ES forecasts on realized portfolio losses."""
    var_arr = np.broadcast_to(np.asarray(var, dtype=np.float64), realized_losses.shape)
    es_arr = np.broadcast_to(np.asarray(es, dtype=np.float64), realized_losses.shape)
    residual = realized_losses - var_arr
    breaches = realized_losses > var_arr
    return {
        'confidence': confidence,
        'targets': int(len(realized_losses)),
        'var_pct': float(100.0 * np.mean(var_arr)),
        'es_pct': float(100.0 * np.mean(es_arr)),
        'breaches': int(np.sum(breaches)),
        'breach_rate_pct': float(100.0 * np.mean(breaches)),
        'nominal_rate_pct': float(100.0 * (1.0 - confidence)),
        'quantile_loss_pct': float(100.0 * np.mean((confidence - (residual < 0)) * residual)),
    }


def risk_rows(paths: np.ndarray, realized_losses: np.ndarray) -> list[dict[str, float | int]]:
    """Compute VaR, ES, breach rate, and pinball loss using benchmarks/historical conventions."""
    losses = np.sort(-terminal_returns(paths))
    rows = []
    for confidence in CONFIDENCE_LEVELS:
        var = _empirical_var(losses, confidence)
        es = _empirical_es(losses, confidence)
        rows.append(score_forecast_arrays(realized_losses, var, es, confidence))
    return rows


def score_backtest_frame(forecasts: pd.DataFrame) -> list[dict[str, float | int]]:
    """Summarize a dated forecast DataFrame (from Rolling Historical or TSFM backtest)."""
    rows = []
    for confidence in CONFIDENCE_LEVELS:
        sub = forecasts.loc[
            forecasts.confidence.eq(confidence)
            & forecasts.estimate_status.eq('ok')
            & forecasts.realization_status.eq('ok')
        ]
        if sub.empty:
            continue
        rows.append(score_forecast_arrays(
            sub['realized_loss'].to_numpy(dtype=np.float64),
            sub['var'].to_numpy(dtype=np.float64),
            sub['es'].to_numpy(dtype=np.float64),
            confidence,
        ))
    return rows


def load_observed(split: str, stocks: list[str], dates: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(DIRECTORY / 'data' / f'{split}_returns.csv', parse_dates=['Date'])
    indexed = frame.set_index('Date').loc[:, stocks]
    split_dates = dates.loc[dates.split.eq(split)].sort_values('window_index')
    windows = np.stack([indexed.loc[row.start_date:row.end_date].to_numpy()
                        for row in split_dates.itertuples()])
    return frame, windows


def load_tsfm_forecasts(split: str, metadata: dict) -> pd.DataFrame | None:
    """Load saved TSFM forecasts for a split if they exist and match preprocessing."""
    tsfm_dir = DIRECTORY / 'results/tsfm'
    forecasts_path = tsfm_dir / f'{split}_forecasts.csv'
    meta_path = tsfm_dir / f'{split}_metadata.json'
    if not (forecasts_path.is_file() and meta_path.is_file()):
        return None
    saved_meta = json.loads(meta_path.read_text())
    if saved_meta.get('protocol', {}).get('preprocessing') != metadata:
        raise ValueError(f'TSFM {split} metadata does not match current data/preprocessing.json')
    return pd.read_csv(forecasts_path, parse_dates=['forecast_origin', 'target_start', 'target_end'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--n-paths', type=int, default=10_000)
    parser.add_argument('--sample-seed', type=int, default=100)
    parser.add_argument('--checkpoint', default='selected',
                        help="'selected' (lowest val MMD) or a snapshot name like 'step_2000'.")
    parser.add_argument('--rolling-lookback', type=int, default=ROLLING_LOOKBACK_SESSIONS)
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Also score the frozen models on the reserved test split.')
    args = parser.parse_args()

    metadata = json.loads((DIRECTORY / 'data/preprocessing.json').read_text())
    stocks = metadata['stock_order']
    mu = np.asarray(metadata['training_mean'], dtype=np.float64)
    scale = np.asarray(metadata['training_std'], dtype=np.float64)
    dates = pd.read_csv(DIRECTORY / 'data/window_dates.csv', parse_dates=['start_date', 'end_date'])

    eval_splits = ('validation', 'test') if args.evaluate_test else ('validation',)
    frames, observed = {}, {}
    for split in ('train', *eval_splits):
        frames[split], observed[split] = load_observed(split, stocks, dates)

    base_hist_kwargs = dict(
        assets=tuple(stocks),
        weights={stock: 1.0 / len(stocks) for stock in stocks},
        return_type='simple',
        horizon_sessions=metadata['window_length'],
        rebalance='daily',
        confidence_levels=CONFIDENCE_LEVELS,
        min_scenarios=100,
    )
    historical_fixed = HistoricalSimulation(
        HistoricalSimulationConfig(**base_hist_kwargs, estimation='fixed')
    ).fit(frames['train'])
    np.testing.assert_allclose(historical_fixed.summarize().scenarios.portfolio_return,
                               terminal_returns(observed['train']), rtol=0, atol=1e-14)

    train_flat = observed['train'].reshape(-1, len(stocks))
    train_mean, train_std = train_flat.mean(0), train_flat.std(0, ddof=1)
    val_std = observed['validation'].reshape(-1, len(stocks)).std(0, ddof=1)
    train_signed, train_abs = acf(observed['train']), acf(observed['train'], True)
    train_corr = np.corrcoef(train_flat.T)
    pairs = np.triu_indices(len(stocks), 1)

    output = DIRECTORY / 'results/evaluation'
    output.mkdir(parents=True, exist_ok=True)

    risk_records, metric_records, solver_records = [], [], []
    rolling_forecasts: dict[str, pd.DataFrame] = {}
    tsfm_forecasts: dict[str, pd.DataFrame] = {}
    samples, signed_curves, abs_curves = {}, {}, {}

    for split in eval_splits:
        realized = -terminal_returns(observed[split])
        for row in risk_rows(observed['train'], realized):
            risk_records.append({**row, 'split': split, 'model': 'historical_fixed', 'seed': -1})

        causal_history = (frames['train'] if split == 'validation'
                          else pd.concat([frames['train'], frames['validation']], ignore_index=True))
        rolling_bt = HistoricalSimulation(HistoricalSimulationConfig(
            **base_hist_kwargs, estimation='rolling', lookback_sessions=args.rolling_lookback,
        )).fit(causal_history).backtest(frames[split])
        rolling_forecasts[split] = rolling_bt.forecasts
        for row in score_backtest_frame(rolling_bt.forecasts):
            risk_records.append({**row, 'split': split,
                                 'model': f'historical_rolling_{args.rolling_lookback}', 'seed': -1})

        tsfm_df = load_tsfm_forecasts(split, metadata)
        if tsfm_df is not None:
            tsfm_forecasts[split] = tsfm_df
            for row in score_backtest_frame(tsfm_df):
                risk_records.append({**row, 'split': split, 'model': 'tsfm_sundial', 'seed': -1})
        else:
            print(f'No saved TSFM forecasts found for {split} (run scripts/run_tsfm.py to include TSFM).')

    for seed in args.seeds:
        run_dir = DIRECTORY / 'results/sde_gan' / f'seed_{seed}'
        if not (run_dir / 'completion.json').is_file():
            raise FileNotFoundError(f'Missing completed run: {run_dir}')
        load_dir = run_dir if args.checkpoint == 'selected' else run_dir / 'checkpoints' / args.checkpoint
        assert json.loads((load_dir / 'preprocessing.json').read_text()) == metadata
        model = SDEGAN.load(load_dir)
        standardized = model.sample(args.n_paths, args.sample_seed)
        paths = np.expm1(standardized.astype(np.float64) * scale + mu)
        assert np.isfinite(paths).all()
        samples[seed] = paths

        n_ref = min(2000, args.n_paths)
        fine_model = SDEGAN.load(load_dir)
        fine_model.config = replace(fine_model.config, generator_dt=0.125)
        fine_paths = np.expm1(fine_model.sample(n_ref, args.sample_seed).astype(np.float64) * scale + mu)
        val_realized = -terminal_returns(observed['validation'])
        for coarse_r, fine_r in zip(risk_rows(paths[:n_ref], val_realized),
                                    risk_rows(fine_paths, val_realized)):
            solver_records.append({
                'seed': seed, 'confidence': coarse_r['confidence'],
                'var_fine_minus_coarse_pp': fine_r['var_pct'] - coarse_r['var_pct'],
                'es_fine_minus_coarse_pp': fine_r['es_pct'] - coarse_r['es_pct'],
            })

        for split in eval_splits:
            realized = -terminal_returns(observed[split])
            for row in risk_rows(paths, realized):
                risk_records.append({**row, 'split': split, 'model': 'sde_gan', 'seed': seed})

        flat = paths.reshape(-1, len(stocks))
        means, stds = flat.mean(0), flat.std(0, ddof=1)
        signed_curves[seed], abs_curves[seed] = acf(paths), acf(paths, True)
        port_ret = terminal_returns(paths)
        metric_records.append({
            'seed': seed,
            'checkpoint': args.checkpoint,
            'selected_step': model.run_metadata['selected_step'],
            'validation_mmd2': model.run_metadata['validation_score'],
            'mean_rmse_in_training_std': float(np.sqrt(np.mean(((means - train_mean) / train_std) ** 2))),
            'vol_rmse_vs_train': float(np.sqrt(np.mean((stds / train_std - 1.0) ** 2))),
            'vol_rmse_vs_validation': float(np.sqrt(np.mean((stds / val_std - 1.0) ** 2))),
            'signed_acf_rmse_vs_train': float(np.sqrt(np.mean((signed_curves[seed] - train_signed) ** 2))),
            'absolute_acf_rmse_vs_train': float(np.sqrt(np.mean((abs_curves[seed] - train_abs) ** 2))),
            'cross_stock_corr_rmse_vs_train': float(np.sqrt(np.mean((np.corrcoef(flat.T) - train_corr)[pairs] ** 2))),
            'portfolio_mean_pct': float(100.0 * port_ret.mean()),
            'portfolio_std_pct': float(100.0 * port_ret.std(ddof=1)),
        })
        print(f'Evaluated seed {seed} ({args.checkpoint}, step {model.run_metadata["selected_step"]})', flush=True)

    risk_df = pd.DataFrame(risk_records)
    metrics_df = pd.DataFrame(metric_records)
    solver_df = pd.DataFrame(solver_records)
    risk_df.to_csv(output / 'risk_scores.csv', index=False)
    metrics_df.to_csv(output / 'distribution_metrics.csv', index=False)
    solver_df.to_csv(output / 'solver_sensitivity.csv', index=False)

    summary = risk_df.loc[risk_df.model.eq('sde_gan')].groupby(['split', 'confidence']).agg(
        seeds=('seed', 'size'),
        var_mean=('var_pct', 'mean'), var_std=('var_pct', 'std'),
        var_range=('var_pct', lambda x: x.max() - x.min()),
        es_mean=('es_pct', 'mean'),
        breach_mean=('breach_rate_pct', 'mean'),
        breach_range=('breach_rate_pct', lambda x: x.max() - x.min()),
        ql_mean=('quantile_loss_pct', 'mean'), ql_std=('quantile_loss_pct', 'std'),
    )
    summary.to_csv(output / 'seed_summary.csv')

    # Plot 1: Validation quantile loss across models and SDE-GAN seeds.
    val_risk = risk_df.loc[risk_df.split.eq('validation')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for confidence, ax in zip(CONFIDENCE_LEVELS, axes):
        sub = val_risk.loc[val_risk.confidence.eq(confidence)]
        gan_rows = sub.loc[sub.model.eq('sde_gan')]
        bench_rows = sub.loc[~sub.model.eq('sde_gan')]
        labels = [row.model for row in bench_rows.itertuples()] + [f'sde_gan s{row.seed}' for row in gan_rows.itertuples()]
        vals = list(bench_rows.quantile_loss_pct) + list(gan_rows.quantile_loss_pct)
        colors = ['C1'] * len(bench_rows) + ['C0'] * len(gan_rows)
        ax.bar(labels, vals, color=colors)
        ax.set(title=f'{confidence:.0%} validation quantile loss', ylabel='Quantile loss × 100')
        ax.tick_params(axis='x', rotation=25)
    fig.savefig(output / 'validation_quantile_loss.png', dpi=150)
    plt.close(fig)

    # Plot 2: Validation time-series of realized losses vs VaR forecasts (conditional & unconditional).
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    roll_val = rolling_forecasts['validation']
    for confidence, ax in zip(CONFIDENCE_LEVELS, axes):
        rsub = roll_val.loc[roll_val.confidence.eq(confidence)].sort_values('target_start')
        t = rsub['target_start']
        ax.plot(t, 100.0 * rsub['realized_loss'], color='black', alpha=0.45, linewidth=1.0, label='Realized 20d loss')
        fixed_var = val_risk.loc[val_risk.model.eq('historical_fixed') & val_risk.confidence.eq(confidence),
                                 'var_pct'].iloc[0]
        ax.axhline(fixed_var, color='grey', linestyle='--', linewidth=1.3, label=f'Fixed HS ({fixed_var:.2f}%)')
        ax.plot(t, 100.0 * rsub['var'], color='C2', linewidth=1.5,
                label=f'Rolling HS {args.rolling_lookback}d')
        gan_vars = val_risk.loc[val_risk.model.eq('sde_gan') & val_risk.confidence.eq(confidence), 'var_pct']
        ax.axhline(gan_vars.mean(), color='C0', linewidth=1.6, label=f'SDE-GAN mean ({gan_vars.mean():.2f}%)')
        ax.axhspan(gan_vars.min(), gan_vars.max(), color='C0', alpha=0.12)
        if 'validation' in tsfm_forecasts:
            tsub = tsfm_forecasts['validation'].loc[
                tsfm_forecasts['validation'].confidence.eq(confidence)
            ].sort_values('target_start')
            ax.plot(tsub['target_start'], 100.0 * tsub['var'], color='C3', linewidth=1.6, label='TSFM (Sundial)')
            breach_pts = tsub.loc[tsub['breach'].fillna(False)]
            ax.scatter(breach_pts['target_start'], 100.0 * breach_pts['realized_loss'],
                       color='C3', s=14, zorder=5, label='TSFM breach')
        if t.min() <= SUNDIAL_RELEASE_DATE <= t.max():
            ax.axvline(SUNDIAL_RELEASE_DATE, color='C3', linestyle=':', alpha=0.7, label='Sundial release (Feb 2025)')
        ax.set(title=f'{confidence:.0%} VaR vs. realized 20-session portfolio loss (validation)',
               ylabel='20-session loss (%)')
        ax.legend(fontsize=8, ncol=3, loc='upper right')
    axes[-1].set_xlabel('Target window start date')
    fig.savefig(output / 'validation_var_timeline.png', dpi=150)
    plt.close(fig)

    # Plot 3: Signed and absolute ACF (multivariate: Observed vs SDE-GAN).
    fig, axes = plt.subplots(2, len(stocks), figsize=(15, 6), constrained_layout=True)
    for row, (kind, curves) in enumerate((('Signed', signed_curves), ('Absolute', abs_curves))):
        for j, stock in enumerate(stocks):
            ax = axes[row, j]
            for split_name, color in (('train', 'black'), ('validation', 'grey')):
                ax.plot(range(1, 6), acf(observed[split_name], row == 1)[:, j], color=color, label=split_name)
            stacked = np.stack([curves[s][:, j] for s in args.seeds])
            ax.plot(range(1, 6), stacked.mean(0), color='C0', label='SDE-GAN mean')
            ax.fill_between(range(1, 6), stacked.min(0), stacked.max(0), color='C0', alpha=0.18)
            ax.set(title=f'{stock}: {kind}', xlabel='Lag', ylabel='ACF', xticks=range(1, 6))
            ax.legend(fontsize=7)
    fig.savefig(output / 'return_acf.png', dpi=150)
    plt.close(fig)

    (output / 'evaluation.json').write_text(json.dumps({
        'seeds': args.seeds, 'n_paths': args.n_paths, 'sample_seed': args.sample_seed,
        'checkpoint': args.checkpoint, 'rolling_lookback': args.rolling_lookback,
        'tsfm_splits_loaded': sorted(tsfm_forecasts), 'test_evaluated': args.evaluate_test,
    }, indent=2) + '\n')

    print('\n=== DISTRIBUTION METRICS (SDE-GAN) ===')
    print(metrics_df.round(4).to_string(index=False))
    print('\n=== RISK SCORES (ALL MODELS) ===')
    print(risk_df.round(4).to_string(index=False))
    print('\n=== SDE-GAN ACROSS-SEED SUMMARY ===')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
