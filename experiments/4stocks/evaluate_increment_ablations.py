"""Compare the fixed ablations with increments and historical simulation on validation."""

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

from sde_gan import SDEGAN

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))
from benchmarks.historical import HistoricalSimulation, HistoricalSimulationConfig, _empirical_var, _empirical_es

GROUPS = {'increments': 'increments', 'GP 5': 'increments_gp5',
          'absolute ACF 1': 'increments_absolute_acf_1'}
N_PATHS, SAMPLE_SEED = 1000, 100


def acf(paths: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Correlate within-path pairs; never join separate windows."""
    a = np.abs(paths) if absolute else paths
    return np.array([[np.corrcoef(a[:, :-lag, j].ravel(), a[:, lag:, j].ravel())[0, 1]
                      for j in range(4)] for lag in range(1, 6)])


def terminal(paths: np.ndarray) -> np.ndarray:
    """Compound a daily-rebalanced equal-weight portfolio."""
    return np.prod(1 + paths.mean(axis=2), axis=1) - 1


def risk_scores(paths: np.ndarray, realized: np.ndarray) -> list[dict]:
    """Use exactly the historical benchmark's empirical VaR and ES conventions."""
    losses = np.sort(-terminal(paths))
    rows = []
    for confidence in (.95, .99):
        var = _empirical_var(losses, confidence)
        residual = realized - var
        rows.append({'confidence': confidence, 'var_pct': 100 * var,
                     'es_pct': 100 * _empirical_es(losses, confidence),
                     'breaches': int(np.sum(realized > var)),
                     'breach_rate_pct': 100 * np.mean(realized > var),
                     'nominal_rate_pct': 100 * (1 - confidence),
                     'quantile_loss_pct': 100 * np.mean((confidence - (residual < 0)) * residual)})
    return rows


def main() -> None:
    """Evaluate every seed using its lowest-MMD checkpoint, without training or test data."""
    results = DIRECTORY / 'results/sde_gan'
    output = results / 'increment_ablations'
    output.mkdir(exist_ok=True)
    metadata = json.loads((DIRECTORY / 'data/preprocessing.json').read_text())
    stocks = metadata['stock_order']
    mu, scale = np.asarray(metadata['training_mean']), np.asarray(metadata['training_std'])
    dates = pd.read_csv(DIRECTORY / 'data/window_dates.csv', parse_dates=['start_date', 'end_date'])
    frames, observed = {}, {}
    for split in ('train', 'validation'):
        frames[split] = pd.read_csv(DIRECTORY / 'data' / f'{split}_returns.csv', parse_dates=['Date'])
        frame = frames[split].set_index('Date').loc[:, stocks]
        observed[split] = np.stack([frame.loc[row.start_date:row.end_date].to_numpy()
                                    for row in dates.loc[dates.split.eq(split)].itertuples()])
    historical = HistoricalSimulation(HistoricalSimulationConfig(
        assets=tuple(stocks), weights=dict(zip(stocks, [.25] * 4)),
        return_type='simple', horizon_sessions=20, rebalance='daily',
        confidence_levels=(.95, .99), estimation='fixed', min_scenarios=100,
    )).fit(frames['train'])
    np.testing.assert_allclose(historical.summarize().scenarios.portfolio_return,
                               terminal(observed['train']), rtol=0, atol=1e-14)
    realized = -terminal(observed['validation'])
    train_flat = observed['train'].reshape(-1, 4)
    train_mean, train_std = train_flat.mean(0), train_flat.std(0, ddof=1)
    val_std = observed['validation'].reshape(-1, 4).std(0, ddof=1)
    train_signed, train_absolute = acf(observed['train']), acf(observed['train'], True)
    train_corr = np.corrcoef(train_flat.T)
    pairs = np.triu_indices(4, 1)
    risk_rows = [{**row, 'experiment': 'historical', 'seed': -1}
                 for row in risk_scores(observed['train'], realized)]
    metrics, stock_rows, checkpoints, sensitivities = [], [], [], []
    samples, signed, absolute = {}, {}, {}
    for group, folder in GROUPS.items():
        for seed in (0, 1, 2):
            run_dir = results / folder / f'seed_{seed}'
            if group != 'increments':
                assert (run_dir / 'completion.json').is_file(), f'Unfinished run: {run_dir}'
            assert json.loads((run_dir / 'preprocessing.json').read_text()) == metadata
            model = SDEGAN.load(run_dir)
            assert model.run_metadata['selection_rule'] == 'lowest_validation_mmd2'
            standardized = model.sample(N_PATHS, SAMPLE_SEED)
            paths = np.expm1(standardized.astype(np.float64) * scale + mu)
            assert np.isfinite(paths).all()
            model.config = replace(model.config, generator_dt=.125)
            fine_standardized = model.sample(N_PATHS, SAMPLE_SEED)
            fine_paths = np.expm1(fine_standardized.astype(np.float64) * scale + mu)
            coarse_risk = risk_scores(paths, realized)
            fine_risk = risk_scores(fine_paths, realized)
            risk_rows.extend({**row, 'experiment': group, 'seed': seed} for row in coarse_risk)
            for coarse, fine in zip(coarse_risk, fine_risk):
                sensitivities.append({'experiment': group, 'seed': seed,
                                      'confidence': coarse['confidence'],
                                      'var_fine_minus_coarse_pp': fine['var_pct'] - coarse['var_pct'],
                                      'es_fine_minus_coarse_pp': fine['es_pct'] - coarse['es_pct'],
                                      'quantile_loss_fine_minus_coarse_pp': fine['quantile_loss_pct'] - coarse['quantile_loss_pct']})
            flat = paths.reshape(-1, 4)
            means, stds = flat.mean(0), flat.std(0, ddof=1)
            signed[group, seed], absolute[group, seed] = acf(paths), acf(paths, True)
            samples[group, seed] = paths
            metrics.append({'experiment': group, 'seed': seed,
                            'selected_step': model.run_metadata['selected_step'],
                            'validation_mmd2': model.run_metadata['validation_score'],
                            'mean_rmse_in_training_std': np.sqrt(np.mean(((means - train_mean) / train_std) ** 2)),
                            'relative_volatility_rmse_vs_train': np.sqrt(np.mean((stds / train_std - 1) ** 2)),
                            'relative_volatility_rmse_vs_validation': np.sqrt(np.mean((stds / val_std - 1) ** 2)),
                            'signed_acf_rmse_vs_train': np.sqrt(np.mean((signed[group, seed] - train_signed) ** 2)),
                            'absolute_acf_rmse_vs_train': np.sqrt(np.mean((absolute[group, seed] - train_absolute) ** 2)),
                            'cross_stock_correlation_rmse_vs_train': np.sqrt(np.mean((np.corrcoef(flat.T) - train_corr)[pairs] ** 2)),
                            'portfolio_mean_pct': 100 * terminal(paths).mean(),
                            'portfolio_std_pct': 100 * terminal(paths).std(ddof=1)})
            for j, stock in enumerate(stocks):
                stock_rows.append({'experiment': group, 'seed': seed, 'stock': stock,
                                   'mean_daily_pct': 100 * means[j], 'daily_std_pct': 100 * stds[j],
                                   'lag1_signed_acf': signed[group, seed][0, j],
                                   'lag1_absolute_acf': absolute[group, seed][0, j]})
            checkpoints.append({'experiment': group, 'seed': seed, 'path': str(run_dir),
                                'selected_step': model.run_metadata['selected_step'],
                                'gradient_penalty_weight': model.config.gradient_penalty_weight,
                                'absolute_temporal_penalty_weight': model.config.absolute_temporal_penalty_weight})
            np.savez_compressed(output / f'{folder}_seed_{seed}.npz',
                                standardized_log_returns=standardized,
                                fine_standardized_log_returns=fine_standardized)
            print(f'Evaluated {group} seed {seed}', flush=True)
    metrics, risk = pd.DataFrame(metrics), pd.DataFrame(risk_rows)
    metrics.to_csv(output / 'run_metrics.csv', index=False)
    risk.to_csv(output / 'validation_risk_scores.csv', index=False)
    pd.DataFrame(stock_rows).to_csv(output / 'stock_metrics.csv', index=False)
    pd.DataFrame(checkpoints).to_csv(output / 'selected_checkpoints.csv', index=False)
    pd.DataFrame(sensitivities).to_csv(output / 'solver_risk_sensitivity.csv', index=False)
    metric_summary = metrics.groupby('experiment').mean(numeric_only=True)
    metric_summary.to_csv(output / 'mean_metrics.csv')
    risk_summary = risk.groupby(['experiment', 'confidence']).agg(
        seeds=('seed', 'size'), mean_quantile_loss_pct=('quantile_loss_pct', 'mean'),
        min_quantile_loss_pct=('quantile_loss_pct', 'min'), max_quantile_loss_pct=('quantile_loss_pct', 'max'),
        mean_breach_rate_pct=('breach_rate_pct', 'mean'),
        mean_var_pct=('var_pct', 'mean'), mean_es_pct=('es_pct', 'mean'))
    risk_summary.to_csv(output / 'mean_risk_scores.csv')
    print(metric_summary.round(5).to_string())
    print(risk_summary.round(5).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for confidence, ax in zip((.95, .99), axes):
        table = risk.loc[(risk.confidence == confidence) & (risk.experiment != 'historical')]
        table.pivot(index='seed', columns='experiment', values='quantile_loss_pct').plot.bar(ax=ax)
        hs = risk.loc[(risk.confidence == confidence) & (risk.experiment == 'historical'), 'quantile_loss_pct'].iloc[0]
        ax.axhline(hs, color='black', linestyle='--', label='historical')
        ax.set(title=f'{confidence:.0%} validation quantile loss', ylabel='Quantile loss × 100')
        ax.legend(fontsize=7)
    fig.savefig(output / 'validation_quantile_loss.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for row, (kind, values) in enumerate((('Signed', signed), ('Absolute', absolute))):
        for j, stock in enumerate(stocks):
            ax = axes[row, j]
            for split, color in (('train', 'black'), ('validation', 'grey')):
                ax.plot(range(1, 6), acf(observed[split], row == 1)[:, j], color=color, label=split)
            for k, group in enumerate(GROUPS):
                curves = np.stack([values[group, seed][:, j] for seed in (0, 1, 2)])
                ax.plot(range(1, 6), curves.mean(0), color=f'C{k}', label=group)
                ax.fill_between(range(1, 6), curves.min(0), curves.max(0), color=f'C{k}', alpha=.12)
            ax.set(title=f'{stock}: {kind}', xlabel='Lag', ylabel='Correlation', xticks=range(1, 6))
            ax.legend(fontsize=6)
    fig.suptitle('Mean ACF across seeds; shading shows seed range, not a confidence interval')
    fig.savefig(output / 'return_acf.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for j, stock in enumerate(stocks):
        ax = axes.flat[j]
        plot_samples = {**observed, **{f'{group}, seed 0': samples[group, 0] for group in GROUPS}}
        for name, paths in plot_samples.items():
            x = np.sort(paths[:, :, j].std(axis=1, ddof=1)) * 100
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=name)
        ax.set(title=stock, xlabel='Within-window daily std (%)', ylabel='CDF')
        ax.legend(fontsize=7)
    fig.savefig(output / 'window_volatility_cdf.png', dpi=150)
    plt.close(fig)
    (output / 'evaluation.json').write_text(json.dumps({
        'n_paths': N_PATHS, 'sample_seed': SAMPLE_SEED, 'seeds': [0, 1, 2],
        'selection_rule': 'lowest_validation_mmd2', 'test_evaluated': False,
        'absolute_loss_units': 'absolute log returns, original centering restored',
        'coarse_dt': .25, 'fine_dt': .125,
    }, indent=2) + '\n')
    print(f'All results saved to {output}', flush=True)


if __name__ == '__main__':
    main()
