"""Unweighted empirical historical simulation on an observed session calendar.

Returns and losses are fractions. Missing rows stay on the session calendar;
neither dates nor economic adjustment/currency conventions are inferred.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import platform
from typing import Any

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class HistoricalSimulationConfig:
    """Historical simulation settings; supplied weights are never normalized.

    Weight sums use absolute tolerance 1e-10 and zero relative tolerance.
    Lookback counts raw session rows, including incomplete rows.
    """

    assets: tuple[str, ...] | None = None
    weights: Mapping[str, float] | None = None
    return_type: str = 'simple'
    horizon_sessions: int = 1
    rebalance: str = 'daily'
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    estimation: str = 'fixed'
    lookback_sessions: int | None = None
    min_scenarios: int = 100

    def __post_init__(self) -> None:
        for name, choices in (
            ('return_type', ('simple', 'log')),
            ('rebalance', ('daily', 'buy_and_hold')),
            ('estimation', ('fixed', 'expanding', 'rolling')),
        ):
            _validate_choice(name, getattr(self, name), choices)
        for name, minimum, allow_none in (
            ('horizon_sessions', 1, False),
            ('min_scenarios', 2, False),
            ('lookback_sessions', 1, True),
        ):
            object.__setattr__(
                self, name, _validate_int(name, getattr(self, name), minimum, allow_none=allow_none))
        if self.estimation == 'rolling' and self.lookback_sessions is None:
            raise ValueError('rolling estimation requires lookback_sessions')
        if self.estimation == 'expanding' and self.lookback_sessions is not None:
            raise ValueError('expanding estimation requires lookback_sessions=None')
        if self.lookback_sessions is not None and self.lookback_sessions < self.horizon_sessions:
            raise ValueError('lookback_sessions must be at least horizon_sessions')
        _validate_assets(self.assets)
        object.__setattr__(self, 'confidence_levels',
                           _validate_confidence_levels(self.confidence_levels))
        object.__setattr__(self, 'weights', _validate_weights(self.weights, self.assets))


@dataclass(frozen=True)
class HistoricalSimulationSummary:
    """Calibration risk estimates, eligible historical blocks, and provenance."""

    risk: pd.DataFrame
    scenarios: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HistoricalSimulationBacktest:
    """Chronological forecasts and descriptive coverage; no independence tests."""

    forecasts: pd.DataFrame
    coverage: pd.DataFrame
    metadata: dict[str, Any]


class HistoricalSimulation:
    """Fit dated raw returns and estimate empirical portfolio losses.

    No method writes files. Backtest calls are independent and never update
    fitted history. The caller supplies intervening observed session rows.
    """

    def __init__(self, config: HistoricalSimulationConfig | None = None) -> None:
        self.config = HistoricalSimulationConfig() if config is None else config
        self._history: pd.DataFrame | None = None
        self._assets: tuple[str, ...] = ()
        self._weights = np.empty(0, dtype=np.float64)
        self._fit_info: dict[str, Any] = {}

    def fit(self, returns: ReturnInput) -> 'HistoricalSimulation':
        """Replace calibration state with a validated snapshot; return self."""
        history, info = _load(returns, self.config.assets, self.config.return_type)
        if history.empty:
            raise ValueError('Fitted history must contain at least one session')
        assets = tuple(history.columns)
        weights = _resolve_weights(assets, self.config.weights)
        self._history, self._fit_info = history, info
        self._assets, self._weights = assets, weights
        return self

    def _require_history(self) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError('Fit HistoricalSimulation before summarizing or backtesting')
        return self._history

    def _scenarios(self, history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        lookback = self.config.lookback_sessions
        if lookback is not None:
            history = history.iloc[-lookback:]
        horizon = self.config.horizon_sessions
        candidates = max(0, len(history) - horizon + 1)
        starts = np.empty(0, dtype=int)
        returns = np.empty(0, dtype=np.float64)
        if candidates:
            # sliding_window_view puts the session dimension last.
            blocks = np.lib.stride_tricks.sliding_window_view(
                history.to_numpy(), horizon, axis=0).transpose(0, 2, 1)
            starts = np.flatnonzero(~np.isnan(blocks).any(axis=(1, 2)))
            returns = _portfolio_returns(blocks[starts], self._weights, self.config.rebalance)
        scenarios = pd.DataFrame({
            'scenario_start': history.index[starts],
            'scenario_end': history.index[starts + horizon - 1],
            'portfolio_return': returns, 'loss': -returns,
        })
        counts = {**_bounds(history), 'n_candidate_blocks': candidates,
                  'n_excluded_blocks': candidates - len(starts), 'n_scenarios': len(starts)}
        return scenarios, counts

    def _risk(self, scenarios: pd.DataFrame) -> list[dict[str, Any]]:
        n = len(scenarios)
        available = n >= self.config.min_scenarios
        losses = np.sort(scenarios.loss.to_numpy())
        records = []
        for confidence in self.config.confidence_levels:
            mass = _tail_mass(n, confidence)
            records.append({
                'method': 'historical_simulation', 'confidence': confidence,
                'horizon_sessions': self.config.horizon_sessions,
                'rebalance': self.config.rebalance,
                'var': _empirical_var(losses, confidence) if available else np.nan,
                'es': _empirical_es(losses, confidence) if available else np.nan,
                'tail_mass_observations': mass, 'small_tail_sample': mass < 20,
                'overlapping_scenarios': self.config.horizon_sessions > 1,
            })
        return records

    def _metadata(self, counts: dict[str, Any],
                  evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
        config = {name: getattr(self.config, name) for name in self.config.__dataclass_fields__
                  if name != 'weights'}
        config['assets'] = list(self._assets)
        config['confidence_levels'] = list(self.config.confidence_levels)
        config['weights'] = dict(zip(self._assets, self._weights.tolist()))
        return {
            'method': 'historical_simulation', 'configuration': config,
            'input_return_type': self.config.return_type, 'loss_units': 'fractional',
            'inputs': {'fit': dict(self._fit_info), **({'evaluation': evaluation}
                       if evaluation is not None else {})},
            'counts': counts,
            'package_versions': {'python': platform.python_version(),
                                 'numpy': np.__version__, 'pandas': pd.__version__},
            'missing_block_policy': 'exclude any block with a missing selected return; preserve raw rows',
            'quantile_convention': 'inverse empirical CDF; no interpolation',
            'es_convention': 'mean of worst 1-confidence mass with fractional boundary',
            'es_integer_mass_tolerance_per_scenario': ES_INTEGER_TOLERANCE,
            'weight_sum_absolute_tolerance': WEIGHT_TOLERANCE,
            'scenario_stride': 1,
            'overlapping_scenarios': self.config.horizon_sessions > 1,
            'overlapping_targets': self.config.horizon_sessions > 1,
        }

    def summarize(self) -> HistoricalSimulationSummary:
        """Summarize fitted history; fail explicitly below min_scenarios."""
        scenarios, counts = self._scenarios(self._require_history())
        if len(scenarios) < self.config.min_scenarios:
            raise ValueError(f'Insufficient history: assets={self._assets}, '
                             f'horizon={self.config.horizon_sessions}, {counts}, '
                             f'min_scenarios={self.config.min_scenarios}')
        with np.errstate(over='raise', invalid='raise'):
            mean = float(scenarios.portfolio_return.mean())
            std = float(scenarios.portfolio_return.std(ddof=1))
        if not np.isfinite([mean, std]).all():
            raise FloatingPointError('Portfolio scenario statistics are nonfinite')
        records = [{**risk, **counts,
                    'portfolio_mean_return': mean, 'portfolio_std_return': std}
                   for risk in self._risk(scenarios)]
        risk = _dated_table(records, list(records[0]), ('history_start', 'history_end'))
        return HistoricalSimulationSummary(risk, scenarios, self._metadata(counts))

    def backtest(self, evaluation_returns: ReturnInput) -> HistoricalSimulationBacktest:
        """Forecast each complete-length evaluation target without future inputs.

        Coverage is descriptive, especially for overlapping multi-session targets.
        Missing realizations and insufficient histories retain their target rows.
        """
        history = self._require_history()
        evaluation, info = _load(evaluation_returns, self._assets, self.config.return_type)
        if len(evaluation) and evaluation.index[0] <= history.index[-1]:
            raise ValueError('Evaluation dates must be strictly after fitted history')
        combined = pd.concat([history, evaluation])
        horizon = self.config.horizon_sessions
        n_targets = max(0, len(evaluation) - horizon + 1)
        records = []
        fixed = None
        if self.config.estimation == 'fixed':
            scenarios, counts = self._scenarios(history)
            fixed = (self._risk(scenarios), counts)
        for start in range(n_targets):
            position = len(history) + start
            if fixed is None:
                scenarios, counts = self._scenarios(combined.iloc[:position])
                risks = self._risk(scenarios)
            else:
                risks, counts = fixed
            block = evaluation.iloc[start:start + horizon].to_numpy()
            realized_ok = not np.isnan(block).any()
            realized = (float(_portfolio_returns(block[None], self._weights,
                                                 self.config.rebalance)[0])
                        if realized_ok else np.nan)
            estimate_ok = counts['n_scenarios'] >= self.config.min_scenarios
            for risk in risks:
                records.append({
                    **risk, **counts, 'forecast_origin': combined.index[position - 1],
                    'target_start': evaluation.index[start],
                    'target_end': evaluation.index[start + horizon - 1],
                    'realized_return': realized, 'realized_loss': -realized,
                    'estimate_status': 'ok' if estimate_ok else 'insufficient_history',
                    'realization_status': 'ok' if realized_ok else 'missing_returns',
                    'breach': bool(-realized > risk['var']) if estimate_ok and realized_ok else pd.NA,
                })
        columns = [
            'method', 'confidence', 'horizon_sessions', 'rebalance', 'var', 'es',
            'tail_mass_observations', 'small_tail_sample', 'overlapping_scenarios',
            'history_start', 'history_end', 'n_history_sessions', 'n_candidate_blocks',
            'n_excluded_blocks', 'n_scenarios', 'forecast_origin', 'target_start',
            'target_end', 'realized_return', 'realized_loss', 'estimate_status',
            'realization_status', 'breach',
        ]
        forecasts = _dated_table(records, columns, (
            'history_start', 'history_end', 'forecast_origin', 'target_start', 'target_end'))
        forecasts['breach'] = forecasts.breach.astype('boolean')
        coverage = _coverage_table(forecasts, self.config.confidence_levels, horizon, n_targets)
        counts = {'n_targets': n_targets, 'n_trailing_excluded': len(evaluation) - n_targets,
                  'n_forecast_rows': len(forecasts),
                  'n_insufficient_history_targets': int(forecasts.loc[
                      forecasts.confidence.eq(self.config.confidence_levels[0]),
                      'estimate_status'].eq('insufficient_history').sum()),
                  'n_missing_realization_targets': int(forecasts.loc[
                      forecasts.confidence.eq(self.config.confidence_levels[0]),
                      'realization_status'].eq('missing_returns').sum())}
        return HistoricalSimulationBacktest(forecasts, coverage, self._metadata(counts, info))
