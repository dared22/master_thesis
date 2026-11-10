"""Unweighted empirical historical simulation on an observed session calendar.

Returns and losses are fractions. Missing rows stay on the session calendar;
neither dates nor economic adjustment/currency conventions are inferred.
"""

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

ReturnInput = str | Path | pd.DataFrame
WEIGHT_TOLERANCE = 1e-10
ES_INTEGER_TOLERANCE = 8 * np.finfo(np.float64).eps


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
            if getattr(self, name) not in choices:
                raise ValueError(f'{name} must be one of {choices}')
        for name, minimum in (('horizon_sessions', 1), ('min_scenarios', 2),
                              ('lookback_sessions', 1)):
            value = getattr(self, name)
            if name == 'lookback_sessions' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
            object.__setattr__(self, name, int(value))
        if self.estimation == 'rolling' and self.lookback_sessions is None:
            raise ValueError('rolling estimation requires lookback_sessions')
        if self.estimation == 'expanding' and self.lookback_sessions is not None:
            raise ValueError('expanding estimation requires lookback_sessions=None')
        if self.lookback_sessions is not None and self.lookback_sessions < self.horizon_sessions:
            raise ValueError('lookback_sessions must be at least horizon_sessions')
        if self.assets is not None:
            if (not isinstance(self.assets, tuple) or not self.assets
                    or any(not isinstance(a, str) or not a or a == 'Date' for a in self.assets)
                    or len(set(self.assets)) != len(self.assets)):
                raise ValueError('assets must be a nonempty tuple of distinct asset names')
        levels = self.confidence_levels
        if (not isinstance(levels, tuple) or not levels
                or any(isinstance(c, bool) or not isinstance(c, Real)
                       or not np.isfinite(c) or not 0 < c < 1 for c in levels)
                or len(set(levels)) != len(levels)):
            raise ValueError('confidence_levels must be distinct finite values in (0, 1)')
        object.__setattr__(self, 'confidence_levels', tuple(float(c) for c in levels))
        if self.weights is not None:
            if not isinstance(self.weights, Mapping):
                raise ValueError('weights must map asset names to finite nonnegative weights')
            weights = dict(self.weights)
            if (not weights or any(not isinstance(a, str) or not a or a == 'Date'
                                   for a in weights)
                    or any(isinstance(w, bool) or not isinstance(w, Real)
                           or not np.isfinite(w) or w < 0 for w in weights.values())
                    or not np.isclose(sum(weights.values()), 1, rtol=0, atol=WEIGHT_TOLERANCE)):
                raise ValueError('weights must be finite, nonnegative, and sum to one within 1e-10')
            if self.assets is not None and set(weights) != set(self.assets):
                raise ValueError('weight keys must match selected assets')
            object.__setattr__(self, 'weights', MappingProxyType(
                {a: float(w) for a, w in weights.items()}))


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


def _load(returns: ReturnInput, assets: tuple[str, ...] | None,
          return_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate selected source returns and retain their canonical fingerprint."""
    source = None
    if isinstance(returns, (str, Path)):
        path = Path(returns)
        with path.open(newline='', encoding='utf-8-sig') as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ValueError('CSV column names must be unique')
        frame = pd.read_csv(path, keep_default_na=False, na_values=[''])
        source = str(path.resolve())
    elif isinstance(returns, pd.DataFrame):
        frame = returns.copy(deep=True)
    else:
        raise ValueError('returns must be a CSV path or pandas DataFrame')
    if not frame.columns.is_unique or 'Date' not in frame:
        raise ValueError('Require unique column names and a Date column')
    selected = tuple(c for c in frame.columns if c != 'Date') if assets is None else assets
    if not selected or any(not isinstance(a, str) or not a for a in selected):
        raise ValueError('Require at least one named selected asset')
    absent = set(selected) - set(frame.columns)
    if absent:
        raise ValueError(f'Missing selected asset columns: {sorted(absent)}')
    parsed = []
    for value in frame['Date']:
        if isinstance(value, str):
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is None:
                raise ValueError(f'Date must be an ISO calendar date: {value!r}')
        elif not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
            raise ValueError(f'Invalid Date: {value!r}')
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f'Invalid Date: {value!r}') from error
        if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
            raise ValueError(f'Date must be timezone-naive and date-only: {value!r}')
        parsed.append(stamp)
    dates = pd.DatetimeIndex(parsed, dtype='datetime64[ns]', name='Date')
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Date values must be unique and increasing')
    columns = {}
    for asset in selected:
        series = frame[asset]
        if any(isinstance(v, (bool, complex, np.bool_, np.complexfloating,
                              date, np.datetime64, pd.Timedelta, np.timedelta64))
               for v in series.dropna()):
            raise ValueError(f'{asset}: returns must be real numeric values')
        try:
            values = pd.to_numeric(series, errors='raise').to_numpy(dtype=np.float64, na_value=np.nan)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f'{asset}: returns must be numeric') from error
        invalid = np.isinf(values)
        if return_type == 'simple':
            invalid |= values < -1
        if invalid.any():
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(f'{asset} on {dates[index].date()}: invalid {return_type} return')
        columns[asset] = values
    validated = pd.DataFrame(columns, index=dates, dtype=np.float64)
    canonical = {'dates': dates.strftime('%Y-%m-%d').tolist(), 'assets': list(selected),
                 'returns': [[None if np.isnan(v) else float(v) for v in row]
                             for row in validated.to_numpy()]}
    fingerprint = hashlib.sha256(json.dumps(
        canonical, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    info = {'source': source, 'fingerprint': fingerprint, **_bounds(validated)}
    if return_type == 'log':
        with np.errstate(over='ignore', invalid='ignore'):
            converted = np.expm1(validated.to_numpy())
        if np.isinf(converted).any():
            raise ValueError('Log return conversion produced nonfinite simple returns')
        validated = pd.DataFrame(converted, index=dates, columns=selected)
    return validated, info


def _bounds(history: pd.DataFrame) -> dict[str, Any]:
    return {'history_start': history.index[0].date().isoformat() if len(history) else None,
            'history_end': history.index[-1].date().isoformat() if len(history) else None,
            'n_history_sessions': len(history)}


def _portfolio_returns(blocks: np.ndarray, weights: np.ndarray, rebalance: str) -> np.ndarray:
    """Compound complete blocks of shape (scenarios, sessions, assets)."""
    with np.errstate(over='raise', invalid='raise'):
        if rebalance == 'daily':
            result = np.prod(1 + blocks @ weights, axis=1) - 1
        else:
            result = (np.prod(1 + blocks, axis=1) - 1) @ weights
    if not np.isfinite(result).all():
        raise FloatingPointError('Portfolio compounding produced nonfinite returns')
    return result


def _empirical_var(sorted_losses: np.ndarray, confidence: float) -> float:
    """First loss whose empirical CDF reaches confidence; no interpolation."""
    cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    return float(sorted_losses[np.searchsorted(cdf, confidence, side='left')])


def _tail_mass(n: int, confidence: float) -> float:
    mass = n * (1 - confidence)
    nearest = round(mass)
    if nearest > 0 and abs(mass - nearest) <= ES_INTEGER_TOLERANCE * n:
        mass = float(nearest)
    return mass


def _empirical_es(sorted_losses: np.ndarray, confidence: float) -> float:
    """Mean of exactly the worst 1-c mass, including a fractional boundary.

    Snap mass near a positive integer within 8 * float64 epsilon * n.
    A positive tail smaller than one observation is never rounded to zero.
    """
    mass = _tail_mass(len(sorted_losses), confidence)
    whole = int(np.floor(mass))
    fraction = mass - whole
    descending = sorted_losses[::-1]
    # Divide before multiplying to preserve even subnormal positive tail masses.
    result = float(np.sum(descending[:whole] / mass))
    if fraction:
        result += (fraction / mass) * float(descending[whole])
    return result


def _dated_table(records: list[dict[str, Any]], columns: list[str],
                 dates: tuple[str, ...]) -> pd.DataFrame:
    table = pd.DataFrame(records, columns=columns)
    for column in dates:
        table[column] = pd.to_datetime(table[column])
    return table


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
        supplied = self.config.weights
        if supplied is not None and set(supplied) != set(assets):
            raise ValueError(f'Weight keys must match selected assets: {assets}')
        weights = np.array([supplied[a] for a in assets] if supplied is not None
                           else [1 / len(assets)] * len(assets), dtype=np.float64)
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
        coverage = []
        for confidence in self.config.confidence_levels:
            rows = forecasts.loc[forecasts.confidence.eq(confidence)]
            comparisons = int(rows.breach.notna().sum())
            breaches = int(rows.breach.sum())
            coverage.append({
                'confidence': confidence, 'horizon_sessions': horizon,
                'n_targets': n_targets, 'n_estimates': int(rows.estimate_status.eq('ok').sum()),
                'n_comparisons': comparisons, 'n_breaches': breaches,
                'breach_rate': breaches / comparisons if comparisons else np.nan,
                'nominal_tail_probability': 1 - confidence,
            })
        counts = {'n_targets': n_targets, 'n_trailing_excluded': len(evaluation) - n_targets,
                  'n_forecast_rows': len(forecasts),
                  'n_insufficient_history_targets': int(forecasts.loc[
                      forecasts.confidence.eq(self.config.confidence_levels[0]),
                      'estimate_status'].eq('insufficient_history').sum()),
                  'n_missing_realization_targets': int(forecasts.loc[
                      forecasts.confidence.eq(self.config.confidence_levels[0]),
                      'realization_status'].eq('missing_returns').sum())}
        return HistoricalSimulationBacktest(forecasts, pd.DataFrame(coverage),
                                            self._metadata(counts, info))
