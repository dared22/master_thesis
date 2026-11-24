"""Shared validation, portfolio compounding, and empirical risk helpers.

Pure NumPy and pandas utilities reused across portfolio risk benchmarks.
"""

from collections.abc import Mapping
import csv
from datetime import date, datetime
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

ReturnInput = str | Path | pd.DataFrame
WEIGHT_TOLERANCE = 1e-10
ES_INTEGER_TOLERANCE = 8 * np.finfo(np.float64).eps


def _validate_choice(name: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f'{name} must be one of {choices}')


def _validate_int(name: str, value: Any, minimum: int, *, allow_none: bool = False) -> int | None:
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _validate_assets(assets: tuple[str, ...] | None) -> None:
    if assets is None:
        return
    if (not isinstance(assets, tuple) or not assets
            or any(not isinstance(a, str) or not a or a == 'Date' for a in assets)
            or len(set(assets)) != len(assets)):
        raise ValueError('assets must be a nonempty tuple of distinct asset names')


def _validate_confidence_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    if (not isinstance(levels, tuple) or not levels
            or any(isinstance(c, bool) or not isinstance(c, Real)
                   or not np.isfinite(c) or not 0 < c < 1 for c in levels)
            or len(set(levels)) != len(levels)):
        raise ValueError('confidence_levels must be distinct finite values in (0, 1)')
    return tuple(float(c) for c in levels)


def _validate_weights(weights: Mapping[str, float] | None,
                      assets: tuple[str, ...] | None) -> MappingProxyType[str, float] | None:
    if weights is None:
        return None
    if not isinstance(weights, Mapping):
        raise ValueError('weights must map asset names to finite nonnegative weights')
    copied = dict(weights)
    if (not copied or any(not isinstance(a, str) or not a or a == 'Date' for a in copied)
            or any(isinstance(w, bool) or not isinstance(w, Real)
                   or not np.isfinite(w) or w < 0 for w in copied.values())
            or not np.isclose(sum(copied.values()), 1, rtol=0, atol=WEIGHT_TOLERANCE)):
        raise ValueError('weights must be finite, nonnegative, and sum to one within 1e-10')
    if assets is not None and set(copied) != set(assets):
        raise ValueError('weight keys must match selected assets')
    return MappingProxyType({a: float(w) for a, w in copied.items()})


def _resolve_weights(assets: tuple[str, ...],
                     supplied: Mapping[str, float] | None) -> np.ndarray:
    if supplied is not None and set(supplied) != set(assets):
        raise ValueError(f'Weight keys must match selected assets: {assets}')
    return np.array([supplied[a] for a in assets] if supplied is not None
                    else [1 / len(assets)] * len(assets), dtype=np.float64)


def _bounds(history: pd.DataFrame) -> dict[str, Any]:
    return {'history_start': history.index[0].date().isoformat() if len(history) else None,
            'history_end': history.index[-1].date().isoformat() if len(history) else None,
            'n_history_sessions': len(history)}


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


def _coverage_table(forecasts: pd.DataFrame, confidence_levels: tuple[float, ...],
                    horizon: int, n_targets: int) -> pd.DataFrame:
    coverage = []
    for confidence in confidence_levels:
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
    return pd.DataFrame(coverage)
