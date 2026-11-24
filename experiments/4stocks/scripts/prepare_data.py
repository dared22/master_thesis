"""Prepare the 4-stock return splits and standardized windows from Data/eligible."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / 'Data/eligible/price_histories.csv'
DATA_DIR = EXPERIMENT_DIR / 'data'

DEFAULT_STOCK_IDS = (
    'NO0013536151',  # KONGSBERG GRUPPEN
    'CY0200352116',  # FRONTLINE
    'NO0010571698',  # WILH, WILHELMSEN A
    'NO0012470089',  # TOMRA SYSTEMS
)
WINDOW_LENGTH = 20
MAX_INTERPOLATION_GAP = 5
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Reviewed event corrections matching EDA/02_sector_dependence.ipynb.
RETURN_CORRECTIONS: dict[tuple[pd.Timestamp, str], float] = {
    (pd.Timestamp('2023-01-30'), 'NO0012953720'): float(np.log((0.0889 + 1.80) / 1.852)),
    (pd.Timestamp('2024-10-31'), 'NO0013601245'): float('nan'),
}


def interpolate_short_price_gaps(
    frame: pd.DataFrame, max_gap: int = MAX_INTERPOLATION_GAP,
) -> tuple[pd.DataFrame, int]:
    """Fill internal missing runs up to max_gap sessions using positive natural cubic splines."""
    filled = frame.copy()
    filled_cells = 0
    for stock_id in frame.columns:
        values = frame[stock_id].to_numpy(dtype=np.float64)
        observed = np.flatnonzero(np.isfinite(values))
        spline = (CubicSpline(observed, values[observed], bc_type='natural', extrapolate=False)
                  if len(observed) >= 4 else None)
        changes = np.diff((~np.isfinite(values)).astype(int), prepend=0, append=0)
        for first, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            internal = first > 0 and stop < len(values)
            if internal and (stop - first) <= max_gap and spline is not None:
                estimates = spline(np.arange(first, stop))
                if np.isfinite(estimates).all() and (estimates > 0).all():
                    filled.iloc[first:stop, filled.columns.get_loc(stock_id)] = estimates
                    filled_cells += int(stop - first)
    return filled, filled_cells


def build_windows(
    standardized: pd.DataFrame, window_length: int,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    """Extract complete contiguous windows on the session grid without bridging gaps."""
    values = standardized.to_numpy(dtype=np.float64)
    dates = standardized.index
    n_candidates = max(0, len(values) - window_length + 1)
    windows, records = [], []
    for start in range(n_candidates):
        block = values[start:start + window_length]
        if np.isfinite(block).all():
            windows.append(block.astype(np.float32))
            records.append({
                'window_index': len(windows) - 1,
                'start_date': dates[start].strftime('%Y-%m-%d'),
                'end_date': dates[start + window_length - 1].strftime('%Y-%m-%d'),
            })
    stacked = (np.stack(windows) if windows
               else np.empty((0, window_length, values.shape[1]), dtype=np.float32))
    return stacked, records, n_candidates


def prepare(
    stock_ids: tuple[str, ...] = DEFAULT_STOCK_IDS,
    source_path: Path = SOURCE_PATH,
    output_dir: Path = DATA_DIR,
    window_length: int = WINDOW_LENGTH,
    split_fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> dict[str, object]:
    """Read Data/eligible/price_histories.csv, interpolate short gaps, and write splits."""
    if not np.isclose(sum(split_fractions), 1.0, atol=1e-9):
        raise ValueError(f'Split fractions must sum to 1.0, got {split_fractions}')
    raw = pd.read_csv(
        source_path,
        usecols=['stock_id', 'Date', 'AdjustedPrice', 'Name'],
        parse_dates=['Date'],
        float_precision='round_trip',
    )
    all_prices = raw.pivot(index='Date', columns='stock_id', values='AdjustedPrice').sort_index()
    missing = all_prices.isna()
    eligible = (all_prices.iloc[-1].notna() & missing.mean().lt(0.05)
                & ~missing.rolling(MAX_INTERPOLATION_GAP + 1).sum().eq(MAX_INTERPOLATION_GAP + 1).any())
    ineligible = [sid for sid in stock_ids if sid not in all_prices.columns or not bool(eligible[sid])]
    if ineligible:
        raise ValueError(f'Requested stock_ids failed the thesis eligibility screen: {ineligible}')

    prices = all_prices.loc[:, list(stock_ids)].copy()
    names_by_id = (raw.drop_duplicates('stock_id').set_index('stock_id')['Name']
                   .astype(str).str.strip().to_dict())
    stock_order = [names_by_id[sid] for sid in stock_ids]
    observed_missing = int(prices.isna().sum().sum())
    interpolated_prices, filled_cells = interpolate_short_price_gaps(prices, MAX_INTERPOLATION_GAP)

    log_returns = np.log(interpolated_prices).diff()
    for (date, sid), corrected in RETURN_CORRECTIONS.items():
        if sid in log_returns.columns and date in log_returns.index:
            log_returns.at[date, sid] = corrected
    # Drop the leading diff row so every split begins with an observable daily return.
    log_returns = log_returns.iloc[1:]
    simple_returns = np.expm1(log_returns)
    simple_returns.columns = stock_order
    log_returns.columns = stock_order

    n_rows = len(simple_returns)
    n_train = round(split_fractions[0] * n_rows)
    n_val = round(split_fractions[1] * n_rows)
    splits = {
        'train': (0, n_train),
        'validation': (n_train, n_train + n_val),
        'test': (n_train + n_val, n_rows),
    }

    train_log = log_returns.iloc[splits['train'][0]:splits['train'][1]]
    training_mean = train_log.mean(axis=0).to_numpy(dtype=np.float64)
    training_std = train_log.std(axis=0, ddof=1).to_numpy(dtype=np.float64)
    if not (np.isfinite(training_mean).all() and np.isfinite(training_std).all() and (training_std > 0).all()):
        raise ValueError('Invalid training standardization statistics')

    output_dir.mkdir(parents=True, exist_ok=True)
    window_arrays: dict[str, np.ndarray] = {}
    date_rows: list[dict[str, object]] = []
    split_metadata: dict[str, dict[str, object]] = {}

    for split_name, (start, stop) in splits.items():
        split_simple = simple_returns.iloc[start:stop]
        split_simple.reset_index().to_csv(output_dir / f'{split_name}_returns.csv', index=False)
        split_std = (log_returns.iloc[start:stop] - training_mean) / training_std
        windows, records, n_candidates = build_windows(split_std, window_length)
        window_arrays[f'{split_name}_windows'] = windows
        for record in records:
            date_rows.append({'split': split_name, **record})
        split_metadata[split_name] = {
            'source': f'{split_name}_returns.csv',
            'first_date': split_simple.index[0].strftime('%Y-%m-%d'),
            'last_date': split_simple.index[-1].strftime('%Y-%m-%d'),
            'rows': int(len(split_simple)),
            'candidates': int(n_candidates),
            'retained': int(len(windows)),
            'rejected': int(n_candidates - len(windows)),
        }

    np.savez(output_dir / 'prepared_windows.npz', **window_arrays)
    pd.DataFrame(date_rows).to_csv(output_dir / 'window_dates.csv', index=False)

    metadata = {
        'source_file': str(source_path.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'stock_ids': list(stock_ids),
        'stock_order': stock_order,
        'max_interpolation_gap': MAX_INTERPOLATION_GAP,
        'observed_missing_prices': observed_missing,
        'interpolated_prices': filled_cells,
        'split_fractions': list(split_fractions),
        'transform': 'log1p',
        'ddof': 1,
        'training_count': [int(train_log[col].notna().sum()) for col in stock_order],
        'training_mean': training_mean.tolist(),
        'training_std': training_std.tolist(),
        'window_length': window_length,
        'stride': 1,
        'observation_times': list(range(window_length)),
        'time_unit': 'market session',
        'calendar': 'Data/eligible/price_histories.csv (XOSL 10-year eligible panel)',
        'splits': split_metadata,
    }
    (output_dir / 'preprocessing.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock-ids', nargs='+', default=list(DEFAULT_STOCK_IDS),
                        help='Ordered stock_id values from Data/eligible/price_histories.csv.')
    args = parser.parse_args()
    metadata = prepare(stock_ids=tuple(args.stock_ids))
    print(f"Prepared {metadata['stock_order']} from {metadata['source_file']}:")
    for name, info in metadata['splits'].items():
        print(f"  {name:10s}: {info['first_date']} .. {info['last_date']} "
              f"({info['rows']} rows -> {info['retained']} windows, {info['rejected']} rejected)")


if __name__ == '__main__':
    main()
