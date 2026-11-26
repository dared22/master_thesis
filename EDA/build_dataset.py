"""Build the canonical ten-year Oslo Bors panel in Data/processed/ from Data/raw/.

Single source of truth for:
- XOSL trading-day filtering (10-year window ending on the latest date in the export)
- ICB sector classification mapping (with company/internal-code inheritance)
- Nonoverlapping ISIN history linking and one-share-class-per-issuer selection
- Thesis eligibility screen (valid final price, <5% missing, max gap <= 5 sessions)
- Short-gap natural cubic-spline price interpolation (internal gaps <= 5 sessions)
- Reviewed corporate-action return corrections (Hunter Group, EAM Solar)

Outputs written to Data/processed/:
- stocks.csv                    111-row issuer/ICB/missingness metadata table
- prices_observed.csv           Wide (2510 x 111) uninterpolated AdjustedPrice panel
- log_returns_observed.csv      Wide (2509 x 111) observed daily log returns + reviewed corrections
- log_returns_interpolated.csv  Wide (2509 x 111) short-gap interpolated log returns + reviewed corrections
- manifest.json                 Provenance, SHA-256 checksums, and contract metadata
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PRICES_PATH = RAW_DIR / "BMG9156K1018_10.09.2026_14.59.34.csv"
SECTOR_MAPPING_PATH = RAW_DIR / "stock_sector_mapping.csv"

CALENDAR_CODE = "XOSL"
CALENDAR_LOOKBACK_YEARS = 15
SAMPLE_YEARS = 10
MAX_MISSING_FRACTION = 0.05
MAX_INTERPOLATION_GAP = 5

ICB_COLUMNS = [
    "icb_industry_name",
    "icb_supersector_name",
    "icb_sector_name",
    "icb_subsector_name",
]

KNOWN_ISIN_LINKS = [
    ["NO0003043309", "NO0013536151"],  # Kongsberg
    ["BMG368191034", "BMG3682E1012", "BMG3682E1921", "CY0200352116"],  # Frontline
    ["NO0005668905", "NO0012470089"],  # Tomra
]

# Reviewed corporate-action event adjustments applied to daily log returns.
RETURN_CORRECTIONS: dict[tuple[pd.Timestamp, str], tuple[float, str]] = {
    (pd.Timestamp("2023-01-30"), "NO0012953720"): (
        float(np.log((0.0889 + 1.80) / 1.852)),
        "Hunter Group: include the NOK 1.80 cash dividend once.",
    ),
    (pd.Timestamp("2024-10-31"), "NO0013601245"): (
        float("nan"),
        "EAM Solar: unresolved share offering adjustment; set return to missing.",
    ),
}


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomic(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a DataFrame to CSV atomically via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False
    ) as dst:
        temporary = Path(dst.name)
    try:
        frame.to_csv(temporary, index=index, date_format="%Y-%m-%d")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def interpolate_short_price_gaps(
    frame: pd.DataFrame, max_gap: int = MAX_INTERPOLATION_GAP
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill internal missing runs up to max_gap sessions with positive natural cubic splines."""
    filled = frame.copy()
    records: list[dict[str, object]] = []
    for stock_id in frame.columns:
        values = frame[stock_id].to_numpy(dtype=np.float64)
        observed = np.flatnonzero(np.isfinite(values))
        spline = (
            CubicSpline(observed, values[observed], bc_type="natural", extrapolate=False)
            if len(observed) >= 4
            else None
        )
        changes = np.diff((~np.isfinite(values)).astype(int), prepend=0, append=0)
        for first, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            internal = first > 0 and stop < len(values)
            eligible_gap = internal and (stop - first) <= max_gap
            accepted = False
            if eligible_gap and spline is not None:
                estimates = spline(np.arange(first, stop))
                if np.isfinite(estimates).all() and (estimates > 0).all():
                    filled.iloc[first:stop, filled.columns.get_loc(stock_id)] = estimates
                    accepted = True
            records.append(
                {
                    "stock_id": stock_id,
                    "start": frame.index[first],
                    "end": frame.index[stop - 1],
                    "sessions": int(stop - first),
                    "internal": internal,
                    "eligible": eligible_gap,
                    "filled": accepted,
                }
            )
    audit = pd.DataFrame(
        records, columns=["stock_id", "start", "end", "sessions", "internal", "eligible", "filled"]
    )
    return filled, audit


def _distinct_ids(values: pd.Series) -> tuple:
    return tuple(sorted(values.dropna().unique()))


def _load_and_screen_raw(
    raw_prices_path: Path, sector_mapping_path: Path
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Filter XOSL sessions, pre-screen companies, attach ICB labels, and return eligible records."""
    identifiers = ["CompanyId", "Internal code"]
    scan_cols = ["Date", "ISIN", "AdjustedPrice", *identifiers]
    rows = pd.read_csv(raw_prices_path, sep=";", usecols=scan_cols, dtype="string")
    rows["Date"] = pd.to_datetime(rows["Date"], format="%Y-%m-%d")

    end = rows["Date"].max()
    calendar_start = end - pd.DateOffset(years=CALENDAR_LOOKBACK_YEARS)
    window_start = end - pd.DateOffset(years=SAMPLE_YEARS)
    calendar = xcals.get_calendar(CALENDAR_CODE, start=calendar_start, end=end)
    non_trading_days = pd.date_range(calendar_start, end).difference(calendar.sessions)
    on_closed_day = rows["Date"].isin(non_trading_days)

    history = rows.loc[~on_closed_day]
    window = history.loc[history["Date"].between(window_start, end)]
    trading_dates = pd.DatetimeIndex(window["Date"].unique()).sort_values()

    # Link ISINs transitively by nonmissing CompanyId or Internal code across the filtered history.
    parent = {isin: isin for isin in history["ISIN"].unique()}

    def company_root(isin: str) -> str:
        while isin != parent[isin]:
            parent[isin] = parent[parent[isin]]
            isin = parent[isin]
        return isin

    for column in identifiers:
        first_isin_by_id: dict[str, str] = {}
        links = history[["ISIN", column]].dropna().drop_duplicates()
        for isin, identifier in links.itertuples(index=False, name=None):
            first_isin = first_isin_by_id.setdefault(identifier, isin)
            parent[company_root(isin)] = company_root(first_isin)

    company_by_isin = {isin: company_root(isin) for isin in parent}
    company = window["ISIN"].map(company_by_isin)
    prices = pd.to_numeric(
        window["AdjustedPrice"].str.replace(",", ".", regex=False), errors="coerce"
    )
    price_values = prices.to_numpy(dtype=float, na_value=np.nan)
    has_price = np.isfinite(price_values) & (price_values > 0)
    company_dates = (
        pd.DataFrame(
            {
                "Company": company[has_price],
                "Session": trading_dates.get_indexer(window.loc[has_price, "Date"]),
            }
        )
        .drop_duplicates()
        .sort_values("Session")
    )

    n_dates = len(trading_dates)
    eligible_companies: set[str] = set()
    for company_id, positions in company_dates.groupby("Company")["Session"]:
        missing_runs = np.diff(np.r_[-1, positions.to_numpy(), n_dates]) - 1
        if missing_runs.max() <= MAX_INTERPOLATION_GAP:
            eligible_companies.add(company_id)

    keep_indices = window.index[company.isin(eligible_companies)]

    # Read full needed columns only for the retained eligible rows.
    full_cols = [
        "Date",
        "ISIN",
        "Name",
        "Sector",
        "CompanyId",
        "SecurityId",
        "Internal code",
        "AdjustedPrice",
        "Turnover",
        "Currency",
    ]
    full_raw = pd.read_csv(
        raw_prices_path,
        sep=";",
        decimal=",",
        usecols=full_cols,
        parse_dates=["Date"],
        float_precision="round_trip",
    )
    eligible_data = full_raw.loc[keep_indices].copy()

    # Map ICB labels by exact ISIN, then inherit via unambiguous shared CompanyId / Internal code.
    eligible_id_str = rows.loc[keep_indices, ["ISIN", *identifiers]].copy()
    snapshot = pd.read_csv(sector_mapping_path, usecols=["ISIN", *ICB_COLUMNS], dtype="string")
    by_isin = snapshot.set_index("ISIN", verify_integrity=True).dropna(subset=ICB_COLUMNS)

    id_keys = ["Identifier type", "Identifier"]
    isin_identifiers = (
        eligible_id_str.melt(
            id_vars="ISIN", value_vars=identifiers, var_name=id_keys[0], value_name=id_keys[1]
        )
        .dropna()
        .drop_duplicates()
    )
    mapped_identifiers = isin_identifiers.join(by_isin, on="ISIN").dropna(subset=ICB_COLUMNS)
    labels_by_identifier = mapped_identifiers[[*id_keys, *ICB_COLUMNS]].drop_duplicates()
    inherited = (
        isin_identifiers.merge(labels_by_identifier, on=id_keys)[["ISIN", *ICB_COLUMNS]]
        .drop_duplicates()
        .drop_duplicates("ISIN", keep=False)
        .set_index("ISIN")
    )
    eligible_isins = pd.Index(eligible_id_str["ISIN"].unique(), name="ISIN")
    classifications = by_isin.reindex(eligible_isins).combine_first(
        inherited.reindex(eligible_isins)
    )[ICB_COLUMNS]

    for col in ICB_COLUMNS:
        eligible_data[col] = eligible_data["ISIN"].map(classifications[col])

    return eligible_data, trading_dates


def _reconstruct_one_class_per_issuer(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join nonoverlapping ISIN spans, select one share class per issuer, and return panels."""
    data = data.sort_values(["Date", "ISIN", "Sector"], kind="stable")
    assert data[["Date", "ISIN"]].notna().all().all()
    assert data["Currency"].dropna().eq("NOK").all()
    sessions = pd.DatetimeIndex(data["Date"].unique(), name="Date")
    session_number = pd.Series(np.arange(len(sessions)), index=sessions)

    profiles = data.groupby("ISIN").agg(
        first=("Date", "min"),
        last=("Date", "max"),
        company=("CompanyId", _distinct_ids),
        security=("SecurityId", _distinct_ids),
        internal=("Internal code", _distinct_ids),
    )
    labels = data.groupby("ISIN")[ICB_COLUMNS].last()
    histories = {isin: {isin} for isin in profiles.index}

    def join_histories(isins: list[str], *, adjacent: bool = False) -> bool:
        members = set().union(*(histories[isin] for isin in isins))
        spans = profiles.loc[sorted(members)].sort_values("first")
        starts = spans["first"].iloc[1:].map(session_number).to_numpy()
        ends = spans["last"].iloc[:-1].map(session_number).to_numpy()
        if not (starts > ends).all():
            return False
        if adjacent:
            groups = [id(histories[isin]) for isin in spans.index]
            cross_component = np.array([a != b for a, b in zip(groups[:-1], groups[1:])])
            span_labels = labels.loc[spans.index]
            if (
                not (starts[cross_component] == ends[cross_component] + 1).all()
                or span_labels.isna().any().any()
                or len(span_labels.drop_duplicates()) != 1
            ):
                return False
        for isin in members:
            histories[isin] = members
        return True

    for aliases in KNOWN_ISIN_LINKS:
        found = [isin for isin in aliases if isin in profiles.index]
        if len(found) > 1:
            assert join_histories(found), found

    for keys, adjacent in [(["company", "security"], False), (["internal"], True)]:
        stable = profiles.loc[profiles[keys].map(len).eq(1).all(axis=1)]
        for _, group in stable.groupby(keys):
            if len(group) > 1:
                join_histories(group.index.tolist(), adjacent=adjacent)

    stock_by_isin: dict[str, str] = {}
    for isin, members in histories.items():
        spans = profiles.loc[sorted(members)].sort_values("last")
        assert spans["first"].iloc[1:].gt(spans["last"].shift().iloc[1:]).all(), isin
        stock_by_isin[isin] = spans.index[-1]
    data["stock_id"] = data["ISIN"].map(stock_by_isin)
    assert data.groupby(["stock_id", "Date"])["ISIN"].nunique().eq(1).all()

    keys = ["stock_id", "Date"]
    grouped = data.groupby(keys)
    records = data.drop_duplicates(keys).set_index(keys).sort_index()
    records["price_conflict"] = grouped["AdjustedPrice"].nunique(dropna=False).gt(1)
    price = records["AdjustedPrice"]
    records["invalid_price"] = ~(np.isfinite(price) & price.gt(0))
    records["AdjustedPrice"] = price.where(~(records["price_conflict"] | records["invalid_price"]))
    turnover = records["Turnover"]
    records["Turnover"] = turnover.where(
        grouped["Turnover"].nunique(dropna=False).eq(1) & np.isfinite(turnover) & turnover.ge(0)
    )

    issuer_members = {stock_id: {stock_id} for stock_id in records.index.unique("stock_id")}
    for identifier in ("CompanyId", "Internal code"):
        links = data[["stock_id", identifier]].dropna().drop_duplicates()
        for _, stock_group in links.groupby(identifier)["stock_id"]:
            members = set().union(*(issuer_members[stock_id] for stock_id in stock_group))
            for stock_id in members:
                issuer_members[stock_id] = members

    class_ranking = (
        records.groupby(level="stock_id")
        .agg(usable_dates=("AdjustedPrice", "count"), total_turnover=("Turnover", "sum"))
        .reset_index()
    )
    class_ranking["issuer"] = class_ranking["stock_id"].map(
        {stock_id: min(members) for stock_id, members in issuer_members.items()}
    )
    class_ranking = class_ranking.sort_values(
        ["usable_dates", "total_turnover", "stock_id"], ascending=[False, False, True]
    )
    keep = class_ranking.drop_duplicates("issuer")["stock_id"]

    records = records.loc[records.index.get_level_values("stock_id").isin(keep)]
    by_stock = data.loc[data["stock_id"].isin(keep)].groupby("stock_id")
    metadata = records.groupby(level="stock_id")[["Name", "CompanyId", "Sector", *ICB_COLUMNS]].last()
    metadata["sector_changed"] = by_stock["Sector"].nunique().gt(1)
    metadata["Name"] = metadata["Name"].astype(str).str.strip()

    grid = pd.MultiIndex.from_product([metadata.index, sessions], names=keys)
    reindexed = records[["ISIN", "AdjustedPrice"]].reindex(grid)

    observed_prices = (
        reindexed["AdjustedPrice"].unstack(level="stock_id").reindex(index=sessions).sort_index(axis=1)
    )
    recorded = (
        reindexed["ISIN"].notna().unstack(level="stock_id").reindex_like(observed_prices).fillna(False)
    )
    metadata = metadata.reindex(observed_prices.columns)
    return observed_prices, recorded, metadata


def _verify_interpolation_invariants(
    observed_prices: pd.DataFrame,
    interpolated_prices: pd.DataFrame,
    observed_returns: pd.DataFrame,
    interpolated_returns: pd.DataFrame,
    gap_audit: pd.DataFrame,
) -> None:
    """Assert that interpolation respects every thesis invariant."""
    obs_p_mask = observed_prices.notna().to_numpy()
    np.testing.assert_array_equal(
        interpolated_prices.to_numpy()[obs_p_mask], observed_prices.to_numpy()[obs_p_mask]
    )
    obs_r_mask = observed_returns.notna().to_numpy()
    np.testing.assert_array_equal(
        interpolated_returns.to_numpy()[obs_r_mask], observed_returns.to_numpy()[obs_r_mask]
    )
    imputed_prices = observed_prices.isna() & interpolated_prices.notna()
    assert interpolated_prices.where(imputed_prices).stack().gt(0).all()
    accepted = gap_audit.loc[gap_audit["filled"].eq(True)]
    assert accepted["internal"].all() and accepted["sessions"].le(MAX_INTERPOLATION_GAP).all()

    # Synthetic gap-length boundary check.
    fixture = pd.DataFrame(
        {
            "price": [
                np.nan,
                100.0,
                np.nan,
                102.0,
                *([np.nan] * 5),
                108.0,
                *([np.nan] * 6),
                115.0,
                116.0,
                np.nan,
            ]
        }
    )
    filled, _ = interpolate_short_price_gaps(fixture, MAX_INTERPOLATION_GAP)
    assert pd.isna(filled.iloc[0, 0]) and pd.isna(filled.iloc[-1, 0])
    assert filled.iloc[2:3, 0].notna().all() and filled.iloc[4:9, 0].notna().all()
    assert filled.iloc[10:16, 0].isna().all()


def build_dataset(
    raw_prices_path: Path = RAW_PRICES_PATH,
    sector_mapping_path: Path = SECTOR_MAPPING_PATH,
    output_dir: Path = PROCESSED_DIR,
) -> dict[str, object]:
    """Run the full preparation pipeline and write Data/processed/ files."""
    eligible_data, _ = _load_and_screen_raw(raw_prices_path, sector_mapping_path)
    all_prices, recorded, metadata = _reconstruct_one_class_per_issuer(eligible_data)

    # Thesis eligibility screen on the one-class-per-issuer histories.
    missing = all_prices.isna()
    eligible_mask = (
        all_prices.iloc[-1].notna()
        & missing.mean().lt(MAX_MISSING_FRACTION)
        & ~missing.rolling(MAX_INTERPOLATION_GAP + 1).sum().eq(MAX_INTERPOLATION_GAP + 1).any()
    )
    observed_prices = all_prices.loc[:, eligible_mask].copy()
    recorded = recorded.loc[:, observed_prices.columns]
    metadata = metadata.loc[observed_prices.columns].copy()
    assert metadata[ICB_COLUMNS].notna().all().all(), "All eligible stocks must have ICB labels"

    interpolated_prices, gap_audit = interpolate_short_price_gaps(
        observed_prices, MAX_INTERPOLATION_GAP
    )

    raw_obs_returns = np.log(observed_prices).diff()
    raw_interp_returns = np.log(interpolated_prices).diff()
    _verify_interpolation_invariants(
        observed_prices, interpolated_prices, raw_obs_returns, raw_interp_returns, gap_audit
    )

    imputed_price_mask = observed_prices.isna() & interpolated_prices.notna()
    imputed_return_mask = raw_obs_returns.isna() & raw_interp_returns.notna()

    # Apply reviewed corporate-action return corrections identically to both panels.
    observed_returns = raw_obs_returns.copy()
    interpolated_returns = raw_interp_returns.copy()
    correction_records = []
    for (date, stock_id), (corrected_value, reason) in RETURN_CORRECTIONS.items():
        assert stock_id in observed_returns.columns and date in observed_returns.index
        assert pd.notna(observed_returns.at[date, stock_id])
        observed_returns.at[date, stock_id] = corrected_value
        interpolated_returns.at[date, stock_id] = corrected_value
        correction_records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "stock_id": stock_id,
                "name": metadata.at[stock_id, "Name"],
                "log_return": None if np.isnan(corrected_value) else corrected_value,
                "reason": reason,
            }
        )

    # Drop the leading all-NaN diff row so return CSVs start with the first observable return session.
    observed_returns = observed_returns.iloc[1:]
    interpolated_returns = interpolated_returns.iloc[1:]
    imputed_return_mask = imputed_return_mask.iloc[1:]

    stocks = (
        metadata.rename(
            columns={"Name": "name", "CompanyId": "company_id", "Sector": "sector"}
        )
        .assign(
            absent_records=(~recorded).sum().astype(int),
            recorded_unavailable_prices=(recorded & observed_prices.isna()).sum().astype(int),
            missing_prices=observed_prices.isna().sum().astype(int),
            estimated_prices=imputed_price_mask.sum().astype(int),
            remaining_missing_prices=interpolated_prices.isna().sum().astype(int),
            estimated_returns=imputed_return_mask.sum().astype(int),
            remaining_missing_returns=interpolated_returns.isna().sum().astype(int),
        )
        .rename_axis("stock_id")
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stocks_path = output_dir / "stocks.csv"
    prices_obs_path = output_dir / "prices_observed.csv"
    returns_obs_path = output_dir / "log_returns_observed.csv"
    returns_interp_path = output_dir / "log_returns_interpolated.csv"

    write_csv_atomic(stocks, stocks_path, index=False)
    write_csv_atomic(observed_prices.reset_index(), prices_obs_path, index=False)
    write_csv_atomic(observed_returns.reset_index(), returns_obs_path, index=False)
    write_csv_atomic(interpolated_returns.reset_index(), returns_interp_path, index=False)

    manifest = {
        "contract": {
            "layout": "wide",
            "date_column": "Date",
            "date_format": "YYYY-MM-DD",
            "asset_identifier": "stock_id",
            "return_type": "log",
            "currency": "NOK",
            "calendar": CALENDAR_CODE,
            "missing_value_policy": "Preserved as empty cells (NaN) on the full XOSL session grid.",
            "retrospective_note": (
                "log_returns_interpolated.csv uses a two-sided natural cubic spline across the "
                "full sample; for strict causal modeling splits, evaluate on log_returns_observed.csv "
                "and fit any training interpolation inside the training slice of prices_observed.csv."
            ),
        },
        "sources": {
            "raw_prices": {
                "path": str(raw_prices_path.relative_to(ROOT)),
                "sha256": sha256_file(raw_prices_path),
            },
            "sector_mapping": {
                "path": str(sector_mapping_path.relative_to(ROOT)),
                "sha256": sha256_file(sector_mapping_path),
            },
        },
        "packages": {
            pkg: version(pkg) for pkg in ("exchange-calendars", "numpy", "pandas", "scipy")
        },
        "sample": {
            "n_stocks": int(observed_prices.shape[1]),
            "n_price_sessions": int(len(observed_prices)),
            "first_price_date": observed_prices.index[0].strftime("%Y-%m-%d"),
            "last_price_date": observed_prices.index[-1].strftime("%Y-%m-%d"),
            "n_return_sessions": int(len(observed_returns)),
            "first_return_date": observed_returns.index[0].strftime("%Y-%m-%d"),
            "last_return_date": observed_returns.index[-1].strftime("%Y-%m-%d"),
            "complete_return_dates_observed": int(observed_returns.notna().all(axis=1).sum()),
            "complete_return_dates_interpolated": int(
                interpolated_returns.notna().all(axis=1).sum()
            ),
        },
        "eligibility": {
            "sample_years": SAMPLE_YEARS,
            "require_final_session": True,
            "max_missing_fraction": MAX_MISSING_FRACTION,
            "max_gap_sessions": MAX_INTERPOLATION_GAP,
            "one_class_per_issuer": True,
        },
        "interpolation": {
            "method": "natural_cubic_spline",
            "max_gap_sessions": MAX_INTERPOLATION_GAP,
            "internal_only": True,
            "positive_only": True,
            "observed_missing_prices": int(observed_prices.isna().sum().sum()),
            "estimated_prices": int(imputed_price_mask.sum().sum()),
            "remaining_missing_prices": int(interpolated_prices.isna().sum().sum()),
            "observed_missing_returns": int(observed_returns.isna().sum().sum()),
            "estimated_returns": int(imputed_return_mask.sum().sum()),
            "remaining_missing_returns": int(interpolated_returns.isna().sum().sum()),
        },
        "return_corrections": correction_records,
        "outputs": {
            path.name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in (stocks_path, prices_obs_path, returns_obs_path, returns_interp_path)
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    manifest = build_dataset()
    sample = manifest["sample"]
    interp = manifest["interpolation"]
    print(
        f"Built Data/processed/ for {sample['n_stocks']} stocks: "
        f"{sample['n_price_sessions']} price sessions ({sample['first_price_date']} .. {sample['last_price_date']}), "
        f"{sample['n_return_sessions']} return sessions ({sample['first_return_date']} .. {sample['last_return_date']})."
    )
    print(
        f"Interpolation filled {interp['estimated_prices']} prices and {interp['estimated_returns']} returns; "
        f"complete return dates: {sample['complete_return_dates_observed']} -> {sample['complete_return_dates_interpolated']}."
    )


if __name__ == "__main__":
    main()
