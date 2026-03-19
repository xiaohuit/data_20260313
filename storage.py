"""
Partitioned Parquet storage layer — safe for continuous append.

Layout on disk:
    data/
      ohlcv/         year=YYYY/month=MM/data.parquet
      indicators/    year=YYYY/month=MM/data.parquet
      macro/         series=<CODE>/year=YYYY/data.parquet
      insider/       ticker=<T>/year=YYYY/data.parquet
      earnings/      ticker=<T>/year=YYYY/data.parquet
      universe/      universe_latest.parquet  (full refresh, small)

Append contract (no data loss, no duplicates):
  1. Determine which partition files the new rows touch.
  2. For each partition: read existing rows → concat new → drop_duplicates
     on the primary-key columns → write back atomically (write to .tmp,
     then os.replace so a crash mid-write never corrupts the live file).
  3. Update the global state checkpoint ONLY after all partitions succeed.

DuckDB can query the entire tree with a single glob:
    SELECT * FROM read_parquet('data/ohlcv/**/*.parquet',
                               hive_partitioning=true)
    WHERE knowledge_timestamp <= '2023-06-30'
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

DATA_ROOT = Path("./data")

# Per-partition-file lock registry — prevents two concurrent daemon jobs from
# racing on the same Parquet file (read→merge→write is not atomic without this).
# e.g. job_earnings and job_implied_q4_eps both append to the earnings table.
_PARTITION_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_META = threading.Lock()   # guards _PARTITION_LOCKS itself


def _partition_lock(path: Path) -> threading.Lock:
    """Return (creating if needed) the Lock for a specific partition file path."""
    key = str(path.resolve())
    with _LOCKS_META:
        if key not in _PARTITION_LOCKS:
            _PARTITION_LOCKS[key] = threading.Lock()
        return _PARTITION_LOCKS[key]


# ── Primary key definitions (used for deduplication) ─────────────────────────

PK = {
    "ohlcv":       ["event_timestamp", "ticker", "frequency"],
    "indicators":  ["event_timestamp", "ticker"],
    "valuations":  ["event_timestamp", "ticker"],
    "macro":       ["event_timestamp", "indicator_code", "revision_number"],
    "insider":     ["ticker_queried", "trade date", "insider name", "trade type", "qty"],
    "earnings":    ["ticker", "period_end", "form"],
    "financials":  ["ticker", "period_end", "form"],
    "dividends":        ["ticker", "year"],
    "events_8k":        ["ticker", "accession_number"],
    "universe_history": ["ticker", "index_name", "action", "event_date"],
    "short_interest":   ["ticker", "settlement_date"],
    "universe":         ["ticker"],
    "sectors":          ["ticker"],
    "quality_metrics":  ["ticker", "period_end", "form"],
}

# Partition columns per table (determines directory nesting)
PARTITION_COLS = {
    "ohlcv":       ["year", "month"],
    "indicators":  ["year", "month"],
    "valuations":  ["year", "month"],
    "macro":       ["indicator_code", "year"],
    "insider":     ["ticker_queried", "year"],
    "earnings":    ["ticker", "year"],
    "financials":  ["ticker", "year"],
    "dividends":        ["ticker"],
    "events_8k":        ["ticker", "year"],
    "universe_history": ["year"],
    "short_interest":   ["ticker", "year"],
    "quality_metrics":  ["year"],
}


def _add_partition_cols(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Add hive-partition columns derived from event_timestamp (or equivalent)."""
    df = df.copy()
    if table in ("ohlcv", "indicators", "valuations"):
        ts = pd.to_datetime(df["event_timestamp"], utc=True)
        df["year"]  = ts.dt.year.astype(str)
        df["month"] = ts.dt.month.apply(lambda m: f"{m:02d}")
    elif table == "macro":
        ts = pd.to_datetime(df["event_timestamp"], utc=True)
        df["year"] = ts.dt.year.astype(str)
    elif table == "insider":
        # Find the best date column (always use column access to get a Series)
        date_col = next(
            (c for c in df.columns if "trade" in c.lower() and "date" in c.lower()),
            next((c for c in df.columns if "knowledge" in c.lower()), None),
        )
        if date_col:
            ts = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        else:
            ts = pd.Series([pd.Timestamp.now(tz="UTC")] * len(df), index=df.index)
        df["year"] = ts.dt.year.fillna(9999).astype(int).astype(str)
    elif table in ("earnings", "financials"):
        if "period_end" in df.columns:
            ts = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
        elif "event_timestamp" in df.columns:
            ts = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
        else:
            ts = pd.Series([pd.Timestamp.now(tz="UTC")] * len(df), index=df.index)
        df["year"] = ts.dt.year.fillna(9999).astype(int).astype(str)
    elif table in ("events_8k", "universe_history", "short_interest"):
        ts = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
        df["year"] = ts.dt.year.fillna(9999).astype(int).astype(str)
    elif table == "quality_metrics":
        # Partition by the period_end year (= fiscal year of the filing)
        ts = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
        df["year"] = ts.dt.year.fillna(9999).astype(int).astype(str)
    return df


def _partition_path(table: str, partition_values: dict[str, str]) -> Path:
    """Build the directory path for a given partition."""
    parts = PARTITION_COLS.get(table, [])
    base = DATA_ROOT / table
    for col in parts:
        val = partition_values.get(col, "unknown")
        base = base / f"{col}={val}"
    return base


def _atomic_write(path: Path, df: pd.DataFrame) -> None:
    """
    Write DataFrame to Parquet atomically (write→rename, never corrupt).

    Type normalisations applied before every write:
      • large_string  → string (utf8)  — fixes pyarrow ≥15 reader incompatibility
        (old pyarrow wrote incorrect repetition-level statistics for large_string
        that newer readers reject with "Repetition level histogram size mismatch")
      • timestamp / date string columns → timestamp[us, UTC] / date32
        so downstream readers (pandas, DuckDB, Arrow Flight) get native types
        instead of opaque VARCHAR and don't need CAST everywhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = _normalise_arrow_types(table)
    pq.write_table(table, tmp_path, compression="snappy")
    os.replace(tmp_path, path)  # atomic on POSIX


# Columns whose string values are full ISO datetimes with timezone.
_TS_COLS: frozenset[str] = frozenset({
    "event_timestamp", "knowledge_timestamp", "ingestion_timestamp",
})
# Columns whose string values are plain dates (YYYY-MM-DD).
_DATE_COLS: frozenset[str] = frozenset({
    "period_end", "settlement_date", "event_date", "filing_date",
})


def _normalise_arrow_types(table: pa.Table) -> pa.Table:
    """
    Normalise Arrow types for cross-version compatibility and evaluator usability:
      • Known timestamp columns (string or large_string) → timestamp[us, UTC]
      • Known date columns     (string or large_string) → date32
      • Any remaining large_string                       → string (utf8)
    """
    new_arrays: list = []
    new_fields: list = []

    for i, field in enumerate(table.schema):
        arr  = table.column(i)
        name = field.name
        t    = field.type
        is_any_string = pa.types.is_large_string(t) or pa.types.is_string(t)

        if is_any_string and name in _TS_COLS:
            try:
                arr   = _cast_ts_array(arr)
                field = field.with_type(pa.timestamp("us", tz="UTC"))
            except Exception:
                arr   = arr.cast(pa.utf8())
                field = field.with_type(pa.string())
        elif is_any_string and name in _DATE_COLS:
            try:
                arr   = _cast_date_array(arr)
                field = field.with_type(pa.date32())
            except Exception:
                arr   = arr.cast(pa.utf8())
                field = field.with_type(pa.string())
        elif pa.types.is_large_string(t):
            arr   = arr.cast(pa.utf8())
            field = field.with_type(pa.string())

        new_arrays.append(arr)
        new_fields.append(field)

    return pa.table(
        {f.name: a for f, a in zip(new_fields, new_arrays)},
        schema=pa.schema(new_fields),
    )


def _cast_ts_array(arr: pa.Array) -> pa.Array:
    """Cast a string array of ISO timestamps to timestamp[us, UTC]."""
    import numpy as np
    # Use pandas for robust ISO parsing (handles +00:00 / Z / naive)
    s = arr.cast(pa.utf8()).to_pandas().astype(str)
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return pa.array(ts, type=pa.timestamp("us", tz="UTC"))


def _cast_date_array(arr: pa.Array) -> pa.Array:
    """Cast a string array of YYYY-MM-DD dates to date32."""
    s  = arr.cast(pa.utf8()).to_pandas().astype(str)
    d  = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").dt.date
    return pa.array(d, type=pa.date32())


def _read_partition(path: Path) -> pd.DataFrame:
    """
    Read an existing partition file, return empty DF if it doesn't exist.

    Falls back to DuckDB when pyarrow cannot read the file (e.g. old files
    written with large_string + incorrect repetition-level statistics).
    After a successful DuckDB read the file is rewritten in-place with the
    current normalised schema so future reads use pyarrow directly.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("pyarrow cannot read %s (%s) — trying DuckDB fallback", path, exc)
    try:
        import duckdb
        conn = duckdb.connect()
        df   = conn.execute(
            f"SELECT * FROM read_parquet('{path}')"
        ).fetchdf()
        conn.close()
        if not df.empty:
            # Rewrite immediately with normalised types so next read uses pyarrow
            _atomic_write(path, df)
            log.info("  Rewrote %s with normalised types (DuckDB fallback)", path)
        return df
    except Exception as exc2:
        log.warning("DuckDB fallback also failed for %s: %s — treating as empty", path, exc2)
        return pd.DataFrame()


def append(table: str, new_df: pd.DataFrame) -> int:
    """
    Idempotently append new_df into the partitioned store for `table`.

    Returns the number of net-new rows written (0 if all were duplicates).
    """
    if new_df is None or new_df.empty:
        return 0

    # Universe and sectors use a single file (tiny, refreshed wholesale)
    if table == "universe":
        path = DATA_ROOT / "universe" / "universe_latest.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _partition_lock(path):
            existing = _read_partition(path)
            combined = _merge(existing, new_df, PK.get("universe", []))
            _atomic_write(path, combined)
        return len(combined) - len(existing)

    if table == "sectors":
        path = DATA_ROOT / "sectors" / "sectors_latest.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _partition_lock(path):
            existing = _read_partition(path)
            combined = _merge(existing, new_df, PK.get("sectors", []))
            _atomic_write(path, combined)
        return len(combined) - len(existing)

    df = _add_partition_cols(new_df, table)
    part_cols = PARTITION_COLS.get(table, [])
    pk_cols = PK.get(table, [])

    total_new = 0
    for part_values, group in df.groupby(part_cols):
        # part_values is a tuple when groupby has multiple keys
        if not isinstance(part_values, tuple):
            part_values = (part_values,)
        pv_dict = dict(zip(part_cols, part_values))

        part_dir  = _partition_path(table, pv_dict)
        part_file = part_dir / "data.parquet"

        # Acquire per-file lock before read-merge-write to prevent two concurrent
        # daemon jobs from corrupting the same partition (race: both read stale
        # data, merge independently, last writer wins and discards the other's rows).
        with _partition_lock(part_file):
            existing = _read_partition(part_file)
            # Only drop purely synthetic date-derived partition columns (year, month).
            # Semantic columns (indicator_code, ticker_queried, ticker) must stay
            # in the file so read_all() can recover them without hive_partitioning.
            SYNTHETIC_COLS = {"year", "month"}
            store_group = group.drop(
                columns=[c for c in part_cols if c in group.columns and c in SYNTHETIC_COLS],
                errors="ignore",
            )
            combined   = _merge(existing, store_group, pk_cols)
            net_new    = len(combined) - len(existing)

            if net_new > 0 or existing.empty:
                _atomic_write(part_file, combined)
                log.debug("  %s partition %s → +%d rows (%d total)",
                          table, pv_dict, net_new, len(combined))
        total_new += max(net_new, 0)

    return total_new


def _merge(existing: pd.DataFrame, new: pd.DataFrame, pk_cols: list[str]) -> pd.DataFrame:
    """
    Merge existing and new DataFrames, deduplicating on pk_cols.
    New rows take precedence (they may have updated values like adj_close corrections).
    """
    if existing.empty:
        return new.copy().reset_index(drop=True)
    if new.empty:
        return existing.copy()

    combined = pd.concat([existing, new], ignore_index=True)

    # Available PK cols (some may not exist in older partitions)
    available_pk = [c for c in pk_cols if c in combined.columns]

    # Resolve type mismatches ONLY on PK columns — drop_duplicates only uses these.
    # Converting ALL object columns (as was done previously) corrupts numeric data by
    # turning None → "None" and NaN → "nan" strings, which silently breaks downstream
    # computations like rolling EPS sums, gross_margin CAGR, etc.
    for col in available_pk:
        if combined[col].dtype == object:
            combined[col] = combined[col].fillna("").astype(str)

    if available_pk:
        # Keep last (= new row) for each primary key
        combined = combined.drop_duplicates(subset=available_pk, keep="last")
    else:
        # No PK columns found — deduplication is impossible; log a warning because
        # repeated appends without dedup will grow the partition without bound.
        log.warning("_merge: no PK columns found in combined data (pk_cols=%s, "
                    "available=%s) — skipping dedup", pk_cols, list(combined.columns))
    return combined.reset_index(drop=True)


def read_all(table: str) -> pd.DataFrame:
    """
    Read the entire table across all partitions (for reporting/validation).
    For production queries use DuckDB directly.
    """
    root = DATA_ROOT / table
    if not root.exists():
        return pd.DataFrame()
    files = sorted(root.rglob("data.parquet"))
    if not files:
        # fallback: check for single-file tables (universe, sectors)
        for alt_name in ("universe_latest.parquet", "sectors_latest.parquet"):
            alt = root / alt_name
            if alt.exists():
                return pd.read_parquet(alt)
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def row_counts() -> dict[str, int]:
    """Quick audit: how many rows are stored per table."""
    counts = {}
    for table in ("ohlcv", "indicators", "valuations", "macro", "insider", "earnings",
                  "financials", "dividends", "events_8k", "universe_history",
                  "short_interest", "universe", "sectors", "quality_metrics"):
        root = DATA_ROOT / table
        if not root.exists():
            counts[table] = 0
            continue
        files = list(root.rglob("*.parquet"))
        total = 0
        for f in files:
            try:
                total += pq.read_metadata(f).num_rows
            except Exception:
                pass
        counts[table] = total
    return counts
