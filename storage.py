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
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

DATA_ROOT = Path("./data")


# ── Primary key definitions (used for deduplication) ─────────────────────────

PK = {
    "ohlcv":      ["event_timestamp", "ticker", "frequency"],
    "indicators": ["event_timestamp", "ticker"],
    "macro":      ["event_timestamp", "indicator_code", "revision_number"],
    "insider":    ["ticker_queried", "trade date", "insider name", "trade type", "qty"],
    "earnings":   ["ticker", "period_end"],
    "universe":   ["ticker"],
}

# Partition columns per table (determines directory nesting)
PARTITION_COLS = {
    "ohlcv":      ["year", "month"],
    "indicators": ["year", "month"],
    "macro":      ["indicator_code", "year"],
    "insider":    ["ticker_queried", "year"],
    "earnings":   ["ticker", "year"],
}


def _add_partition_cols(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Add hive-partition columns derived from event_timestamp (or equivalent)."""
    df = df.copy()
    if table in ("ohlcv", "indicators"):
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
    elif table == "earnings":
        if "period_end" in df.columns:
            ts = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
        elif "event_timestamp" in df.columns:
            ts = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
        else:
            ts = pd.Series([pd.Timestamp.now(tz="UTC")] * len(df), index=df.index)
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
    """Write DataFrame to Parquet atomically (write→rename, never corrupt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_path, compression="snappy")
    os.replace(tmp_path, path)  # atomic on POSIX


def _read_partition(path: Path) -> pd.DataFrame:
    """Read an existing partition file, return empty DF if it doesn't exist."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        log.warning("Could not read %s: %s — treating as empty", path, exc)
        return pd.DataFrame()


def append(table: str, new_df: pd.DataFrame) -> int:
    """
    Idempotently append new_df into the partitioned store for `table`.

    Returns the number of net-new rows written (0 if all were duplicates).
    """
    if new_df is None or new_df.empty:
        return 0

    # Universe uses a single file (tiny, refreshed wholesale)
    if table == "universe":
        path = DATA_ROOT / "universe" / "universe_latest.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_partition(path)
        combined = _merge(existing, new_df, PK.get("universe", []))
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

    # Resolve column type mismatches before dedup
    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = combined[col].astype(str)

    # Available PK cols (some may not exist in older partitions)
    available_pk = [c for c in pk_cols if c in combined.columns]
    if available_pk:
        # Keep last (= new row) for each primary key
        combined = combined.drop_duplicates(subset=available_pk, keep="last")
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
        # fallback: check for universe_latest.parquet
        alt = root / "universe_latest.parquet"
        if alt.exists():
            return pd.read_parquet(alt)
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def row_counts() -> dict[str, int]:
    """Quick audit: how many rows are stored per table."""
    counts = {}
    for table in ("ohlcv", "indicators", "macro", "insider", "earnings", "universe"):
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
