"""
migrate_parquet.py — one-shot migration of existing Parquet files.

Fixes:
  1. large_string → string (utf8)     fixes pyarrow ≥15 reader incompatibility
  2. timestamp strings → timestamp[us, UTC]   native type for PiT filtering
  3. date strings     → date32                native type for date comparison

Uses DuckDB to read (works regardless of pyarrow version), then rewrites
each file with the storage._atomic_write path so they are normalised.
Run once after upgrading pyarrow; safe to re-run (idempotent).

Usage:
    python migrate_parquet.py [--dry-run] [--table ohlcv] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from storage import _atomic_write  # reuse the normalised write path

DATA_ROOT = Path("./data")

TABLES = [
    "ohlcv", "indicators", "valuations", "macro", "insider",
    "earnings", "financials", "dividends", "events_8k",
    "universe_history", "short_interest", "quality_metrics",
    "universe", "sectors",
]


def migrate_file(path: Path) -> tuple[Path, str]:
    """
    Read one parquet file with DuckDB and rewrite with normalised types.
    Returns (path, status) where status is 'ok', 'skipped', or 'error:<msg>'.
    """
    try:
        conn = duckdb.connect()
        df   = conn.execute(
            f"SELECT * FROM read_parquet('{path}')"
        ).fetchdf()
        conn.close()
    except Exception as e:
        return path, f"error:duckdb:{e}"

    if df.empty:
        return path, "skipped:empty"

    try:
        _atomic_write(path, df)
        return path, "ok"
    except Exception as e:
        return path, f"error:write:{e}"


def collect_files(tables: list[str]) -> list[Path]:
    files: list[Path] = []
    for table in tables:
        root = DATA_ROOT / table
        if not root.exists():
            continue
        files.extend(root.rglob("*.parquet"))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate parquet files to normalised schema")
    parser.add_argument("--dry-run",  action="store_true",
                        help="List files that would be migrated without touching them")
    parser.add_argument("--table",    nargs="+", default=TABLES,
                        help="Only migrate specific tables (default: all)")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel worker threads (default: 4)")
    args = parser.parse_args()

    files = collect_files(args.table)
    print(f"Found {len(files)} parquet files across {len(args.table)} tables")

    if args.dry_run:
        for f in files:
            print(f"  {f}")
        return

    ok = errors = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(migrate_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            path, status = future.result()
            rel = path.relative_to(DATA_ROOT)
            if status == "ok":
                ok += 1
                if i % 50 == 0 or i == len(files):
                    print(f"  [{i}/{len(files)}]  {ok} ok  {errors} errors  {skipped} skipped")
            elif status.startswith("skipped"):
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR  {rel}: {status}")

    print(f"\nDone.  {ok} migrated  /  {skipped} skipped  /  {errors} errors")
    if errors:
        print("Check logs above for error details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
