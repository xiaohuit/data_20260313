"""
Parquet cold-storage archive.

Hot data (< 2 years) lives in TimescaleDB for fast SQL queries.
Cold data (>= 2 years) is exported to Parquet and queried via DuckDB.

This module handles:
  - Exporting TimescaleDB partitions to partitioned Parquet (year/month layout).
  - Reading archived Parquet files via DuckDB with the SAME PiT filtering
    semantics as the SQL loader.
  - Transparent hot+cold querying so PiTDataLoader doesn't need to know
    whether data is in Postgres or Parquet.

Parquet layout:
  {PARQUET_ROOT}/
    market_ohlcv/
      year=2019/month=01/data.parquet
      year=2019/month=02/data.parquet
      ...
    macro_indicators/
      year=2019/...
    financial_events/
      category=EARNINGS/year=2019/...
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from financial_pipeline.config import CONFIG

log = logging.getLogger(__name__)

PARQUET_ROOT = Path(CONFIG.parquet_root)


# ── Export from TimescaleDB to Parquet ────────────────────────────────────────

async def export_to_parquet(
    table_name: str,
    session_factory,
    cutoff_date: datetime | None = None,
) -> int:
    """
    Export rows older than `cutoff_date` from `table_name` to Parquet.
    Default cutoff: 2 years ago.
    Returns number of rows exported.
    """
    if cutoff_date is None:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=730)

    from sqlalchemy import text

    total_rows = 0
    # Export month by month to keep file sizes manageable
    current = datetime(cutoff_date.year - 6, 1, 1, tzinfo=timezone.utc)
    while current < cutoff_date:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        part_dir = PARQUET_ROOT / table_name / f"year={current.year}" / f"month={current.month:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "data.parquet"

        if out_path.exists():
            current = next_month
            continue   # already exported

        query = f"""
            SELECT * FROM {table_name}
            WHERE event_timestamp >= :start AND event_timestamp < :end
        """
        async with session_factory() as session:
            from sqlalchemy import text as sql_text
            result = await session.execute(
                sql_text(query), {"start": current, "end": next_month}
            )
            rows = result.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=result.keys())
                df.to_parquet(out_path, index=False, compression="snappy")
                total_rows += len(df)
                log.info("Exported %d rows to %s", len(df), out_path)

        current = next_month

    return total_rows


# ── DuckDB hot+cold query layer ───────────────────────────────────────────────

class ParquetReader:
    """
    Reads Parquet archives using DuckDB. Applies the same PiT filter
    (knowledge_timestamp <= as_of) as the SQL loader.
    """

    def __init__(self) -> None:
        try:
            import duckdb  # type: ignore
            self._duck = duckdb.connect(database=":memory:")
            self._duck.execute("INSTALL parquet; LOAD parquet;")
            self._available = True
        except ImportError:
            log.warning("duckdb not installed — cold Parquet queries unavailable")
            self._available = False

    def query_ohlcv(
        self,
        tickers: list[str],
        as_of: datetime,
        lookback_start: datetime,
    ) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        glob = str(PARQUET_ROOT / "market_ohlcv" / "**" / "*.parquet")
        ticker_list = ", ".join(f"'{t}'" for t in tickers)
        sql = f"""
            SELECT event_timestamp, ticker, adj_close, close, volume
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE ticker IN ({ticker_list})
              AND frequency = '1D'
              AND event_timestamp >= TIMESTAMPTZ '{lookback_start.isoformat()}'
              AND event_timestamp <= TIMESTAMPTZ '{as_of.isoformat()}'
              AND knowledge_timestamp <= TIMESTAMPTZ '{as_of.isoformat()}'
            ORDER BY event_timestamp
        """
        return self._duck.execute(sql).df()

    def query_macro(
        self,
        as_of: datetime,
        lookback_start: datetime,
    ) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        glob = str(PARQUET_ROOT / "macro_indicators" / "**" / "*.parquet")
        sql = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY indicator_code, event_timestamp
                        ORDER BY revision_number DESC
                    ) AS rn
                FROM read_parquet('{glob}', hive_partitioning=true)
                WHERE knowledge_timestamp <= TIMESTAMPTZ '{as_of.isoformat()}'
                  AND event_timestamp >= TIMESTAMPTZ '{lookback_start.isoformat()}'
            )
            SELECT indicator_code, event_timestamp, value
            FROM ranked WHERE rn = 1
            ORDER BY indicator_code, event_timestamp
        """
        return self._duck.execute(sql).df()
