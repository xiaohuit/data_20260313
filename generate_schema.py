"""
generate_schema.py — write data/_schema.json describing all tables.

The schema file lets an evaluator understand the data layout without
reading any parquet file.  It documents:
  • column names and types (as Arrow type strings)
  • primary key columns (for deduplication semantics)
  • partition columns (directory structure)
  • time semantics (event_timestamp vs knowledge_timestamp)
  • row counts per table

Usage:
    python generate_schema.py
Outputs:
    data/_schema.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

DATA_ROOT = Path("./data")

# ── Static schema definitions ────────────────────────────────────────────────
# Describes the intended schema.  Actual types are filled in from live files.

TABLE_META: dict[str, dict] = {
    "ohlcv": {
        "description": "Daily OHLCV price data (adjusted) for all universe tickers",
        "primary_key": ["event_timestamp", "ticker", "frequency"],
        "partition_cols": ["year", "month"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": (
            "event_timestamp = market open UTC for that trading day.  "
            "knowledge_timestamp = 16:00 ET (market close) on the same day — "
            "prices are only observable after market close."
        ),
    },
    "indicators": {
        "description": "Daily technical indicators (RSI, MACD, Bollinger, etc.) per ticker",
        "primary_key": ["event_timestamp", "ticker"],
        "partition_cols": ["year", "month"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": "Computed from OHLCV; same PiT timing as ohlcv.",
    },
    "valuations": {
        "description": (
            "Daily valuation ratios (P/E TTM, P/B, EV/EBITDA, P/S, FCF yield, "
            "dividend yield) joined with fundamentals"
        ),
        "primary_key": ["event_timestamp", "ticker"],
        "partition_cols": ["year", "month"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": (
            "knowledge_timestamp = 16:00 ET (market close).  "
            "Fundamental inputs (TTM EPS, revenue) are looked up with "
            "merge_asof(direction='backward') on their own knowledge_timestamps, "
            "so no look-ahead bias."
        ),
    },
    "macro": {
        "description": (
            "Macro time series: Fed Funds, 2Y/10Y yields, CPI, CPI YoY, "
            "M2, GDP growth, unemployment, consumer confidence, VIX, oil (WTI)"
        ),
        "primary_key": ["event_timestamp", "indicator_code", "revision_number"],
        "partition_cols": ["indicator_code", "year"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": (
            "knowledge_timestamp = event_timestamp + publication_lag "
            "(series-specific, 2–45 days).  Use knowledge_timestamp for PiT."
        ),
    },
    "insider": {
        "description": "SEC Form 4 insider buy/sell transactions",
        "primary_key": ["ticker_queried", "trade date", "insider name", "trade type", "qty"],
        "partition_cols": ["ticker_queried", "year"],
        "event_timestamp_col": "trade date",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": "knowledge_timestamp = SEC filing date of the Form 4.",
    },
    "earnings": {
        "description": (
            "Quarterly EPS actuals from SEC EDGAR (10-Q/10-K) or yfinance, "
            "plus analyst estimates where available"
        ),
        "primary_key": ["ticker", "period_end", "form"],
        "partition_cols": ["ticker", "year"],
        "event_timestamp_col": "period_end",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": (
            "knowledge_timestamp = EDGAR filed date.  "
            "For yfinance-sourced rows = period_end + 45 days (conservative)."
        ),
    },
    "financials": {
        "description": (
            "Fundamental financials (income stmt, balance sheet, cash flow) "
            "from SEC EDGAR XBRL or yfinance.  One row per (ticker, period_end, form)."
        ),
        "primary_key": ["ticker", "period_end", "form"],
        "partition_cols": ["ticker", "year"],
        "event_timestamp_col": "period_end",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": (
            "knowledge_timestamp = EDGAR filed date.  "
            "form = '10-Q' (quarterly) or '10-K' (annual)."
        ),
    },
    "dividends": {
        "description": "Annual dividend summary: DPS, growth rate, payout ratio, streak",
        "primary_key": ["ticker", "year"],
        "partition_cols": ["ticker"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": "event_timestamp = Dec 31 of the calendar year.",
    },
    "events_8k": {
        "description": "SEC 8-K material event filings (M&A, earnings, guidance, etc.)",
        "primary_key": ["ticker", "accession_number"],
        "partition_cols": ["ticker", "year"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": "knowledge_timestamp = SEC filed datetime.",
    },
    "universe_history": {
        "description": "Index membership change history (S&P 500 adds/removes)",
        "primary_key": ["ticker", "index_name", "action", "event_date"],
        "partition_cols": ["year"],
        "event_timestamp_col": "event_date",
        "knowledge_timestamp_col": None,
        "time_note": "No knowledge_timestamp — addition/removal dates are public same-day.",
    },
    "short_interest": {
        "description": "Bimonthly short interest from FINRA (shares short, days-to-cover)",
        "primary_key": ["ticker", "settlement_date"],
        "partition_cols": ["ticker", "year"],
        "event_timestamp_col": "event_timestamp",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": (
            "knowledge_timestamp = fetch time (datetime.now(UTC)) because "
            "yfinance only serves current FINRA data after official publication."
        ),
    },
    "quality_metrics": {
        "description": (
            "Derived quality scores per filing: gross margin CAGR, "
            "revenue CAGR, ROIC, Piotroski F-score, Altman Z-score"
        ),
        "primary_key": ["ticker", "period_end", "form"],
        "partition_cols": ["year"],
        "event_timestamp_col": "period_end",
        "knowledge_timestamp_col": "knowledge_timestamp",
        "time_note": "Same PiT semantics as financials — knowledge_timestamp = filed date.",
    },
    "universe": {
        "description": "Current universe of tracked tickers (S&P 500 + extensions)",
        "primary_key": ["ticker"],
        "partition_cols": [],
        "event_timestamp_col": None,
        "knowledge_timestamp_col": None,
        "time_note": "Single file, full refresh.  No time dimension.",
    },
    "sectors": {
        "description": "Sector/industry/country classification per ticker",
        "primary_key": ["ticker"],
        "partition_cols": [],
        "event_timestamp_col": None,
        "knowledge_timestamp_col": None,
        "time_note": "Single file, full refresh.",
    },
}


def _sample_file(table: str) -> Path | None:
    """Return any one parquet file for the table."""
    root = DATA_ROOT / table
    if not root.exists():
        return None
    files = sorted(root.rglob("*.parquet"))
    return files[-1] if files else None


def _arrow_type_str(t) -> str:
    """Human-readable Arrow type string."""
    import pyarrow as pa
    if pa.types.is_timestamp(t):
        return f"timestamp[{t.unit}, {t.tz or 'naive'}]"
    if pa.types.is_date(t):
        return "date32"
    return str(t)


def _live_columns(table: str) -> list[dict] | None:
    """Introspect actual column names and types from a live file."""
    f = _sample_file(table)
    if f is None:
        return None
    try:
        # Try pyarrow first, fall back to DuckDB
        try:
            schema = pq.read_schema(f)
            return [
                {"name": field.name, "type": _arrow_type_str(field.type)}
                for field in schema
                if field.name not in ("year", "month")   # synthetic partition cols
            ]
        except Exception:
            pass
        conn = duckdb.connect()
        desc = conn.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{f}')"
        ).fetchdf()
        conn.close()
        return [
            {"name": r["column_name"], "type": r["column_type"]}
            for _, r in desc.iterrows()
            if r["column_name"] not in ("year", "month")
        ]
    except Exception:
        return None


def _row_count(table: str) -> int | None:
    try:
        conn  = duckdb.connect()
        count = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet(
                '{DATA_ROOT}/{table}/**/*.parquet',
                hive_partitioning=true, union_by_name=true
            )
        """).fetchone()[0]
        conn.close()
        return int(count)
    except Exception:
        return None


def build_schema() -> dict:
    tables_out = {}
    for name, meta in TABLE_META.items():
        cols     = _live_columns(name)
        row_cnt  = _row_count(name)
        entry = {
            "description":              meta["description"],
            "primary_key":              meta["primary_key"],
            "partition_cols":           meta["partition_cols"],
            "event_timestamp_col":      meta["event_timestamp_col"],
            "knowledge_timestamp_col":  meta["knowledge_timestamp_col"],
            "time_semantics":           meta["time_note"],
            "row_count":                row_cnt,
            "columns":                  cols,
        }
        tables_out[name] = entry
        print(f"  {name:<20} rows={row_cnt or '?':>10}  "
              f"cols={len(cols) if cols else '?'}")

    return {
        "schema_version": "1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "data_root":      str(DATA_ROOT),
        "pit_note": (
            "All tables carry both event_timestamp (when the fact occurred) "
            "and knowledge_timestamp (earliest moment it was publicly available). "
            "For point-in-time correct backtests filter on knowledge_timestamp <= t, "
            "not event_timestamp."
        ),
        "tables": tables_out,
    }


def main() -> None:
    print("Scanning tables…")
    schema = build_schema()
    out = DATA_ROOT / "_schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(schema, f, indent=2, default=str)
    print(f"\nSchema written to {out}")


if __name__ == "__main__":
    main()
