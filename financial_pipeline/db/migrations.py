"""
DDL bootstrap + TimescaleDB hypertable setup.

Run once against a fresh TimescaleDB instance:
    python -m financial_pipeline.db.migrations

Idempotent — safe to re-run.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from financial_pipeline.config import CONFIG
from financial_pipeline.db.models import Base

log = logging.getLogger(__name__)

# Hypertable definitions: (table_name, time_column, chunk_interval)
HYPERTABLES = [
    ("market_ohlcv",         "event_timestamp", "7 days"),
    ("technical_indicators", "event_timestamp", "7 days"),
    ("options_chain",        "event_timestamp", "7 days"),
    ("macro_indicators",     "event_timestamp", "30 days"),
    ("financial_events",     "event_timestamp", "30 days"),
]

# Continuous aggregates for common resampling targets
# (TimescaleDB materialises these incrementally)
CONTINUOUS_AGGREGATES = [
    # Weekly OHLCV
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS market_ohlcv_weekly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 week', event_timestamp)  AS bucket,
        ticker,
        first(open,  event_timestamp)           AS open,
        max(high)                               AS high,
        min(low)                                AS low,
        last(close,  event_timestamp)           AS close,
        last(adj_close, event_timestamp)        AS adj_close,
        sum(volume)                             AS volume,
        last(knowledge_timestamp, event_timestamp) AS knowledge_timestamp
    FROM market_ohlcv
    WHERE frequency = '1D'
    GROUP BY bucket, ticker
    WITH NO DATA;
    """,
    # Monthly OHLCV
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS market_ohlcv_monthly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 month', event_timestamp) AS bucket,
        ticker,
        first(open,  event_timestamp)           AS open,
        max(high)                               AS high,
        min(low)                                AS low,
        last(close,  event_timestamp)           AS close,
        last(adj_close, event_timestamp)        AS adj_close,
        sum(volume)                             AS volume,
        last(knowledge_timestamp, event_timestamp) AS knowledge_timestamp
    FROM market_ohlcv
    WHERE frequency = '1D'
    GROUP BY bucket, ticker
    WITH NO DATA;
    """,
]

RETENTION_POLICIES = [
    # Keep raw intraday data for 2 years in hot storage; older data → Parquet
    ("market_ohlcv",  "730 days"),
]


async def bootstrap() -> None:
    engine = create_async_engine(CONFIG.db.url, echo=CONFIG.db.echo)

    async with engine.begin() as conn:
        # 1. Ensure TimescaleDB extension is available
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        log.info("timescaledb extension confirmed")

        # 2. Create all tables from ORM metadata
        await conn.run_sync(Base.metadata.create_all)
        log.info("ORM tables created (idempotent)")

        # 3. Convert to hypertables (idempotent via create_if_not_exists flag)
        for table, col, chunk in HYPERTABLES:
            await conn.execute(text(f"""
                SELECT create_hypertable(
                    '{table}', '{col}',
                    chunk_time_interval => INTERVAL '{chunk}',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """))
            log.info("hypertable ready: %s", table)

        # 4. Continuous aggregates
        for ddl in CONTINUOUS_AGGREGATES:
            try:
                await conn.execute(text(ddl.strip()))
            except Exception as exc:
                # View already exists
                log.debug("Continuous aggregate already exists: %s", exc)

        # 5. Retention policies
        for table, interval in RETENTION_POLICIES:
            await conn.execute(text(f"""
                SELECT add_retention_policy(
                    '{table}',
                    INTERVAL '{interval}',
                    if_not_exists => TRUE
                );
            """))

    await engine.dispose()
    log.info("Bootstrap complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(bootstrap())
