"""
Prefect 2.x pipeline orchestration.

Flows:
  1. `bootstrap_flow`  — one-time historical backfill (run once).
  2. `daily_flow`      — incremental daily update (scheduled at market close).
  3. `weekly_flow`     — deeper weekly enrichment (13F, Congress, macro).

Each flow is composed of independent tasks that can be parallelized.
Failed tasks are retried without re-running successful ones (Prefect handles
this via task caching).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from financial_pipeline.config import CONFIG
from financial_pipeline.crawlers.alternative import (
    CongressTradeCrawler,
    InsiderTradeCrawler,
    Portfolio13FCrawler,
)
from financial_pipeline.crawlers.fundamental import EarningsCrawler, SECEdgarCrawler
from financial_pipeline.crawlers.macro import FOMCCrawler, FREDCrawler
from financial_pipeline.crawlers.market import (
    OHLCVCrawler,
    OptionsChainCrawler,
    TechnicalIndicatorCrawler,
)
from financial_pipeline.crawlers.universe import UniverseCrawler
from financial_pipeline.db.migrations import bootstrap as db_bootstrap
from financial_pipeline.loader.pit_loader import PiTDataLoader

log = logging.getLogger(__name__)

UTC = timezone.utc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _years_ago(n: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=n * 365)


async def _get_universe() -> list[str]:
    """Return the current combined S&P500 + NDX100 universe."""
    loader = PiTDataLoader()
    async with loader._sf() as session:
        return await loader._universe_crawler.get_combined_universe_at(
            session, datetime.now(UTC)
        )


# ── Prefect task wrappers ─────────────────────────────────────────────────────
# If Prefect is not installed, these fall back to plain async functions.

try:
    from prefect import flow, task
    from prefect.tasks import task_input_hash
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Minimal stubs so the rest of the file is importable without Prefect
    def task(fn=None, **_):
        return fn if fn else (lambda f: f)
    def flow(fn=None, **_):
        return fn if fn else (lambda f: f)
    log.warning("prefect not installed — flows will run as plain coroutines")


# ── Individual tasks ──────────────────────────────────────────────────────────

@task(name="refresh_universe", retries=3, retry_delay_seconds=60)
async def task_refresh_universe() -> dict:
    crawler = UniverseCrawler()
    return await crawler.refresh()


@task(name="backfill_ohlcv", retries=3, retry_delay_seconds=120)
async def task_backfill_ohlcv(
    tickers: list[str],
    start: datetime,
    end: datetime,
    frequency: str = "1D",
) -> None:
    crawler = OHLCVCrawler(frequency=frequency)
    await crawler.backfill(tickers, start, end)


@task(name="backfill_indicators", retries=2)
async def task_backfill_indicators(
    tickers: list[str], start: datetime, end: datetime
) -> None:
    crawler = TechnicalIndicatorCrawler()
    await crawler.backfill(tickers, start, end)


@task(name="backfill_options", retries=2)
async def task_backfill_options(tickers: list[str]) -> None:
    crawler = OptionsChainCrawler()
    await crawler.ingest_live(tickers)


@task(name="backfill_sec", retries=3, retry_delay_seconds=60)
async def task_backfill_sec(
    tickers: list[str], start: datetime, end: datetime
) -> None:
    crawler = SECEdgarCrawler()
    await crawler.backfill(tickers, start, end)


@task(name="backfill_earnings", retries=2)
async def task_backfill_earnings(tickers: list[str]) -> None:
    crawler = EarningsCrawler()
    await crawler.backfill(tickers)


@task(name="backfill_fred", retries=3, retry_delay_seconds=120)
async def task_backfill_fred(start: datetime, end: datetime) -> None:
    crawler = FREDCrawler()
    await crawler.backfill_all(start, end)


@task(name="backfill_fomc", retries=3)
async def task_backfill_fomc(start_year: int = 2018) -> None:
    crawler = FOMCCrawler()
    await crawler.backfill(start_year)


@task(name="backfill_insider", retries=2, retry_delay_seconds=60)
async def task_backfill_insider(tickers: list[str]) -> None:
    crawler = InsiderTradeCrawler()
    await crawler.backfill(tickers)


@task(name="backfill_13f", retries=3, retry_delay_seconds=120)
async def task_backfill_13f(start: datetime, end: datetime) -> None:
    crawler = Portfolio13FCrawler()
    await crawler.backfill_all(start, end)


@task(name="backfill_congress", retries=2, retry_delay_seconds=60)
async def task_backfill_congress(start: datetime, end: datetime) -> None:
    crawler = CongressTradeCrawler()
    await crawler.backfill(start, end)


# ── Bootstrap flow (run once) ─────────────────────────────────────────────────

@flow(name="bootstrap", log_prints=True)
async def bootstrap_flow(history_years: int = 6) -> None:
    """
    One-time historical backfill. Run this once to seed the database.
    Approximate runtime: 6–24 hours depending on universe size and rate limits.

    Steps (in order of dependency):
      1. DB schema + TimescaleDB setup
      2. Universe (needed before tickers known)
      3. All historical data in parallel batches
    """
    log.info("=== BOOTSTRAP START (history=%d years) ===", history_years)
    end = datetime.now(UTC)
    start = _years_ago(history_years)

    # 1. Database setup
    await db_bootstrap()

    # 2. Universe (must come first — determines ticker list)
    await task_refresh_universe()
    tickers = await _get_universe()
    log.info("Universe: %d tickers", len(tickers))

    # 3. Split into batches to respect rate limits
    batch_size = 50
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    # 4. OHLCV (daily) — foundation for everything else
    for batch in ticker_batches:
        await task_backfill_ohlcv(batch, start, end, "1D")

    # 5. Parallel: technical indicators + fundamentals + macro
    await asyncio.gather(
        # Indicators (computed from OHLCV, so runs after)
        *[task_backfill_indicators(batch, start, end) for batch in ticker_batches],
        # Macro is ticker-independent
        task_backfill_fred(start, end),
        task_backfill_fomc(start_year=end.year - history_years),
    )

    # 6. Fundamentals (SEC has rate limits — sequential batches)
    for batch in ticker_batches:
        await task_backfill_sec(batch, start, end)
        await asyncio.sleep(2)   # extra courtesy delay for SEC

    await asyncio.gather(
        *[task_backfill_earnings(batch) for batch in ticker_batches],
    )

    # 7. Alternative data
    await asyncio.gather(
        *[task_backfill_insider(batch) for batch in ticker_batches],
        task_backfill_congress(start, end),
        task_backfill_13f(start, end),
    )

    log.info("=== BOOTSTRAP COMPLETE ===")


# ── Daily incremental flow ────────────────────────────────────────────────────

@flow(name="daily_update", log_prints=True)
async def daily_flow() -> None:
    """
    Scheduled daily at 18:00 ET (after market close + 2h buffer).
    Updates: OHLCV, options snapshot, earnings, insider filings.
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=5)   # small overlap for reliability

    await task_refresh_universe()
    tickers = await _get_universe()
    batch_size = 50
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    await asyncio.gather(
        *[task_backfill_ohlcv(batch, start, end, "1D") for batch in batches],
    )
    await asyncio.gather(
        *[task_backfill_indicators(batch, start, end) for batch in batches],
        *[task_backfill_earnings(batch) for batch in batches],
        *[task_backfill_insider(batch) for batch in batches],
        task_backfill_fred(start, end),
    )
    # Options snapshot (live only)
    for batch in batches:
        await task_backfill_options(batch)

    log.info("Daily update complete.")


# ── Weekly enrichment flow ────────────────────────────────────────────────────

@flow(name="weekly_enrichment", log_prints=True)
async def weekly_flow() -> None:
    """
    Scheduled Sundays. Pulls slower-moving data: 13F, Congress, SEC filings.
    """
    end = datetime.now(UTC)
    start = end - timedelta(days=100)   # ~1 quarter overlap

    tickers = await _get_universe()
    batch_size = 50
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    await asyncio.gather(
        task_backfill_13f(start, end),
        task_backfill_congress(start, end),
        task_backfill_fomc(start_year=end.year - 2),
    )
    for batch in batches:
        await task_backfill_sec(batch, start, end)
        await asyncio.sleep(1)

    log.info("Weekly enrichment complete.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if cmd == "bootstrap":
        asyncio.run(bootstrap_flow(history_years=6))
    elif cmd == "weekly":
        asyncio.run(weekly_flow())
    else:
        asyncio.run(daily_flow())
