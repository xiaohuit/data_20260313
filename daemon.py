"""
24/7 continuous data pipeline daemon.

Run:
    python daemon.py           # start the daemon (blocks forever)
    python daemon.py --reset   # wipe checkpoints and re-fetch from scratch
    python daemon.py --status  # print current checkpoint state and row counts
    python daemon.py --once    # run all jobs once right now and exit (useful for CI)

Schedule (all times US/Eastern):
  ┌──────────────────────────────────────────────────────────────────┐
  │  Job               │ When                   │ Why                │
  ├──────────────────────────────────────────────────────────────────┤
  │ ohlcv_daily        │ Mon-Fri 18:05          │ After market close │
  │ indicators         │ Mon-Fri 18:30          │ After OHLCV done   │
  │ earnings           │ Mon-Fri 18:35          │ After market close │
  │ insider_trades     │ Mon-Fri 20:00          │ SEC filing lag     │
  │ macro_fred         │ Daily   07:00          │ FRED publishes AM  │
  │ universe           │ Mon     08:00          │ Weekly refresh     │
  │ congress_trades    │ Sun     10:00          │ Weekly disclosure  │
  └──────────────────────────────────────────────────────────────────┘

Resilience:
  - Each job runs in its own try/except — one failure never kills the daemon.
  - APScheduler uses a thread-pool executor so slow jobs don't block others.
  - Checkpoint is only advanced AFTER a successful write to disk.
  - On restart the daemon resumes from the last checkpoint automatically.
  - Rotating log file at logs/daemon.log (10 MB × 5 files).
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

import storage
import state as state_mod
from run_crawler import (
    TICKERS,
    FRED_SERIES,
    fetch_ohlcv_one,
    compute_indicators_one,
    fetch_earnings_one,
    fetch_fred_series,
    fetch_insider_trades,
    fetch_sp500_universe,
    _market_close_utc,
)

# ── Logging — lazy setup called from main() only ──────────────────────────────
# Module-level basicConfig is a no-op when imported by other scripts (e.g.
# validate.py) that already configured the root logger.  We attach the file
# handler explicitly in _configure_logging() instead.

LOG_DIR = Path("./logs")
log = logging.getLogger("daemon")


def _configure_logging() -> None:
    """Wire up console + rotating-file handlers.  Safe to call more than once."""
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if this function is somehow called twice
    handler_paths = {
        getattr(h, "baseFilename", None) for h in root.handlers
    }
    log_path = str((LOG_DIR / "daemon.log").resolve())

    if log_path not in handler_paths:
        fh = logging.handlers.RotatingFileHandler(
            LOG_DIR / "daemon.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Console handler (stdout) — always add when called from main()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    # Only add if no StreamHandler already present
    if not any(isinstance(h, logging.StreamHandler) and
               getattr(h, 'stream', None) is sys.stdout
               for h in root.handlers):
        root.addHandler(ch)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

STATE = state_mod.StateStore()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_async(coro) -> object:
    """Run a coroutine from a sync APScheduler job."""
    return asyncio.run(coro)


def _since(job: str, key: str = "") -> datetime:
    """Return the start date for an incremental fetch."""
    return STATE.get_last_fetched(job, key) - timedelta(days=1)
    # -1 day overlap ensures we never miss a bar due to timezone edge cases


def _log_job(name: str, rows: int, status: str = "OK") -> None:
    log.info("%-22s  %-4s  +%d rows written", name, status, rows)


# ── Job: OHLCV daily bars ─────────────────────────────────────────────────────

def job_ohlcv() -> None:
    log.info("── JOB: ohlcv_daily ──────────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    for ticker in TICKERS:
        since = _since("ohlcv", ticker)
        try:
            df = fetch_ohlcv_one(ticker)   # always fetches full history, sliced below
            if df.empty:
                continue
            df["event_timestamp_dt"] = pd.to_datetime(df["event_timestamp"], utc=True)
            df = df[df["event_timestamp_dt"] >= since].drop(columns=["event_timestamp_dt"])
            n = storage.append("ohlcv", df)
            total += n
            if n:
                log.debug("  ohlcv %-6s +%d", ticker, n)
            STATE.set_last_fetched("ohlcv", ticker, now)
        except Exception as exc:
            log.error("  ohlcv %s FAILED: %s", ticker, exc)

    _log_job("ohlcv_daily", total)


# ── Job: Technical indicators ─────────────────────────────────────────────────

def job_indicators() -> None:
    log.info("── JOB: indicators ───────────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    for ticker in TICKERS:
        since = _since("indicators", ticker)
        try:
            # Read OHLCV for this ticker from storage (last 300 days for warmup)
            ohlcv_all = storage.read_all("ohlcv")
            if ohlcv_all.empty:
                break
            sub = ohlcv_all[ohlcv_all["ticker"] == ticker].copy()
            sub["event_ts"] = pd.to_datetime(sub["event_timestamp"], utc=True)
            # Need 200 bars of warmup before the slice for SMA200
            earliest_needed = since - timedelta(days=300)
            sub = sub[sub["event_ts"] >= earliest_needed].sort_values("event_ts")

            if len(sub) < 30:
                continue

            df = compute_indicators_one(ticker, sub)
            if df.empty:
                continue
            df["event_ts_dt"] = pd.to_datetime(df["event_timestamp"], utc=True)
            df = df[df["event_ts_dt"] >= since].drop(columns=["event_ts_dt"])
            n = storage.append("indicators", df)
            total += n
            STATE.set_last_fetched("indicators", ticker, now)
        except Exception as exc:
            log.error("  indicators %s FAILED: %s", ticker, exc)

    _log_job("indicators", total)


# ── Job: Macro / FRED ─────────────────────────────────────────────────────────

def job_macro() -> None:
    log.info("── JOB: macro_fred ───────────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    async def _fetch_all():
        tasks = [
            fetch_fred_series(code, sid)
            for code, sid in FRED_SERIES.items()
        ]
        return await asyncio.gather(*tasks)

    frames = _run_async(_fetch_all())
    for code, df in zip(FRED_SERIES.keys(), frames):
        if df.empty:
            continue
        since = _since("macro", code)
        df["event_ts_dt"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")
        df = df[df["event_ts_dt"] >= since].drop(columns=["event_ts_dt"])
        n = storage.append("macro", df)
        total += n
        STATE.set_last_fetched("macro", code, now)

    _log_job("macro_fred", total)


# ── Job: Insider trades ───────────────────────────────────────────────────────

def job_insider() -> None:
    log.info("── JOB: insider_trades ───────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    async def _fetch_all():
        frames = []
        for ticker in TICKERS[:10]:   # OpenInsider: be courteous
            df = await fetch_insider_trades(ticker)
            frames.append((ticker, df))
            await asyncio.sleep(2.0)
        return frames

    results = _run_async(_fetch_all())
    for ticker, df in results:
        if df.empty:
            continue
        n = storage.append("insider", df)
        total += n
        if n:
            log.debug("  insider %-6s +%d", ticker, n)
        STATE.set_last_fetched("insider", ticker, now)

    _log_job("insider_trades", total)


# ── Job: Earnings ─────────────────────────────────────────────────────────────

def job_earnings() -> None:
    log.info("── JOB: earnings ─────────────────────────────────────")
    total = 0
    now = datetime.now(UTC)
    loop = asyncio.new_event_loop()

    for ticker in TICKERS:
        try:
            df = fetch_earnings_one(ticker)
            if df.empty:
                continue
            # Standardise date column to "period_end" for PK
            # yfinance uses "quarter" as the index name after reset_index()
            col_map = {}
            for c in df.columns:
                if c.lower() in ("quarter", "period", "date", "index") or \
                   "period" in c.lower() or "quarter" in c.lower():
                    col_map[c] = "period_end"
                    break
            if col_map:
                df = df.rename(columns=col_map)
            n = storage.append("earnings", df)
            total += n
            STATE.set_last_fetched("earnings", ticker, now)
        except Exception as exc:
            log.error("  earnings %s FAILED: %s", ticker, exc)

    loop.close()
    _log_job("earnings", total)


# ── Job: Universe refresh ─────────────────────────────────────────────────────

def job_universe() -> None:
    log.info("── JOB: universe ─────────────────────────────────────")
    try:
        df = _run_async(fetch_sp500_universe())
        n = storage.append("universe", df)
        STATE.set_last_fetched("universe")
        _log_job("universe", n)
    except Exception as exc:
        log.error("  universe FAILED: %s", exc)


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def job_heartbeat() -> None:
    counts = storage.row_counts()
    parts = "  ".join(f"{k}={v:,}" for k, v in counts.items())
    log.info("♥  HEARTBEAT  %s", parts)


# ── Status report ─────────────────────────────────────────────────────────────

def print_status() -> None:
    print("\n── Row counts ───────────────────────────────────────────")
    for k, v in storage.row_counts().items():
        print(f"  {k:<15} {v:>10,} rows")

    print("\n── Checkpoints ──────────────────────────────────────────")
    all_state = STATE.get_all()
    for job, keys in all_state.items():
        for key, ts in (keys.items() if isinstance(keys, dict) else [("_", keys)]):
            print(f"  {job}/{key:<20}  last={ts}")
    print()


# ── Migrate flat files → partitioned Parquet (one-time) ──────────────────────

def migrate_flat_files() -> None:
    """
    If the old flat CSV/Parquet files exist from the initial run,
    migrate them into the new partitioned layout and remove the flat files.
    This runs once on first daemon start.
    """
    migrations = [
        (Path("data/ohlcv/ohlcv_daily.parquet"),   "ohlcv"),
        (Path("data/indicators/technical_indicators.parquet"), "indicators"),
        (Path("data/macro/macro_series.parquet"),   "macro"),
        (Path("data/insider/insider_trades.parquet"), "insider"),
        (Path("data/earnings/earnings_history.parquet"), "earnings"),
    ]
    for flat_path, table in migrations:
        if not flat_path.exists():
            continue
        log.info("Migrating %s → partitioned/%s …", flat_path, table)
        try:
            df = pd.read_parquet(flat_path)
            # Rename columns to match current schema
            if table == "earnings" and "period_end" not in df.columns:
                for c in df.columns:
                    if c.lower() in ("quarter", "period", "date", "index") or \
                       "period" in c.lower() or "quarter" in c.lower():
                        df = df.rename(columns={c: "period_end"})
                        break
            n = storage.append(table, df)
            log.info("  → %d rows migrated", n)
            # Rename flat file so it's not re-processed (keep as backup)
            flat_path.rename(flat_path.with_suffix(".parquet.migrated"))
        except Exception as exc:
            log.error("  Migration failed for %s: %s", flat_path, exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    executors = {"default": ThreadPoolExecutor(max_workers=4)}
    scheduler = BlockingScheduler(executors=executors, timezone=ET)

    # ── Daily equity jobs (weekdays only) ────────────────────────────────────
    scheduler.add_job(job_ohlcv,      "cron", day_of_week="mon-fri",
                      hour=18, minute=5,  id="ohlcv_daily",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_indicators, "cron", day_of_week="mon-fri",
                      hour=18, minute=30, id="indicators",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_earnings,   "cron", day_of_week="mon-fri",
                      hour=18, minute=35, id="earnings",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_insider,    "cron", day_of_week="mon-fri",
                      hour=20, minute=0,  id="insider_trades",
                      max_instances=1, misfire_grace_time=3600)

    # ── Macro: every day at 7 AM (FRED publishes morning releases) ───────────
    scheduler.add_job(job_macro,      "cron",
                      hour=7, minute=0,  id="macro_fred",
                      max_instances=1, misfire_grace_time=3600)

    # ── Universe: every Monday morning ───────────────────────────────────────
    scheduler.add_job(job_universe,   "cron", day_of_week="mon",
                      hour=8, minute=0,  id="universe",
                      max_instances=1, misfire_grace_time=7200)

    # ── Heartbeat: every hour ────────────────────────────────────────────────
    scheduler.add_job(job_heartbeat,  "interval", hours=1, id="heartbeat")

    return scheduler


def run_all_once() -> None:
    """Run every job immediately in sequence (for --once mode)."""
    log.info("Running all jobs once …")
    for fn in (job_universe, job_ohlcv, job_indicators,
               job_macro, job_earnings, job_insider):
        try:
            fn()
        except Exception as exc:
            log.error("%s crashed: %s", fn.__name__, exc)
    print_status()


def main() -> None:
    _configure_logging()
    args = sys.argv[1:]

    if "--status" in args:
        print_status()
        return

    if "--reset" in args:
        STATE.reset()
        log.info("All checkpoints cleared.")
        return

    # Migrate old flat files on first start
    migrate_flat_files()

    if "--once" in args:
        run_all_once()
        return

    # ── Daemon mode ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Financial Data Pipeline Daemon starting …")
    log.info("  PID: %d  |  Logs: %s", Path("/proc/self").exists() and
             int(open("/proc/self/stat").read().split()[0]) if Path("/proc/self/stat").exists()
             else 0, LOG_DIR / "daemon.log")
    log.info("=" * 60)

    # Run all jobs immediately on startup to catch up on any missed data
    log.info("Initial catch-up run …")
    for fn in (job_universe, job_ohlcv, job_indicators,
               job_macro, job_earnings, job_insider):
        try:
            fn()
        except Exception as exc:
            log.error("Startup job %s failed: %s", fn.__name__, exc)

    # Print initial status
    print_status()

    # Build and start the scheduler
    scheduler = build_scheduler()

    def _shutdown(sig, frame):
        log.info("Signal %s received — shutting down gracefully …", sig)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Scheduler started. Registered jobs:")
    for job in scheduler.get_jobs():
        log.info("  %-22s  trigger=%s", job.id, job.trigger)

    scheduler.start()   # blocks here — next_run_time is set after this returns


if __name__ == "__main__":
    main()
