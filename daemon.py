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
  │ valuations         │ Mon-Fri 18:45          │ After indicators   │
  │ implied_q4_eps     │ Mon-Fri 18:40          │ After earnings     │
  │ dividends          │ Mon-Fri 19:00          │ After earnings     │
  │ events_8k          │ Mon-Fri 20:30          │ After insider      │
  │ insider_trades     │ Mon-Fri 20:00          │ SEC filing lag     │
  │ macro_fred         │ Daily   07:00          │ FRED publishes AM  │
  │ short_interest     │ Mon     09:00          │ Weekly snapshot    │
  │ universe_history   │ Mon     08:30          │ Weekly changes     │
  │ financials         │ Sat     06:00          │ Weekly, low urgency│
  │ delisted_backfill  │ Sat     07:30          │ After financials   │
  │ quality_metrics    │ Sat     08:00          │ After financials   │
  │ universe           │ Mon     08:00          │ Weekly refresh     │
  │ sectors            │ Mon     08:45          │ After universe     │
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
    TICKERS,           # minimal bootstrap list (fallback only)
    FRED_SERIES,
    fetch_ohlcv_one,
    compute_indicators_one,
    compute_valuations_one,
    compute_dividends_one,
    fetch_earnings_one,
    fetch_8k_one,
    fetch_financials_edgar,
    fetch_fred_series,
    fetch_insider_trades,
    fetch_universe_changes,
    fetch_short_interest_one,
    fetch_full_universe,
    fetch_sectors,
    compute_quality_metrics_one,
    compute_implied_q4_eps_one,
    fetch_financials_yfinance,
    NON_EDGAR_TICKERS,
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


def _get_tickers() -> list[str]:
    """
    Return the current ticker universe from the stored universe table.
    This reflects the live S&P 500 + NASDAQ 100 composition as of the last
    universe job run (every Monday 08:00).
    Falls back to the minimal bootstrap TICKERS list on first ever startup
    before any universe fetch has succeeded.
    """
    df = storage.read_all("universe")
    if not df.empty and "ticker" in df.columns:
        tickers = df["ticker"].dropna().unique().tolist()
        log.debug("_get_tickers: %d tickers from stored universe", len(tickers))
        return tickers
    log.warning("_get_tickers: universe table empty — using bootstrap list (%d tickers)", len(TICKERS))
    return list(TICKERS)


# ── Job: OHLCV daily bars ─────────────────────────────────────────────────────

def job_ohlcv() -> None:
    log.info("── JOB: ohlcv_daily ──────────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    tickers = _get_tickers()
    log.info("  Processing %d tickers", len(tickers))
    for ticker in tickers:
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

    tickers = _get_tickers()
    log.info("  Processing %d tickers", len(tickers))
    ohlcv_all = storage.read_all("ohlcv")   # read once, share across all tickers
    for ticker in tickers:
        since = _since("indicators", ticker)
        try:
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

    # ── Derive CPI_YOY (true 12-month %) from the full stored CPI_INDEX ──────
    # Recompute and fully replace every run so the series is always correct.
    cpi_all = storage.read_all("macro")
    cpi_all = cpi_all[cpi_all["indicator_code"] == "CPI_INDEX"].copy()
    if not cpi_all.empty:
        cpi_all["event_ts"] = pd.to_datetime(cpi_all["event_timestamp"], format="ISO8601", errors="coerce")
        cpi_all = (cpi_all.sort_values("event_ts")
                          .drop_duplicates("event_ts", keep="last")
                          .reset_index(drop=True))
        cpi_all["value_12m"] = cpi_all["value"].shift(12)
        cpi_yoy_df = cpi_all.dropna(subset=["value_12m"]).copy()
        cpi_yoy_df["value"] = ((cpi_yoy_df["value"] - cpi_yoy_df["value_12m"])
                               / cpi_yoy_df["value_12m"] * 100).round(4)
        cpi_yoy_df["indicator_code"] = "CPI_YOY"
        cpi_yoy_df["series_id"]      = "CPIAUCSL_YOY"
        today_date = pd.Timestamp.today().normalize().date()
        cpi_yoy_df["knowledge_timestamp"] = cpi_yoy_df["event_ts"].apply(
            lambda t: str(min((t + pd.Timedelta(days=45)).date(), today_date))
        )
        cpi_yoy_df = cpi_yoy_df.drop(columns=["event_ts", "value_12m"])
        # Always replace CPI_YOY partition to prevent stale rows accumulating
        import shutil as _shutil
        cpi_yoy_dir = storage.DATA_ROOT / "macro" / "indicator_code=CPI_YOY"
        if cpi_yoy_dir.exists():
            _shutil.rmtree(cpi_yoy_dir)
        n_yoy = storage.append("macro", cpi_yoy_df)
        total += n_yoy
        log.info("  CPI_YOY derived: %d rows (12-month YoY %%)", n_yoy)

    _log_job("macro_fred", total)


# ── Job: Insider trades ───────────────────────────────────────────────────────

def job_insider() -> None:
    log.info("── JOB: insider_trades ───────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    async def _fetch_all():
        frames = []
        # OpenInsider: limit to 50 tickers per run, rotate through the universe
        # daily so the full list is covered over the week
        import datetime as _dt
        day_of_year = _dt.date.today().timetuple().tm_yday
        all_tickers = _get_tickers()
        batch_size = 50
        start_idx = (day_of_year * batch_size) % max(len(all_tickers), 1)
        batch = (all_tickers + all_tickers)[start_idx:start_idx + batch_size]
        log.info("  Insider batch: tickers %d-%d of %d",
                 start_idx, start_idx + len(batch), len(all_tickers))
        for ticker in batch:   # OpenInsider: be courteous
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

    tickers = _get_tickers()
    log.info("  Processing %d tickers (EDGAR + yfinance)", len(tickers))
    for ticker in tickers:
        try:
            df = fetch_earnings_one(ticker)
            if df.empty:
                continue
            n = storage.append("earnings", df)
            total += n
            STATE.set_last_fetched("earnings", ticker, now)
        except Exception as exc:
            log.error("  earnings %s FAILED: %s", ticker, exc)

    _log_job("earnings", total)


# ── Job: Implied Q4 EPS ───────────────────────────────────────────────────────

def job_implied_q4_eps() -> None:
    """
    Derive implied Q4 EPS = annual_EPS − (Q1 + Q2 + Q3) for every ticker
    and store as synthetic 10-Q rows so the TTM P/E rolling window is complete.

    Companies don't file a 10-Q for fiscal Q4 — only a 10-K.  Without this
    fix, the rolling-4-quarter TTM P/E systematically overstates P/E by 10-15%
    during the gap between the annual filing and the next Q1 filing.

    Runs Mon-Fri 18:40 — right after job_earnings completes.
    """
    log.info("── JOB: implied_q4_eps ───────────────────────────────")
    total = 0
    now   = datetime.now(UTC)

    earn_all = storage.read_all("earnings")
    if earn_all.empty:
        log.warning("  implied_q4_eps: earnings table empty — skipping")
        return

    tickers = list(set(_get_tickers()) |
                   set(earn_all["ticker"].unique() if not earn_all.empty else []))
    log.info("  Computing implied Q4 for %d tickers", len(tickers))

    for ticker in tickers:
        try:
            df = compute_implied_q4_eps_one(
                ticker,
                earn_all[earn_all["ticker"] == ticker],
            )
            if df.empty:
                continue
            n = storage.append("earnings", df)
            total += n
            if n:
                log.debug("  implied_q4 %-6s +%d", ticker, n)
        except Exception as exc:
            log.error("  implied_q4 %s FAILED: %s", ticker, exc)

    _log_job("implied_q4_eps", total)


# ── Job: Valuation ratios ─────────────────────────────────────────────────────

def job_valuations() -> None:
    """
    Compute daily P/E, P/B, EV/EBITDA, FCF yield, P/S, dividend yield
    for every ticker.  Pure in-memory computation — no external API calls.
    Incremental: only re-derives days since last checkpoint.
    """
    log.info("── JOB: valuations ───────────────────────────────────")
    total = 0
    now   = datetime.now(UTC)

    # Load all source tables once (shared across tickers)
    ohlcv_all = storage.read_all("ohlcv")
    earn_all  = storage.read_all("earnings")
    fin_all   = storage.read_all("financials")

    if ohlcv_all.empty:
        log.warning("  valuations: ohlcv table empty — skipping")
        return

    tickers = _get_tickers()
    log.info("  Computing valuations for %d tickers", len(tickers))

    for ticker in tickers:
        since = _since("valuations", ticker)
        try:
            df = compute_valuations_one(
                ticker,
                ohlcv_all[ohlcv_all["ticker"] == ticker],
                earn_all[earn_all["ticker"] == ticker],
                fin_all[fin_all["ticker"] == ticker],
            )
            if df.empty:
                continue
            df["event_ts_dt"] = pd.to_datetime(df["event_timestamp"], utc=True)
            df = df[df["event_ts_dt"] >= since].drop(columns=["event_ts_dt"])
            n = storage.append("valuations", df)
            total += n
            STATE.set_last_fetched("valuations", ticker, now)
        except Exception as exc:
            log.error("  valuations %s FAILED: %s", ticker, exc)

    _log_job("valuations", total)


# ── Job: 8-K material event filings ──────────────────────────────────────────

def job_events_8k() -> None:
    """
    Fetch 8-K / 8-K/A filing metadata from SEC EDGAR submissions API.
    Incremental: each ticker is checkpointed; only newly filed 8-Ks are stored.
    Runs Mon-Fri 20:30 — after market-hours filings have had time to land.
    """
    log.info("── JOB: events_8k ────────────────────────────────────")
    total = 0
    now   = datetime.now(UTC)

    tickers = _get_tickers()
    log.info("  Fetching 8-K filings for %d tickers (SEC EDGAR)", len(tickers))

    for ticker in tickers:
        since = _since("events_8k", ticker)
        try:
            df = fetch_8k_one(ticker)
            if df.empty:
                continue
            # Filter to only new filings since last checkpoint
            df["filing_ts"] = pd.to_datetime(df["event_timestamp"], utc=True)
            df = df[df["filing_ts"] >= since].drop(columns=["filing_ts"])
            n = storage.append("events_8k", df)
            total += n
            if n:
                log.debug("  events_8k %-6s +%d", ticker, n)
            STATE.set_last_fetched("events_8k", ticker, now)
            import time as _time; _time.sleep(0.15)   # SEC rate limit
        except Exception as exc:
            log.error("  events_8k %s FAILED: %s", ticker, exc)

    _log_job("events_8k", total)


# ── Job: Fundamental financials ───────────────────────────────────────────────

def job_financials() -> None:
    log.info("── JOB: financials ───────────────────────────────────")
    total = 0
    now = datetime.now(UTC)

    tickers = _get_tickers()
    log.info("  Processing %d tickers (SEC EDGAR XBRL)", len(tickers))
    for ticker in tickers:
        try:
            # Primary: SEC EDGAR XBRL for US filers
            if ticker not in NON_EDGAR_TICKERS:
                df = fetch_financials_edgar(ticker)
            else:
                df = pd.DataFrame()

            # Fallback: yfinance for non-EDGAR filers (foreign companies)
            # Also use as supplement if EDGAR returned nothing
            if df.empty:
                df = fetch_financials_yfinance(ticker)
                if not df.empty:
                    log.debug("  financials %-6s using yfinance fallback", ticker)

            if df.empty:
                continue
            n = storage.append("financials", df)
            total += n
            if n:
                log.debug("  financials %-6s +%d", ticker, n)
            STATE.set_last_fetched("financials", ticker, now)
        except Exception as exc:
            log.error("  financials %s FAILED: %s", ticker, exc)

    _log_job("financials", total)


# ── Job: Dividend history ─────────────────────────────────────────────────────

def job_dividends() -> None:
    """
    Compute annual dividend history (DPS, growth rates, payout ratio, streak)
    for every ticker.  Pure in-memory computation — no external API calls.
    Incremental: only re-derives tickers whose checkpoint is stale.
    """
    log.info("── JOB: dividends ────────────────────────────────────")
    total = 0
    now   = datetime.now(UTC)

    ohlcv_all = storage.read_all("ohlcv")
    earn_all  = storage.read_all("earnings")

    if ohlcv_all.empty:
        log.warning("  dividends: ohlcv table empty — skipping")
        return

    tickers = _get_tickers()
    log.info("  Computing dividend history for %d tickers", len(tickers))

    for ticker in tickers:
        try:
            df = compute_dividends_one(
                ticker,
                ohlcv_all[ohlcv_all["ticker"] == ticker],
                earn_all[earn_all["ticker"] == ticker],
            )
            if df.empty:
                continue
            n = storage.append("dividends", df)
            total += n
            if n:
                log.debug("  dividends %-6s +%d", ticker, n)
            STATE.set_last_fetched("dividends", ticker, now)
        except Exception as exc:
            log.error("  dividends %s FAILED: %s", ticker, exc)

    _log_job("dividends", total)


# ── Job: Universe refresh ─────────────────────────────────────────────────────

def job_universe() -> None:
    log.info("── JOB: universe ─────────────────────────────────────")
    try:
        df = _run_async(fetch_full_universe())
        if df.empty:
            log.warning("  universe: no data returned")
            return
        n = storage.append("universe", df)
        tickers = df["ticker"].nunique()
        log.info("  universe: %d unique tickers stored", tickers)
        STATE.set_last_fetched("universe")
        _log_job("universe", n)
    except Exception as exc:
        log.error("  universe FAILED: %s", exc)


# ── Job: Universe membership history ─────────────────────────────────────────

def job_universe_history() -> None:
    """
    Fetch S&P 500 and NASDAQ 100 membership changes from Wikipedia.
    Stores every addition and removal event with date and reason.
    Runs weekly (Monday 08:30) just after the universe refresh.
    """
    log.info("── JOB: universe_history ─────────────────────────────")
    try:
        df = fetch_universe_changes()
        if df.empty:
            log.warning("  universe_history: no data returned")
            return
        n = storage.append("universe_history", df)
        added   = (df["action"] == "added").sum()
        removed = (df["action"] == "removed").sum()
        log.info("  universe_history: %d added events, %d removed events stored",
                 added, removed)
        STATE.set_last_fetched("universe_history")
        _log_job("universe_history", n)
    except Exception as exc:
        log.error("  universe_history FAILED: %s", exc)


# ── Job: Short interest snapshots ─────────────────────────────────────────────

def job_short_interest() -> None:
    """
    Fetch current + prior-month short interest for every ticker via yfinance.
    Running weekly builds a time series going forward.

    Note: FINRA consolidated CDN returns HTTP 403; yfinance is the free
    substitute providing settlement-date short interest and prior month data.
    """
    log.info("── JOB: short_interest ───────────────────────────────")
    total = 0
    now   = datetime.now(UTC)

    tickers = _get_tickers()
    log.info("  Fetching short interest for %d tickers", len(tickers))

    for ticker in tickers:
        try:
            df = fetch_short_interest_one(ticker)
            if df.empty:
                continue
            n = storage.append("short_interest", df)
            total += n
            STATE.set_last_fetched("short_interest", ticker, now)
            time.sleep(0.25)
        except Exception as exc:
            log.error("  short_interest %s FAILED: %s", ticker, exc)

    _log_job("short_interest", total)


# ── Job: Delisted ticker data backfill ────────────────────────────────────────

def job_delisted_backfill() -> None:
    """
    Fetch OHLCV, earnings, and financials for tickers that were historically
    in the S&P 500 or NASDAQ 100 but have since been removed (acquired, merged,
    delisted, or dropped for underperformance).

    Adds negative/different examples to training data, partially correcting the
    survivorship bias inherent in using only the current index membership.

    Incremental: state checkpoint tracks which tickers have been processed.
    Processes up to 50 new tickers per run (runs Saturdays 07:30).
    Data is stored in the existing ohlcv/earnings/financials tables — same schema.
    """
    log.info("── JOB: delisted_backfill ────────────────────────────")

    hist = storage.read_all("universe_history")
    if hist.empty:
        log.info("  universe_history empty — run job_universe_history first")
        return

    removed_tickers  = set(hist[hist["action"] == "removed"]["ticker"].unique())
    current_tickers  = set(_get_tickers())
    epoch = datetime(2010, 1, 1, tzinfo=UTC)

    # Only process tickers that are removed AND not in the current live universe
    # AND not already checkpointed (i.e. not yet fetched)
    candidates = sorted(
        t for t in removed_tickers - current_tickers
        if STATE.get_last_fetched("delisted_backfill", t) <= epoch
    )

    if not candidates:
        log.info("  All %d removed tickers already backfilled", len(removed_tickers - current_tickers))
        return

    batch = candidates[:50]
    log.info("  Backfilling %d delisted tickers (%d remaining after this run)",
             len(batch), len(candidates) - len(batch))

    total_ohlcv = total_earn = total_fin = 0
    for ticker in batch:
        try:
            df_ohlcv = fetch_ohlcv_one(ticker)
            if not df_ohlcv.empty:
                total_ohlcv += storage.append("ohlcv", df_ohlcv)

            df_earn = fetch_earnings_one(ticker)
            if not df_earn.empty:
                total_earn += storage.append("earnings", df_earn)

            df_fin = fetch_financials_edgar(ticker)
            if not df_fin.empty:
                total_fin += storage.append("financials", df_fin)

            STATE.set_last_fetched("delisted_backfill", ticker)
            log.debug("  delisted %-6s  ohlcv=%d earn=%d fin=%d",
                      ticker, total_ohlcv, total_earn, total_fin)
            time.sleep(0.5)   # be courteous to yfinance + EDGAR
        except Exception as exc:
            log.error("  delisted_backfill %s FAILED: %s", ticker, exc)

    log.info("  delisted_backfill done: +%d ohlcv  +%d earnings  +%d financials",
             total_ohlcv, total_earn, total_fin)


# ── Job: Sector / industry classification ─────────────────────────────────────

def job_sectors() -> None:
    """
    Fetch sector, industry, and market-cap category for all tickers via yfinance.
    Stores a single flat file (sectors_latest.parquet) — refreshed weekly.
    Sectors rarely change, so weekly is more than sufficient.
    """
    log.info("── JOB: sectors ──────────────────────────────────────")
    try:
        tickers = _get_tickers()
        log.info("  Fetching sector classification for %d tickers", len(tickers))
        df = fetch_sectors(tickers)
        if df.empty:
            log.warning("  sectors: no data returned")
            return
        n = storage.append("sectors", df)
        log.info("  sectors: %d tickers stored", df["ticker"].nunique())
        STATE.set_last_fetched("sectors")
        _log_job("sectors", n)
    except Exception as exc:
        log.error("  sectors FAILED: %s", exc)


# ── Job: Quality metrics ───────────────────────────────────────────────────────

def job_quality_metrics() -> None:
    """
    Compute fundamental quality metrics (ROIC, FCF yield, revenue CAGR, etc.)
    for every ticker from existing financials + OHLCV data.
    Pure in-memory computation — no external API calls.
    Runs Saturday 08:00, after financials and delisted_backfill.
    """
    log.info("── JOB: quality_metrics ──────────────────────────────")
    total = 0
    now   = datetime.now(UTC)

    fin_all   = storage.read_all("financials")
    ohlcv_all = storage.read_all("ohlcv")

    if fin_all.empty:
        log.warning("  quality_metrics: financials table empty — skipping")
        return

    tickers = _get_tickers()
    # Also include delisted tickers that have financials
    if not fin_all.empty:
        all_fin_tickers = set(fin_all["ticker"].unique())
        tickers = list(set(tickers) | all_fin_tickers)
    log.info("  Computing quality metrics for %d tickers", len(tickers))

    for ticker in tickers:
        try:
            df = compute_quality_metrics_one(
                ticker,
                fin_all[fin_all["ticker"] == ticker],
                ohlcv_all[ohlcv_all["ticker"] == ticker] if not ohlcv_all.empty else pd.DataFrame(),
            )
            if df.empty:
                continue
            n = storage.append("quality_metrics", df)
            total += n
            if n:
                log.debug("  quality_metrics %-6s +%d", ticker, n)
            STATE.set_last_fetched("quality_metrics", ticker, now)
        except Exception as exc:
            log.error("  quality_metrics %s FAILED: %s", ticker, exc)

    _log_job("quality_metrics", total)


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

    scheduler.add_job(job_valuations, "cron", day_of_week="mon-fri",
                      hour=18, minute=45, id="valuations",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_dividends,  "cron", day_of_week="mon-fri",
                      hour=19, minute=0,  id="dividends",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_earnings,   "cron", day_of_week="mon-fri",
                      hour=18, minute=35, id="earnings",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_implied_q4_eps, "cron", day_of_week="mon-fri",
                      hour=18, minute=40, id="implied_q4_eps",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_insider,    "cron", day_of_week="mon-fri",
                      hour=20, minute=0,  id="insider_trades",
                      max_instances=1, misfire_grace_time=3600)

    scheduler.add_job(job_events_8k,  "cron", day_of_week="mon-fri",
                      hour=20, minute=30, id="events_8k",
                      max_instances=1, misfire_grace_time=3600)

    # ── Macro: every day at 7 AM (FRED publishes morning releases) ───────────
    scheduler.add_job(job_macro,      "cron",
                      hour=7, minute=0,  id="macro_fred",
                      max_instances=1, misfire_grace_time=3600)

    # ── Fundamentals: every Saturday morning (weekly, low urgency) ───────────
    scheduler.add_job(job_financials, "cron", day_of_week="sat",
                      hour=6, minute=0,  id="financials",
                      max_instances=1, misfire_grace_time=7200)

    # ── Universe: every Monday morning ───────────────────────────────────────
    scheduler.add_job(job_universe,         "cron", day_of_week="mon",
                      hour=8, minute=0,  id="universe",
                      max_instances=1, misfire_grace_time=7200)

    scheduler.add_job(job_universe_history, "cron", day_of_week="mon",
                      hour=8, minute=30, id="universe_history",
                      max_instances=1, misfire_grace_time=7200)

    scheduler.add_job(job_short_interest,   "cron", day_of_week="mon",
                      hour=9, minute=0,  id="short_interest",
                      max_instances=1, misfire_grace_time=7200)

    # ── Delisted backfill: every Saturday (50 tickers per run) ───────────────
    scheduler.add_job(job_delisted_backfill,"cron", day_of_week="sat",
                      hour=7, minute=30, id="delisted_backfill",
                      max_instances=1, misfire_grace_time=7200)

    scheduler.add_job(job_quality_metrics,  "cron", day_of_week="sat",
                      hour=8, minute=0,  id="quality_metrics",
                      max_instances=1, misfire_grace_time=7200)

    scheduler.add_job(job_sectors,          "cron", day_of_week="mon",
                      hour=8, minute=45, id="sectors",
                      max_instances=1, misfire_grace_time=7200)

    # ── Heartbeat: every hour ────────────────────────────────────────────────
    scheduler.add_job(job_heartbeat,  "interval", hours=1, id="heartbeat")

    return scheduler


def run_all_once() -> None:
    """Run every job immediately in sequence (for --once mode).

    Dependency order matters for correctness on first/reset runs:
      universe         → provides ticker list for all subsequent jobs
      ohlcv            → price data required by indicators, valuations, dividends
      financials       → balance sheet/CF required by valuations, quality_metrics
      earnings         → EPS required by valuations, dividends, implied_q4
      implied_q4_eps   → must run after earnings (derives from earnings table)
      indicators       → requires ohlcv
      valuations       → requires ohlcv + earnings + financials
      dividends        → requires ohlcv + earnings
      quality_metrics  → requires financials (+ ohlcv for market-cap lookup)
    """
    log.info("Running all jobs once …")
    for fn in (job_universe, job_universe_history, job_short_interest,
               job_ohlcv,
               job_financials,         # before valuations / quality_metrics
               job_earnings,           # before implied_q4, valuations, dividends
               job_implied_q4_eps,     # after earnings
               job_macro,
               job_indicators,         # after ohlcv
               job_valuations,         # after ohlcv + earnings + financials
               job_dividends,          # after ohlcv + earnings
               job_events_8k, job_insider,
               job_delisted_backfill,  # fills ohlcv/earnings/financials for removed tickers
               job_quality_metrics,    # after financials
               job_sectors):
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

    # Run all jobs immediately on startup to catch up on any missed data.
    # Order matters: dependencies must complete before derived jobs run.
    log.info("Initial catch-up run …")
    for fn in (job_universe, job_universe_history, job_short_interest,
               job_ohlcv,
               job_financials,         # before valuations / quality_metrics
               job_earnings,           # before implied_q4, valuations, dividends
               job_implied_q4_eps,     # after earnings
               job_macro,
               job_indicators,         # after ohlcv
               job_valuations,         # after ohlcv + earnings + financials
               job_dividends,          # after ohlcv + earnings
               job_events_8k, job_insider,
               job_delisted_backfill,  # fills ohlcv/earnings/financials for removed tickers
               job_quality_metrics,    # after financials
               job_sectors):
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
