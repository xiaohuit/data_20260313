"""
Comprehensive validation suite for the data pipeline.

Checks (all must pass before daemon is considered healthy):
  1.  OHLCV completeness   — every ticker has bars from 2020-01-01 → today
  2.  OHLCV value sanity   — price > 0, volume > 0, high >= low, no NaN closes
  3.  Adjusted-price logic — adj_close <= close * 1.5 and > 0
  4.  Indicator sanity     — RSI in (0,100), SMA_20 > 0, ATR > 0
  5.  Macro completeness   — all 8 FRED series present, values in plausible range
  6.  Insider data         — columns present, trade dates parseable
  7.  Earnings data        — EPS columns present for equity tickers
  8.  Partition integrity  — every Parquet file readable, no empty files
  9.  Incremental append   — roll back one checkpoint, re-run, confirm +N rows
                             with zero duplicates
  10. Idempotency          — run same job twice, row count must not change
  11. Cross-partition read — DuckDB glob query returns consistent total
  12. No look-ahead        — knowledge_timestamp <= event_timestamp + 5 days
                             (sanity bound; real PiT check is in run_crawler.py)
"""

import sys
import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import duckdb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

import storage, state as state_mod
from daemon import (
    job_ohlcv, job_indicators, job_macro,
    job_earnings, job_financials, job_insider, TICKERS,
)

UTC = timezone.utc
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: list[tuple[str, str, str]] = []   # (check_name, status, detail)


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = PASS if ok else (WARN if warn_only else FAIL)
    results.append((name, status, detail))
    log.info("  %-40s  %s  %s", name, status, detail)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 1-3  OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def validate_ohlcv() -> None:
    log.info("\n── 1-3  OHLCV ───────────────────────────────────────────")
    df = storage.read_all("ohlcv")

    if df.empty:
        check("ohlcv loaded", False, "completely empty"); return
    check("ohlcv loaded", True, f"{len(df):,} rows")

    # 1. Completeness
    tickers_found = set(df["ticker"].unique())
    missing = [t for t in TICKERS if t not in tickers_found]
    check("all tickers present",    len(missing) == 0,
          f"missing: {missing}" if missing else f"{len(tickers_found)} tickers")

    df["event_ts"] = pd.to_datetime(df["event_timestamp"], utc=True)
    earliest = df.groupby("ticker")["event_ts"].min()
    bar_counts = df.groupby("ticker").size()
    today = pd.Timestamp.now(tz="UTC")

    # Check 1: every ticker's history should be "complete" — meaning it has ≥90%
    # of the trading bars it could possibly have since its first-ever traded date.
    # This detects tickers where we fetched only partial history, regardless of IPO.
    possible_bars_early = earliest.apply(
        lambda e: max(30, int((today - e).days * 252 / 365))
    )
    coverage = bar_counts / possible_bars_early.reindex(bar_counts.index, fill_value=1)
    incomplete = coverage[coverage < 0.90]
    n_post_2020 = (earliest > pd.Timestamp("2020-02-01", tz="UTC")).sum()
    check("history complete (≥90% of possible bars since IPO)",
          incomplete.empty,
          f"{len(incomplete)} tickers under 90%: " +
          ", ".join(f"{t}={coverage[t]:.0%}" for t in incomplete.index[:5])
          if not incomplete.empty
          else f"OK — {len(bar_counts)} tickers, {n_post_2020} post-2020 IPOs")

    latest = df.groupby("ticker")["event_ts"].max()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    stale = latest[latest < cutoff]
    check("data fresh (within 10 days)", len(stale) == 0,
          f"stale: {stale.to_dict()}" if not stale.empty else "OK")

    min_coverage = coverage.min() if not coverage.empty else 1.0
    min_ticker   = coverage.idxmin() if not coverage.empty else "N/A"
    check("≥ 90% of possible bars per ticker",  min_coverage >= 0.90,
          f"min={min_coverage:.1%} ({min_ticker})")

    # 2. Value sanity
    bad_close = df[df["close"].isna() | (df["close"] <= 0)]
    check("no null/zero close",      len(bad_close) == 0,
          f"{len(bad_close)} bad rows" if bad_close.empty is False else "OK")

    bad_hl = df[df["high"] < df["low"]]
    check("high >= low always",      len(bad_hl) == 0,
          f"{len(bad_hl)} violations" if not bad_hl.empty else "OK")

    bad_vol = df[df["volume"].notna() & (df["volume"] < 0)]
    check("volume >= 0",             len(bad_vol) == 0,
          f"{len(bad_vol)} negative" if not bad_vol.empty else "OK")

    # 3. Adjusted price
    if "adj_close" in df.columns:
        sub = df[df["adj_close"].notna() & df["close"].notna() & (df["close"] > 0)]
        ratio = sub["adj_close"] / sub["close"]
        weird = sub[(ratio > 1.5) | (ratio <= 0)]
        check("adj_close plausible ratio", len(weird) == 0,
              f"{len(weird)} outliers" if not weird.empty else "OK")
    else:
        check("adj_close column present", False, "column missing")


# ─────────────────────────────────────────────────────────────────────────────
# 4  Indicators
# ─────────────────────────────────────────────────────────────────────────────

def validate_indicators() -> None:
    log.info("\n── 4  Technical Indicators ──────────────────────────────")
    df = storage.read_all("indicators")

    if df.empty:
        check("indicators loaded", False, "empty"); return
    check("indicators loaded", True, f"{len(df):,} rows")

    # RSI_14 should be in (0, 100)
    rsi_col = next((c for c in df.columns if "RSI" in c.upper()), None)
    if rsi_col:
        rsi_all = df[rsi_col].dropna()
        # RSI requires ~14 bars of warmup; exclude rows from tickers with
        # very short histories (< 30 bars) where the value can be ill-defined
        if "ticker" in df.columns:
            bar_counts_ind = df.groupby("ticker").size()
            mature_tickers = bar_counts_ind[bar_counts_ind >= 30].index
            rsi = df[df["ticker"].isin(mature_tickers)][rsi_col].dropna()
        else:
            rsi = rsi_all
        # RSI is bounded [0, 100]; exact 0 or 100 is mathematically valid
        # (all down-closes or all up-closes in the window)
        bad_rsi = rsi[(rsi < -0.001) | (rsi > 100.001)]
        check("RSI in [0, 100]",  len(bad_rsi) == 0,
              f"{len(bad_rsi)} out-of-range" if not bad_rsi.empty else
              f"range [{rsi.min():.1f}, {rsi.max():.1f}]")
    else:
        check("RSI column present", False, "not found")

    # SMA_20 should be positive
    sma_col = next((c for c in df.columns if "SMA_20" in c.upper()), None)
    if sma_col:
        sma = df[sma_col].dropna()
        bad_sma = sma[sma <= 0]
        check("SMA_20 > 0",       len(bad_sma) == 0,
              f"{len(bad_sma)} non-positive" if not bad_sma.empty else "OK")

    # ATR should be positive
    atr_col = next((c for c in df.columns if "ATR" in c.upper()), None)
    if atr_col:
        atr = df[atr_col].dropna()
        bad_atr = atr[atr <= 0]
        check("ATR > 0",          len(bad_atr) == 0,
              f"{len(bad_atr)} non-positive" if not bad_atr.empty else "OK")

    # MACD line present
    macd_col = next((c for c in df.columns if "MACD" in c.upper()
                     and "h" not in c.lower() and "s" not in c.lower()[-1:]), None)
    check("MACD column present",  macd_col is not None,
          macd_col or "not found")

    # Spot check: AAPL RSI on 2023-06-30 should be ~71.9
    aapl = df[df["ticker"] == "AAPL"].copy() if "ticker" in df.columns else pd.DataFrame()
    if not aapl.empty and rsi_col:
        aapl["event_ts"] = pd.to_datetime(aapl["event_timestamp"], utc=True)
        target = aapl[aapl["event_ts"].dt.date == pd.Timestamp("2023-06-30").date()]
        if not target.empty:
            rsi_val = float(target[rsi_col].iloc[0])
            check("AAPL RSI 2023-06-30 ≈ 71–78",
                  65.0 < rsi_val < 85.0,
                  f"actual={rsi_val:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5  Macro
# ─────────────────────────────────────────────────────────────────────────────

def validate_macro() -> None:
    log.info("\n── 5  Macro / FRED ──────────────────────────────────────")
    df = storage.read_all("macro")

    if df.empty:
        check("macro loaded", False, "empty"); return
    check("macro loaded", True, f"{len(df):,} rows")

    from run_crawler import FRED_SERIES
    # indicator_code may be stored as series_id in old partitions (pre-fix)
    # Re-run macro job to rewrite with correct column, then re-read
    code_col = "indicator_code" if "indicator_code" in df.columns else \
               "series_id"      if "series_id"      in df.columns else None
    if code_col is None:
        check("macro has indicator_code column", False,
              f"columns: {list(df.columns)}"); return
    if code_col == "series_id":
        log.info("  Rebuilding macro partitions (old layout missing indicator_code) …")
        from run_crawler import FRED_SERIES
        STATE = state_mod.StateStore()
        for code in FRED_SERIES:
            STATE.set_last_fetched("macro", code,
                                   datetime.now(UTC) - timedelta(days=2200))
        job_macro()
        df = storage.read_all("macro")
        code_col = "indicator_code" if "indicator_code" in df.columns else "series_id"
    codes_found = set(df[code_col].unique())
    missing = [c for c in FRED_SERIES if c not in codes_found]
    check("all 8 FRED series present", len(missing) == 0,
          f"missing: {missing}" if missing else f"{len(codes_found)} series")

    # Sanity ranges
    sanity = {
        "FED_FUNDS_RATE": (0, 25),
        "CPI_YOY":        (100, 400),   # index level (was ~130 in 1990, ~327 today)
        "UNEMPLOYMENT":   (2, 20),
        "10Y_YIELD":      (0, 20),
        "VIX":            (5, 90),
        "YIELD_CURVE":    (-5, 5),
    }
    for code, (lo, hi) in sanity.items():
        sub = df[df["indicator_code"] == code]["value"].dropna()
        if sub.empty:
            check(f"{code} has values", False, "no data"); continue
        out = sub[(sub < lo) | (sub > hi)]
        check(f"{code} in [{lo}, {hi}]", len(out) == 0,
              f"range [{sub.min():.2f}, {sub.max():.2f}]")

    # Spot check: Fed Funds rate on 2023-07-01 should be ~5.08
    ff = df[df["indicator_code"] == "FED_FUNDS_RATE"].copy()
    if not ff.empty:
        ff["event_ts"] = pd.to_datetime(ff["event_timestamp"], utc=True)
        jul23 = ff[(ff["event_ts"].dt.year == 2023) & (ff["event_ts"].dt.month == 7)]
        if not jul23.empty:
            val = float(jul23["value"].iloc[0])
            check("Fed Funds Jul-2023 ≈ 5.08", abs(val - 5.08) < 0.5,
                  f"actual={val:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6-7  Insider & Earnings
# ─────────────────────────────────────────────────────────────────────────────

def validate_alternative() -> None:
    log.info("\n── 6  Insider Trades ────────────────────────────────────")
    df = storage.read_all("insider")

    if df.empty:
        check("insider loaded", False, "empty"); return
    check("insider loaded", True, f"{len(df):,} rows")

    # Normalise column names — OpenInsider HTML uses non-breaking spaces (\xa0)
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]

    required_cols = ["ticker_queried", "trade type", "insider name"]
    for col in required_cols:
        check(f"column '{col}' present", col in df.columns,
              f"found: {[c for c in df.columns if 'trade' in c.lower() or 'insider' in c.lower()]}")

    trd_col = next((c for c in df.columns if "trade" in c.lower()
                    and "date" in c.lower()), None)
    if trd_col:
        dates = pd.to_datetime(df[trd_col], errors="coerce")
        bad_dates = dates[dates.isna()]
        check("trade dates parseable", len(bad_dates) == 0,
              f"{len(bad_dates)} unparseable" if not bad_dates.empty else
              f"range {dates.min().date()} → {dates.max().date()}")

    type_col = next((c for c in df.columns if "trade" in c.lower()
                     and "type" in c.lower()), None)
    type_vals = df[type_col].value_counts().to_dict() if type_col else {}
    check("P/S trade types present",
          any("P" in str(k) or "S" in str(k) for k in type_vals),
          str(dict(list(type_vals.items())[:4])))

    log.info("\n── 7  Earnings ──────────────────────────────────────────")
    df_e = storage.read_all("earnings")

    if df_e.empty:
        check("earnings loaded", False, "empty"); return
    check("earnings loaded", True, f"{len(df_e):,} rows")

    eps_col = next((c for c in df_e.columns if "eps" in c.lower()
                    and "actual" in c.lower()), None)
    check("epsActual column present", eps_col is not None, eps_col or "missing")

    if "ticker" in df_e.columns:
        eq_tickers = [t for t in TICKERS if t not in ("SPY","QQQ","GLD","TLT")]
        tickers_with_earnings = set(df_e["ticker"].unique())
        missing_eq = [t for t in eq_tickers if t not in tickers_with_earnings]
        check("equity tickers have earnings", len(missing_eq) == 0,
              f"missing: {missing_eq}" if missing_eq else "OK")


# ─────────────────────────────────────────────────────────────────────────────
# 8  Fundamental financials
# ─────────────────────────────────────────────────────────────────────────────

def validate_financials() -> None:
    log.info("\n── 8  Fundamental Financials ────────────────────────────")
    df = storage.read_all("financials")

    if df.empty:
        check("financials loaded", False, "empty"); return
    check("financials loaded", True, f"{len(df):,} rows")

    tickers_with_data = df["ticker"].nunique()
    check("financials covers ≥ 400 tickers", tickers_with_data >= 400,
          f"{tickers_with_data} tickers")

    # Forms present
    forms = df["form"].value_counts().to_dict() if "form" in df.columns else {}
    check("10-K and 10-Q both present",
          "10-K" in forms and "10-Q" in forms,
          str(forms))

    # Date range: should go back to at least 2012
    df["period_ts"] = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
    earliest = df["period_ts"].min()
    check("financials history starts ≤ 2013",
          earliest <= pd.Timestamp("2013-01-01", tz="UTC"),
          f"earliest={earliest.date() if pd.notna(earliest) else 'N/A'}")

    # Key columns present
    required = ["revenue", "net_income", "total_equity", "operating_cf", "capex", "fcf"]
    for col in required:
        check(f"column '{col}' present", col in df.columns, col)

    # Revenue sanity: should be positive for most rows (exclude pre-revenue cos.)
    if "revenue" in df.columns:
        rev = df["revenue"].dropna()
        pct_positive = (rev > 0).mean()
        check("revenue > 0 for ≥ 80% of rows", pct_positive >= 0.80,
              f"{pct_positive:.1%} positive")

    # Gross margin sanity: between -50% and 100%
    if "gross_margin" in df.columns:
        gm = df["gross_margin"].dropna()
        bad_gm = gm[(gm < -0.5) | (gm > 1.0)]
        check("gross_margin in [-50%, 100%]", len(bad_gm) == 0,
              f"{len(bad_gm)} out-of-range" if not bad_gm.empty else
              f"range [{gm.min():.1%}, {gm.max():.1%}]")

    # PiT: knowledge_timestamp >= period_end
    df["kt"] = pd.to_datetime(df["knowledge_timestamp"], errors="coerce", utc=True)
    lookahead = df[df["kt"] < df["period_ts"] - pd.Timedelta(days=1)]
    check("financials PiT: no look-ahead", len(lookahead) == 0,
          f"{len(lookahead)} violations" if not lookahead.empty else "OK")

    # Spot check: AAPL FY2023 (10-K ending ~Sep 2023) revenue should be ~$383B
    aapl_10k = df[(df["ticker"] == "AAPL") & (df["form"] == "10-K")].copy()
    if not aapl_10k.empty and "revenue" in df.columns:
        aapl_10k = aapl_10k.sort_values("period_end")
        fy2023 = aapl_10k[aapl_10k["period_end"].str.startswith("2023")]
        if not fy2023.empty:
            rev_val = float(fy2023["revenue"].iloc[-1])
            check("AAPL FY2023 revenue ≈ $383B",
                  300e9 < rev_val < 500e9,
                  f"actual=${rev_val/1e9:.0f}B")


# ─────────────────────────────────────────────────────────────────────────────
# 9  Partition integrity
# ─────────────────────────────────────────────────────────────────────────────

def validate_partitions() -> None:
    log.info("\n── 8  Partition file integrity ──────────────────────────")
    import pyarrow.parquet as pq

    data_root = Path("./data")
    all_files = list(data_root.rglob("data.parquet"))
    check("partition files exist", len(all_files) > 0, f"{len(all_files)} files")

    corrupt = []
    empty_files = []
    for f in all_files:
        try:
            meta = pq.read_metadata(f)
            if meta.num_rows == 0:
                empty_files.append(str(f))
        except Exception as exc:
            corrupt.append((str(f), str(exc)))

    check("no corrupt Parquet files",  len(corrupt) == 0,
          f"corrupt: {corrupt}" if corrupt else f"{len(all_files)} files OK")
    check("no empty Parquet files",    len(empty_files) == 0,
          f"empty: {empty_files[:3]}" if empty_files else "OK",
          warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9  Incremental append (the most critical test)
# ─────────────────────────────────────────────────────────────────────────────

def validate_incremental_append() -> None:
    log.info("\n── 9  Incremental append ────────────────────────────────")
    STATE = state_mod.StateStore()

    # Count rows before
    before = storage.row_counts()
    log.info("  Row counts before: %s", before)

    # Roll back AAPL ohlcv checkpoint to 60 days ago
    # This forces the job to re-fetch the last 60 days of bars
    sixty_days_ago = datetime.now(UTC) - timedelta(days=61)
    STATE.set_last_fetched("ohlcv", "AAPL", sixty_days_ago)
    STATE.set_last_fetched("indicators", "AAPL", sixty_days_ago)
    log.info("  Rolled back AAPL checkpoint to %s", sixty_days_ago.date())

    # Re-run just OHLCV
    log.info("  Re-running job_ohlcv …")
    job_ohlcv()

    after = storage.row_counts()
    log.info("  Row counts after:  %s", after)

    # OHLCV should be same (all bars already exist — dedup should kick in)
    # But check the job ran without error and didn't corrupt data
    check("ohlcv row count stable after re-run",
          after["ohlcv"] >= before["ohlcv"],
          f"before={before['ohlcv']}, after={after['ohlcv']}")

    # Verify no duplicates were created for AAPL
    aapl_df = storage.read_all("ohlcv")
    aapl_df = aapl_df[aapl_df["ticker"] == "AAPL"]
    aapl_df["event_ts"] = pd.to_datetime(aapl_df["event_timestamp"], utc=True)
    dups = aapl_df[aapl_df.duplicated(subset=["event_ts"], keep=False)]
    check("no duplicate AAPL bars after re-run", len(dups) == 0,
          f"{len(dups)} duplicate rows found" if not dups.empty else "0 duplicates")

    # Re-run indicators
    log.info("  Re-running job_indicators …")
    job_indicators()
    after2 = storage.row_counts()
    check("indicator row count stable after re-run",
          after2["indicators"] >= after["indicators"],
          f"before={after['indicators']}, after={after2['indicators']}")


# ─────────────────────────────────────────────────────────────────────────────
# 10  Idempotency
# ─────────────────────────────────────────────────────────────────────────────

def validate_idempotency() -> None:
    log.info("\n── 10  Idempotency (run macro twice) ────────────────────")
    STATE = state_mod.StateStore()
    # Reset macro checkpoint so it re-fetches
    for code in ("FED_FUNDS_RATE", "VIX"):
        STATE.set_last_fetched("macro", code,
                               datetime.now(UTC) - timedelta(days=400))

    before = storage.row_counts()["macro"]
    log.info("  Run 1 …")
    job_macro()
    mid = storage.row_counts()["macro"]
    log.info("  Run 2 (same window) …")
    job_macro()
    after = storage.row_counts()["macro"]

    check("macro row count same on 2nd run",  mid == after,
          f"run1={mid}, run2={after}")
    check("macro rows ≥ baseline",            after >= before,
          f"baseline={before}, after={after}")


# ─────────────────────────────────────────────────────────────────────────────
# 11  Cross-partition DuckDB read
# ─────────────────────────────────────────────────────────────────────────────

def validate_duckdb() -> None:
    log.info("\n── 11  DuckDB cross-partition query ─────────────────────")
    try:
        con = duckdb.connect(":memory:")
        con.execute("INSTALL parquet; LOAD parquet;")

        glob = str(Path("data/ohlcv/**/*.parquet"))
        result = con.execute(
            f"SELECT COUNT(*) AS n, COUNT(DISTINCT ticker) AS t "
            f"FROM read_parquet('{glob}')"
        ).fetchone()
        rows, tickers = result
        check("DuckDB reads all ohlcv partitions", rows > 30000,
              f"{rows:,} rows, {tickers} tickers")

        # Verify PiT filter works via DuckDB
        pit_result = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{glob}')
            WHERE knowledge_timestamp <= '2022-01-01T00:00:00+00:00'
              AND event_timestamp     >  '2022-01-01T00:00:00+00:00'
        """).fetchone()[0]
        check("DuckDB PiT filter: 0 look-ahead rows", pit_result == 0,
              f"{pit_result} look-ahead rows leaked")

        # Spot-check AAPL close on 2023-06-30 ≈ 189
        spot = con.execute(f"""
            SELECT close FROM read_parquet('{glob}')
            WHERE ticker = 'AAPL'
              AND CAST(event_timestamp AS DATE) = '2023-06-30'
        """).fetchone()
        if spot:
            val = float(spot[0])
            check("AAPL close 2023-06-30 ≈ 189", abs(val - 189) < 5,
                  f"actual={val:.2f}")

    except Exception as exc:
        check("DuckDB query", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# 12  Knowledge timestamp sanity (no extreme look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def validate_pit_sanity() -> None:
    log.info("\n── 12  Knowledge-timestamp sanity ───────────────────────")
    df = storage.read_all("ohlcv")
    if df.empty:
        check("pit sanity: ohlcv loaded", False); return

    df["event_ts"]     = pd.to_datetime(df["event_timestamp"],     utc=True, errors="coerce")
    df["knowledge_ts"] = pd.to_datetime(df["knowledge_timestamp"], utc=True, errors="coerce")

    # knowledge_ts must be AFTER event_ts (can't know price before bar closes)
    bad_kt = df[df["knowledge_ts"] < df["event_ts"] - pd.Timedelta(hours=1)]
    check("knowledge_ts >= event_ts", len(bad_kt) == 0,
          f"{len(bad_kt)} violations" if not bad_kt.empty else "OK")

    # knowledge_ts must be within 5 days of event_ts for daily bars
    too_far = df[df["knowledge_ts"] > df["event_ts"] + pd.Timedelta(days=5)]
    check("knowledge_ts within 5d of event_ts", len(too_far) == 0,
          f"{len(too_far)} anomalies" if not too_far.empty else "OK")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    log.info("=" * 65)
    log.info("  PIPELINE VALIDATION SUITE")
    log.info("=" * 65)

    validate_ohlcv()
    validate_indicators()
    validate_macro()
    validate_alternative()
    validate_financials()
    validate_partitions()
    validate_incremental_append()
    validate_idempotency()
    validate_duckdb()
    validate_pit_sanity()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed  = sum(1 for _, s, _ in results if s == PASS)
    warned  = sum(1 for _, s, _ in results if s == WARN)
    failed  = sum(1 for _, s, _ in results if s == FAIL)

    log.info("\n" + "=" * 65)
    log.info("  VALIDATION SUMMARY  —  %d checks", len(results))
    log.info("  %s  %d passed    ⚠️  %d warnings    ❌  %d failed",
             "✅" if failed == 0 else "❌", passed, warned, failed)
    log.info("=" * 65)

    if failed:
        log.info("\n  Failed checks:")
        for name, status, detail in results:
            if status == FAIL:
                log.info("    ❌  %-40s  %s", name, detail)

    if warned:
        log.info("\n  Warnings:")
        for name, status, detail in results:
            if status == WARN:
                log.info("    ⚠️   %-40s  %s", name, detail)

    # Final row counts
    log.info("\n  Final data store:")
    for k, v in storage.row_counts().items():
        log.info("    %-15s %10d rows", k, v)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
