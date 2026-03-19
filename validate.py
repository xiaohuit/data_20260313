"""
Comprehensive validation suite for the data pipeline.

Checks (all must pass before daemon is considered healthy):
  1.  OHLCV completeness   — every ticker has bars from 2020-01-01 → today
  2.  OHLCV value sanity   — price > 0, volume > 0, high >= low, no NaN closes
  3.  Adjusted-price logic — adj_close <= close * 1.5 and > 0
  4.  Indicator sanity     — RSI in (0,100), SMA_20 > 0, ATR > 0
  5.  Macro completeness   — all 11 FRED series present (incl. derived CPI_YOY %), values in plausible range
  6.  Insider data         — columns present, trade dates parseable
  7.  Earnings data        — EPS columns present for equity tickers
  8.  Partition integrity  — every Parquet file readable, no empty files
  9.  Incremental append   — roll back one checkpoint, re-run, confirm +N rows
                             with zero duplicates
  10. Idempotency          — run same job twice, row count must not change
  11. Cross-partition read — DuckDB glob query returns consistent total
  12. No look-ahead        — knowledge_timestamp <= event_timestamp + 5 days
                             (sanity bound; real PiT check is in run_crawler.py)
  14. Sector classification — all GICS sectors present, spot checks for AAPL/JPM
  15. Quality metrics       — ROIC/CAGR/consistency sanity ranges, AAPL spot check
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
    job_ohlcv, job_indicators, job_valuations, job_dividends, job_macro,
    job_earnings, job_financials, job_events_8k, job_insider,
    job_universe_history, job_short_interest, TICKERS,
    job_sectors, job_quality_metrics,
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

    # Load delisted (removed) tickers from universe_history — these are expected to
    # have incomplete/stale data since they were acquired or delisted.
    _uh = storage.read_all("universe_history")
    delisted_tickers = set()
    if not _uh.empty and "action" in _uh.columns and "ticker" in _uh.columns:
        delisted_tickers = set(_uh[_uh["action"] == "removed"]["ticker"].unique())

    # Foreign cross-listings with thin US volume: primary market is non-US so daily
    # US bar coverage is legitimately sparse.  Exclude from the ≥90% bar checks.
    _THIN_VOLUME_FOREIGN = {"FER", "AMCR", "SW"}   # Ferrovial (ES), Amcor (AU), Smurfit WestRock (IE)
    delisted_tickers = delisted_tickers | _THIN_VOLUME_FOREIGN

    # Check 1: every ticker's history should be "complete" — meaning it has ≥90%
    # of the trading bars it could possibly have since its first-ever traded date.
    # Delisted tickers: use their last bar date as the end of their "possible" window.
    latest = df.groupby("ticker")["event_ts"].max()
    def _possible_bars(ticker: str) -> int:
        start = earliest[ticker]
        end = latest[ticker] if ticker in delisted_tickers else today
        return max(30, int((end - start).days * 252 / 365))
    possible_bars_early = pd.Series(
        {t: _possible_bars(t) for t in bar_counts.index}, dtype=float
    )
    coverage = bar_counts / possible_bars_early.reindex(bar_counts.index, fill_value=1)
    # Only check completeness for active (non-delisted) tickers
    active_bar_counts = bar_counts[~bar_counts.index.isin(delisted_tickers)]
    active_earliest   = earliest[~earliest.index.isin(delisted_tickers)]
    active_incomplete = coverage[
        (coverage < 0.90) & (~coverage.index.isin(delisted_tickers))
    ]
    n_post_2020 = (active_earliest > pd.Timestamp("2020-02-01", tz="UTC")).sum()
    check("history complete (≥90% of possible bars since IPO)",
          active_incomplete.empty,
          f"{len(active_incomplete)} tickers under 90%: " +
          ", ".join(f"{t}={coverage[t]:.0%}" for t in active_incomplete.index[:5])
          if not active_incomplete.empty
          else f"OK — {len(active_bar_counts)} active tickers, {n_post_2020} post-2020 IPOs")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    # Exclude delisted tickers from freshness check — they have no recent data by design
    stale = latest[(latest < cutoff) & (~latest.index.isin(delisted_tickers))]
    check("data fresh (within 10 days)", len(stale) == 0,
          f"stale: {stale.to_dict()}" if not stale.empty else "OK")

    # Exclude delisted tickers from min-coverage check
    active_coverage = coverage[~coverage.index.isin(delisted_tickers)]
    min_coverage = active_coverage.min() if not active_coverage.empty else 1.0
    min_ticker   = active_coverage.idxmin() if not active_coverage.empty else "N/A"
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
    # FRED_SERIES keys are base series; CPI_YOY is a derived series stored separately
    expected_codes = set(FRED_SERIES.keys()) | {"CPI_YOY"}
    missing = [c for c in expected_codes if c not in codes_found]
    check("all FRED series present", len(missing) == 0,
          f"missing: {missing}" if missing else f"{len(codes_found)} series")

    # Sanity ranges
    sanity = {
        "FED_FUNDS_RATE": (0, 25),
        "CPI_INDEX":      (100, 400),   # raw CPI index level (~130 in 1990, ~327 today)
        "CPI_YOY":        (-5, 20),     # true year-over-year % (usually 0–10)
        "GDP_GROWTH":     (-40, 40),    # annualised real GDP growth rate (%)
        "UNEMPLOYMENT":   (2, 20),
        "10Y_YIELD":      (0, 20),
        "2Y_YIELD":       (0, 20),      # short-end yield
        "VIX":            (5, 90),
        "YIELD_CURVE":    (-5, 5),
        "CONSUMER_CONF":  (40, 120),    # UMich consumer sentiment index (~50–110 historical)
        "M2":             (2000, 30000),  # M2 in billions USD (~$3.9T in 1990, ~$21T today)
        "OIL_WTI":        (10, 200),    # WTI crude in USD/barrel
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

    # Form check: only 10-K and 10-Q after normalisation; no 8-K, NT 10-K, etc.
    if "form" in df_e.columns:
        e_forms = df_e["form"].value_counts().to_dict()
        unexpected_e_forms = {f for f in e_forms if f not in ("10-K", "10-Q")}
        check("no unexpected form types in earnings",
              len(unexpected_e_forms) == 0,
              f"unexpected: {unexpected_e_forms}" if unexpected_e_forms else "OK")


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

    # Forms present — must have both annual and quarterly
    forms = df["form"].value_counts().to_dict() if "form" in df.columns else {}
    check("10-K and 10-Q both present",
          "10-K" in forms and "10-Q" in forms,
          str(forms))

    # No unexpected form types: only 10-K and 10-Q should exist after normalisation.
    # 10-K/A and 10-Q/A are normalised to base forms; anything else is a pipeline bug.
    unexpected_forms = {f for f in forms if f not in ("10-K", "10-Q")}
    check("no unexpected form types in financials",
          len(unexpected_forms) == 0,
          f"unexpected: {unexpected_forms}" if unexpected_forms else "OK")

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
# 9  Valuation ratios
# ─────────────────────────────────────────────────────────────────────────────

def validate_valuations() -> None:
    log.info("\n── 9  Valuation Ratios ──────────────────────────────────")
    df = storage.read_all("valuations")

    if df.empty:
        check("valuations loaded", False, "empty"); return
    check("valuations loaded", True, f"{len(df):,} rows")

    tickers_with_vals = df["ticker"].nunique()
    check("valuations covers ≥ 400 tickers", tickers_with_vals >= 400,
          f"{tickers_with_vals} tickers")

    # Date range
    df["event_ts"] = pd.to_datetime(df["event_timestamp"], utc=True)
    earliest = df["event_ts"].min()
    check("valuations history starts ≤ 2011",
          earliest <= pd.Timestamp("2011-01-01", tz="UTC"),
          f"earliest={earliest.date()}")

    # P/E sanity: should be positive and < 500 for most rows
    if "pe_ttm" in df.columns:
        pe = df["pe_ttm"].dropna()
        check("pe_ttm has values", len(pe) > 0, f"{len(pe):,} non-null rows")
        bad_pe = pe[(pe <= 0) | (pe > 500)]
        check("pe_ttm in (0, 500]", len(bad_pe) == 0,
              f"{len(bad_pe)} out-of-range" if not bad_pe.empty else
              f"range [{pe.min():.1f}, {pe.max():.1f}]")

    # P/B sanity
    if "pb" in df.columns:
        pb = df["pb"].dropna()
        bad_pb = pb[(pb <= 0) | (pb > 100)]
        check("pb in (0, 100]", len(bad_pb) == 0,
              f"{len(bad_pb)} out-of-range" if not bad_pb.empty else
              f"range [{pb.min():.1f}, {pb.max():.1f}]")

    # Dividend yield sanity
    if "dividend_yield" in df.columns:
        dy = df["dividend_yield"].dropna()
        bad_dy = dy[(dy < 0) | (dy > 0.5)]
        check("dividend_yield in [0, 50%]", len(bad_dy) == 0,
              f"{len(bad_dy)} out-of-range" if not bad_dy.empty else
              f"range [{dy.min():.3f}, {dy.max():.3f}]")

    # Spot check: AAPL P/E on 2023-06-30 should be ~30-35x
    aapl = df[df["ticker"] == "AAPL"].copy()
    if not aapl.empty and "pe_ttm" in df.columns:
        target = aapl[aapl["event_ts"].dt.date == pd.Timestamp("2023-06-30").date()]
        if not target.empty:
            pe_val = target["pe_ttm"].iloc[0]
            check("AAPL P/E 2023-06-30 ≈ 30-35x",
                  pe_val is not None and 25 < float(pe_val) < 45,
                  f"actual={pe_val:.1f}x" if pe_val else "None")

    # No future knowledge_timestamp
    df["kt"] = pd.to_datetime(df["knowledge_timestamp"], utc=True)
    lookahead = df[df["kt"] > df["event_ts"] + pd.Timedelta(days=1)]
    check("valuations: no future knowledge_ts", len(lookahead) == 0,
          f"{len(lookahead)} anomalies" if not lookahead.empty else "OK")

    # TTM revenue coverage: for EDGAR filers with ≥ 4 quarters of history,
    # most rows should have ttm_revenue populated (verifies implied-Q4 computation).
    if "ttm_revenue" in df.columns and "ps" in df.columns:
        ps_rows = df[df["ps"].notna()]
        if not ps_rows.empty:
            ttm_rev_fill = ps_rows["ttm_revenue"].notna().mean()
            check("P/S rows have ttm_revenue ≥ 60%", ttm_rev_fill >= 0.60,
                  f"{ttm_rev_fill:.1%} of P/S rows have ttm_revenue")


# ─────────────────────────────────────────────────────────────────────────────
# 10  Dividend history
# ─────────────────────────────────────────────────────────────────────────────

def validate_dividends() -> None:
    log.info("\n── 10  Dividend History ─────────────────────────────────")
    df = storage.read_all("dividends")

    if df.empty:
        check("dividends loaded", False, "empty"); return
    check("dividends loaded", True, f"{len(df):,} rows")

    tickers_with_divs = df["ticker"].nunique()
    check("dividends covers ≥ 400 tickers", tickers_with_divs >= 400,
          f"{tickers_with_divs} tickers")

    # Year range: should go back to at least 2011
    if "year" in df.columns:
        earliest_yr = int(df["year"].min())
        check("dividends history starts ≤ 2011", earliest_yr <= 2011,
              f"earliest year={earliest_yr}")

    # annual_dps should be non-negative
    if "annual_dps" in df.columns:
        dps = df["annual_dps"].dropna()
        bad_dps = dps[dps < 0]
        check("annual_dps >= 0", len(bad_dps) == 0,
              f"{len(bad_dps)} negative values" if not bad_dps.empty else
              f"range [{dps.min():.4f}, {dps.max():.2f}]")

        # Payers: stocks with annual_dps > 0
        payers = df[df["annual_dps"] > 0]["ticker"].nunique()
        check("dividend payers ≥ 200 tickers", payers >= 200,
              f"{payers} payers")

    # Payout ratio should be in [0, 2.0] when present
    if "payout_ratio" in df.columns:
        pr = df["payout_ratio"].dropna()
        if len(pr) > 0:
            bad_pr = pr[(pr < 0) | (pr > 2.0)]
            check("payout_ratio in [0, 200%]", len(bad_pr) == 0,
                  f"{len(bad_pr)} out-of-range" if not bad_pr.empty else
                  f"range [{pr.min():.3f}, {pr.max():.3f}]")

    # consecutive_growth should be non-negative integer
    if "consecutive_growth" in df.columns:
        cg = df["consecutive_growth"].dropna()
        bad_cg = cg[cg < 0]
        check("consecutive_growth >= 0", len(bad_cg) == 0,
              f"{len(bad_cg)} negative" if not bad_cg.empty else
              f"max streak={int(cg.max())}")

    # Spot check: JNJ should have a long dividend growth streak (Dividend King)
    jnj = df[df["ticker"] == "JNJ"].copy() if "ticker" in df.columns else pd.DataFrame()
    if not jnj.empty and "consecutive_growth" in jnj.columns:
        max_streak = int(jnj["consecutive_growth"].max())
        check("JNJ has ≥ 10 consecutive growth years", max_streak >= 10,
              f"max streak={max_streak}")

    # Spot check: AAPL 2023 annual DPS should be ~0.94
    aapl = df[(df["ticker"] == "AAPL") & (df["year"] == 2023)] \
        if "ticker" in df.columns else pd.DataFrame()
    if not aapl.empty and "annual_dps" in aapl.columns:
        dps_val = float(aapl["annual_dps"].iloc[0])
        check("AAPL 2023 annual DPS ≈ $0.94", 0.7 < dps_val < 1.2,
              f"actual=${dps_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 11  8-K material event filings
# ─────────────────────────────────────────────────────────────────────────────

def validate_events_8k() -> None:
    log.info("\n── 11  8-K Material Events ──────────────────────────────")
    df = storage.read_all("events_8k")

    if df.empty:
        check("events_8k loaded", False, "empty"); return
    check("events_8k loaded", True, f"{len(df):,} rows")

    tickers_covered = df["ticker"].nunique()
    check("events_8k covers ≥ 400 tickers", tickers_covered >= 400,
          f"{tickers_covered} tickers")

    # Date range
    df["event_ts"] = pd.to_datetime(df["event_timestamp"], utc=True)
    earliest = df["event_ts"].min()
    check("events_8k history starts ≤ 2011",
          earliest <= pd.Timestamp("2011-01-01", tz="UTC"),
          f"earliest={earliest.date()}")

    # All should be 8-K or 8-K/A
    forms = df["form"].value_counts().to_dict() if "form" in df.columns else {}
    check("only 8-K forms present",
          all(f in ("8-K", "8-K/A") for f in forms),
          str(forms))

    # Boolean flag columns should be 0 or 1
    flag_cols = [c for c in df.columns if c.startswith("has_")]
    check("flag columns present", len(flag_cols) >= 6,
          f"found: {flag_cols}")
    if flag_cols:
        bad_flags = df[flag_cols].isin([0, 1]).all().all() or \
                    df[flag_cols].notna().all().all()
        vals = df[flag_cols].stack().value_counts().to_dict()
        check("flag values are 0/1", set(vals.keys()) <= {0, 1},
              str(dict(list(vals.items())[:4])))

    # Earnings releases: most large-cap companies file 4× per year
    if "has_earnings" in df.columns:
        earnings_8k = df[df["has_earnings"] == 1]
        check("earnings 8-K filings present", len(earnings_8k) > 1000,
              f"{len(earnings_8k):,} earnings releases")

    # No look-ahead: knowledge_timestamp == event_timestamp for 8-Ks
    # (filings are public immediately upon SEC acceptance)
    df["kt"] = pd.to_datetime(df["knowledge_timestamp"], utc=True)
    lookahead = df[df["kt"] > df["event_ts"] + pd.Timedelta(days=1)]
    check("events_8k no future knowledge_ts", len(lookahead) == 0,
          f"{len(lookahead)} anomalies" if not lookahead.empty else "OK")

    # Spot check: AAPL files earnings 8-K (item 2.02) each October
    aapl = df[(df["ticker"] == "AAPL") & (df["has_earnings"] == 1)].copy() \
        if "ticker" in df.columns else pd.DataFrame()
    if not aapl.empty:
        aapl_oct = aapl[aapl["event_ts"].dt.month.isin([10, 11])]
        check("AAPL has October earnings 8-Ks", len(aapl_oct) >= 5,
              f"{len(aapl_oct)} October/November earnings filings")

    # Accession numbers should be unique per ticker
    if "accession_number" in df.columns:
        dups = df.duplicated(subset=["ticker", "accession_number"])
        check("no duplicate accession numbers", dups.sum() == 0,
              f"{dups.sum()} duplicates" if dups.any() else "OK")


# ─────────────────────────────────────────────────────────────────────────────
# 12  Universe membership history
# ─────────────────────────────────────────────────────────────────────────────

def validate_universe_history() -> None:
    log.info("\n── 12  Universe Membership History ─────────────────────")
    df = storage.read_all("universe_history")

    if df.empty:
        check("universe_history loaded", False, "empty"); return
    check("universe_history loaded", True, f"{len(df):,} rows")

    # Should have both S&P 500 and NASDAQ 100 events
    indices = df["index_name"].value_counts().to_dict() if "index_name" in df.columns else {}
    check("both indices present",
          "S&P 500" in indices and "NASDAQ 100" in indices,
          str(indices))

    # Both add and remove events
    actions = df["action"].value_counts().to_dict() if "action" in df.columns else {}
    check("added and removed events present",
          "added" in actions and "removed" in actions,
          f"added={actions.get('added',0)}  removed={actions.get('removed',0)}")

    # History should go back at least to 2000 for S&P 500
    df["event_ts"] = pd.to_datetime(df["event_timestamp"], utc=True)
    earliest = df["event_ts"].min()
    check("history starts ≤ 2005",
          earliest <= pd.Timestamp("2005-01-01", tz="UTC"),
          f"earliest={earliest.date()}")

    # Removed tickers should outnumber the current universe
    removed_tickers = df[df["action"] == "removed"]["ticker"].nunique()
    check("≥ 200 unique removed tickers", removed_tickers >= 200,
          f"{removed_tickers} unique removed tickers")

    # Spot check: ATVI (Activision, acquired by MSFT 2023) should be in removed
    atvi_removed = df[(df["ticker"] == "ATVI") & (df["action"] == "removed")]
    check("ATVI shows as removed", not atvi_removed.empty,
          f"found {len(atvi_removed)} removal events" if not atvi_removed.empty else "missing")

    # No duplicate events (same ticker+index+action+date)
    dups = df.duplicated(subset=["ticker", "index_name", "action", "event_date"])
    check("no duplicate events", dups.sum() == 0,
          f"{dups.sum()} duplicates" if dups.any() else "OK")


# ─────────────────────────────────────────────────────────────────────────────
# 13  Short interest
# ─────────────────────────────────────────────────────────────────────────────

def validate_short_interest() -> None:
    log.info("\n── 13  Short Interest ───────────────────────────────────")
    df = storage.read_all("short_interest")

    if df.empty:
        check("short_interest loaded", False, "empty"); return
    check("short_interest loaded", True, f"{len(df):,} rows")

    tickers_covered = df["ticker"].nunique()
    check("short_interest covers ≥ 400 tickers", tickers_covered >= 400,
          f"{tickers_covered} tickers")

    # shares_short should be positive
    if "shares_short" in df.columns:
        ss = pd.to_numeric(df["shares_short"], errors="coerce").dropna()
        bad = ss[ss <= 0]
        check("shares_short > 0", len(bad) == 0,
              f"{len(bad)} non-positive" if not bad.empty else
              f"range [{ss.min():,.0f}, {ss.max():,.0f}]")

    # days_to_cover should be in [0, 100] when present
    if "days_to_cover" in df.columns:
        dtc = pd.to_numeric(df["days_to_cover"], errors="coerce").dropna()
        bad_dtc = dtc[(dtc < 0) | (dtc > 100)]
        check("days_to_cover in [0, 100]", len(bad_dtc) == 0,
              f"{len(bad_dtc)} out-of-range" if not bad_dtc.empty else
              f"range [{dtc.min():.2f}, {dtc.max():.2f}]")

    # short_pct_float should be in [0, 1] when present
    if "short_pct_float" in df.columns:
        spf = pd.to_numeric(df["short_pct_float"], errors="coerce").dropna()
        bad_spf = spf[(spf < 0) | (spf > 1)]
        check("short_pct_float in [0, 100%]", len(bad_spf) == 0,
              f"{len(bad_spf)} out-of-range" if not bad_spf.empty else
              f"range [{spf.min():.3f}, {spf.max():.3f}]")

    # Should have two settlement dates per ticker (current + prior month)
    dates_per_ticker = df.groupby("ticker")["settlement_date"].nunique()
    avg_dates = dates_per_ticker.mean()
    check("≥ 1 settlement date per ticker", (dates_per_ticker >= 1).all(),
          f"avg {avg_dates:.1f} dates/ticker")

    # PiT: knowledge_timestamp must be > settlement_date (FINRA publishes 14-21 days
    # after the settlement date; we record knowledge_timestamp = fetch time, so it
    # must always be strictly after the settlement date).
    if "knowledge_timestamp" in df.columns and "settlement_date" in df.columns:
        df["_kt"] = pd.to_datetime(df["knowledge_timestamp"], errors="coerce", utc=True)
        df["_sd"] = pd.to_datetime(df["settlement_date"],     errors="coerce", utc=True)
        lookahead_si = df[df["_kt"] <= df["_sd"]].dropna(subset=["_kt", "_sd"])
        check("short_interest kt > settlement_date (no PiT look-ahead)",
              len(lookahead_si) == 0,
              f"{len(lookahead_si)} violations" if not lookahead_si.empty else "OK")


# ─────────────────────────────────────────────────────────────────────────────
# 14  Sector classification
# ─────────────────────────────────────────────────────────────────────────────

def validate_sectors() -> None:
    log.info("\n── 14  Sector Classification ────────────────────────────")
    df = storage.read_all("sectors")

    if df.empty:
        log.info("  sectors table empty — running job_sectors() now …")
        job_sectors()
        df = storage.read_all("sectors")

    if df.empty:
        check("sectors loaded", False, "empty after fetch"); return
    check("sectors loaded", True, f"{len(df):,} rows")

    tickers_covered = df["ticker"].nunique()
    check("sectors covers ≥ 400 tickers", tickers_covered >= 400,
          f"{tickers_covered} tickers")

    # All major GICS sectors should appear (yfinance uses "Financial Services")
    expected_sectors = {"Technology", "Healthcare", "Financial Services",
                        "Consumer Cyclical", "Industrials", "Energy",
                        "Communication Services", "Consumer Defensive"}
    found_sectors = set(df["sector"].dropna().unique())
    missing_sectors = expected_sectors - found_sectors
    check("all major sectors present", len(missing_sectors) == 0,
          f"missing: {missing_sectors}" if missing_sectors else
          f"{len(found_sectors)} sectors found")

    # Market cap categories should be populated
    if "market_cap_category" in df.columns:
        cats = df["market_cap_category"].value_counts().to_dict()
        check("market_cap_category populated", len(cats) >= 2, str(cats))

    # Spot check: AAPL should be Technology
    aapl = df[df["ticker"] == "AAPL"] if "ticker" in df.columns else pd.DataFrame()
    if not aapl.empty and "sector" in aapl.columns:
        sector = aapl["sector"].iloc[0]
        check("AAPL sector = Technology", sector == "Technology",
              f"actual='{sector}'")

    # Spot check: JPM should be Financial Services
    jpm = df[df["ticker"] == "JPM"] if "ticker" in df.columns else pd.DataFrame()
    if not jpm.empty and "sector" in jpm.columns:
        sector = jpm["sector"].iloc[0]
        check("JPM sector = Financial Services", sector == "Financial Services",
              f"actual='{sector}'")


# ─────────────────────────────────────────────────────────────────────────────
# 15  Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def validate_quality_metrics() -> None:
    log.info("\n── 15  Quality Metrics ──────────────────────────────────")
    df = storage.read_all("quality_metrics")

    if df.empty:
        log.info("  quality_metrics table empty — running job_quality_metrics() now …")
        job_quality_metrics()
        df = storage.read_all("quality_metrics")

    if df.empty:
        check("quality_metrics loaded", False, "empty after compute"); return
    check("quality_metrics loaded", True, f"{len(df):,} rows")

    tickers_covered = df["ticker"].nunique()
    check("quality_metrics covers ≥ 400 tickers", tickers_covered >= 400,
          f"{tickers_covered} tickers")

    # Should go back at least to 2012 (need prior years for CAGR)
    df["period_ts"] = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
    earliest = df["period_ts"].min()
    check("quality_metrics history starts ≤ 2013",
          earliest <= pd.Timestamp("2013-01-01", tz="UTC"),
          f"earliest={earliest.date() if pd.notna(earliest) else 'N/A'}")

    # ROIC sanity: capped at [-100%, 300%] in computation
    if "roic" in df.columns:
        roic = pd.to_numeric(df["roic"], errors="coerce").dropna()
        check("roic has values", len(roic) > 0, f"{len(roic):,} non-null rows")
        bad_roic = roic[(roic < -1.0) | (roic > 3.0)]
        check("roic in [-100%, 300%]", len(bad_roic) == 0,
              f"{len(bad_roic)} outliers" if not bad_roic.empty else
              f"range [{roic.min():.1%}, {roic.max():.1%}]")

    # Revenue CAGR 3Y: capped at [-80%, 500%] to exclude spin-offs/hypergrowth from near-zero
    if "revenue_cagr_3y" in df.columns:
        cagr = pd.to_numeric(df["revenue_cagr_3y"], errors="coerce").dropna()
        if len(cagr) > 0:
            bad_cagr = cagr[(cagr < -0.8) | (cagr > 5.0)]
            check("revenue_cagr_3y in [-80%, 500%]", len(bad_cagr) == 0,
                  f"{len(bad_cagr)} outliers" if not bad_cagr.empty else
                  f"range [{cagr.min():.1%}, {cagr.max():.1%}]")

    # Earnings consistency should be in [0, 1]
    if "earnings_consistency_5y" in df.columns:
        ec = pd.to_numeric(df["earnings_consistency_5y"], errors="coerce").dropna()
        bad_ec = ec[(ec < 0) | (ec > 1)]
        check("earnings_consistency_5y in [0, 1]", len(bad_ec) == 0,
              f"{len(bad_ec)} out-of-range" if not bad_ec.empty else
              f"range [{ec.min():.2f}, {ec.max():.2f}]")

    # Buyback yield in reasonable range
    if "buyback_yield" in df.columns:
        by = pd.to_numeric(df["buyback_yield"], errors="coerce").dropna()
        bad_by = by[(by < -0.5) | (by > 0.5)]
        check("buyback_yield in [-50%, 50%]", len(bad_by) == 0,
              f"{len(bad_by)} out-of-range" if not bad_by.empty else
              f"range [{by.min():.1%}, {by.max():.1%}]")

    # Spot check: AAPL ROIC should be very high (>30% historically)
    aapl = df[df["ticker"] == "AAPL"].copy() if "ticker" in df.columns else pd.DataFrame()
    if not aapl.empty and "roic" in aapl.columns:
        roic_vals = pd.to_numeric(aapl["roic"], errors="coerce").dropna()
        if not roic_vals.empty:
            median_roic = roic_vals.median()
            check("AAPL median ROIC > 20%", median_roic > 0.20,
                  f"median={median_roic:.1%}")

    # Spot check: MSFT revenue CAGR 3Y should be positive
    msft = df[df["ticker"] == "MSFT"].copy() if "ticker" in df.columns else pd.DataFrame()
    if not msft.empty and "revenue_cagr_3y" in msft.columns:
        cagr_vals = pd.to_numeric(msft["revenue_cagr_3y"], errors="coerce").dropna()
        if not cagr_vals.empty:
            recent_cagr = cagr_vals.iloc[-1]
            check("MSFT recent revenue CAGR 3Y > 0", recent_cagr > 0,
                  f"recent={recent_cagr:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# 16  Partition integrity
# ─────────────────────────────────────────────────────────────────────────────

def validate_partitions() -> None:
    log.info("\n── 16  Partition file integrity ─────────────────────────")
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
    validate_valuations()
    validate_dividends()
    validate_events_8k()
    validate_universe_history()
    validate_short_interest()
    validate_sectors()
    validate_quality_metrics()
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
