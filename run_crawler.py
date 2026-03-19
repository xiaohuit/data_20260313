"""
Standalone crawler runner — no database required.

Downloads real data for a representative set of tickers and saves to:
  ./data/ohlcv/          CSV + Parquet per ticker
  ./data/indicators/     CSV per ticker
  ./data/macro/          CSV (FRED series)
  ./data/insider/        CSV per ticker
  ./data/earnings/       CSV per ticker
  ./data/universe.csv    S&P 500 current members
  ./data/pit_snapshot/   PiT snapshot sample (JSON)

Also runs a PiT correctness check to prove zero look-ahead bias.
"""

import asyncio
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

UTC = timezone.utc
DATA_DIR = Path("./data")
# NOTE: Do NOT use a module-level END constant — it would freeze at import time
# and cause OHLCV/FRED fetches to stop at daemon start rather than "now".
# Each fetch function computes its own end date at call time.

# OHLCV/indicators/earnings: 15 years of daily bars gives models enough
# data to learn across multiple market cycles (2008 recovery, 2020 crash, etc.)
START_EQUITY = datetime(2010, 1, 1, tzinfo=UTC)

# Macro (FRED): go back to 1990 — covers multiple recessions, rate cycles,
# and inflation regimes that are critical for macro-aware models
START_MACRO = datetime(1990, 1, 1, tzinfo=UTC)

# Backward-compat alias used by any code that still references START
START = START_EQUITY

# Minimal bootstrap list — used ONLY when no universe has ever been stored yet.
# The real universe is fetched live from Wikipedia (S&P 500 + NASDAQ 100) and
# stored in data/universe/; subsequent runs read from there.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "JNJ",  "XOM",  "WMT",   "BAC",  "UNH",  "CVX",  "HD",
    "PG",   "MA",   "V",     "NFLX", "AVGO",
]

# Browser-like headers — prevents Wikipedia 403 on automated requests
_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# SEC EDGAR requires: "Company Name contact@email.com" format
# See https://www.sec.gov/os/accessing-edgar-data
_EDGAR_HEADERS = {
    "User-Agent": "FinancialDataPipeline research@financialpipeline.example.com",
    "Accept": "application/json",
    "Host": "data.sec.gov",
}

FRED_SERIES = {
    "FED_FUNDS_RATE": "FEDFUNDS",
    "CPI_INDEX":      "CPIAUCSL",   # raw CPI index level; CPI_YOY (%) is derived below
    "GDP_GROWTH":     "A191RL1Q225SBEA",
    "UNEMPLOYMENT":   "UNRATE",
    "10Y_YIELD":      "GS10",
    "2Y_YIELD":       "GS2",        # short-end yields; enables independent 2Y signal
    "VIX":            "VIXCLS",
    "YIELD_CURVE":    "T10Y2Y",
    "CONSUMER_CONF":  "UMCSENT",
    "M2":             "M2SL",       # M2 money supply — liquidity/QE regime indicator
    "OIL_WTI":        "DCOILWTICO", # WTI crude oil price — energy costs, inflation signal
}

# Publication lag per series (days after reference period before data is publicly released).
# Used to set knowledge_timestamp correctly when no FRED API key is available.
# Daily market series are available same/next day; monthly macro series lag 30–45 days.
_FRED_PUB_LAG = {
    "VIX":            1,    # VIX is a real-time market index
    "YIELD_CURVE":    1,    # T10Y2Y updated daily
    "10Y_YIELD":      1,    # GS10 updated daily
    "2Y_YIELD":       1,    # GS2 updated daily (Fed H.15 release)
    "OIL_WTI":        2,    # EIA publishes DCOILWTICO daily with ~1-2 day lag
    "FED_FUNDS_RATE": 35,   # FEDFUNDS: effective rate published ~35 days after month end
    "CPI_INDEX":      45,   # CPI released ~45 days after reference month end
    "UNEMPLOYMENT":   35,   # BLS Employment Situation: first Friday of following month
    "GDP_GROWTH":     90,   # BEA advance GDP estimate: ~30 days; use 90 for revision
    "CONSUMER_CONF":  35,   # UMich consumer sentiment: ~35 days
    "M2":             45,   # M2: published ~5 weeks after reference month end
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Universe from Wikipedia (live, dynamic — reflects actual index changes)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_wiki_table(url: str, table_id: str, index_name: str) -> pd.DataFrame:
    """
    Synchronous: fetch a Wikipedia constituents table using browser-like
    headers to avoid 403. Raises on failure so the caller can handle fallback.
    """
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), attrs={"id": table_id})
    if not tables:
        raise ValueError(f"No table id='{table_id}' on {url}")
    df_raw = tables[0]
    df_raw.columns = [c.strip() for c in df_raw.columns]

    ticker_col  = next((c for c in df_raw.columns
                        if "symbol" in c.lower() or "ticker" in c.lower()),
                       df_raw.columns[0])
    company_col = next((c for c in df_raw.columns
                        if "security" in c.lower() or "company" in c.lower()),
                       df_raw.columns[1] if len(df_raw.columns) > 1 else df_raw.columns[0])
    sector_col  = next((c for c in df_raw.columns if "sector" in c.lower()), None)
    sub_col     = next((c for c in df_raw.columns
                        if "sub" in c.lower() and "industry" in c.lower()), None)

    rows = []
    for _, r in df_raw.iterrows():
        ticker = str(r[ticker_col]).replace(".", "-").strip()
        if not ticker or ticker.lower() == "nan":
            continue
        rows.append({
            "ticker":              ticker,
            "company":             str(r[company_col]),
            "sector":              str(r[sector_col]) if sector_col else "",
            "sub_industry":        str(r[sub_col]) if sub_col else "",
            "index":               index_name,
            "added_date":          pd.Timestamp.today().date().isoformat(),
            "knowledge_timestamp": datetime.now(UTC).isoformat(),
        })
    return pd.DataFrame(rows)


async def fetch_sp500_universe() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia."""
    log.info("📋  Fetching S&P 500 universe from Wikipedia …")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(
            None, lambda: _parse_wiki_table(url, "constituents", "S&P 500")
        )
        log.info("    → %d S&P 500 members", len(df))
        return df
    except Exception as exc:
        log.warning("    S&P 500 Wikipedia fetch failed: %s", exc)
        return pd.DataFrame()


async def fetch_ndx100_universe() -> pd.DataFrame:
    """Fetch current NASDAQ 100 constituents from Wikipedia."""
    log.info("📋  Fetching NASDAQ 100 universe from Wikipedia …")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(
            None, lambda: _parse_wiki_table(url, "constituents", "NASDAQ 100")
        )
        log.info("    → %d NASDAQ 100 members", len(df))
        return df
    except Exception as exc:
        log.warning("    NASDAQ 100 Wikipedia fetch failed: %s", exc)
        return pd.DataFrame()


async def fetch_full_universe() -> pd.DataFrame:
    """
    Fetch S&P 500 + NASDAQ 100 from Wikipedia, merge and deduplicate.

    Fallback chain:
      1. Live Wikipedia fetch (both indices, ~600 tickers before dedup)
      2. Last stored universe from data/universe/ (if fetch fails)
      3. Minimal TICKERS list (bootstrap only — should never reach this)
    """
    import storage as _storage  # imported here to avoid circular imports

    sp500, ndx100 = await asyncio.gather(
        fetch_sp500_universe(),
        fetch_ndx100_universe(),
    )

    live_frames = [f for f in [sp500, ndx100] if not f.empty]
    if live_frames:
        combined = pd.concat(live_frames, ignore_index=True)
        # Deduplicate by ticker — keep S&P 500 entry when both have a row
        combined = combined.drop_duplicates(subset=["ticker"], keep="first")
        log.info("    → %d unique tickers (S&P 500 + NASDAQ 100)", len(combined))
        return combined

    # Both fetches failed — use last stored universe so jobs aren't blocked
    stored = _storage.read_all("universe")
    if not stored.empty:
        log.warning(
            "    Wikipedia unreachable — using %d previously stored tickers",
            stored["ticker"].nunique(),
        )
        return stored

    # Absolute last resort: bootstrap list
    log.error("    No universe available from any source — using bootstrap list")
    return pd.DataFrame([
        {"ticker": t, "company": t, "sector": "", "sub_industry": "",
         "index": "bootstrap", "added_date": pd.Timestamp.today().date().isoformat(),
         "knowledge_timestamp": datetime.now(UTC).isoformat()}
        for t in TICKERS
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 2. OHLCV + adjusted prices
# ─────────────────────────────────────────────────────────────────────────────

def _market_close_utc(ts: pd.Timestamp) -> str:
    """Approximate market close UTC for a trading day."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    close_et = datetime(ts.year, ts.month, ts.day, 16, 0, 0, tzinfo=ET)
    return close_et.astimezone(UTC).isoformat()


def fetch_ohlcv_one(ticker: str) -> pd.DataFrame:
    end = datetime.now(UTC)
    t = yf.Ticker(ticker)
    df = t.history(
        start=START_EQUITY.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        actions=True,
    )
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Rename yfinance adj close column
    if "adj_close" not in df.columns and "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})

    df["ticker"] = ticker
    df["frequency"] = "1D"
    df["event_timestamp"] = df.index.map(lambda ts: ts.isoformat())
    df["knowledge_timestamp"] = df.index.map(_market_close_utc)
    df["ingestion_timestamp"] = datetime.now(UTC).isoformat()
    df["source"] = "yfinance"
    return df.reset_index(drop=True)


async def fetch_all_ohlcv(tickers: list[str]) -> pd.DataFrame:
    log.info("📈  Downloading OHLCV for %d tickers (%s → %s) …",
             len(tickers), START_EQUITY.date(), datetime.now(UTC).date())
    loop = asyncio.get_running_loop()
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = await loop.run_in_executor(None, fetch_ohlcv_one, ticker)
        if not df.empty:
            frames.append(df)
            log.info("    [%2d/%d] %-6s  %d bars", i, len(tickers), ticker, len(df))
        else:
            log.warning("    [%2d/%d] %-6s  NO DATA", i, len(tickers), ticker)
        await asyncio.sleep(0.3)   # rate limit courtesy
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Technical indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators_one(ticker: str, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta
    df = ohlcv_df.copy()
    df = df.set_index(pd.to_datetime(df["event_timestamp"], utc=True))
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume",
                             "adj_close": "Adj Close"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Compute indicators
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)

    # Keep only indicator columns
    indicator_cols = [c for c in df.columns if c not in
                      ["Open", "High", "Low", "Close", "Volume"]]
    result = df[indicator_cols].copy()
    result["ticker"] = ticker
    result["event_timestamp"] = result.index.map(lambda ts: ts.isoformat())
    result["knowledge_timestamp"] = result.index.map(_market_close_utc)
    return result.reset_index(drop=True)


async def compute_all_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    log.info("📊  Computing technical indicators …")
    loop = asyncio.get_running_loop()
    frames = []
    for ticker in ohlcv["ticker"].unique():
        sub = ohlcv[ohlcv["ticker"] == ticker]
        df = await loop.run_in_executor(None, compute_indicators_one, ticker, sub)
        if not df.empty:
            frames.append(df)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d indicator rows across %d tickers", len(result),
             result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Valuation ratios (derived — no external API, computed from stored data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_valuations_one(
    ticker: str,
    ohlcv_sub: pd.DataFrame,   # OHLCV rows for this ticker only
    earn_sub: pd.DataFrame,    # earnings rows for this ticker only
    fin_sub: pd.DataFrame,     # financials rows for this ticker only
) -> pd.DataFrame:
    """
    Compute daily valuation ratios for one ticker using PiT-correct joins.

    For each trading day D, only financial data with knowledge_timestamp <= D
    is used — no look-ahead bias.

    Ratios computed:
      pe_ttm       — Price / Trailing-12-month EPS (last 4 quarterly reports, implied Q4)
      pb           — Price / Book value per share
      ps           — Price / TTM revenue per share  (4Q rolling with implied Q4; annual fallback)
      ev_ebitda    — Enterprise value / EBITDA          (last 10-K, annual convention)
      fcf_yield    — TTM FCF per share / Price   (4Q rolling with implied Q4; annual fallback)
      dividend_yield — Trailing-12-month dividends / Price
    """
    if ohlcv_sub.empty:
        return pd.DataFrame()

    # ── 1. Daily price series ─────────────────────────────────────────────────
    price = ohlcv_sub.copy()
    price["date"] = pd.to_datetime(price["event_timestamp"], utc=True)
    price = price.sort_values("date")[["date", "close", "dividends"]].reset_index(drop=True)
    if len(price) < 5:
        return pd.DataFrame()

    # Trailing-12-month dividends (rolling 252 trading days)
    price["div_ttm"] = price["dividends"].fillna(0).rolling(252, min_periods=1).sum()

    # ── 2. TTM EPS — rolling sum of last 4 quarterly EPS reports ─────────────
    # Guard: non-EDGAR tickers (ASML, ARM, etc.) have no "form" column when
    # fetch_earnings_one fell back to yfinance-only data.  Use all rows as
    # quarterly proxies in that case (yfinance earnings_history is quarterly).
    earn_q = (earn_sub[earn_sub["form"] == "10-Q"].copy()
              if "form" in earn_sub.columns else earn_sub.copy())
    ttm_eps_lkp = pd.DataFrame(columns=["kt", "ttm_eps"])
    if not earn_q.empty and "epsactual" in earn_q.columns:
        earn_q["kt"] = pd.to_datetime(earn_q["knowledge_timestamp"], utc=True)
        # Force numeric: guards against "None"/"nan" strings written by _merge in
        # older pipeline versions or after concat of mixed-type parquet partitions.
        earn_q["epsactual"] = pd.to_numeric(earn_q["epsactual"], errors="coerce")
        earn_q = earn_q.sort_values("kt").dropna(subset=["epsactual"])
        # min_periods=4 ensures we only report TTM when all 4 quarters are known —
        # fewer quarters underestimate TTM EPS and inflate P/E by up to 2–3×.
        earn_q["ttm_eps"] = earn_q["epsactual"].rolling(4, min_periods=4).sum()
        ttm_eps_lkp = earn_q[["kt", "ttm_eps"]].dropna().drop_duplicates("kt")

    # ── 3. Balance sheet: most recent filing of any type, PiT ────────────────
    bs_cols = ["total_equity", "shares_outstanding", "lt_debt", "cash"]
    fin_bs = fin_sub[fin_sub[bs_cols].notna().any(axis=1)].copy() if not fin_sub.empty else pd.DataFrame()
    bs_lkp = pd.DataFrame(columns=["kt"] + bs_cols)
    if not fin_bs.empty:
        fin_bs["kt"] = pd.to_datetime(fin_bs["knowledge_timestamp"], utc=True)
        fin_bs = fin_bs.sort_values("kt").drop_duplicates("kt", keep="last")
        bs_lkp = fin_bs[["kt"] + [c for c in bs_cols if c in fin_bs.columns]]

    # ── 4a. TTM flow metrics: rolling 4-quarter sum with implied Q4 ───────────
    # P/S and FCF yield use TTM (4-quarter sum) rather than the last 10-K annual
    # figure, which can be 6-11 months stale mid-fiscal-year.
    #
    # Naive rolling(4) over 10-Q rows only is WRONG: companies file 3 10-Qs per
    # year (Q1/Q2/Q3); the 4th slot would be Q3 of the prior fiscal year instead
    # of Q4 of the prior fiscal year.  _ttm_flow_series() computes an implied
    # Q4 = annual_10K − Q1 − Q2 − Q3 (same pattern as implied Q4 EPS) so the
    # rolling window is always 4 true consecutive fiscal quarters.
    ttm_rev_lkp = _ttm_flow_series(fin_sub, "revenue")
    ttm_fcf_lkp = _ttm_flow_series(fin_sub, "fcf")

    # ── 4b. Annual flow metrics: most recent 10-K, PiT ───────────────────────
    # Keep operating_income and D&A from annual 10-K for EV/EBITDA — this ratio
    # is conventionally computed on a trailing-twelve-month or annual basis and
    # using it from the most recent annual filing is industry standard.
    ann_flow_cols = ["revenue", "operating_income", "depreciation_amortization", "fcf"]
    fin_annual = (fin_sub[fin_sub["form"] == "10-K"].copy()
                  if (not fin_sub.empty and "form" in fin_sub.columns)
                  else pd.DataFrame())
    flow_lkp = pd.DataFrame(columns=["kt"] + ann_flow_cols)
    if not fin_annual.empty:
        fin_annual["kt"] = pd.to_datetime(fin_annual["knowledge_timestamp"], utc=True)
        fin_annual = fin_annual.sort_values("kt").drop_duplicates("kt", keep="last")
        flow_lkp = fin_annual[["kt"] + [c for c in ann_flow_cols if c in fin_annual.columns]]

    # ── 5. PiT joins via merge_asof ───────────────────────────────────────────
    base = price.copy()

    if not ttm_eps_lkp.empty:
        base = pd.merge_asof(base, ttm_eps_lkp, left_on="date", right_on="kt",
                             direction="backward").drop(columns=["kt"], errors="ignore")
    else:
        base["ttm_eps"] = None

    if not bs_lkp.empty:
        base = pd.merge_asof(base, bs_lkp, left_on="date", right_on="kt",
                             direction="backward").drop(columns=["kt"], errors="ignore")
    else:
        for c in bs_cols:
            base[c] = None

    if not flow_lkp.empty:
        base = pd.merge_asof(base, flow_lkp, left_on="date", right_on="kt",
                             direction="backward").drop(columns=["kt"], errors="ignore")
    else:
        for c in ann_flow_cols:
            base[c] = None

    # Merge TTM revenue (preferred for P/S) — falls back to annual revenue if
    # < 4 quarterly filings are available (non-EDGAR tickers, early history).
    if not ttm_rev_lkp.empty:
        base = pd.merge_asof(base, ttm_rev_lkp, left_on="date", right_on="kt",
                             direction="backward").drop(columns=["kt"], errors="ignore")
    else:
        base["ttm_revenue"] = None

    # Merge TTM FCF (preferred for FCF yield).
    if not ttm_fcf_lkp.empty:
        base = pd.merge_asof(base, ttm_fcf_lkp, left_on="date", right_on="kt",
                             direction="backward").drop(columns=["kt"], errors="ignore")
    else:
        base["ttm_fcf"] = None

    # ── 6. Derived metrics ────────────────────────────────────────────────────
    def _col(name: str) -> pd.Series:
        """Return column as float64 Series, all-NaN if column absent."""
        if name in base.columns:
            return pd.to_numeric(base[name], errors="coerce")
        return pd.Series(float("nan"), index=base.index, dtype=float)

    def _sdiv(a: pd.Series, b: pd.Series, lo: float = None, hi: float = None) -> pd.Series:
        """Safe element-wise division; clip to [lo, hi] if given."""
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        out = pd.Series(float("nan"), index=a.index, dtype=float)
        mask = b.notna() & (b != 0) & a.notna()
        out[mask] = a[mask] / b[mask]
        if lo is not None:
            out[out < lo] = float("nan")
        if hi is not None:
            out[out > hi] = float("nan")
        return out

    close  = pd.to_numeric(base["close"], errors="coerce")
    shares = _col("shares_outstanding")

    base["market_cap"]           = close * shares
    base["book_value_per_share"] = _sdiv(_col("total_equity"), shares)

    # P/S: prefer TTM revenue (4-quarter rolling sum) over annual 10-K revenue.
    # Fall back to annual if TTM is not available (< 4 quarters of history or
    # non-EDGAR tickers that only report annual data).
    ttm_rev  = _col("ttm_revenue")
    ann_rev  = _col("revenue")
    eff_rev  = ttm_rev.where(ttm_rev.notna(), ann_rev)   # TTM preferred, annual fallback
    base["revenue_per_share"]    = _sdiv(eff_rev,          shares)

    # FCF per share: prefer TTM FCF over annual 10-K FCF.
    ttm_fcf_s = _col("ttm_fcf")
    ann_fcf   = _col("fcf")
    eff_fcf   = ttm_fcf_s.where(ttm_fcf_s.notna(), ann_fcf)
    base["fcf_per_share"]        = _sdiv(eff_fcf,          shares)

    lt_debt = _col("lt_debt").fillna(0)
    cash    = _col("cash").fillna(0)
    base["enterprise_value"]     = base["market_cap"] + lt_debt - cash

    # P/E TTM
    base["pe_ttm"]  = _sdiv(close, _col("ttm_eps"),               lo=0,   hi=500)
    # P/B
    base["pb"]      = _sdiv(close, _col("book_value_per_share"),   lo=0,   hi=100)
    # P/S (uses TTM or annual revenue per share)
    base["ps"]      = _sdiv(close, _col("revenue_per_share"),      lo=0,   hi=200)
    # EV/EBITDA: uses annual operating_income + D&A (industry convention)
    ebitda = _col("operating_income").fillna(0) + _col("depreciation_amortization").fillna(0)
    ebitda[ebitda <= 0] = float("nan")
    base["ebitda"]    = ebitda
    base["ev_ebitda"] = _sdiv(_col("enterprise_value"), ebitda,    lo=0,   hi=300)
    # FCF yield (uses TTM or annual FCF per share)
    base["fcf_yield"]      = _sdiv(_col("fcf_per_share"), close,   lo=-0.5, hi=0.5)
    # Dividend yield
    base["dividend_yield"] = _sdiv(_col("div_ttm"), close,         lo=0,   hi=0.5)

    # ── 7. Final output ───────────────────────────────────────────────────────
    base["ticker"]               = ticker
    base["event_timestamp"]      = base["date"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    # knowledge_timestamp = market close (when close price becomes observable).
    # Using midnight UTC would introduce look-ahead if a consumer filters by
    # exact UTC time — consistent with OHLCV and indicators which use _market_close_utc.
    base["knowledge_timestamp"]  = base["date"].apply(_market_close_utc)
    base["source"]               = "computed"

    out_cols = [
        "ticker", "event_timestamp", "knowledge_timestamp", "source",
        "close", "market_cap", "enterprise_value", "shares_outstanding",
        "ttm_eps", "book_value_per_share", "revenue_per_share", "fcf_per_share",
        "revenue", "ttm_revenue", "fcf", "ttm_fcf", "ebitda",
        "pe_ttm", "pb", "ps", "ev_ebitda", "fcf_yield", "dividend_yield",
    ]
    out = base[[c for c in out_cols if c in base.columns]].copy()
    str_cols = {"ticker", "event_timestamp", "knowledge_timestamp", "source"}
    for col in out.columns:
        if col not in str_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def compute_all_valuations(tickers: list[str],
                           ohlcv_all: pd.DataFrame,
                           earn_all: pd.DataFrame,
                           fin_all: pd.DataFrame) -> pd.DataFrame:
    """Compute daily valuation ratios for all tickers. Tables are pre-loaded."""
    log.info("💹  Computing valuation ratios for %d tickers …", len(tickers))
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = compute_valuations_one(
            ticker,
            ohlcv_all[ohlcv_all["ticker"] == ticker],
            earn_all[earn_all["ticker"] == ticker],
            fin_all[fin_all["ticker"] == ticker],
        )
        if not df.empty:
            frames.append(df)
        if i % 100 == 0:
            log.info("    valuations %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d valuation rows across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Universe membership changes (S&P 500 + NASDAQ 100 — Wikipedia)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_universe_changes() -> pd.DataFrame:
    """
    Fetch historical S&P 500 and NASDAQ 100 membership changes from Wikipedia.

    Returns one row per event (addition and removal are separate rows), covering
    the full available Wikipedia history — typically back to ~2000 for S&P 500
    and several years for NASDAQ 100.

    Schema: ticker, company, index_name, action ("added"/"removed"),
            event_date, event_timestamp, knowledge_timestamp, reason, source

    Used for:
      1. PiT-correct universe reconstruction — what stocks were in the index
         on any given date for backtesting.
      2. Delisted/removed ticker discovery — survivorship bias correction.
    """
    sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100",                   "NASDAQ 100"),
    ]
    records = []
    _EMPTY = {"", "—", "–", "N/A", "n/a", "None", "none"}

    for url, index_name in sources:
        try:
            resp = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"id": "changes"})
            if not table:
                # Fallback: Wikipedia occasionally renames the table ID.
                # Try finding any wikitable whose preceding header mentions "changes".
                for hdr in soup.find_all(["h2", "h3"]):
                    if "change" in hdr.get_text(strip=True).lower():
                        table = hdr.find_next("table", class_="wikitable")
                        if table:
                            break
            if not table:
                log.warning("  No #changes table found for %s — skipping", index_name)
                continue

            # Use <tbody> rows only — skips the two-level <thead>
            tbody = table.find("tbody") or table
            rows  = tbody.find_all("tr")

            for row in rows:
                cells = [c.get_text(separator=" ", strip=True)
                         for c in row.find_all("td")]
                if len(cells) < 4:
                    continue

                # Col layout: 0=date, 1=add_ticker, 2=add_company,
                #             3=rem_ticker, 4=rem_company, 5=reason
                date_text   = cells[0]
                event_date  = pd.to_datetime(date_text, errors="coerce")
                if pd.isna(event_date):
                    continue
                event_date_str = event_date.date().isoformat()
                ts = f"{event_date_str}T00:00:00+00:00"

                add_ticker  = cells[1].replace(".", "-").strip() if len(cells) > 1 else ""
                add_company = cells[2].strip()                   if len(cells) > 2 else ""
                rem_ticker  = cells[3].replace(".", "-").strip() if len(cells) > 3 else ""
                rem_company = cells[4].strip()                   if len(cells) > 4 else ""
                reason      = cells[5].strip()                   if len(cells) > 5 else ""

                if add_ticker and add_ticker not in _EMPTY:
                    records.append({
                        "ticker":              add_ticker,
                        "company":             add_company,
                        "index_name":          index_name,
                        "action":              "added",
                        "event_date":          event_date_str,
                        "event_timestamp":     ts,
                        "knowledge_timestamp": ts,
                        "reason":              reason,
                        "source":              "Wikipedia",
                    })

                if rem_ticker and rem_ticker not in _EMPTY:
                    records.append({
                        "ticker":              rem_ticker,
                        "company":             rem_company,
                        "index_name":          index_name,
                        "action":              "removed",
                        "event_date":          event_date_str,
                        "event_timestamp":     ts,
                        "knowledge_timestamp": ts,
                        "reason":              reason,
                        "source":              "Wikipedia",
                    })

        except Exception as exc:
            log.warning("  Universe changes %s failed: %s", index_name, exc)

    if not records:
        return pd.DataFrame()

    df = (pd.DataFrame(records)
            .sort_values("event_date")
            .drop_duplicates(subset=["ticker", "index_name", "action", "event_date"],
                             keep="last")
            .reset_index(drop=True))
    log.info("    → %d membership events  (added=%d  removed=%d)",
             len(df),
             (df["action"] == "added").sum(),
             (df["action"] == "removed").sum())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Short interest (yfinance — current + prior-month snapshots)
#
# Note: FINRA's consolidated short interest CDN is access-restricted (HTTP 403).
# yfinance provides current settlement date + prior month in Ticker.info,
# giving two data points per fetch.  Running weekly builds a time series.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_short_interest_one(ticker: str) -> pd.DataFrame:
    """
    Fetch current and prior-month short interest for one ticker via yfinance.

    Returns 1–2 rows:
      - current  : settlement_date from dateShortInterest, with days_to_cover
                   and short_pct_float
      - prior    : settlement_date from sharesShortPreviousMonthDate, shares only

    Columns: ticker, settlement_date, event_timestamp, knowledge_timestamp,
             shares_short, days_to_cover, short_pct_float, float_shares, source
    """
    records: list[dict] = []
    try:
        info = yf.Ticker(ticker).info
        # PiT: FINRA publishes short interest ~2-3 weeks after the settlement date.
        # yfinance serves the data only AFTER publication, so knowledge_timestamp
        # = now (fetch time) is correct.  Using settlement_date as knowledge_timestamp
        # would introduce a ~14-21 day look-ahead bias in backtests.
        fetch_kt = datetime.now(UTC).isoformat()

        shares_short    = info.get("sharesShort")
        short_date_ts   = info.get("dateShortInterest")
        short_ratio     = info.get("shortRatio")
        short_pct_float = info.get("shortPercentOfFloat")
        float_shares    = info.get("floatShares")

        if shares_short and short_date_ts:
            settle = pd.to_datetime(short_date_ts, unit="s").date()
            ts = f"{settle.isoformat()}T00:00:00+00:00"
            records.append({
                "ticker":              ticker,
                "settlement_date":     settle.isoformat(),
                "event_timestamp":     ts,
                "knowledge_timestamp": fetch_kt,   # known when yfinance published it
                "shares_short":        int(shares_short),
                "days_to_cover":       float(short_ratio)     if short_ratio     else None,
                "short_pct_float":     float(short_pct_float) if short_pct_float else None,
                "float_shares":        int(float_shares)      if float_shares    else None,
                "source":              "yfinance",
            })

        prior_shares  = info.get("sharesShortPriorMonth")
        prior_date_ts = info.get("sharesShortPreviousMonthDate")
        if prior_shares and prior_date_ts:
            prior_settle = pd.to_datetime(prior_date_ts, unit="s").date()
            pts = f"{prior_settle.isoformat()}T00:00:00+00:00"
            records.append({
                "ticker":              ticker,
                "settlement_date":     prior_settle.isoformat(),
                "event_timestamp":     pts,
                "knowledge_timestamp": fetch_kt,   # prior month also known at fetch time
                "shares_short":        int(prior_shares),
                "days_to_cover":       None,
                "short_pct_float":     None,
                "float_shares":        int(float_shares) if float_shares else None,
                "source":              "yfinance_prior",
            })

    except Exception as exc:
        log.debug("Short interest %s: %s", ticker, exc)

    return pd.DataFrame(records) if records else pd.DataFrame()


async def fetch_all_short_interest(tickers: list[str]) -> pd.DataFrame:
    log.info("📉  Fetching short interest for %d tickers (yfinance) …", len(tickers))
    loop   = asyncio.get_running_loop()
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = await loop.run_in_executor(None, fetch_short_interest_one, ticker)
        if not df.empty:
            frames.append(df)
        await asyncio.sleep(0.25)
        if i % 100 == 0:
            log.info("    short_interest %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d short interest rows across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4d. Dividend history (derived — computed from OHLCV dividends + earnings EPS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dividends_one(
    ticker: str,
    ohlcv_sub: pd.DataFrame,   # OHLCV rows for this ticker only
    earn_sub: pd.DataFrame,    # earnings rows for this ticker only
) -> pd.DataFrame:
    """
    Compute annual dividend summary for one ticker.

    Returns one row per complete calendar year (≥ 200 trading bars) with:
      annual_dps         - total dividends per share paid in the calendar year
      dps_growth_1y      - year-over-year % change in annual DPS
      dps_growth_3y      - 3-year CAGR of annual DPS
      dps_growth_5y      - 5-year CAGR of annual DPS
      payout_ratio       - annual_dps / annual_eps (sum of 4 quarterly 10-Q EPS)
      consecutive_growth - streak of consecutive years with increasing DPS
      is_payer           - 1 if annual_dps > 0 else 0

    Knowledge-timestamp = Dec 31 of each year (annual total known at year-end).
    PiT-correct: only data available on Dec 31 is used.
    """
    if ohlcv_sub.empty:
        return pd.DataFrame()

    price = ohlcv_sub.copy()
    price["date"] = pd.to_datetime(price["event_timestamp"], utc=True)
    price = price.sort_values("date")

    if "dividends" not in price.columns:
        return pd.DataFrame()

    price["dividends"] = pd.to_numeric(price["dividends"], errors="coerce").fillna(0)
    price["cal_year"] = price["date"].dt.year

    # Annual DPS and bar counts per calendar year
    annual_dps = price.groupby("cal_year")["dividends"].sum()
    bar_counts  = price.groupby("cal_year").size()

    # Only include complete calendar years (≥ 200 trading bars)
    # This excludes partial first/last years and the current in-progress year
    complete_years = bar_counts[bar_counts >= 200].index
    annual_dps = annual_dps[annual_dps.index.isin(complete_years)]

    if annual_dps.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"year": annual_dps.index.astype(int),
                        "annual_dps": annual_dps.values})
    df = df.sort_values("year").reset_index(drop=True)

    # Growth rates via CAGR
    def _cagr(series: pd.Series, n: int) -> pd.Series:
        prior  = series.shift(n)
        result = pd.Series(float("nan"), index=series.index, dtype=float)
        mask   = prior.notna() & (prior > 0) & series.notna() & (series >= 0)
        result[mask] = (series[mask] / prior[mask]) ** (1.0 / n) - 1
        return result

    df["dps_growth_1y"] = _cagr(df["annual_dps"], 1)
    df["dps_growth_3y"] = _cagr(df["annual_dps"], 3)
    df["dps_growth_5y"] = _cagr(df["annual_dps"], 5)

    # Consecutive years of DPS growth (streak ending at each row)
    grew = df["annual_dps"] > df["annual_dps"].shift(1)
    streak, current = [], 0
    for g in grew:
        current = (current + 1) if g else 0
        streak.append(current)
    df["consecutive_growth"] = streak

    df["is_payer"] = (df["annual_dps"] > 0).astype(int)

    # Payout ratio: annual_dps / annual_eps (sum of 4 quarterly 10-Q EPS)
    df["payout_ratio"] = float("nan")
    if not earn_sub.empty and "epsactual" in earn_sub.columns:
        eq = earn_sub[earn_sub["form"] == "10-Q"].copy() \
            if "form" in earn_sub.columns else earn_sub.copy()
        if not eq.empty:
            eq["epsactual"] = pd.to_numeric(eq["epsactual"], errors="coerce")
            eq = eq.dropna(subset=["epsactual"])
            # Use period_end (fiscal quarter end) rather than knowledge_timestamp
            # to assign EPS to the correct fiscal year.  The implied Q4 filing
            # (filed in Feb of the following year) has knowledge_timestamp.year =
            # next_year, which would put it in the wrong payout-ratio bucket and
            # make the denominator 3 quarters short → payout overstated by 33%.
            eq["period_ts"] = pd.to_datetime(eq["period_end"], utc=True, errors="coerce")
            eq = eq.dropna(subset=["period_ts"])
            eq["eps_year"] = eq["period_ts"].dt.year
            # Dedup: if the same period_end appears multiple times (restated),
            # keep the most recently filed version before summing.
            if "knowledge_timestamp" in eq.columns:
                eq = eq.sort_values("knowledge_timestamp").drop_duplicates(
                    subset=["period_end"], keep="last"
                )
            ann_eps = eq.groupby("eps_year")["epsactual"].sum().reset_index()
            ann_eps.columns = ["year", "annual_eps"]
            df = df.merge(ann_eps, on="year", how="left")
            mask = df["annual_eps"].notna() & (df["annual_eps"] > 0)
            df.loc[mask, "payout_ratio"] = (
                df.loc[mask, "annual_dps"] / df.loc[mask, "annual_eps"]
            )
            # Clamp: payout > 200% or < 0 are data artifacts
            df.loc[df["payout_ratio"] > 2.0, "payout_ratio"] = float("nan")
            df.loc[df["payout_ratio"] < 0,   "payout_ratio"] = float("nan")
            df = df.drop(columns=["annual_eps"])

    # Timestamps: Dec 31 of each year (full-year total is known at year-end)
    df["event_timestamp"]     = df["year"].apply(lambda y: f"{y}-12-31T00:00:00+00:00")
    df["knowledge_timestamp"] = df["event_timestamp"]
    df["ticker"] = ticker
    df["source"] = "computed"

    out_cols = [
        "ticker", "year", "event_timestamp", "knowledge_timestamp", "source",
        "annual_dps", "dps_growth_1y", "dps_growth_3y", "dps_growth_5y",
        "payout_ratio", "consecutive_growth", "is_payer",
    ]
    return df[[c for c in out_cols if c in df.columns]].reset_index(drop=True)


def compute_all_dividends(tickers: list[str],
                          ohlcv_all: pd.DataFrame,
                          earn_all: pd.DataFrame) -> pd.DataFrame:
    """Compute annual dividend history for all tickers. Tables are pre-loaded."""
    log.info("💰  Computing dividend history for %d tickers …", len(tickers))
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = compute_dividends_one(
            ticker,
            ohlcv_all[ohlcv_all["ticker"] == ticker],
            earn_all[earn_all["ticker"] == ticker],
        )
        if not df.empty:
            frames.append(df)
        if i % 100 == 0:
            log.info("    dividends %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d dividend rows across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Macro data (FRED via direct HTTP — no API key needed for basic series)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_fred_series(code: str, series_id: str) -> pd.DataFrame:
    """
    Fetch a FRED series via the public observations endpoint.
    Uses FRED API if key available, else falls back to public endpoint.
    """
    import os
    api_key = os.environ.get("FRED_API_KEY", "")
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&observation_start={START_MACRO.strftime('%Y-%m-%d')}"
        f"&observation_end={end_date}"
        f"&api_key={api_key}&file_type=json"
    ) if api_key else (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    )

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)

        records = []
        if api_key and resp.status_code == 200:
            data = resp.json()
            for obs in data.get("observations", []):
                try:
                    val = float(obs["value"])
                    records.append({
                        "indicator_code":      code,
                        "series_id":           series_id,
                        "event_timestamp":     obs["date"],
                        "knowledge_timestamp": obs["realtime_start"],
                        "value":               val,
                        "revision_number":     0,
                        "source":              "FRED_API",
                    })
                except (ValueError, KeyError):
                    continue
        elif not api_key and resp.status_code == 200:
            # CSV fallback — FRED CSV has columns: DATE, <series_id>
            from io import StringIO
            df_raw = pd.read_csv(StringIO(resp.text))
            # Normalise column names (FRED uses DATE + series_id as header)
            df_raw.columns = [c.strip() for c in df_raw.columns]
            # FRED CSV uses 'observation_date' or 'DATE' for the date column
            date_col  = next((c for c in df_raw.columns
                              if c.upper() in ("DATE", "OBSERVATION_DATE")), None)
            val_col   = next((c for c in df_raw.columns
                              if c.upper() not in ("DATE", "OBSERVATION_DATE")), None)
            if date_col is None or val_col is None:
                log.warning("    FRED %s: unexpected CSV columns %s", code, list(df_raw.columns))
            else:
                df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
                df_raw = df_raw.dropna(subset=[date_col])
                df_raw = df_raw[df_raw[date_col] >= pd.Timestamp(START_MACRO.date())]
                for _, row in df_raw.iterrows():
                    raw_val = str(row[val_col]).strip()
                    if raw_val in (".", "", "nan"):
                        continue
                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue
                    # knowledge_timestamp = event_timestamp + publication lag.
                    # Monthly macro data is NOT available on the reference date itself;
                    # it is released days-to-weeks later.  Using the event date directly
                    # would introduce look-ahead bias in PiT-correct backtests.
                    pub_lag  = _FRED_PUB_LAG.get(code, 35)
                    kt_date  = row[date_col].date() + pd.Timedelta(days=pub_lag)
                    # Never exceed today
                    kt_date  = min(kt_date, pd.Timestamp.today().date())
                    records.append({
                        "indicator_code":      code,
                        "series_id":           series_id,
                        "event_timestamp":     str(row[date_col].date()),
                        "knowledge_timestamp": str(kt_date),
                        "value":               val,
                        "revision_number":     0,
                        "source":              "FRED_CSV",
                    })
        return pd.DataFrame(records)
    except Exception as exc:
        log.warning("    FRED %s (%s): %s", code, series_id, exc)
        return pd.DataFrame()


async def fetch_all_macro() -> pd.DataFrame:
    log.info("🌍  Fetching macro data from FRED …")
    tasks = [fetch_fred_series(code, sid) for code, sid in FRED_SERIES.items()]
    frames = await asyncio.gather(*tasks)
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        log.warning("    No macro data returned from FRED")
        return pd.DataFrame()
    result = pd.concat(non_empty, ignore_index=True)
    if not result.empty:
        # ── Derive CPI_YOY (true year-over-year %) from CPI_INDEX ────────────
        cpi = result[result["indicator_code"] == "CPI_INDEX"].copy()
        if not cpi.empty:
            cpi["event_ts"] = pd.to_datetime(cpi["event_timestamp"], errors="coerce")
            cpi = cpi.sort_values("event_ts").dropna(subset=["event_ts"])
            cpi["value_12m_ago"] = cpi["value"].shift(12)
            cpi_yoy = cpi.dropna(subset=["value_12m_ago"]).copy()
            cpi_yoy["value"] = (
                (cpi_yoy["value"] - cpi_yoy["value_12m_ago"]) / cpi_yoy["value_12m_ago"] * 100
            ).round(4)
            cpi_yoy["indicator_code"] = "CPI_YOY"
            cpi_yoy["series_id"]      = "CPIAUCSL_YOY"
            cpi_yoy = cpi_yoy.drop(columns=["event_ts", "value_12m_ago"])
            result = pd.concat([result, cpi_yoy], ignore_index=True)

        for code, grp in result.groupby("indicator_code"):
            log.info("    %-20s  %d observations", code, len(grp))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Insider trades (OpenInsider)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_insider_trades(ticker: str) -> pd.DataFrame:
    url = (
        f"http://openinsider.com/screener?s={ticker}"
        f"&fd=1826&fdr=&td=0&tdr=&xp=1&xs=1"
        f"&sortcol=0&cnt=100&action=1"
    )
    try:
        async with httpx.AsyncClient(
            timeout=20, follow_redirects=True,
            headers={"User-Agent": "ResearchCrawler research@example.com"},
        ) as client:
            resp = await client.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            return pd.DataFrame()
        headers = [th.get_text(strip=True).replace('\xa0', ' ').strip().lower() for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue
            row = {h: cells[i].get_text(strip=True) for i, h in enumerate(headers) if i < len(cells)}
            row["ticker_queried"] = ticker
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as exc:
        log.debug("Insider %s: %s", ticker, exc)
        return pd.DataFrame()


async def fetch_all_insider(tickers: list[str]) -> pd.DataFrame:
    log.info("🔍  Fetching insider trades for %d tickers …", len(tickers))
    frames = []
    # Limit to 5 tickers to be courteous to OpenInsider
    for ticker in tickers[:5]:
        df = await fetch_insider_trades(ticker)
        if not df.empty:
            frames.append(df)
            log.info("    %-6s  %d insider trades", ticker, len(df))
        await asyncio.sleep(2.0)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Earnings history — SEC EDGAR XBRL (historical) + yfinance (estimates)
# ─────────────────────────────────────────────────────────────────────────────

# EDGAR ticker→CIK map (loaded once, shared across all calls)
_EDGAR_CIK_MAP: dict[str, str] = {}

def _load_edgar_cik_map() -> dict[str, str]:
    """Fetch the EDGAR ticker→CIK mapping (free, no key required)."""
    global _EDGAR_CIK_MAP
    if _EDGAR_CIK_MAP:
        return _EDGAR_CIK_MAP
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={**_EDGAR_HEADERS, "Host": "www.sec.gov"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        _EDGAR_CIK_MAP = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
        log.info("    EDGAR CIK map loaded: %d tickers", len(_EDGAR_CIK_MAP))
    except Exception as exc:
        log.warning("    EDGAR CIK map fetch failed: %s", exc)
    return _EDGAR_CIK_MAP


def fetch_earnings_edgar(ticker: str) -> pd.DataFrame:
    """
    Fetch historical quarterly EPS (diluted) from SEC EDGAR XBRL.
    Returns actual reported EPS going back to ~2010 for S&P 500 companies.
    No API key required. Rate-limit: 10 req/s max per SEC fair-use policy.
    """
    cik_map = _load_edgar_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return pd.DataFrame()
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(
            url,
            headers=_EDGAR_HEADERS,
            timeout=30,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        facts = resp.json()
        # Try diluted EPS first, fall back to basic EPS
        eps_data = (
            facts.get("facts", {})
                 .get("us-gaap", {})
                 .get("EarningsPerShareDiluted",
                      facts.get("facts", {})
                           .get("us-gaap", {})
                           .get("EarningsPerShareBasic", {}))
        )
        if not eps_data:
            return pd.DataFrame()

        rows = []
        for unit_type, entries in eps_data.get("units", {}).items():
            for e in entries:
                # Accept 10-Q, 10-K and their amendments (10-Q/A, 10-K/A).
                # Normalise amended forms to base form so restatements replace
                # originals via PK dedup on (ticker, period_end, form).
                raw_form = e.get("form", "")
                base_form = raw_form.replace("/A", "")
                if base_form not in ("10-Q", "10-K"):
                    continue
                form = base_form  # store normalised form for consistent dedup
                # "end" = period end date; "filed" = when SEC received the filing
                period_end = e.get("end")
                filed_date = e.get("filed")
                val        = e.get("val")
                if not period_end or val is None:
                    continue

                # ── CRITICAL: filter by period duration ──────────────────────
                # XBRL filings include BOTH quarterly (3-month) and YTD cumulative
                # entries with the same period_end.  For Q3 specifically, there is
                # a 3-month entry (Q3 EPS) and a 9-month entry (YTD EPS) — both
                # with period_end = Q3 end date and form = "10-Q".  Without this
                # filter, the dedup may keep the 9-month YTD value, causing the
                # TTM P/E rolling sum to triple-count Q1/Q2 earnings.
                # If start_date is absent for a 10-Q, we cannot verify it is
                # quarterly so we skip it (absence typically indicates cumulative).
                start_date = e.get("start")
                if form == "10-Q" and not start_date:
                    continue   # cannot verify duration — skip to be safe
                if start_date:
                    try:
                        duration = (pd.Timestamp(period_end) - pd.Timestamp(start_date)).days
                        if form == "10-Q" and not (60 <= duration <= 105):
                            continue   # skip YTD cumulative entries in quarterly filings
                        if form == "10-K" and not (330 <= duration <= 400):
                            continue   # skip transition-period / partial-year filings
                    except Exception:
                        pass   # unparseable dates — let the row through

                try:
                    epsactual = float(val)
                except (ValueError, TypeError):
                    continue   # skip non-numeric XBRL values (rare but possible)
                rows.append({
                    "ticker":              ticker,
                    "period_end":          period_end,
                    "epsactual":           epsactual,
                    "epsestimate":         None,   # EDGAR has no consensus estimates
                    "epsdifference":       None,
                    "surprisepercent":     None,
                    "form":                form,
                    "event_timestamp":     period_end,
                    # PiT: the value was only known after the SEC filing date
                    # PiT: use filing date when available. If the XBRL entry lacks
                    # a "filed" field (rare), fall back to period_end + 45 days so
                    # we don't accidentally front-run the earnings announcement.
                    "knowledge_timestamp": filed_date or (
                        pd.Timestamp(period_end) + pd.Timedelta(days=45)
                    ).date().isoformat(),
                    "source":              "SEC_EDGAR",
                })
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Keep only rows within our study window
        df["period_ts"] = pd.to_datetime(df["period_end"], errors="coerce")
        df = df[df["period_ts"] >= pd.Timestamp(START_EQUITY.date())]
        # De-duplicate: same (ticker, period_end, form) — keep latest filing
        df = df.sort_values("knowledge_timestamp").drop_duplicates(
            subset=["ticker", "period_end", "form"], keep="last"
        )
        return df.drop(columns=["period_ts"]).reset_index(drop=True)
    except Exception as exc:
        log.debug("EDGAR earnings %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_earnings_one(ticker: str) -> pd.DataFrame:
    """
    Fetch earnings: EDGAR for full history, yfinance for recent estimates.
    Merges both sources — EDGAR actual EPS wins on conflict.
    """
    # --- SEC EDGAR: actual reported EPS, full history since ~2010 ---
    edgar_df = fetch_earnings_edgar(ticker)

    # --- yfinance: last 4 quarters with analyst estimates & surprise ---
    yf_df = pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is not None and not hist.empty:
            hist = hist.copy().reset_index()
            hist["ticker"] = ticker
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
            # Standardise date column name
            for c in hist.columns:
                if c.lower() in ("quarter", "period", "date", "index") or \
                   "period" in c.lower() or "quarter" in c.lower():
                    hist = hist.rename(columns={c: "period_end"})
                    break
            if "period_end" in hist.columns:
                hist["period_end"] = hist["period_end"].astype(str).str[:10]
                hist["event_timestamp"] = hist["period_end"]
                # PiT: knowledge_timestamp = period_end + 45 days (conservative
                # estimate for when quarterly earnings are publicly announced).
                # Companies typically report 3-6 weeks after quarter end; 45 days
                # is safely conservative.  Using period_end directly would be
                # look-ahead bias of ~3-4 weeks for non-EDGAR tickers.
                _today = datetime.now(UTC).date().isoformat()
                hist["knowledge_timestamp"] = hist["period_end"].apply(
                    lambda d: min(
                        (pd.Timestamp(d) + pd.Timedelta(days=45)).date().isoformat(),
                        _today,
                    )
                )
                hist["source"] = "yfinance"
                # Set form="10-Q" so the earnings PK (ticker, period_end, form)
                # deduplicates correctly if EDGAR data is later added for the same ticker.
                if "form" not in hist.columns:
                    hist["form"] = "10-Q"
                yf_df = hist
    except Exception as exc:
        log.debug("yfinance earnings %s: %s", ticker, exc)

    # Merge: prefer EDGAR for actuals, yfinance for estimates
    if edgar_df.empty and yf_df.empty:
        return pd.DataFrame()
    if edgar_df.empty:
        return yf_df
    if yf_df.empty:
        return edgar_df

    # Merge on period_end: fill in estimates from yfinance where available
    merged = edgar_df.copy()
    yf_lookup = yf_df.set_index("period_end")
    for col in ("epsestimate", "epsdifference", "surprisepercent"):
        if col in yf_lookup.columns:
            merged[col] = merged["period_end"].map(yf_lookup[col])
    return merged


async def fetch_all_earnings(tickers: list[str]) -> pd.DataFrame:
    log.info("💰  Fetching earnings history (SEC EDGAR + yfinance) …")
    loop = asyncio.get_running_loop()
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = await loop.run_in_executor(None, fetch_earnings_one, ticker)
        if not df.empty:
            frames.append(df)
        # SEC fair-use: max 10 req/s — 0.15s delay keeps us well under
        await asyncio.sleep(0.15)
        if i % 50 == 0:
            log.info("    earnings %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d earnings records across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6a-extra. Implied Q4 EPS (derived: annual EPS − Q1 − Q2 − Q3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_implied_q4_eps_one(ticker: str, earn_sub: pd.DataFrame) -> pd.DataFrame:
    """
    For each fiscal year with a 10-K, derive implied Q4 EPS:
        implied_Q4 = annual_EPS(10-K) − Q1_EPS − Q2_EPS − Q3_EPS

    Companies don't file a separate 10-Q for fiscal Q4, so this implied value
    fills the gap in the TTM P/E rolling-4-quarter calculation.

    The synthetic row is stored with:
        form               = '10-Q'        (picked up by TTM calculation)
        period_end         = 10-K period_end
        knowledge_timestamp = 10-K knowledge_timestamp  (PiT-correct)
        source             = 'implied_q4'
    """
    if earn_sub is None or earn_sub.empty:
        return pd.DataFrame()

    eps_col = next((c for c in earn_sub.columns if "epsactual" in c.lower()), None)
    if eps_col is None:
        return pd.DataFrame()

    earn_sub = earn_sub.copy()
    earn_sub["_period_ts"] = pd.to_datetime(earn_sub["period_end"], errors="coerce", utc=True)

    # Non-EDGAR tickers (ASML, ARM, etc.) may have no "form" column (yfinance-only).
    # For implied Q4 we need both annual (10-K) and quarterly (10-Q) rows.
    # If form is absent, we cannot distinguish annual from quarterly, so skip.
    if "form" not in earn_sub.columns:
        return pd.DataFrame()
    annual    = earn_sub[earn_sub["form"] == "10-K"].dropna(subset=["_period_ts", eps_col])
    quarterly = earn_sub[earn_sub["form"] == "10-Q"].dropna(subset=["_period_ts", eps_col])

    if annual.empty or quarterly.empty:
        return pd.DataFrame()

    implied_rows = []
    for _, ann in annual.iterrows():
        fiscal_end = ann["_period_ts"]
        annual_eps = pd.to_numeric(ann[eps_col], errors="coerce")
        if pd.isna(annual_eps):
            continue

        # Find Q1+Q2+Q3 in the 9 months before fiscal year end
        window = quarterly[
            (quarterly["_period_ts"] > fiscal_end - pd.Timedelta(days=280)) &
            (quarterly["_period_ts"] < fiscal_end)
        ]
        if len(window) < 3:
            continue   # not enough quarters — skip to avoid spurious estimates

        # Use the 3 quarters closest to fiscal year end
        window = window.nlargest(3, "_period_ts")
        q_eps = pd.to_numeric(window[eps_col], errors="coerce").dropna()
        if len(q_eps) < 3:
            continue
        q1q2q3 = q_eps.sum()

        implied_q4 = float(annual_eps) - float(q1q2q3)

        # Sanity: implied Q4 magnitude should be roughly in line with the average
        # per-quarter magnitude of Q1+Q2+Q3.  Use per-quarter absolute average so
        # loss companies (where q1q2q3 < 0) are not incorrectly rejected — computing
        # avg_q = abs(sum)/3 understates the per-quarter magnitude when quarters have
        # mixed signs, while sum of abs / 3 is the true average quarter magnitude.
        avg_q_abs = sum(abs(float(x)) for x in q_eps) / len(q_eps)
        if avg_q_abs > 0 and abs(implied_q4) > avg_q_abs * 6:
            continue   # implausible magnitude — skip (e.g. massive write-down quarter)

        implied_rows.append({
            "ticker":               ticker,
            "period_end":           ann["period_end"],
            "epsactual":            round(implied_q4, 4),
            "epsestimate":          None,
            "epsdifference":        None,
            "surprisepercent":      None,
            "form":                 "10-Q",
            "event_timestamp":      ann["period_end"],
            "knowledge_timestamp":  ann["knowledge_timestamp"],
            "source":               "implied_q4",
        })

    if not implied_rows:
        return pd.DataFrame()
    return pd.DataFrame(implied_rows)


def compute_all_implied_q4_eps(tickers: list[str], earn_all: pd.DataFrame) -> pd.DataFrame:
    """Compute implied Q4 EPS rows for all tickers. earn_all is pre-loaded."""
    log.info("📐  Computing implied Q4 EPS for %d tickers …", len(tickers))
    frames = []
    for ticker in tickers:
        df = compute_implied_q4_eps_one(
            ticker,
            earn_all[earn_all["ticker"] == ticker] if not earn_all.empty else pd.DataFrame(),
        )
        if not df.empty:
            frames.append(df)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d implied Q4 rows across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6a-extra-2. yfinance financial fallback (for non-EDGAR/foreign filers)
# ─────────────────────────────────────────────────────────────────────────────

# Tickers that don't file 10-K/10-Q with SEC EDGAR (foreign filers, recent IPOs)
NON_EDGAR_TICKERS = {"ASML", "ARM", "PDD", "CCEP", "FER"}

_YF_INC_MAP: dict[str, list[str]] = {
    "revenue":                   ["Total Revenue", "Revenue"],
    "gross_profit":              ["Gross Profit"],
    "operating_income":          ["Operating Income", "EBIT"],
    "net_income":                ["Net Income", "Net Income Common Stockholders"],
    "interest_expense":          ["Interest Expense", "Interest Expense Non Operating"],
    "depreciation_amortization": ["Reconciled Depreciation",
                                  "Depreciation And Amortization"],
}
_YF_BS_MAP: dict[str, list[str]] = {
    "total_assets":        ["Total Assets"],
    "total_equity":        ["Stockholders Equity",
                            "Total Equity Gross Minority Interest"],
    "lt_debt":             ["Long Term Debt",
                            "Long Term Debt And Capital Lease Obligation"],
    "cash":                ["Cash And Cash Equivalents",
                            "Cash Cash Equivalents And Short Term Investments"],
    "shares_outstanding":  ["Ordinary Shares Number", "Share Issued"],
}
_YF_CF_MAP: dict[str, list[str]] = {
    "operating_cf": ["Operating Cash Flow"],
    "capex":        ["Capital Expenditure", "Purchase Of Ppe"],
}


def fetch_financials_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetch annual fundamental data from yfinance.

    Used as:
      1. Primary source for non-EDGAR filers (ASML, ARM, PDD, CCEP, FER).
      2. Fallback for EDGAR-covered tickers with sparse balance sheet data.

    PiT knowledge_timestamp: 90 days after fiscal year end (conservative
    estimate for foreign 20-F filers; real lag is 90-120 days).
    """
    try:
        t = yf.Ticker(ticker)
        inc = t.financials       # rows=metrics, cols=fiscal year-end dates
        bs  = t.balance_sheet
        cf  = t.cashflow

        def _pick(df_src, names: list[str], date_col) -> float | None:
            if df_src is None or df_src.empty:
                return None
            for name in names:
                if name in df_src.index and date_col in df_src.columns:
                    val = df_src.at[name, date_col]
                    try:
                        f = float(val)
                        return f if not pd.isna(f) else None
                    except Exception:
                        pass
            return None

        # Union of all date columns across all three statements
        all_dates: set = set()
        for src in (inc, bs, cf):
            if src is not None and not src.empty:
                all_dates.update(src.columns.tolist())

        if not all_dates:
            return pd.DataFrame()

        rows = []
        now_utc = datetime.now(UTC)
        for date_col in sorted(all_dates):
            try:
                period_ts = pd.Timestamp(date_col)
            except Exception:
                continue
            if period_ts < pd.Timestamp("2010-01-01"):
                continue

            # PiT: 90 days after fiscal year end, capped at today
            filing_ts = min(period_ts + pd.Timedelta(days=90),
                            pd.Timestamp(now_utc.date()))
            period_str  = period_ts.date().isoformat()
            filing_str  = filing_ts.isoformat()

            row: dict = {
                "ticker":               ticker,
                "period_end":           period_str,
                "form":                 "10-K",   # annual equivalent
                "event_timestamp":      period_str,
                "knowledge_timestamp":  filing_str,
                "source":               "yfinance",
            }

            for our_col, names in _YF_INC_MAP.items():
                row[our_col] = _pick(inc, names, date_col)
            for our_col, names in _YF_BS_MAP.items():
                row[our_col] = _pick(bs,  names, date_col)
            for our_col, names in _YF_CF_MAP.items():
                row[our_col] = _pick(cf,  names, date_col)

            # Derived metrics
            def _sdiv(a, b):
                try:
                    return float(a) / float(b) if b and float(b) != 0 else None
                except Exception:
                    return None

            op_cf  = row.get("operating_cf")
            capex  = row.get("capex")
            if op_cf is not None and capex is not None:
                # yfinance reports capex as negative outflow — subtract magnitude
                row["fcf"] = float(op_cf) - abs(float(capex))

            row["gross_margin"]   = _sdiv(row.get("gross_profit"),  row.get("revenue"))
            row["net_margin"]     = _sdiv(row.get("net_income"),     row.get("revenue"))
            row["roe"]            = _sdiv(row.get("net_income"),     row.get("total_equity"))
            row["debt_to_equity"] = _sdiv(row.get("lt_debt"),        row.get("total_equity"))
            row["fcf_margin"]     = _sdiv(row.get("fcf"),            row.get("revenue"))

            # Require at least revenue or net_income to keep the row
            if row.get("revenue") is None and row.get("net_income") is None:
                continue
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Dedup: keep latest per (ticker, period_end, form)
        df = df.sort_values("knowledge_timestamp").drop_duplicates(
            subset=["ticker", "period_end", "form"], keep="last"
        )
        return df.reset_index(drop=True)

    except Exception as exc:
        log.debug("fetch_financials_yfinance %s: %s", ticker, exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 6b. 8-K material event filings (SEC EDGAR submissions API — free, no key)
# ─────────────────────────────────────────────────────────────────────────────

# 8-K item codes that matter most for long-term fundamental investors
_8K_ITEM_FLAGS = {
    "has_earnings":          "2.02",   # Results of Operations (earnings release)
    "has_exec_change":       "5.02",   # Director / Officer departure or appointment
    "has_ma":                "2.01",   # Completion of Acquisition / Disposition
    "has_material_agreement":"1.01",   # Entry into Material Definitive Agreement
    "has_debt_obligation":   "2.03",   # Creation of Direct Financial Obligation
    "has_auditor_change":    "4.01",   # Change of Certifying Accountant (red flag)
    "has_bankruptcy":        "1.03",   # Bankruptcy or Receivership (red flag)
    "has_delisting":         "3.01",   # Notice of Delisting (red flag)
}


def fetch_8k_one(ticker: str) -> pd.DataFrame:
    """
    Fetch 8-K filing metadata for one ticker from SEC EDGAR submissions API.

    Uses https://data.sec.gov/submissions/CIK{cik}.json — free, no key required.
    Paginates through all available history (back to ~2010 for S&P 500 companies).

    Returns one row per 8-K / 8-K/A filing with:
      accession_number   — unique EDGAR filing ID
      filing_date        — date filing was accepted by SEC (= knowledge_timestamp)
      items              — raw item string, e.g. "2.02,7.01"
      has_earnings       — item 2.02 present (earnings release)
      has_exec_change    — item 5.02 present (executive departure/appointment)
      has_ma             — item 2.01 present (M&A completion)
      has_material_agreement — item 1.01
      has_debt_obligation    — item 2.03
      has_auditor_change     — item 4.01
      has_bankruptcy         — item 1.03
      has_delisting          — item 3.01

    SEC rate limit: ≤ 10 req/s.  Caller sleeps 0.15s between tickers.
    Pagination requests within this function sleep 0.12s each.
    """
    cik_map = _load_edgar_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return pd.DataFrame()

    records: list[dict] = []

    def _parse_filings_block(block: dict) -> None:
        """Extract 8-K rows from one filings block (recent or paginated)."""
        forms      = block.get("form",          [])
        dates      = block.get("filingDate",    [])
        accessions = block.get("accessionNumber", [])
        items_list = block.get("items",         [])

        for i, form in enumerate(forms):
            if form not in ("8-K", "8-K/A"):
                continue
            filing_date = dates[i]      if i < len(dates)      else None
            accession   = accessions[i] if i < len(accessions) else None
            items_raw   = items_list[i] if i < len(items_list) else ""

            if not filing_date or not accession:
                continue
            if filing_date < "2010-01-01":
                continue   # outside study window — also stops pagination early

            item_set = {x.strip() for x in str(items_raw).split(",") if x.strip()}
            ts = f"{filing_date}T00:00:00+00:00"

            row = {
                "ticker":            ticker,
                "cik":               cik.lstrip("0"),
                "accession_number":  accession,
                "form":              form,
                "event_timestamp":   ts,
                "knowledge_timestamp": ts,   # EDGAR filings are public immediately
                "items":             str(items_raw),
                "source":            "SEC_EDGAR_SUBMISSIONS",
            }
            for flag, code in _8K_ITEM_FLAGS.items():
                row[flag] = int(code in item_set)
            records.append(row)

    try:
        # ── Primary submissions URL ───────────────────────────────────────────
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_EDGAR_HEADERS,
            timeout=30,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()

        _parse_filings_block(data.get("filings", {}).get("recent", {}))

        # ── Pagination for older filings ──────────────────────────────────────
        for page_file in data.get("filings", {}).get("files", []):
            fname = page_file.get("name", "")
            if not fname:
                continue
            time.sleep(0.12)
            pr = requests.get(
                f"https://data.sec.gov/submissions/{fname}",
                headers=_EDGAR_HEADERS,
                timeout=30,
            )
            if pr.status_code != 200:
                continue
            page_data = pr.json()
            # Pagination files have the same flat structure as filings.recent
            _parse_filings_block(page_data)
            # If the oldest filing on this page is already before 2010, stop
            dates_on_page = page_data.get("filingDate", [])
            if dates_on_page and min(dates_on_page) < "2010-01-01":
                break

    except Exception as exc:
        log.debug("8-K %s: %s", ticker, exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate on primary key (handles 8-K/A amendments)
    df = df.sort_values("knowledge_timestamp").drop_duplicates(
        subset=["ticker", "accession_number"], keep="last"
    )
    return df.reset_index(drop=True)


async def fetch_all_8k(tickers: list[str]) -> pd.DataFrame:
    log.info("📋  Fetching 8-K filings (SEC EDGAR) for %d tickers …", len(tickers))
    loop = asyncio.get_running_loop()
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = await loop.run_in_executor(None, fetch_8k_one, ticker)
        if not df.empty:
            frames.append(df)
        await asyncio.sleep(0.15)
        if i % 50 == 0:
            log.info("    8-K %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d 8-K filings across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fundamental financials (SEC EDGAR XBRL — income stmt, balance sheet, CF)
# ─────────────────────────────────────────────────────────────────────────────

# Priority-ordered XBRL tags per metric (first tag with data wins)
_GAAP_TAGS: dict[str, list[str]] = {
    # ── Income statement ──────────────────────────────────────────────────────
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenue",                # pre-ASC 606 filers, some banks
        "RevenuesNet",                 # some retail/service companies
    ],
    "gross_profit":      ["GrossProfit"],
    "operating_income":  ["OperatingIncomeLoss"],
    "net_income":        ["NetIncomeLoss"],
    "interest_expense":  ["InterestExpense", "InterestAndDebtExpense"],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    # ── Balance sheet (instant — point-in-time) ───────────────────────────────
    "total_assets":  ["Assets"],
    "total_equity":  [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        "Equity",
    ],
    "lt_debt":  ["LongTermDebt", "LongTermDebtNoncurrent"],
    "cash":     [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    # ── Cash flow statement ───────────────────────────────────────────────────
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex":        ["PaymentsToAcquirePropertyPlantAndEquipment"],
    # ── Share count ───────────────────────────────────────────────────────────
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssuedAndOutstanding",
    ],
}


def _ttm_flow_series(fin_sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute a PiT TTM (trailing-twelve-month) series for a flow metric
    (revenue or fcf) using quarterly incremental 10-Q values plus an implied
    Q4 = annual_10K_value − Q1 − Q2 − Q3.

    This mirrors the implied-Q4-EPS approach: companies do not file a 10-Q
    for fiscal Q4, so a naive rolling(4) over 10-Q rows skips Q4 entirely and
    substitutes Q3 of the prior fiscal year — systematically understating TTM
    for growing companies and misstating it for seasonal ones.

    Returns DataFrame[kt, ttm_{metric}] sorted by kt, ready for merge_asof.
    """
    col = f"ttm_{metric}"
    empty = pd.DataFrame(columns=["kt", col])
    if fin_sub.empty or "form" not in fin_sub.columns or metric not in fin_sub.columns:
        return empty

    fin = fin_sub.copy()
    fin["_pe"] = pd.to_datetime(fin["period_end"], utc=True, errors="coerce")
    fin["_kt"] = pd.to_datetime(fin["knowledge_timestamp"], utc=True, errors="coerce")
    fin["_val"] = pd.to_numeric(fin[metric], errors="coerce")
    fin = fin.dropna(subset=["_pe", "_kt", "_val"])
    if fin.empty:
        return empty

    annual  = fin[fin["form"] == "10-K"].copy()
    quarterly = fin[fin["form"] == "10-Q"].copy()

    # ── Build a complete quarterly timeline including implied Q4 ──────────────
    # Each record: (fiscal period_end, knowledge_timestamp, quarterly_value)
    records: list[tuple] = [(r["_pe"], r["_kt"], r["_val"]) for _, r in quarterly.iterrows()]

    for _, ann in annual.iterrows():
        ann_pe  = ann["_pe"]
        ann_kt  = ann["_kt"]
        ann_val = ann["_val"]

        # Q1/Q2/Q3 whose period_end falls within the annual's fiscal year.
        # 370-day window handles all standard fiscal year lengths.
        fy_start = ann_pe - pd.Timedelta(days=370)
        q_in_fy  = quarterly[
            (quarterly["_pe"] > fy_start) & (quarterly["_pe"] <= ann_pe)
        ]

        if len(q_in_fy) != 3:
            # Require exactly 3 quarters so implied Q4 is unambiguous.
            # (< 3 means some quarters are missing; > 3 is a data anomaly.)
            continue

        q4_implied = ann_val - q_in_fy["_val"].sum()
        # Q4 period_end ≡ the annual period_end.
        # kt = max(10-K filing date, latest Q1/Q2/Q3 filing date):
        # The TTM value depends on all three quarterly values; if the 10-K
        # is filed before the Q3 10-Q (rare but occurs with amendments),
        # serving the implied-Q4 TTM before Q3 is public is look-ahead bias.
        latest_component_kt = max(ann_kt, q_in_fy["_kt"].max())
        records.append((ann_pe, latest_component_kt, q4_implied))

    if not records:
        return empty

    df = pd.DataFrame(records, columns=["_pe", "_kt", "_val"])
    # Sort by period_end, keep the latest filing when duplicates exist (amendments)
    df = (df.sort_values(["_pe", "_kt"])
            .drop_duplicates("_pe", keep="last")
            .reset_index(drop=True))

    # Rolling 4-quarter sum in fiscal period order → true TTM
    df[col] = df["_val"].rolling(4, min_periods=4).sum()
    df = df.dropna(subset=[col])

    # Return kt-sorted table for merge_asof(direction="backward").
    # drop_duplicates on kt: if two different fiscal period_ends share the same
    # filing date (rare but possible for same-day filings), keep the one from
    # the later fiscal period (already last after sort_values(["_pe","_kt"])).
    return (df[["_kt", col]]
              .rename(columns={"_kt": "kt"})
              .sort_values("kt")
              .drop_duplicates("kt", keep="last")
              .reset_index(drop=True))


def _extract_xbrl_metric(
    us_gaap: dict, tags: list[str], duration_check: bool = True
) -> dict[tuple, tuple]:
    """
    Merge data from ALL candidate tags into {(period_end, form): (value, filed_date)}.
    Companies often switch XBRL tags over time (e.g. AAPL used SalesRevenueNet
    through 2017, then RevenueFromContractWithCustomerExcludingAssessedTax from
    2018 onward after ASC 606 adoption). Merging all tags gives full history.

    Conflict resolution when multiple tags report the same (period_end, form):
      - Earlier tag in the priority list wins (it is preferred semantically).
      - Within the same tag, the most recently filed revision wins.

    duration_check: if True (default), filters out YTD cumulative entries in
      10-Q filings by checking the period duration.  Set False for instant
      (balance-sheet) tags that have no start date.
    """
    merged: dict[tuple, tuple] = {}
    for priority, tag in enumerate(tags):
        data = us_gaap.get(tag, {})
        if not data:
            continue
        for unit_vals in data.get("units", {}).values():
            for e in unit_vals:
                # Accept 10-Q/A and 10-K/A; normalise to base form so amended
                # filings replace originals via PK dedup on (period_end, form).
                raw_form = e.get("form", "")
                base_form = raw_form.replace("/A", "")
                if base_form not in ("10-K", "10-Q"):
                    continue
                form = base_form
                period_end = e.get("end")
                filed      = e.get("filed", "")
                val        = e.get("val")
                if not period_end or val is None:
                    continue

                # Filter by period duration to avoid picking up YTD cumulative entries.
                # 10-Q filings include both quarterly (3-month) and YTD cumulative entries
                # for the same period_end.  Income-statement/CF metrics should be per-quarter
                # so that annual aggregations don't double-count earlier quarters.
                # If start_date is absent for a 10-Q with duration_check, skip it —
                # absence typically means the entry is cumulative with no range declared.
                if duration_check:
                    start_date = e.get("start")
                    if form == "10-Q" and not start_date:
                        continue   # cannot verify duration — skip to be safe
                    if start_date:
                        try:
                            dur = (pd.Timestamp(period_end) - pd.Timestamp(start_date)).days
                            if form == "10-Q" and not (60 <= dur <= 105):
                                continue   # YTD cumulative — skip
                            if form == "10-K" and not (330 <= dur <= 400):
                                continue   # transition-period filing — skip
                        except Exception:
                            pass   # unparseable dates — let through

                key = (period_end, form)
                existing = merged.get(key)
                if existing is None:
                    merged[key] = (val, filed, priority)
                else:
                    ex_priority, ex_filed = existing[2], existing[1]
                    # Lower priority index = preferred tag; within same tag keep latest filed
                    if priority < ex_priority or (priority == ex_priority and filed > ex_filed):
                        merged[key] = (val, filed, priority)
    # Strip priority from result
    return {k: (v[0], v[1]) for k, v in merged.items()}


def fetch_financials_edgar(ticker: str) -> pd.DataFrame:
    """
    Fetch full fundamental financials from SEC EDGAR XBRL in one API call.
    Returns one row per (ticker, period_end, form) with:
      - Income statement: revenue, gross_profit, operating_income, net_income,
                          interest_expense, depreciation_amortization
      - Balance sheet:    total_assets, total_equity, lt_debt, cash
      - Cash flow:        operating_cf, capex
      - Share count:      shares_outstanding
      - Derived:          fcf, gross_margin, net_margin, roe, debt_to_equity,
                          fcf_margin
    knowledge_timestamp = SEC filing date (proper PiT — no look-ahead).
    """
    cik_map = _load_edgar_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return pd.DataFrame()
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        us_gaap = resp.json().get("facts", {}).get("us-gaap", {})

        # Balance-sheet tags are "instant" values (no start date) — duration check N/A.
        # Income statement and cash-flow tags are "duration" and need the filter to
        # avoid picking up YTD cumulative entries (e.g., 9-month Q3 revenue).
        _INSTANT_TAGS = {"total_assets", "total_equity", "lt_debt", "cash",
                         "shares_outstanding"}

        # Extract all metrics into {(period_end, form): (value, filed)} dicts
        metric_data: dict[str, dict[tuple, tuple]] = {
            metric: _extract_xbrl_metric(us_gaap, tags,
                                         duration_check=(metric not in _INSTANT_TAGS))
            for metric, tags in _GAAP_TAGS.items()
        }

        # Union of all (period_end, form) keys across every metric
        all_keys: set[tuple] = set()
        for d in metric_data.values():
            all_keys.update(d.keys())
        if not all_keys:
            return pd.DataFrame()

        rows = []
        for (period_end, form) in all_keys:
            # Use the latest filing date seen across any metric for this period
            filed_dates = [
                metric_data[m][(period_end, form)][1]
                for m in metric_data
                if (period_end, form) in metric_data[m]
            ]
            filed = max(filed_dates) if filed_dates else period_end

            row: dict = {
                "ticker":               ticker,
                "period_end":           period_end,
                "form":                 form,
                "event_timestamp":      period_end,
                "knowledge_timestamp":  filed,
                "source":               "SEC_EDGAR",
            }
            for metric in _GAAP_TAGS:
                entry = metric_data[metric].get((period_end, form))
                row[metric] = entry[0] if entry else None
            rows.append(row)

        df = pd.DataFrame(rows)

        # Restrict to study window
        df["_period_ts"] = pd.to_datetime(df["period_end"], errors="coerce")
        df = df[df["_period_ts"] >= pd.Timestamp(START_EQUITY.date())].copy()

        # ── Derived metrics ───────────────────────────────────────────────────
        def _div(a, b):
            try:
                return float(a) / float(b) if b and float(b) != 0 else None
            except Exception:
                return None

        # capex from PaymentsToAcquirePropertyPlantAndEquipment should be positive
        # per US GAAP taxonomy, but some filers report it as a negative outflow.
        # Using abs() matches the yfinance path and is correct under both conventions.
        df["fcf"]           = df.apply(
            lambda r: (float(r["operating_cf"]) - abs(float(r["capex"])))
            if r["operating_cf"] is not None and r["capex"] is not None else None, axis=1)
        df["gross_margin"]  = df.apply(lambda r: _div(r["gross_profit"],  r["revenue"]),       axis=1)
        df["net_margin"]    = df.apply(lambda r: _div(r["net_income"],     r["revenue"]),       axis=1)
        df["roe"]           = df.apply(lambda r: _div(r["net_income"],     r["total_equity"]),  axis=1)
        df["debt_to_equity"]= df.apply(lambda r: _div(r["lt_debt"],        r["total_equity"]),  axis=1)
        df["fcf_margin"]    = df.apply(lambda r: _div(r["fcf"],            r["revenue"]),       axis=1)

        # Null-out gross_margin values that are clearly XBRL tag mismatches
        # (gross_profit from one segment vs. total revenue from another).
        # Gross margin cannot exceed 100% — if it does, revenue and gross_profit
        # are from different XBRL scopes and the ratio is meaningless.
        df.loc[
            df["gross_margin"].notna() &
            ((df["gross_margin"] >= 1.0) | (df["gross_margin"] < -1.0)),
            "gross_margin"
        ] = None
        df.loc[
            df["fcf_margin"].notna() &
            ((df["fcf_margin"] > 5.0) | (df["fcf_margin"] < -5.0)),
            "fcf_margin"
        ] = None
        df.loc[
            df["net_margin"].notna() &
            ((df["net_margin"] > 5.0) | (df["net_margin"] < -5.0)),
            "net_margin"
        ] = None

        # PiT guard: knowledge_timestamp must be >= period_end
        # EDGAR occasionally has quirky filed dates on restated/transition filings
        df["_kt"] = pd.to_datetime(df["knowledge_timestamp"], errors="coerce")
        df["_pe"] = pd.to_datetime(df["period_end"],          errors="coerce")
        mask = df["_kt"].notna() & df["_pe"].notna() & (df["_kt"] < df["_pe"])
        df.loc[mask, "knowledge_timestamp"] = df.loc[mask, "period_end"]
        df = df.drop(columns=["_kt", "_pe"])

        # Dedup: same (ticker, period_end, form) — keep latest filed revision
        df = df.sort_values("knowledge_timestamp").drop_duplicates(
            subset=["ticker", "period_end", "form"], keep="last"
        )
        return df.drop(columns=["_period_ts"]).reset_index(drop=True)

    except Exception as exc:
        log.debug("EDGAR financials %s: %s", ticker, exc)
        return pd.DataFrame()


async def fetch_all_financials(tickers: list[str]) -> pd.DataFrame:
    log.info("📊  Fetching fundamental financials (SEC EDGAR XBRL) …")
    loop = asyncio.get_running_loop()
    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = await loop.run_in_executor(None, fetch_financials_edgar, ticker)
        if not df.empty:
            frames.append(df)
        await asyncio.sleep(0.15)   # SEC fair-use: max 10 req/s
        if i % 50 == 0:
            log.info("    financials %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d financial records across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. PiT correctness check
# ─────────────────────────────────────────────────────────────────────────────

def run_pit_check(ohlcv: pd.DataFrame) -> bool:
    """
    Prove zero look-ahead bias:
      For any given as_of date, filtering knowledge_timestamp <= as_of
      must never return a bar from the future.
    """
    log.info("🔒  Running Point-in-Time correctness checks …")
    passed = True

    # Test 5 random as_of dates
    test_dates = pd.to_datetime([
        "2021-06-15", "2022-01-03", "2022-09-30",
        "2023-03-15", "2023-12-01",
    ], utc=True)

    df = ohlcv.copy()
    df["knowledge_ts"] = pd.to_datetime(df["knowledge_timestamp"], utc=True)
    df["event_ts"]     = pd.to_datetime(df["event_timestamp"],     utc=True)

    for as_of in test_dates:
        visible = df[df["knowledge_ts"] <= as_of]
        future_leak = visible[visible["event_ts"] > as_of + timedelta(days=1)]
        n_bars = len(visible)
        n_tickers = visible["ticker"].nunique() if n_bars > 0 else 0

        if len(future_leak) > 0:
            log.error("  ❌  as_of=%s: LOOK-AHEAD BIAS — %d future bars leaked!",
                      as_of.date(), len(future_leak))
            passed = False
        else:
            log.info("  ✅  as_of=%s: %d bars, %d tickers — CLEAN (0 leaks)",
                     as_of.date(), n_bars, n_tickers)

        # Extra: check that adj_close is not suspiciously far in the future
        if n_bars > 0:
            max_event = visible["event_ts"].max()
            assert max_event <= as_of + timedelta(days=2), \
                f"Event timestamp {max_event} exceeds as_of {as_of}"

    # Survivorship bias check: count unique tickers visible at each date
    log.info("\n  Survivorship-bias sanity (ticker count by date):")
    for as_of in test_dates:
        n = df[df["knowledge_ts"] <= as_of]["ticker"].nunique()
        log.info("    as_of %s → %d tickers visible", as_of.date(), n)

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sector / industry classification  (yfinance, static reference table)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sectors(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch sector, industry, GICS sub-industry, and market-cap category for
    every ticker via yfinance.  Returns one row per ticker.

    Schema:
        ticker, sector, industry, gics_sub_industry,
        market_cap_category, country, exchange,
        knowledge_timestamp, source
    """
    now_ts = datetime.now(UTC).isoformat()
    rows = []
    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap") or 0
            if   mc > 200e9: cap_cat = "mega"
            elif mc >  10e9: cap_cat = "large"
            elif mc >   2e9: cap_cat = "mid"
            else:            cap_cat = "small"

            rows.append({
                "ticker":              ticker,
                "sector":              info.get("sector")              or "",
                "industry":            info.get("industry")            or "",
                "gics_sub_industry":   info.get("industryKey")         or "",
                "market_cap_category": cap_cat,
                "country":             info.get("country")             or "",
                "exchange":            info.get("exchange")            or "",
                "knowledge_timestamp": now_ts,
                "source":              "yfinance",
            })
            time.sleep(0.15)
        except Exception as exc:
            log.debug("fetch_sectors %s: %s", ticker, exc)
        if i % 100 == 0:
            log.info("  sectors %d/%d …", i, len(tickers))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Quality metrics  (computed from existing financials + ohlcv)
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_metrics_one(
    ticker: str,
    fin_sub: pd.DataFrame,   # all financials rows for this ticker
    ohlcv_sub: pd.DataFrame, # OHLCV rows for this ticker (for market-cap lookup)
) -> pd.DataFrame:
    """
    Compute fundamental quality metrics for one ticker, one row per annual
    10-K filing.  All metrics are PiT-correct: only data available at
    knowledge_timestamp is used.

    Metrics:
        roic              — Return on Invested Capital  (NOPAT / invested_capital)
        fcf_yield_filing  — FCF / market_cap at filing date
        net_debt_ebitda   — (lt_debt - cash) / (operating_income + D&A)
        revenue_cagr_1y   — Annualised revenue growth vs. prior 10-K
        revenue_cagr_3y   — Annualised revenue growth vs. 3 prior 10-Ks
        revenue_cagr_5y   — Annualised revenue growth vs. 5 prior 10-Ks
        eps_cagr_1y       — Annualised EPS growth vs. prior 10-K
        eps_cagr_3y       — Annualised EPS growth vs. 3 prior 10-Ks
        gross_margin_curr — gross_margin at this filing
        gross_margin_3y_avg — avg gross_margin over prior 3 10-Ks
        gross_margin_trend  — curr - 3y_avg  (positive = improving)
        buyback_yield     — (shares_t-1 - shares_t) / shares_t-1  (positive=buyback)
        earnings_consistency_5y — fraction of last 5 years with net_income growth
    """
    if fin_sub is None or fin_sub.empty:
        return pd.DataFrame()

    # ── 1. Annual filings only, sorted by period_end ──────────────────────────
    ann = fin_sub[fin_sub["form"] == "10-K"].copy()
    if ann.empty:
        return pd.DataFrame()

    ann["period_ts"] = pd.to_datetime(ann["period_end"], errors="coerce", utc=True)
    ann["kt"]        = pd.to_datetime(ann["knowledge_timestamp"], errors="coerce", utc=True)
    ann = ann.dropna(subset=["period_ts"]).sort_values("period_ts").reset_index(drop=True)

    # ── 2. OHLCV price lookup table (for market cap at filing date) ───────────
    price_lkp = pd.DataFrame()
    if ohlcv_sub is not None and not ohlcv_sub.empty:
        p = ohlcv_sub.copy()
        p["pts"] = pd.to_datetime(p["event_timestamp"], utc=True)
        price_lkp = p[["pts", "close"]].sort_values("pts")

    def _close_at(kt: pd.Timestamp) -> float | None:
        """Most recent close on or before the filing date."""
        if price_lkp.empty or pd.isna(kt):
            return None
        before = price_lkp[price_lkp["pts"] <= kt]
        return float(before["close"].iloc[-1]) if not before.empty else None

    def _safe_div(a, b) -> float | None:
        try:
            a, b = float(a), float(b)
            return a / b if b != 0 else None
        except Exception:
            return None

    def _cagr(v_now, v_past, years: int) -> float | None:
        try:
            v_now, v_past = float(v_now), float(v_past)
            if v_past <= 0 or v_now <= 0 or years <= 0:
                return None
            return (v_now / v_past) ** (1.0 / years) - 1.0
        except Exception:
            return None

    # ── 3. Build one output row per 10-K ─────────────────────────────────────
    rows = []
    for i, row in ann.iterrows():
        def _get(col):
            v = row.get(col)
            try: return float(v) if v is not None and str(v) != "nan" else None
            except Exception: return None

        op_inc  = _get("operating_income")
        da      = _get("depreciation_amortization")
        eq      = _get("total_equity")
        lt_debt = _get("lt_debt")  or 0.0
        cash    = _get("cash")     or 0.0
        fcf     = _get("fcf")
        shares  = _get("shares_outstanding")
        rev     = _get("revenue")
        net_inc = _get("net_income")
        gm      = _get("gross_margin")

        # ROIC — NOPAT ≈ operating_income × (1 - tax_rate).
        # US statutory rate was 35% through 2017, 21% from 2018 (TCJA).
        # Using a year-based rate avoids ~22% relative overstatement in pre-2018 ROIC.
        _period_year = pd.Timestamp(row["period_end"]).year if row.get("period_end") else 2018
        _tax_rate    = 0.35 if _period_year <= 2017 else 0.21
        nopat    = op_inc * (1.0 - _tax_rate) if op_inc is not None else None
        inv_cap  = (eq or 0.0) + lt_debt - cash
        roic     = _safe_div(nopat, inv_cap) if inv_cap and abs(inv_cap) > 1e6 else None
        # Cap ROIC: values outside [-100%, 300%] indicate near-zero invested capital
        if roic is not None and (roic < -1.0 or roic > 3.0):
            roic = None

        # FCF yield at filing date
        close_at_filing = _close_at(row["kt"])
        mktcap = (close_at_filing * shares) if (close_at_filing and shares) else None
        fcf_yield = _safe_div(fcf, mktcap)
        if fcf_yield is not None and abs(fcf_yield) > 1.0:
            fcf_yield = None   # implausible

        # Net debt / EBITDA
        ebitda     = (op_inc or 0.0) + (da or 0.0)
        net_debt   = lt_debt - cash
        nd_ebitda  = _safe_div(net_debt, ebitda) if ebitda and abs(ebitda) > 1e6 else None
        if nd_ebitda is not None and abs(nd_ebitda) > 50:
            nd_ebitda = None   # implausible

        # Revenue CAGRs — look back in sorted annual history
        def _prior_rev(n_years: int):
            j = i - n_years
            if j < 0 or j not in ann.index:
                # find by position
                pos = ann.index.get_loc(i)
                if pos < n_years:
                    return None
                return ann.iloc[pos - n_years].get("revenue")
            return ann.loc[j].get("revenue")

        def _cap_cagr(v):
            # Cap CAGR at [-80%, 500%] — beyond this range it's a spin-off, restatement,
            # or hypergrowth from near-zero base (not a meaningful compound rate)
            if v is None: return None
            return v if -0.8 <= v <= 5.0 else None

        rev_1y = _cap_cagr(_cagr(rev, _prior_rev(1), 1)) if rev else None
        rev_3y = _cap_cagr(_cagr(rev, _prior_rev(3), 3)) if rev else None
        rev_5y = _cap_cagr(_cagr(rev, _prior_rev(5), 5)) if rev else None

        # EPS CAGRs
        eps_now = _safe_div(net_inc, shares)
        def _prior_eps(n_years: int):
            pos = ann.index.get_loc(i)
            if pos < n_years:
                return None
            r = ann.iloc[pos - n_years]
            ni, sh = r.get("net_income"), r.get("shares_outstanding")
            return _safe_div(ni, sh)

        eps_1y = _cap_cagr(_cagr(eps_now, _prior_eps(1), 1)) if eps_now else None
        eps_3y = _cap_cagr(_cagr(eps_now, _prior_eps(3), 3)) if eps_now else None

        # Gross margin trend
        pos = ann.index.get_loc(i)
        def _safe_float(v) -> float | None:
            """Convert to float, returning None for None/'None'/'nan'/NaN."""
            if v is None:
                return None
            try:
                f = float(v)
                return f if not pd.isna(f) else None
            except (ValueError, TypeError):
                return None
        prior_gms = [
            _safe_float(ann.iloc[pos - k].get("gross_margin"))
            for k in range(1, 4)
            if pos >= k
        ]
        prior_gms = [v for v in prior_gms if v is not None]
        gm_3y_avg = sum(prior_gms) / len(prior_gms) if prior_gms else None
        gm_trend  = (gm - gm_3y_avg) if (gm is not None and gm_3y_avg is not None) else None

        # Buyback yield — share count change vs. prior year
        buyback_yield = None
        if pos >= 1:
            prior_shares = ann.iloc[pos - 1].get("shares_outstanding")
            if prior_shares and shares:
                try:
                    buyback_yield = (float(prior_shares) - float(shares)) / float(prior_shares)
                    if abs(buyback_yield) > 0.5:   # implausible (> 50% change)
                        buyback_yield = None
                except Exception:
                    pass

        # Earnings consistency (last 5 years with positive YoY net income growth)
        earn_consistency = None
        if pos >= 1:
            lookback = min(5, pos)
            growths = []
            for k in range(1, lookback + 1):
                ni_now  = ann.iloc[pos - k + 1].get("net_income")
                ni_prev = ann.iloc[pos - k    ].get("net_income")
                if ni_now is not None and ni_prev is not None:
                    try:
                        growths.append(float(ni_now) > float(ni_prev))
                    except Exception:
                        pass
            if growths:
                earn_consistency = sum(growths) / len(growths)

        rows.append({
            "ticker":                  ticker,
            "period_end":              row.get("period_end"),
            "form":                    "10-K",
            "event_timestamp":         row.get("period_end"),
            "knowledge_timestamp":     row.get("knowledge_timestamp"),
            "source":                  "computed",
            "roic":                    roic,
            "fcf_yield_filing":        fcf_yield,
            "net_debt_ebitda":         nd_ebitda,
            "revenue_cagr_1y":         rev_1y,
            "revenue_cagr_3y":         rev_3y,
            "revenue_cagr_5y":         rev_5y,
            "eps_cagr_1y":             eps_1y,
            "eps_cagr_3y":             eps_3y,
            "gross_margin_curr":       gm,
            "gross_margin_3y_avg":     gm_3y_avg,
            "gross_margin_trend":      gm_trend,
            "buyback_yield":           buyback_yield,
            "earnings_consistency_5y": earn_consistency,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_all_quality_metrics(
    tickers: list[str],
    fin_all: pd.DataFrame,
    ohlcv_all: pd.DataFrame,
) -> pd.DataFrame:
    """Compute quality metrics for all tickers. Tables are pre-loaded."""
    log.info("📊  Computing quality metrics for %d tickers …", len(tickers))
    frames = []
    for i, ticker in enumerate(tickers, 1):
        try:
            df = compute_quality_metrics_one(
                ticker,
                fin_all[fin_all["ticker"] == ticker] if not fin_all.empty else pd.DataFrame(),
                ohlcv_all[ohlcv_all["ticker"] == ticker] if not ohlcv_all.empty else pd.DataFrame(),
            )
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            log.debug("quality_metrics %s: %s", ticker, exc)
        if i % 100 == 0:
            log.info("    quality_metrics %d/%d …", i, len(tickers))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("    → %d quality metric rows across %d tickers",
             len(result), result["ticker"].nunique() if not result.empty else 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 10. Build a sample PiT snapshot (like the backtester would)
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_snapshot(
    ohlcv: pd.DataFrame,
    indicators: pd.DataFrame,
    macro: pd.DataFrame,
    as_of_str: str = "2023-06-30",
) -> dict:
    as_of = pd.Timestamp(as_of_str, tz="UTC")
    lookback_start = as_of - pd.Timedelta(days=252)

    log.info("📦  Building PiT snapshot as_of=%s …", as_of.date())

    # ── Prices (PiT filtered) ─────────────────────────────────────────────────
    df = ohlcv.copy()
    df["knowledge_ts"] = pd.to_datetime(df["knowledge_timestamp"], utc=True)
    df["event_ts"]     = pd.to_datetime(df["event_timestamp"],     utc=True)

    price_df = df[
        (df["knowledge_ts"] <= as_of) &
        (df["event_ts"] >= lookback_start) &
        (df["event_ts"] <= as_of)
    ].copy()

    if "adj_close" in price_df.columns:
        close_col = "adj_close"
    else:
        close_col = "close"

    price_wide = price_df.pivot_table(
        index="event_ts", columns="ticker", values=close_col, aggfunc="last"
    )

    # ── Latest indicators per ticker ──────────────────────────────────────────
    latest_indicators = {}
    if not indicators.empty:
        ind = indicators.copy()
        ind["knowledge_ts"] = pd.to_datetime(ind["knowledge_timestamp"], utc=True)
        ind["event_ts"]     = pd.to_datetime(ind["event_timestamp"],     utc=True)
        visible_ind = ind[(ind["knowledge_ts"] <= as_of) & (ind["event_ts"] <= as_of)]
        if not visible_ind.empty:
            latest = visible_ind.sort_values("event_ts").groupby("ticker").last()
            ind_cols = [c for c in latest.columns if c not in
                        ["ticker", "event_timestamp", "knowledge_timestamp",
                         "knowledge_ts", "event_ts"]]
            for ticker in latest.index:
                latest_indicators[ticker] = {
                    c: round(float(v), 4)
                    for c, v in latest.loc[ticker, ind_cols].items()
                    if pd.notna(v)
                }

    # ── Latest macro values ───────────────────────────────────────────────────
    latest_macro = {}
    if not macro.empty:
        m = macro.copy()
        m["knowledge_ts"] = pd.to_datetime(m["knowledge_timestamp"], utc=True)
        visible_m = m[m["knowledge_ts"] <= as_of]
        if not visible_m.empty:
            for code, grp in visible_m.groupby("indicator_code"):
                grp_sorted = grp.sort_values("event_timestamp")
                latest_val = grp_sorted["value"].iloc[-1]
                latest_macro[code] = round(float(latest_val), 4)

    # ── Returns (last 52 weeks) ───────────────────────────────────────────────
    returns_52w = {}
    if not price_wide.empty and len(price_wide) > 10:
        pct = price_wide.pct_change().dropna(how="all")
        cum_ret = (1 + pct).prod() - 1
        for ticker in cum_ret.index:
            if pd.notna(cum_ret[ticker]):
                returns_52w[ticker] = round(float(cum_ret[ticker]) * 100, 2)

    snapshot = {
        "as_of":             as_of_str,
        "universe_size":     price_wide.shape[1] if not price_wide.empty else 0,
        "price_bars_in_window": len(price_df),
        "price_shape":       list(price_wide.shape) if not price_wide.empty else [0, 0],
        "latest_prices":     {
            t: round(float(price_wide[t].dropna().iloc[-1]), 2)
            for t in price_wide.columns
            if not price_wide[t].dropna().empty
        },
        "returns_52w_pct":   returns_52w,
        "macro":             latest_macro,
        "indicators_sample": {
            t: v for i, (t, v) in enumerate(latest_indicators.items()) if i < 3
        },
    }
    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save everything to disk
# ─────────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, subdir: str, filename: str) -> None:
    if df is None or df.empty:
        log.warning("    (skipping %s/%s — empty)", subdir, filename)
        return
    path = DATA_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / filename
    df.to_csv(csv_path, index=False)
    # Also save Parquet for analysis
    parquet_path = path / filename.replace(".csv", ".parquet")
    df.to_parquet(parquet_path, index=False, compression="snappy")
    log.info("    💾  Saved %s  (%d rows, %.1f KB)",
             csv_path, len(df), csv_path.stat().st_size / 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("=" * 65)
    log.info("  FINANCIAL DATA PIPELINE — STANDALONE CRAWLER RUN")
    log.info("  Equity start: %s  |  Macro start: %s  |  End: %s",
             START_EQUITY.date(), START_MACRO.date(), datetime.now(UTC).date())
    log.info("=" * 65)

    # 1. Universe (live from Wikipedia: S&P 500 + NASDAQ 100)
    universe_df = await fetch_full_universe()
    save(universe_df, "universe", "sp500_ndx100_current.csv")
    live_tickers = universe_df["ticker"].tolist() if not universe_df.empty else TICKERS
    log.info("  Universe: %d tickers", len(live_tickers))

    # 2. OHLCV
    ohlcv = await fetch_all_ohlcv(live_tickers)
    save(ohlcv, "ohlcv", "ohlcv_daily.csv")

    if ohlcv.empty:
        log.error("No OHLCV data — aborting remaining steps.")
        return

    # 3. Technical indicators
    indicators = await compute_all_indicators(ohlcv)
    save(indicators, "indicators", "technical_indicators.csv")

    # 4. Macro
    macro = await fetch_all_macro()
    save(macro, "macro", "macro_series.csv")

    # 5. Insider trades (limit to 20 tickers to be courteous to OpenInsider)
    insider = await fetch_all_insider(live_tickers[:20])
    save(insider, "insider", "insider_trades.csv")

    # 6. Earnings
    earnings = await fetch_all_earnings(live_tickers)
    save(earnings, "earnings", "earnings_history.csv")

    # 7. Fundamental financials (income stmt, balance sheet, cash flow)
    financials = await fetch_all_financials(live_tickers)
    save(financials, "financials", "fundamentals.csv")

    # 8. PiT correctness
    log.info("")
    pit_ok = run_pit_check(ohlcv)

    # 8. Sample PiT snapshot
    log.info("")
    snapshot = build_sample_snapshot(ohlcv, indicators, macro, "2023-06-30")
    snap_path = DATA_DIR / "pit_snapshot"
    snap_path.mkdir(parents=True, exist_ok=True)
    with open(snap_path / "snapshot_2023-06-30.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info("    💾  PiT snapshot saved to %s", snap_path / "snapshot_2023-06-30.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("  RESULTS SUMMARY")
    log.info("=" * 65)
    log.info("  Universe rows:       %d", len(universe_df))
    log.info("  OHLCV rows:          %d  (%d tickers)",
             len(ohlcv), ohlcv["ticker"].nunique())
    log.info("  Indicator rows:      %d", len(indicators))
    log.info("  Macro rows:          %d  (%d series)",
             len(macro), macro["indicator_code"].nunique() if not macro.empty else 0)
    log.info("  Insider trade rows:  %d", len(insider))
    log.info("  Earnings rows:       %d", len(earnings))
    log.info("  Financials rows:     %d  (%d tickers)",
             len(financials), financials["ticker"].nunique() if not financials.empty else 0)
    log.info("  PiT check:           %s", "✅ PASS" if pit_ok else "❌ FAIL")
    log.info("")
    log.info("  PiT snapshot (2023-06-30):")
    log.info("    Universe visible:    %d tickers", snapshot["universe_size"])
    log.info("    Price bars in window:%d", snapshot["price_bars_in_window"])
    log.info("    Macro indicators:    %s",
             {k: v for k, v in list(snapshot["macro"].items())[:4]})
    log.info("    52-week return AAPL: %s%%",
             snapshot["returns_52w_pct"].get("AAPL", "N/A"))
    log.info("    52-week return NVDA: %s%%",
             snapshot["returns_52w_pct"].get("NVDA", "N/A"))
    log.info("=" * 65)
    log.info("  All files written to: %s", DATA_DIR.resolve())
    log.info("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
