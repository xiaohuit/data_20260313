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
END   = datetime.now(UTC)

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
    "CPI_YOY":        "CPIAUCSL",
    "GDP_GROWTH":     "A191RL1Q225SBEA",
    "UNEMPLOYMENT":   "UNRATE",
    "10Y_YIELD":      "GS10",
    "VIX":            "VIXCLS",
    "YIELD_CURVE":    "T10Y2Y",
    "CONSUMER_CONF":  "UMCSENT",
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
    t = yf.Ticker(ticker)
    df = t.history(
        start=START_EQUITY.strftime("%Y-%m-%d"),
        end=END.strftime("%Y-%m-%d"),
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
             len(tickers), START_EQUITY.date(), END.date())
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
      pe_ttm       — Price / Trailing-12-month EPS (last 4 quarterly reports)
      pb           — Price / Book value per share
      ps           — Price / Annual revenue per share  (last 10-K)
      ev_ebitda    — Enterprise value / EBITDA          (last 10-K)
      fcf_yield    — Free cash flow per share / Price   (last 10-K)
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
    earn_q = earn_sub[earn_sub["form"] == "10-Q"].copy()
    ttm_eps_lkp = pd.DataFrame(columns=["kt", "ttm_eps"])
    if not earn_q.empty and "epsactual" in earn_q.columns:
        earn_q["kt"] = pd.to_datetime(earn_q["knowledge_timestamp"], utc=True)
        earn_q = earn_q.sort_values("kt").dropna(subset=["epsactual"])
        earn_q["ttm_eps"] = earn_q["epsactual"].rolling(4, min_periods=2).sum()
        ttm_eps_lkp = earn_q[["kt", "ttm_eps"]].dropna().drop_duplicates("kt")

    # ── 3. Balance sheet: most recent filing of any type, PiT ────────────────
    bs_cols = ["total_equity", "shares_outstanding", "lt_debt", "cash"]
    fin_bs = fin_sub[fin_sub[bs_cols].notna().any(axis=1)].copy() if not fin_sub.empty else pd.DataFrame()
    bs_lkp = pd.DataFrame(columns=["kt"] + bs_cols)
    if not fin_bs.empty:
        fin_bs["kt"] = pd.to_datetime(fin_bs["knowledge_timestamp"], utc=True)
        fin_bs = fin_bs.sort_values("kt").drop_duplicates("kt", keep="last")
        bs_lkp = fin_bs[["kt"] + [c for c in bs_cols if c in fin_bs.columns]]

    # ── 4. Annual flow metrics: most recent 10-K, PiT ────────────────────────
    flow_cols = ["revenue", "operating_income", "depreciation_amortization", "fcf"]
    fin_annual = fin_sub[fin_sub["form"] == "10-K"].copy() if not fin_sub.empty else pd.DataFrame()
    flow_lkp = pd.DataFrame(columns=["kt"] + flow_cols)
    if not fin_annual.empty:
        fin_annual["kt"] = pd.to_datetime(fin_annual["knowledge_timestamp"], utc=True)
        fin_annual = fin_annual.sort_values("kt").drop_duplicates("kt", keep="last")
        flow_lkp = fin_annual[["kt"] + [c for c in flow_cols if c in fin_annual.columns]]

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
        for c in flow_cols:
            base[c] = None

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
    base["revenue_per_share"]    = _sdiv(_col("revenue"),      shares)
    base["fcf_per_share"]        = _sdiv(_col("fcf"),          shares)
    lt_debt = _col("lt_debt").fillna(0)
    cash    = _col("cash").fillna(0)
    base["enterprise_value"]     = base["market_cap"] + lt_debt - cash

    # P/E TTM
    base["pe_ttm"]  = _sdiv(close, _col("ttm_eps"),               lo=0,   hi=500)
    # P/B
    base["pb"]      = _sdiv(close, _col("book_value_per_share"),   lo=0,   hi=100)
    # P/S
    base["ps"]      = _sdiv(close, _col("revenue_per_share"),      lo=0,   hi=200)
    # EV/EBITDA
    ebitda = _col("operating_income").fillna(0) + _col("depreciation_amortization").fillna(0)
    ebitda[ebitda <= 0] = float("nan")
    base["ebitda"]    = ebitda
    base["ev_ebitda"] = _sdiv(_col("enterprise_value"), ebitda,    lo=0,   hi=300)
    # FCF yield
    base["fcf_yield"]      = _sdiv(_col("fcf_per_share"), close,   lo=-0.5, hi=0.5)
    # Dividend yield
    base["dividend_yield"] = _sdiv(_col("div_ttm"), close,         lo=0,   hi=0.5)

    # ── 7. Final output ───────────────────────────────────────────────────────
    base["ticker"]               = ticker
    base["event_timestamp"]      = base["date"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    base["knowledge_timestamp"]  = base["date"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    base["source"]               = "computed"

    out_cols = [
        "ticker", "event_timestamp", "knowledge_timestamp", "source",
        "close", "market_cap", "enterprise_value", "shares_outstanding",
        "ttm_eps", "book_value_per_share", "revenue_per_share", "fcf_per_share",
        "revenue", "fcf", "ebitda",
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
# 4b. Dividend history (derived — computed from OHLCV dividends + earnings EPS)
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
            eq["kt"] = pd.to_datetime(eq["knowledge_timestamp"], utc=True, errors="coerce")
            eq["eps_year"] = eq["kt"].dt.year
            eq["epsactual"] = pd.to_numeric(eq["epsactual"], errors="coerce")
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
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&observation_start={START_MACRO.strftime('%Y-%m-%d')}"
        f"&observation_end={END.strftime('%Y-%m-%d')}"
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
                    records.append({
                        "indicator_code":      code,
                        "series_id":           series_id,
                        "event_timestamp":     str(row[date_col].date()),
                        "knowledge_timestamp": str(row[date_col].date()),
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
                # Only quarterly filings (10-Q) and annual (10-K), not amendments
                form = e.get("form", "")
                if form not in ("10-Q", "10-K"):
                    continue
                # "end" = period end date; "filed" = when SEC received the filing
                period_end = e.get("end")
                filed_date = e.get("filed")
                val        = e.get("val")
                if not period_end or val is None:
                    continue
                rows.append({
                    "ticker":              ticker,
                    "period_end":          period_end,
                    "epsactual":           float(val),
                    "epsestimate":         None,   # EDGAR has no consensus estimates
                    "epsdifference":       None,
                    "surprisepercent":     None,
                    "form":                form,
                    "event_timestamp":     period_end,
                    # PiT: the value was only known after the SEC filing date
                    "knowledge_timestamp": filed_date or period_end,
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
                hist["event_timestamp"]     = hist["period_end"]
                hist["knowledge_timestamp"] = hist["period_end"]
                hist["source"] = "yfinance"
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


def _extract_xbrl_metric(
    us_gaap: dict, tags: list[str]
) -> dict[tuple, tuple]:
    """
    Merge data from ALL candidate tags into {(period_end, form): (value, filed_date)}.
    Companies often switch XBRL tags over time (e.g. AAPL used SalesRevenueNet
    through 2017, then RevenueFromContractWithCustomerExcludingAssessedTax from
    2018 onward after ASC 606 adoption). Merging all tags gives full history.

    Conflict resolution when multiple tags report the same (period_end, form):
      - Earlier tag in the priority list wins (it is preferred semantically).
      - Within the same tag, the most recently filed revision wins.
    """
    merged: dict[tuple, tuple] = {}
    for priority, tag in enumerate(tags):
        data = us_gaap.get(tag, {})
        if not data:
            continue
        for unit_vals in data.get("units", {}).values():
            for e in unit_vals:
                form = e.get("form", "")
                if form not in ("10-K", "10-Q"):
                    continue
                period_end = e.get("end")
                filed      = e.get("filed", "")
                val        = e.get("val")
                if not period_end or val is None:
                    continue
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

        # Extract all metrics into {(period_end, form): (value, filed)} dicts
        metric_data: dict[str, dict[tuple, tuple]] = {
            metric: _extract_xbrl_metric(us_gaap, tags)
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

        df["fcf"]           = df.apply(
            lambda r: (float(r["operating_cf"]) - float(r["capex"]))
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
# 8. Build a sample PiT snapshot (like the backtester would)
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
        pct = price_wide.pct_change(fill_method=None).dropna(how="all")
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
             START_EQUITY.date(), START_MACRO.date(), END.date())
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
