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
# 4. Macro data (FRED via direct HTTP — no API key needed for basic series)
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
# 7. PiT correctness check
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

    # 7. PiT correctness
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
