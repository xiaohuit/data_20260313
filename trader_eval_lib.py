"""
trader_eval_lib.py  —  Shared library for 13-F manager evaluation
==================================================================

Contains all manager-agnostic logic:
  - SEC EDGAR helpers
  - Cache helpers
  - CUSIP resolution (shared map)
  - Price loading
  - Portfolio construction
  - Performance computation
  - JSON schema building
  - Text report printing

Manager-specific constants (CIK, name, entity, strategy) live in each
individual evaluator (analyze_buffett.py, analyze_ackman.py, …).
"""

from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import requests

# ── Shared constants ──────────────────────────────────────────────────────────

SEC_HEADERS = {"User-Agent": "financial-research analyst@research.local"}
DATA_ROOT   = Path("./data")
CACHE_DIR   = Path("./data/13f_cache")

# Quarters with fewer than this many positions are flagged as confidential
CONFIDENTIAL_THRESHOLD = 10

# ── Shared CUSIP → ticker map ─────────────────────────────────────────────────
# Covers known Berkshire Hathaway, Pershing Square, and other common holdings.

CUSIP_MAP: dict[str, str] = {
    # ── Berkshire Hathaway known holdings ─────────────────────────────────────
    "037833100": "AAPL",   # Apple
    "060505104": "BAC",    # Bank of America
    "025816109": "AXP",    # American Express
    "191216100": "KO",     # Coca-Cola
    "166764100": "CVX",    # Chevron
    "674599105": "OXY",    # Occidental Petroleum
    "615369105": "MCO",    # Moody's Corporation
    "609207105": "MDLZ",   # Mondelez International
    "60920310X": "MCO",    # Moody's (alt CUSIP)
    "49826K102": "KHC",    # Kraft Heinz (old CUSIP)
    "500754106": "KHC",    # Kraft Heinz (current)
    "23918K108": "DVA",    # DaVita
    "92343E102": "VRSN",   # VeriSign
    "902973304": "USB",    # US Bancorp
    "064058100": "BK",     # Bank of New York Mellon
    "428236103": "HPQ",    # HP Inc
    "92826C839": "V",      # Visa
    "57636Q104": "MA",     # Mastercard A
    "57636Q206": "MA",     # Mastercard B
    "00507V109": "ATVI",   # Activision Blizzard
    "92556H207": "PARA",   # Paramount Global
    "02005N100": "ALLY",   # Ally Financial
    "37045V100": "GM",     # General Motors
    "12767X107": "CB",     # Chubb
    "H1467J104": "CB",     # Chubb (Bermuda CUSIP)
    "833445115": "SNOW",   # Snowflake
    "023135106": "AMZN",   # Amazon
    "47233W109": "JEF",    # Jefferies Financial
    "867224108": "SU",     # Suncor Energy
    "G8475D109": "SU",     # Suncor (alt)
    "G6683N103": "NU",     # Nu Holdings
    "G2519Y108": "NU",     # Nu Holdings (alt)
    "90384S303": "ULTA",   # Ulta Beauty
    "543900102": "LPX",    # Louisiana-Pacific (old CUSIP)
    "546347105": "LPX",    # Louisiana-Pacific (current)
    "26884L109": "ELV",    # Elevance Health
    "26884L208": "ELV",    # Elevance Health (alt)
    "74144T108": "PSX",    # Phillips 66
    "172967424": "C",      # Citigroup
    "22160K105": "COST",   # Costco
    "594918104": "MSFT",   # Microsoft
    "78468R107": "SPY",    # SPDR S&P 500
    "46625H100": "JPM",    # JPMorgan
    "G8056D109": "LNG",    # Cheniere Energy
    "88160R101": "TSLA",   # Tesla
    "501889208": "ABBV",   # AbbVie
    "254687106": "DIS",    # Disney
    "693475105": "PNC",    # PNC Financial
    "30303M102": "META",   # Meta Platforms
    "744320102": "LOW",    # Lowe's
    "268311107": "EMR",    # Emerson Electric
    "G4124C109": "HEI",    # HEICO
    "422806208": "HEI",    # HEICO (alt)
    "30071E207": "EXPE",   # Expedia
    "02079K305": "GOOGL",  # Alphabet A
    "02079K107": "GOOGL",  # Alphabet A (alt)
    "038222105": "GOOGL",  # Alphabet (old pre-split)
    # Resolved via OpenFIGI
    "047726302": "BATRK",  # Atlanta Braves Holdings C
    "14040H105": "COF",    # Capital One Financial
    "16119P108": "CHTR",   # Charter Communications
    "21036P108": "STZ",    # Constellation Brands A
    "25754A201": "DPZ",    # Domino's Pizza
    "501044101": "KR",     # Kroger
    "512816109": "LAMR",   # Lamar Advertising
    "526057104": "LEN",    # Lennar Corp A
    "526057302": "LEN",    # Lennar Corp B
    "530909100": "LLYVA",  # Liberty Live Holdings A
    "530909308": "LLYVK",  # Liberty Live Holdings C
    "62944T105": "NVR",    # NVR Inc
    "670346105": "NUE",    # Nucor Corp
    "829933100": "SIRI",   # Sirius XM
    "91324P102": "UNH",    # UnitedHealth Group
    "25243Q205": "DEO",    # Diageo ADR
    "531229755": "LBTYA",  # Liberty Broadband
    "67106420":  "OXY",    # OXY (alt CUSIP)
    "478160104": "JNJ",    # Johnson & Johnson
    "931142103": "WMT",    # Walmart
    "459200101": "IBM",    # IBM
    "742718109": "PG",     # Procter & Gamble
    "126650100": "CVS",    # CVS Health
    "097023105": "BN",     # Brookfield Asset Mgmt
    "344849104": "FLT",    # Fleetcor
    "369550108": "GEN",    # Gen Digital
    "585055106": "MED",    # MEDIFAST
    "110122108": "BRK.B",  # Berkshire B
    "050863106": "AXTA",   # Axalta Coating
    "G0176J109": "ATVI",   # Activision (alt)

    # ── Ackman / Pershing Square known holdings ────────────────────────────────
    "205887102": "CNI",    # Canadian National Railway
    "20030N101": "CNP",    # CenterPoint Energy (old position)
    "169656105": "CMG",    # Chipotle Mexican Grill
    "43300A203": "HLT",    # Hilton Worldwide
    "76027X102": "HHH",    # Howard Hughes Holdings
    "76027X309": "HHH",    # Howard Hughes (old class)
    "76131D103": "QSR",    # Restaurant Brands International
    "G0083B108": "QSR",    # Restaurant Brands (Cayman)
    "654106103": "NKE",    # Nike
    "11282X113": "BN",     # Brookfield Asset Mgmt (alt)
    "11271J107": "BAM",    # Brookfield Asset Mgmt
    "17275R102": "CPRI",   # Capri Holdings
    "20825C104": "CP",     # Canadian Pacific (now CPKC)
    "G1902G107": "CPKC",   # CPKC (merged CP+KCS)
    "55262C100": "LPX",    # Louisiana Pacific (old position)
    "532457108": "LOW",    # Lowe's (old position)  — duplicate key handled above
    "92279V303": "VRX",    # Valeant — notorious loss
    "716748108": "PSX",    # Phillips 66
    "464286107": "IRM",    # Iron Mountain
    "55027E102": "UMG",    # Universal Music (Dutch listing)
    "656553104": "NOC",    # Northrop Grumman
    "11284V105": "BX",     # Blackstone
    "G09702104": "FNF",    # FNF Group (Fidelity National)
    "34956Q109": "NET",    # Cloudflare
    "718172109": "PM",     # Philip Morris
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(cik: str, period_end: str) -> Path:
    return CACHE_DIR / cik / f"{period_end}.parquet"


def _load_cached_holdings(cik: str, period_end: str) -> Optional[pd.DataFrame]:
    p = _cache_path(cik, period_end)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_cached_holdings(cik: str, period_end: str, df: pd.DataFrame) -> None:
    p = _cache_path(cik, period_end)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _load_cached_filings(cik: str) -> Optional[pd.DataFrame]:
    p = CACHE_DIR / cik / "_filings.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_cached_filings(cik: str, df: pd.DataFrame) -> None:
    p = CACHE_DIR / cik / "_filings.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


# ── SEC EDGAR helpers ─────────────────────────────────────────────────────────

def _sec_get(url: str, retry: int = 3) -> requests.Response:
    for attempt in range(retry):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retry - 1:
                raise
            time.sleep(1)
    raise RuntimeError(f"Failed: {url}")


def get_13f_filings(cik: str, use_cache: bool = True) -> pd.DataFrame:
    """Return all 13-F-HR filings for the CIK, oldest first."""
    if use_cache:
        cached = _load_cached_filings(cik)
        if cached is not None:
            # Check if cache is stale: if most recent filing is < 90 days old,
            # re-fetch in case a new quarter was filed
            latest = pd.to_datetime(cached["filing_date"].max())
            if (pd.Timestamp.now() - latest).days < 90:
                return cached

    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _sec_get(url).json()

    rows = []
    def _collect(filings_block: dict) -> None:
        for form, fd, acc, period in zip(
            filings_block.get("form", []),
            filings_block.get("filingDate", []),
            filings_block.get("accessionNumber", []),
            filings_block.get("reportDate", []),
        ):
            if form in ("13F-HR", "13F-HR/A"):
                rows.append({"form": form, "period_end": period,
                              "filing_date": fd, "accession_number": acc})

    _collect(data.get("filings", {}).get("recent", {}))
    for page_meta in data.get("filings", {}).get("files", []):
        try:
            page = _sec_get(
                f"https://data.sec.gov/submissions/{page_meta['name']}"
            ).json()
            _collect(page)
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.sort_values(["period_end", "filing_date"])
            .drop_duplicates("period_end", keep="last")
            .reset_index(drop=True))

    if use_cache:
        _save_cached_filings(cik, df)
    return df


def parse_13f_holdings(
    accession_number: str,
    cik_numeric: str,
    period_end: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download and parse the infotable XML for one 13-F filing.
    Returns: cusip, company_name, shares, value_raw, share_type, discretion.
    """
    cik_str = cik_numeric.lstrip("0") or "0"

    if use_cache and period_end:
        cached = _load_cached_holdings(f"CIK{cik_numeric.zfill(10)}", period_end)
        if cached is not None:
            return cached

    acc_nodash = accession_number.replace("-", "")
    index_url  = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_str}/"
        f"{acc_nodash}/{accession_number}-index.htm"
    )
    try:
        idx_html = _sec_get(index_url).text
    except Exception:
        return pd.DataFrame()

    xml_candidates = re.findall(
        r'href="(/Archives/edgar/data/[^"]+\.xml)"',
        idx_html, re.IGNORECASE,
    )
    xml_url = None
    for href in xml_candidates:
        fname = href.split("/")[-1].lower()
        if fname != "primary_doc.xml" and "xsl" not in href:
            xml_url = "https://www.sec.gov" + href
            break
    if xml_url is None:
        for href in xml_candidates:
            if "primary_doc" not in href.lower():
                xml_url = "https://www.sec.gov" + href
                break
    if xml_url is None:
        return pd.DataFrame()

    try:
        xml_text = _sec_get(xml_url).text
        df = _parse_infotable_xml(xml_text)
    except Exception as e:
        print(f"  Warning: {xml_url}: {e}")
        return pd.DataFrame()

    if use_cache and period_end and not df.empty:
        _save_cached_holdings(f"CIK{cik_numeric.zfill(10)}", period_end, df)
    return df


def _parse_infotable_xml(xml_text: str) -> pd.DataFrame:
    xml_text = xml_text.replace(' xmlns="', ' xmlns_ignored="')
    root = ET.fromstring(xml_text)
    rows = []
    for entry in root.iter():
        if entry.tag.split("}")[-1].lower() != "infotable":
            continue
        row: dict = {}
        for child in entry:
            ctag = child.tag.split("}")[-1].lower()
            if ctag == "nameofissuer":
                row["company_name"] = (child.text or "").strip()
            elif ctag == "cusip":
                row["cusip"] = (child.text or "").strip()
            elif ctag == "value":
                try:
                    row["value_raw"] = int(str(child.text).replace(",", "").strip())
                except Exception:
                    row["value_raw"] = None
            elif ctag == "shrsorprnamt":
                for sub in child:
                    stag = sub.tag.split("}")[-1].lower()
                    if stag == "sshprnamt":
                        try:
                            row["shares"] = int(str(sub.text).replace(",", "").strip())
                        except Exception:
                            row["shares"] = None
                    elif stag == "sshprnamttype":
                        row["share_type"] = (sub.text or "").strip()
            elif ctag == "investmentdiscretion":
                row["discretion"] = (child.text or "").strip()
        if row.get("cusip"):
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── CUSIP → ticker resolution ─────────────────────────────────────────────────

def resolve_tickers(cusips: list[str]) -> dict[str, str]:
    """Map CUSIPs to tickers using hardcoded map then OpenFIGI for misses."""
    result: dict[str, str] = {}
    misses: list[str] = []
    for c in cusips:
        if c in CUSIP_MAP:
            result[c] = CUSIP_MAP[c]
        else:
            misses.append(c)

    for i in range(0, len(misses), 10):
        batch = misses[i:i+10]
        try:
            payload = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
            resp = requests.post(
                "https://api.openfigi.com/v3/mapping",
                json=payload,
                headers={**SEC_HEADERS, "Content-Type": "application/json"},
                timeout=20,
            )
            if resp.status_code == 200:
                for cusip, entry in zip(batch, resp.json()):
                    data = entry.get("data", [])
                    for item in data:
                        if ("Common" in item.get("securityType", "") and
                                item.get("exchCode", "") in ("US", "UN", "UW", "UA", "UQ")):
                            result[cusip] = item["ticker"]
                            break
                    else:
                        if data:
                            result[cusip] = data[0].get("ticker", "")
            time.sleep(0.3)
        except Exception:
            pass
    return result


# ── Price data via DuckDB ─────────────────────────────────────────────────────

def load_prices(tickers: list[str]) -> pd.DataFrame:
    """Load daily adj_close for tickers from the OHLCV parquet store."""
    conn = duckdb.connect()
    tl   = ", ".join(f"'{t}'" for t in tickers)
    try:
        df = conn.execute(f"""
            SELECT ticker,
                   CAST(event_timestamp AS DATE) AS date,
                   adj_close AS close
            FROM read_parquet('{DATA_ROOT}/ohlcv/**/*.parquet',
                              hive_partitioning=true, union_by_name=true)
            WHERE ticker IN ({tl})
              AND adj_close IS NOT NULL AND adj_close > 0
            ORDER BY ticker, date
        """).fetchdf()
        return df
    finally:
        conn.close()


def _price_on(prices: pd.DataFrame, ticker: str, target_date: date) -> Optional[float]:
    """Last available close on or before target_date."""
    sub = prices[
        (prices["ticker"] == ticker) & (prices["date"] <= pd.Timestamp(target_date))
    ]
    return float(sub.iloc[-1]["close"]) if not sub.empty else None


# ── Portfolio construction with change detection ──────────────────────────────

def build_portfolio_history(
    filings: pd.DataFrame,
    cik_numeric: str,
    start_year: int = 2015,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download, parse, and aggregate holdings for each filing.
    Adds position_change column: new / held / increased / decreased / closed.
    Returns long-format DataFrame, one row per (period_end, cusip).
    """
    filings = filings[filings["period_end"] >= f"{start_year}-01-01"].copy()
    if verbose:
        print(f"Processing {len(filings)} filings from {start_year}…")

    all_quarters: list[pd.DataFrame] = []

    prev_shares: dict[str, int] = {}   # cusip → shares from last quarter

    for _, row in filings.iterrows():
        period  = row["period_end"]
        fdate   = row["filing_date"]
        acc     = row["accession_number"]
        if verbose:
            print(f"  {period}  (filed {fdate})", end="", flush=True)

        raw = parse_13f_holdings(
            acc, cik_numeric=cik_numeric,
            period_end=period, use_cache=use_cache,
        )
        if raw.empty:
            if verbose:
                print("  → no data")
            prev_shares = {}
            continue

        # Keep equity shares only; aggregate across manager subsidiaries
        eq = raw[
            raw["share_type"].isin(["SH", "SH ", ""]) | raw["share_type"].isna()
        ].copy()
        if eq.empty:
            if verbose:
                print("  → no equity rows")
            prev_shares = {}
            continue

        holdings = (
            eq.groupby(["cusip", "company_name"], as_index=False)
            .agg(shares=("shares", "sum"), value_raw=("value_raw", "sum"))
        )
        holdings["share_type"] = "SH"

        # Ticker resolution
        cmap = resolve_tickers(holdings["cusip"].tolist())
        holdings["ticker"] = holdings["cusip"].map(cmap)

        # Portfolio weight
        total_val = holdings["value_raw"].sum()
        holdings["pct_portfolio"] = (
            holdings["value_raw"] / total_val if total_val > 0 else float("nan")
        )

        # Position change vs previous quarter
        def _change(row_) -> str:
            c = row_["cusip"]
            s = row_["shares"] or 0
            if c not in prev_shares:
                return "new"
            prev = prev_shares[c]
            if prev == 0:
                return "new"
            ratio = s / prev
            if ratio > 1.05:
                return "increased"
            if ratio < 0.95:
                return "decreased"
            return "held"

        holdings["position_change"] = holdings.apply(_change, axis=1)

        # Detect closed positions (in prev but not in this quarter)
        current_cusips = set(holdings["cusip"])
        closed_cusips  = set(prev_shares) - current_cusips
        if closed_cusips:
            closed_rows = []
            for c in closed_cusips:
                closed_rows.append({
                    "cusip":           c,
                    "company_name":    "",
                    "shares":          0,
                    "value_raw":       0,
                    "share_type":      "SH",
                    "ticker":          cmap.get(c),
                    "pct_portfolio":   0.0,
                    "position_change": "closed",
                })
            holdings = pd.concat(
                [holdings, pd.DataFrame(closed_rows)], ignore_index=True
            )

        holdings["period_end"]  = period
        holdings["filing_date"] = fdate
        all_quarters.append(holdings)

        # Update prev_shares (only non-closed positions)
        prev_shares = {
            r["cusip"]: r["shares"]
            for _, r in holdings[holdings["position_change"] != "closed"].iterrows()
            if r["shares"]
        }

        if verbose:
            n_pos    = (holdings["position_change"] != "closed").sum()
            resolved = holdings["ticker"].notna().sum()
            n_new    = (holdings["position_change"] == "new").sum()
            n_closed = (holdings["position_change"] == "closed").sum()
            print(f"  → {n_pos} pos  ({n_new} new, {n_closed} closed)  "
                  f"{resolved} tickers resolved")

        time.sleep(0.12)

    if not all_quarters:
        return pd.DataFrame()
    return pd.concat(all_quarters, ignore_index=True)


# ── Holding quarter index ─────────────────────────────────────────────────────

def add_holding_quarter(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Add holding_quarter: how many consecutive quarters the position has been held
    as of each period_end (1 = just opened this quarter).
    """
    portfolio = portfolio.copy()
    portfolio["holding_quarter"] = 0

    periods = sorted(portfolio["period_end"].unique())
    counters: dict[str, int] = {}   # cusip → consecutive quarters held

    for period in periods:
        mask = portfolio["period_end"] == period
        q    = portfolio[mask].copy()

        new_counters: dict[str, int] = {}
        for i, row in q.iterrows():
            c = row["cusip"]
            if row["position_change"] == "closed":
                continue
            new_counters[c] = counters.get(c, 0) + 1
            portfolio.at[i, "holding_quarter"] = new_counters[c]

        counters = new_counters

    return portfolio


# ── Performance computation ───────────────────────────────────────────────────

def compute_performance(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    copycat: bool = False,
    confidential_threshold: int = CONFIDENTIAL_THRESHOLD,
) -> dict:
    """
    Compute quarterly and full holding-period performance.

    Quarterly view: entry at Q-end (or filing date if --copycat),
                    exit at next Q-end.
    Holding period view: entry at first quarter the position appears,
                         exit at the last quarter before it's closed
                         (or current date if still held).
    """
    pf = portfolio.copy()
    pf["period_end"]  = pd.to_datetime(pf["period_end"])
    pf["filing_date"] = pd.to_datetime(pf["filing_date"])

    # Only active positions for return calculations
    active = pf[
        (pf["position_change"] != "closed") &
        pf["ticker"].notna() &
        (pf["ticker"] != "")
    ].copy()

    periods = sorted(active["period_end"].unique())
    available_tickers = set(prices["ticker"].unique())

    # ── Per-position portfolio value with prices ──────────────────────────────
    # Used to compute coverage % per quarter
    coverage_by_quarter: dict[str, float] = {}
    for period in periods:
        qdf = active[active["period_end"] == period]
        total_val = qdf["value_raw"].sum()
        if total_val <= 0:
            coverage_by_quarter[str(period.date())] = 0.0
            continue
        covered = qdf[qdf["ticker"].isin(available_tickers)]["value_raw"].sum()
        coverage_by_quarter[str(period.date())] = float(covered / total_val)

    # ── Quarterly position returns ────────────────────────────────────────────
    pos_rows: list[dict] = []
    for i, period in enumerate(periods):
        qdf = active[active["period_end"] == period].copy()

        entry_date = (
            qdf["filing_date"].iloc[0].date() if copycat else period.date()
        )
        exit_date = (
            periods[i + 1].date() if i + 1 < len(periods) else date.today()
        )

        is_conf = len(qdf) < confidential_threshold

        spy_entry = _price_on(prices, "SPY", entry_date)
        spy_exit  = _price_on(prices, "SPY", exit_date)
        spy_ret   = ((spy_exit / spy_entry) - 1) if (spy_entry and spy_exit) else None

        for _, pos in qdf.iterrows():
            ticker = pos["ticker"]
            entry  = _price_on(prices, ticker, entry_date)
            exit_p = _price_on(prices, ticker, exit_date)
            if entry is None or exit_p is None or entry <= 0:
                continue
            pos_ret = (exit_p / entry) - 1
            alpha   = ((1 + pos_ret) / (1 + spy_ret) - 1) if spy_ret is not None else None
            pos_rows.append({
                "period_end":       str(period.date()),
                "filing_date":      str(pos["filing_date"].date()),
                "ticker":           ticker,
                "cusip":            pos.get("cusip", ""),
                "company_name":     pos.get("company_name", ""),
                "shares":           int(pos.get("shares") or 0),
                "value_usd":        float(pos.get("value_raw") or 0),
                "pct_portfolio":    float(pos.get("pct_portfolio") or 0),
                "position_change":  pos.get("position_change", ""),
                "holding_quarter":  int(pos.get("holding_quarter") or 0),
                "is_confidential":  bool(is_conf),
                "entry_date":       str(entry_date),
                "exit_date":        str(exit_date),
                "entry_price":      round(entry, 4),
                "exit_price":       round(exit_p, 4),
                "return":           round(pos_ret, 6),
                "benchmark_return": round(spy_ret, 6) if spy_ret is not None else None,
                "alpha":            round(alpha, 6) if alpha is not None else None,
            })

    pos_df = pd.DataFrame(pos_rows)

    # ── Quarterly portfolio (value-weighted) ──────────────────────────────────
    qtly_rows: list[dict] = []
    for i, period in enumerate(periods):
        pstr = str(period.date())
        qdf  = pos_df[pos_df["period_end"] == pstr].copy()
        qdf  = qdf.dropna(subset=["return", "pct_portfolio"])
        total_w = qdf["pct_portfolio"].sum()
        if total_w <= 0:
            continue
        qdf["w"] = qdf["pct_portfolio"] / total_w
        port_ret = float((qdf["w"] * qdf["return"]).sum())
        spy_ret  = qdf["benchmark_return"].iloc[0] if len(qdf) else None
        is_conf  = len(active[active["period_end"] == period]) < confidential_threshold
        filing_d = str(active[active["period_end"] == period]["filing_date"].iloc[0].date())
        qtly_rows.append({
            "period_end":       pstr,
            "filing_date":      filing_d,
            "portfolio_return": round(port_ret, 6),
            "benchmark_return": round(spy_ret, 6) if spy_ret is not None else None,
            "alpha":            round((1 + port_ret) / (1 + spy_ret) - 1, 6) if spy_ret is not None else None,
            "n_positions":      int(len(active[active["period_end"] == period])),
            "coverage_pct":     round(coverage_by_quarter.get(pstr, 0), 4),
            "is_confidential":  bool(is_conf),
        })
    qtly_df = pd.DataFrame(qtly_rows)

    # ── Full holding period returns ───────────────────────────────────────────
    # Deduplicate by (ticker, period_end): when the same ticker appears under
    # multiple CUSIPs (e.g. two share classes both resolved to "GOOGL"), keep
    # only the dominant row (largest value_raw) so run-detection isn't confused.
    pf_hp = (
        pf[pf["ticker"].notna() & (pf["ticker"] != "")]
        .sort_values("value_raw", ascending=False)
        .drop_duplicates(subset=["ticker", "period_end"], keep="first")
        .sort_values(["ticker", "period_end"])
    )

    hp_rows: list[dict] = []
    for ticker, grp in pf_hp.groupby("ticker"):
        if not ticker or pd.isna(ticker):
            continue
        grp = grp.sort_values("period_end")

        # Identify contiguous holding runs (position may be closed then re-opened)
        runs: list[list] = []
        current_run: list = []
        for _, r in grp.iterrows():
            if r["position_change"] == "new":
                if current_run:
                    runs.append(current_run)
                current_run = [r]
            elif r["position_change"] == "closed":
                if current_run:
                    runs.append(current_run)
                current_run = []
            else:
                if current_run:
                    current_run.append(r)
                else:
                    current_run = [r]   # started before our analysis window
        if current_run:
            runs.append(current_run)

        for run in runs:
            first   = run[0]
            last    = run[-1]
            is_open = (last["position_change"] != "closed")

            open_q  = str(pd.Timestamp(first["period_end"]).date())
            close_q = None if is_open else str(pd.Timestamp(last["period_end"]).date())

            open_date  = pd.Timestamp(first["period_end"]).date()
            close_date = date.today() if is_open else pd.Timestamp(last["period_end"]).date()

            open_p  = _price_on(prices, ticker, open_date)
            close_p = _price_on(prices, ticker, close_date)

            spy_o   = _price_on(prices, "SPY", open_date)
            spy_c   = _price_on(prices, "SPY", close_date)

            if not (open_p and close_p and open_p > 0):
                continue

            tot_ret  = (close_p / open_p) - 1
            spy_ret  = ((spy_c / spy_o) - 1) if (spy_o and spy_c) else None
            alpha    = ((1 + tot_ret) / (1 + spy_ret) - 1) if spy_ret is not None else None
            peak_wt  = float(grp[grp["position_change"] != "closed"]["pct_portfolio"].max())

            hp_rows.append({
                "ticker":             ticker,
                "cusip":              str(first.get("cusip", "")),
                "company_name":       str(first.get("company_name", "")),
                "open_quarter":       open_q,
                "close_quarter":      close_q,
                "is_open":            bool(is_open),
                "open_price":         round(open_p, 4),
                "close_price":        round(close_p, 4),
                "total_return":       round(tot_ret, 6),
                "benchmark_return":   round(spy_ret, 6) if spy_ret is not None else None,
                "alpha":              round(alpha, 6) if alpha is not None else None,
                "duration_quarters":  len(run),
                "peak_pct_portfolio": round(peak_wt, 4),
            })

    hp_df = pd.DataFrame(hp_rows)

    # ── Summary metrics (exclude confidential quarters from aggregates) ────────
    clean_qtly = qtly_df[~qtly_df["is_confidential"]]
    clean_pos  = pos_df[~pos_df["is_confidential"]]

    n_q = len(clean_qtly)
    if n_q > 0:
        q_rets   = clean_qtly["portfolio_return"].dropna()
        spy_q    = clean_qtly["benchmark_return"].dropna()
        port_ann = float((np.prod(1 + q_rets)) ** (4 / n_q) - 1) if len(q_rets) else None
        spy_ann  = float((np.prod(1 + spy_q))  ** (4 / n_q) - 1) if len(spy_q)  else None
        # Max drawdown on quarterly portfolio returns
        cum = (1 + q_rets).cumprod()
        peak = cum.cummax()
        max_dd = float(((cum - peak) / peak).min()) if len(cum) else None
    else:
        port_ann = spy_ann = max_dd = None

    alp_ann = (port_ann - spy_ann) if (port_ann is not None and spy_ann is not None) else None

    alpha_q = clean_qtly["alpha"].dropna()
    ir = (float(alpha_q.mean() / alpha_q.std()) * np.sqrt(4)
          if len(alpha_q) > 1 and alpha_q.std() > 0 else None)

    completed = clean_pos.dropna(subset=["return", "alpha"])
    hit_rate     = float((completed["return"] > 0).mean()) if len(completed) else None
    hit_vs_spy   = float((completed["alpha"]  > 0).mean()) if len(completed) else None
    avg_ret      = float(completed["return"].mean())        if len(completed) else None
    avg_alpha    = float(completed["alpha"].mean())         if len(completed) else None

    top5_wt = float(
        active.groupby("period_end")
        .apply(lambda g: g.nlargest(5, "pct_portfolio")["pct_portfolio"].sum(),
               include_groups=False)
        .mean()
    ) if not active.empty else None

    # Coverage stats
    tickers_needed  = set(active["ticker"].dropna().unique())
    missing_tickers = sorted(tickers_needed - available_tickers - {""})
    conf_quarters   = [r["period_end"] for r in qtly_rows if r["is_confidential"]]

    portfolio_metrics = {
        "annualized_return":          round(port_ann, 4) if port_ann is not None else None,
        "annualized_benchmark_return":round(spy_ann,  4) if spy_ann  is not None else None,
        "annualized_alpha":           round(alp_ann,  4) if alp_ann  is not None else None,
        "information_ratio":          round(ir,       4) if ir       is not None else None,
        "max_drawdown_quarterly":     round(max_dd,   4) if max_dd   is not None else None,
    }
    position_metrics = {
        "total_position_quarters":  len(completed),
        "unique_tickers":           int(active["ticker"].nunique()),
        "hit_rate":                 round(hit_rate,   4) if hit_rate   is not None else None,
        "hit_rate_vs_benchmark":    round(hit_vs_spy, 4) if hit_vs_spy is not None else None,
        "avg_quarterly_return":     round(avg_ret,    4) if avg_ret    is not None else None,
        "avg_quarterly_alpha":      round(avg_alpha,  4) if avg_alpha  is not None else None,
        "avg_top5_concentration":   round(top5_wt,    4) if top5_wt   is not None else None,
    }
    coverage = {
        "n_quarters":                  len(qtly_rows),
        "n_clean_quarters":            len(clean_qtly),
        "confidential_quarters":       conf_quarters,
        "tickers_with_prices":         int(len(tickers_needed - set(missing_tickers))),
        "tickers_missing_prices":      missing_tickers,
        "avg_coverage_pct":            round(float(qtly_df["coverage_pct"].mean()), 4),
    }

    return {
        "portfolio_metrics": portfolio_metrics,
        "position_metrics":  position_metrics,
        "coverage":          coverage,
        "quarterly_returns": qtly_df,
        "positions":         pos_df,
        "holding_periods":   hp_df,
    }


# ── JSON schema output ────────────────────────────────────────────────────────

def build_json_schema(
    results: dict,
    manager_meta: dict,
    start_year: int = 2015,
    copycat: bool = False,
) -> dict:
    """
    Assemble the machine-readable JSON schema.

    Parameters
    ----------
    results      : dict returned by compute_performance()
    manager_meta : {"name": ..., "entity": ..., "cik": ..., "strategy": ...}
    start_year   : first year of the analysis window
    copycat      : whether filing-date entry prices were used
    """
    qtly = results["quarterly_returns"]
    pos  = results["positions"]
    hp   = results["holding_periods"]

    schema = {
        "schema_version": "1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "manager":        manager_meta,
        "methodology": {
            "entry":      "filing_date_close" if copycat else "quarter_end_close",
            "exit":       "next_quarter_end_close",
            "benchmark":  "SPY",
            "weighting":  "value_weighted_13f",
            "start_year": start_year,
        },
        "coverage":          results["coverage"],
        "portfolio_metrics": results["portfolio_metrics"],
        "position_metrics":  results["position_metrics"],
        "quarterly_returns": qtly.to_dict(orient="records"),
        "positions":         pos.to_dict(orient="records"),
        "holding_periods":   hp.to_dict(orient="records"),
    }
    return schema


def save_json(schema: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(schema, f, indent=2, default=str)


# ── Human-readable report ─────────────────────────────────────────────────────

def print_report(
    results: dict,
    schema: dict,
    manager_meta: dict,
    copycat: bool = False,
) -> None:
    pm  = results["portfolio_metrics"]
    psm = results["position_metrics"]
    cov = results["coverage"]
    qtly = results["quarterly_returns"]
    pos  = results["positions"]
    hp   = results["holding_periods"]

    print("\n" + "=" * 72)
    print(f"  {manager_meta['name']} / {manager_meta['entity']}  —  13-F Performance Report")
    label = "Entry: filing-date close (copycat)" if copycat else "Entry: quarter-end close (stock-picking skill)"
    print(f"  {label}")
    print("=" * 72)

    print(f"\n  Coverage  ({cov['n_quarters']} quarters total, "
          f"{cov['n_clean_quarters']} usable)")
    print(f"    Tickers with prices : {cov['tickers_with_prices']}")
    print(f"    Missing prices      : {', '.join(cov['tickers_missing_prices']) or 'none'}")
    print(f"    Avg price coverage  : {cov['avg_coverage_pct']*100:.0f}% of portfolio value")
    print(f"    Confidential qtrs   : {cov['confidential_quarters'] or 'none'}")

    print("\n  Portfolio metrics (confidential quarters excluded)")
    _pct = lambda v: f"{v*100:+.1f}%" if v is not None else "n/a"
    _f2  = lambda v: f"{v:.2f}"       if v is not None else "n/a"
    print(f"    Annualized return   : {_pct(pm['annualized_return'])}")
    print(f"    Annualized SPY      : {_pct(pm['annualized_benchmark_return'])}")
    print(f"    Annualized alpha    : {_pct(pm['annualized_alpha'])}")
    print(f"    Information ratio   : {_f2(pm['information_ratio'])}")
    print(f"    Max quarterly DD    : {_pct(pm['max_drawdown_quarterly'])}")

    print("\n  Position metrics")
    print(f"    Position-quarters   : {psm['total_position_quarters']}")
    print(f"    Unique tickers      : {psm['unique_tickers']}")
    print(f"    Hit rate            : {psm['hit_rate']*100:.1f}%" if psm['hit_rate'] else "    Hit rate            : n/a")
    print(f"    Beat SPY rate       : {psm['hit_rate_vs_benchmark']*100:.1f}%" if psm['hit_rate_vs_benchmark'] else "    Beat SPY rate       : n/a")
    print(f"    Avg top-5 weight    : {psm['avg_top5_concentration']*100:.1f}%" if psm['avg_top5_concentration'] else "    Avg top-5 weight    : n/a")

    # Quarterly table
    print("\n  Quarterly returns")
    print(f"  {'Quarter':<12} {'Portfolio':>10} {'SPY':>8} {'Alpha':>8} "
          f"{'#Pos':>5} {'Cover':>6} {'Note'}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*5} {'-'*6} {'-'*8}")
    for _, qr in qtly.iterrows():
        pr_s = f"{qr['portfolio_return']*100:+.1f}%" if pd.notna(qr['portfolio_return']) else "n/a"
        sp_s = f"{qr['benchmark_return']*100:+.1f}%" if pd.notna(qr['benchmark_return']) else "n/a"
        al_s = f"{qr['alpha']*100:+.1f}%"            if pd.notna(qr['alpha'])            else "n/a"
        cv_s = f"{qr['coverage_pct']*100:.0f}%"
        flag = " [CONF]" if qr["is_confidential"] else ""
        print(f"  {qr['period_end']:<12} {pr_s:>10} {sp_s:>8} {al_s:>8} "
              f"{qr['n_positions']:>5} {cv_s:>6}{flag}")

    # Holding periods table
    print("\n  Full holding period returns (top/bottom by alpha)")
    hpd = hp.dropna(subset=["alpha"]).copy()
    hpd["status"] = hpd["is_open"].map({True: "OPEN", False: "closed"})
    top5_hp   = hpd.nlargest(5,  "alpha")
    bot5_hp   = hpd.nsmallest(5, "alpha")
    hdr = f"  {'Ticker':<8} {'Opened':<12} {'Closed':<12} {'Qtrs':>5} {'Total':>8} {'Alpha':>8}  Status"
    sep = f"  {'-'*8} {'-'*12} {'-'*12} {'-'*5} {'-'*8} {'-'*8}  {'-'*8}"
    print("\n  Top 5:")
    print(hdr); print(sep)
    for _, r in top5_hp.iterrows():
        print(f"  {r['ticker']:<8} {str(r['open_quarter']):<12} "
              f"{str(r['close_quarter'] or '-'):<12} {r['duration_quarters']:>5} "
              f"{r['total_return']*100:>+7.1f}% {r['alpha']*100:>+7.1f}%  {r['status']}")
    print("\n  Bottom 5:")
    print(hdr); print(sep)
    for _, r in bot5_hp.iterrows():
        print(f"  {r['ticker']:<8} {str(r['open_quarter']):<12} "
              f"{str(r['close_quarter'] or '-'):<12} {r['duration_quarters']:>5} "
              f"{r['total_return']*100:>+7.1f}% {r['alpha']*100:>+7.1f}%  {r['status']}")

    # Position change breakdown
    print("\n  Position change breakdown (latest quarter excl. closed)")
    latest_q = pos[pos["period_end"] == pos["period_end"].max()]
    change_counts = latest_q["position_change"].value_counts()
    for chg, cnt in change_counts.items():
        print(f"    {chg:<12}: {cnt}")

    print("=" * 72)
