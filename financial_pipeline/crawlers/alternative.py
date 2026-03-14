"""
Alternative / sentiment data crawlers.

Three sources:
  1. OpenInsider  — insider buying/selling filings (Form 4).
  2. SEC EDGAR 13F — institutional holdings changes (Buffett, Ackman, etc.).
  3. Capitol Trades — US Congressional trading disclosures.

PiT rules:
  - Form 4 (insider): knowledge_timestamp = SEC filing date (2-business-day
    deadline after the transaction). event_timestamp = transaction date.
  - 13F: knowledge_timestamp = filing date (≤45 days after quarter end).
         event_timestamp = quarter end date.
  - Congress: knowledge_timestamp = disclosure date (up to 45 days after trade).
              event_timestamp = trade date.

All are stored in `financial_events` with appropriate data_category.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, date, timedelta
from typing import Any

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from financial_pipeline.config import CONFIG, DataCategory
from financial_pipeline.crawlers.base import (
    TokenBucket,
    bulk_upsert,
    make_session_factory,
    stable_source_id,
    utcnow,
    with_retry,
)
from financial_pipeline.db.models import FinancialEvent

log = logging.getLogger(__name__)


# ── OpenInsider crawler ───────────────────────────────────────────────────────

class InsiderTradeCrawler:
    """
    Scrapes OpenInsider.com for Form 4 insider buy/sell filings.

    OpenInsider structures its data as an HTML table sortable by date.
    We request the CSV export endpoint which gives clean columnar data.

    knowledge_timestamp = filing_date (when SEC accepted the Form 4).
    event_timestamp     = trade_date  (when the insider actually traded).
    """

    CSV_URL = (
        "http://openinsider.com/screener?s={ticker}"
        "&o=&pl=&ph=&ll=&lh=&fd=1826&fdr=&td=0&tdr=&fdlyl=&fdlyh="
        "&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999"
        "&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
        "&sortcol=0&cnt=500&action=1"
    )

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()
        self._bucket = TokenBucket(CONFIG.rate_limits.open_insider)

    async def fetch_for_ticker(self, ticker: str) -> list[dict]:
        await self._bucket.acquire()
        url = self.CSV_URL.format(ticker=ticker)
        try:
            async with httpx.AsyncClient(
                timeout=30,
                headers={"User-Agent": CONFIG.api_keys.sec_user_agent},
                follow_redirects=True,
            ) as client:
                resp = await with_retry(client.get, url)
        except Exception as exc:
            log.warning("OpenInsider %s: %s", ticker, exc)
            return []

        return self._parse(resp.text, ticker)

    @staticmethod
    def _parse(html: str, ticker: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            return []

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        records: list[dict] = []

        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue
            data = {h: cells[i].get_text(strip=True) for i, h in enumerate(headers) if i < len(cells)}

            try:
                trade_date = datetime.strptime(data.get("trade date", ""), "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                filing_date = datetime.strptime(data.get("filing date", ""), "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                try:
                    trade_date = pd.to_datetime(data.get("trade date", "")).replace(tzinfo=timezone.utc)
                    filing_date = pd.to_datetime(data.get("filing date", "")).replace(tzinfo=timezone.utc)
                except Exception:
                    continue

            price_str = re.sub(r"[,$]", "", data.get("price", "") or "")
            qty_str = re.sub(r"[,+]", "", data.get("qty", "") or "")
            val_str = re.sub(r"[,$+]", "", data.get("value", "") or "").replace("K", "000").replace("M", "000000")

            try:
                price = float(price_str) if price_str else None
            except ValueError:
                price = None
            try:
                qty = int(qty_str) if qty_str else None
            except ValueError:
                qty = None
            try:
                value = float(val_str) if val_str else None
            except ValueError:
                value = None

            trade_type = data.get("trade type", "").upper()  # P = Purchase, S = Sale
            insider_name = data.get("insider name", "")
            title = data.get("title", "")

            sid = stable_source_id(
                {"ticker": ticker, "insider": insider_name,
                 "trade_date": str(trade_date.date()), "qty": str(qty), "type": trade_type},
                ["ticker", "insider", "trade_date", "qty", "type"],
            )

            records.append({
                "event_timestamp":    trade_date,
                "knowledge_timestamp": filing_date,
                "ticker":             ticker,
                "data_category":      DataCategory.INSIDER_TRADE,
                "data_source":        "openinsider",
                "headline": (
                    f"{insider_name} ({title}) {trade_type} {qty:,} "
                    f"shares of {ticker} @ ${price:.2f}"
                    if qty and price else f"{insider_name} insider trade {ticker}"
                ),
                "payload": {
                    "insider_name":   insider_name,
                    "title":          title,
                    "trade_type":     trade_type,   # P/S/A/D/F/...
                    "price":          price,
                    "qty":            qty,
                    "value":          value,
                    "owned_after":    data.get("owned after trade"),
                    "delta_own_pct":  data.get("delta own"),
                    "schema_version": 1,
                },
                "schema_version": 1,
                "source_id":          sid,
            })
        return records

    async def backfill(self, tickers: list[str]) -> None:
        async with self._sf() as session:
            for ticker in tickers:
                records = await self.fetch_for_ticker(ticker)
                if records:
                    await bulk_upsert(
                        session,
                        FinancialEvent,
                        records,
                        conflict_columns=["source_id"],
                        update_columns=["payload"],
                    )
                    log.info("Insider %s: %d trades", ticker, len(records))


# ── SEC 13F Crawler ───────────────────────────────────────────────────────────

# Notable institutional investors with their CIKs
FAMOUS_INVESTORS: dict[str, str] = {
    "berkshire_hathaway": "0001067983",   # Warren Buffett
    "pershing_square":    "0001336528",   # Bill Ackman
    "bridgewater":        "0001350694",   # Ray Dalio
    "renaissance_tech":   "0001037389",   # Jim Simons
    "third_point":        "0001040273",   # Dan Loeb
    "david_tepper":       "0001418814",   # Appaloosa
    "soros_fund":         "0001029160",   # George Soros
    "tiger_global":       "0001167483",
}

EDGAR_BASE = "https://data.sec.gov"


class Portfolio13FCrawler:
    """
    Fetches 13F-HR filings from SEC EDGAR for famous institutional investors.

    knowledge_timestamp = 13F filing date (≤45 days after quarter end).
    event_timestamp     = quarter end date (the period the holdings reflect).

    Returns changes in portfolio holdings, not absolute positions, to make
    the signal PiT-safe (only available after filing).
    """

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()
        self._bucket = TokenBucket(CONFIG.rate_limits.sec_edgar)

    async def backfill_investor(
        self,
        investor_name: str,
        cik: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        await self._bucket.acquire()
        headers = {"User-Agent": CONFIG.api_keys.sec_user_agent}
        submissions_url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"

        try:
            async with httpx.AsyncClient(headers=headers, timeout=30) as client:
                resp = await with_retry(client.get, submissions_url)
                sub_data = resp.json()
        except Exception as exc:
            log.warning("13F submissions %s: %s", investor_name, exc)
            return []

        records = await self._process_submissions(
            sub_data, investor_name, cik, start, end
        )
        return records

    async def _process_submissions(
        self,
        sub_data: dict,
        investor_name: str,
        cik: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        recent = sub_data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        period_dates = recent.get("reportDate", [])
        documents = recent.get("primaryDocument", [])

        all_records: list[dict] = []
        headers = {"User-Agent": CONFIG.api_keys.sec_user_agent}

        for form, filing_date_str, accession, period_str, doc in zip(
            forms, dates, accessions, period_dates, documents
        ):
            if form not in ("13F-HR", "13F-HR/A"):
                continue
            try:
                filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                period_dt = datetime.strptime(period_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if filing_dt < start or filing_dt > end:
                continue

            # Fetch the XML filing to extract holdings
            acc_clean = accession.replace("-", "")
            holdings_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{doc}"
            )

            await self._bucket.acquire()
            try:
                async with httpx.AsyncClient(headers=headers, timeout=60) as client:
                    holdings_resp = await with_retry(client.get, holdings_url)
                holdings = self._parse_13f_xml(holdings_resp.text)
            except Exception as exc:
                log.debug("13F XML parse %s/%s: %s", investor_name, accession, exc)
                holdings = []

            source_id = f"13f_{acc_clean}"
            all_records.append({
                "event_timestamp":    period_dt,
                "knowledge_timestamp": filing_dt,
                "ticker":             None,   # portfolio-level, not ticker-specific
                "data_category":      DataCategory.PORTFOLIO_13F,
                "data_source":        "SEC_EDGAR_13F",
                "headline":           f"{investor_name} 13F filing {filing_date_str}",
                "payload": {
                    "investor_name":  investor_name,
                    "cik":            cik,
                    "period":         period_str,
                    "filing_date":    filing_date_str,
                    "form":           form,
                    "accession":      accession,
                    "holdings":       holdings[:500],   # cap at 500 positions
                    "holdings_count": len(holdings),
                    "schema_version": 1,
                },
                "schema_version": 1,
                "source_id":          source_id,
            })
        return all_records

    @staticmethod
    def _parse_13f_xml(xml_text: str) -> list[dict]:
        """Parse 13F-HR XML holdings into a list of position dicts."""
        soup = BeautifulSoup(xml_text, "xml")
        holdings: list[dict] = []
        for info in soup.find_all("infoTable"):
            try:
                ticker_el = info.find("shrsOrPrnAmt")
                holdings.append({
                    "company_name": (info.find("nameOfIssuer") or BeautifulSoup("", "html.parser")).get_text(strip=True),
                    "cusip":        (info.find("cusip") or BeautifulSoup("", "html.parser")).get_text(strip=True),
                    "value_1000":   int((info.find("value") or BeautifulSoup("0", "html.parser")).get_text(strip=True).replace(",", "")),
                    "shares":       int((info.find("sshPrnamt") or BeautifulSoup("0", "html.parser")).get_text(strip=True).replace(",", "")),
                    "shares_type":  (info.find("sshPrnamtType") or BeautifulSoup("", "html.parser")).get_text(strip=True),
                    "investment_discretion": (info.find("investmentDiscretion") or BeautifulSoup("", "html.parser")).get_text(strip=True),
                    "voting_authority_sole": int((info.find("Sole") or BeautifulSoup("0", "html.parser")).get_text(strip=True).replace(",", "")),
                })
            except (ValueError, AttributeError):
                continue
        return holdings

    async def backfill_all(self, start: datetime, end: datetime) -> None:
        async with self._sf() as session:
            for name, cik in FAMOUS_INVESTORS.items():
                log.info("13F: processing %s (CIK %s)", name, cik)
                records = await self.backfill_investor(name, cik, start, end)
                if records:
                    await bulk_upsert(
                        session,
                        FinancialEvent,
                        records,
                        conflict_columns=["source_id"],
                        update_columns=["payload", "headline"],
                    )
                    log.info("  %s → %d filings", name, len(records))


# ── Congressional Trades Crawler ──────────────────────────────────────────────

class CongressTradeCrawler:
    """
    Fetches Congressional trading disclosures from Capitol Trades API / house.gov.

    knowledge_timestamp = disclosure_date (when the politician filed the report,
                          up to 45 days after the transaction per STOCK Act).
    event_timestamp     = transaction_date (when the trade actually occurred).

    This gap is a genuine information delay: a trader in 2023 could NOT have
    known about a Congress member's June trade until the August disclosure.
    """

    # Capitol Trades provides a public JSON API
    CAPITOL_TRADES_API = "https://www.capitoltrades.com/trades?pageSize=1000&page={page}"

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()
        self._bucket = TokenBucket(CONFIG.rate_limits.congress_trades)

    async def backfill(self, start: datetime, end: datetime, max_pages: int = 50) -> None:
        async with self._sf() as session:
            all_records: list[dict] = []
            for page in range(1, max_pages + 1):
                records = await self._fetch_page(page, start, end)
                if not records:
                    break
                all_records.extend(records)
                log.debug("Congress page %d: %d trades", page, len(records))

            if all_records:
                await bulk_upsert(
                    session,
                    FinancialEvent,
                    all_records,
                    conflict_columns=["source_id"],
                    update_columns=["payload"],
                )
                log.info("Congress trades: %d records stored", len(all_records))

    async def _fetch_page(
        self, page: int, start: datetime, end: datetime
    ) -> list[dict]:
        await self._bucket.acquire()
        url = self.CAPITOL_TRADES_API.format(page=page)
        try:
            async with httpx.AsyncClient(
                timeout=30,
                headers={"User-Agent": CONFIG.api_keys.sec_user_agent},
                follow_redirects=True,
            ) as client:
                resp = await with_retry(client.get, url)
                data = resp.json()
        except Exception as exc:
            log.warning("Capitol Trades page %d: %s", page, exc)
            return []

        trades = data.get("data", []) if isinstance(data, dict) else data
        if not trades:
            return []

        records: list[dict] = []
        for trade in trades:
            try:
                trade_date_str = trade.get("transactionDate") or trade.get("txDate", "")
                disclose_date_str = trade.get("disclosureDate") or trade.get("filedDate", "")
                ticker = (trade.get("ticker") or trade.get("asset", {}).get("ticker", "")).upper()
                if not ticker:
                    continue
                trade_dt = pd.to_datetime(trade_date_str).replace(tzinfo=timezone.utc)
                disclose_dt = pd.to_datetime(disclose_date_str).replace(tzinfo=timezone.utc)
                if disclose_dt < start or disclose_dt > end:
                    continue
            except Exception:
                continue

            politician = (
                trade.get("politician", {}).get("name")
                or trade.get("representerName", "Unknown")
            )
            party = trade.get("politician", {}).get("party", "")
            chamber = trade.get("politician", {}).get("chamber", "")
            tx_type = trade.get("txType", trade.get("transactionType", ""))
            amount_min = trade.get("amountMin") or trade.get("size", {}).get("min")
            amount_max = trade.get("amountMax") or trade.get("size", {}).get("max")

            sid = stable_source_id(
                {"politician": politician, "ticker": ticker,
                 "trade_date": str(trade_dt.date()), "type": tx_type},
                ["politician", "ticker", "trade_date", "type"],
            )

            records.append({
                "event_timestamp":    trade_dt,
                "knowledge_timestamp": disclose_dt,
                "ticker":             ticker,
                "data_category":      DataCategory.CONGRESS_TRADE,
                "data_source":        "capitol_trades",
                "headline": (
                    f"{politician} ({party}-{chamber}) {tx_type} "
                    f"${amount_min:,}-${amount_max:,} of {ticker}"
                    if amount_min and amount_max
                    else f"{politician} traded {ticker}"
                ),
                "payload": {
                    "politician":     politician,
                    "party":          party,
                    "chamber":        chamber,   # House / Senate
                    "state":          trade.get("politician", {}).get("state", ""),
                    "transaction_type": tx_type,
                    "amount_min":     amount_min,
                    "amount_max":     amount_max,
                    "asset_name":     trade.get("asset", {}).get("name", ""),
                    "delay_days": (disclose_dt - trade_dt).days,
                    "schema_version": 1,
                },
                "schema_version": 1,
                "source_id":          sid,
            })
        return records
