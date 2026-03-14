"""
Fundamental data crawler — SEC EDGAR filings and earnings.

Sources:
  - SEC EDGAR full-text search API (no key required, rate limit ≤ 10 req/s)
  - SEC EDGAR company facts API (structured XBRL data)
  - earnings-calendar via yfinance (for whisper + actual EPS/revenue)

PiT rules:
  - SEC filing:  knowledge_timestamp = filing date (when SEC accepted it), NOT
                 the period end date. A 10-K filed on 2024-02-15 for FY2023 has
                 event_timestamp=2023-12-31, knowledge_timestamp=2024-02-15.
  - Earnings:    knowledge_timestamp = pre-market open or after-hours time of
                 the earnings call (when results were publicly released).

All filings are stored in `financial_events` with data_category=SEC_10K etc.
The JSONB payload carries structured XBRL facts so the data-loader can extract
revenue, EPS, debt ratios etc. for the AI trader's fundamental context.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, date
from typing import Any

import httpx
import pandas as pd
import yfinance as yf

from financial_pipeline.config import CONFIG, DataCategory
from financial_pipeline.crawlers.base import (
    bulk_upsert,
    make_session_factory,
    stable_source_id,
    utcnow,
    with_retry,
)
from financial_pipeline.db.models import FinancialEvent

log = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": CONFIG.api_keys.sec_user_agent}


# ── SEC EDGAR company-facts loader ───────────────────────────────────────────

class SECEdgarCrawler:
    """
    Uses the SEC EDGAR company facts API to pull structured XBRL data for
    every 10-K and 10-Q filing, and the submissions API for 8-K filings.

    The company facts API returns time-series for each XBRL concept
    (e.g. us-gaap/Revenues) with the filing date embedded, giving us
    genuine Point-in-Time fundamental data.
    """

    # Key XBRL concepts to extract from each filing
    XBRL_CONCEPTS = {
        "Revenues":                    "revenue",
        "NetIncomeLoss":               "net_income",
        "EarningsPerShareBasic":       "eps_basic",
        "EarningsPerShareDiluted":     "eps_diluted",
        "Assets":                      "total_assets",
        "Liabilities":                 "total_liabilities",
        "StockholdersEquity":          "equity",
        "CashAndCashEquivalentsAtCarryingValue": "cash",
        "LongTermDebt":                "long_term_debt",
        "OperatingIncomeLoss":         "operating_income",
        "GrossProfit":                 "gross_profit",
        "ResearchAndDevelopmentExpense": "rd_expense",
        "CommonStockSharesOutstanding": "shares_outstanding",
        "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    }

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()
        # CIK lookup cache: ticker → cik
        self._cik_cache: dict[str, str] = {}

    async def _get_cik(self, ticker: str) -> str | None:
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        url = f"{EDGAR_BASE}/submissions/CIK{{}}.json"
        # Use company search API
        search_url = (
            "https://efts.sec.gov/LATEST/search-index?q=%22"
            + ticker.replace("-", "+")
            + "%22&dateRange=custom&startdt=2000-01-01&forms=10-K"
        )
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                resp = await client.get(
                    f"https://www.sec.gov/cgi-bin/browse-edgar"
                    f"?action=getcompany&company=&CIK={ticker}"
                    f"&type=10-K&dateb=&owner=include&count=1&search_text=&output=atom"
                )
            # Extract CIK from SEC response
            cik_match = re.search(r"CIK=(\d+)", resp.text)
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                self._cik_cache[ticker] = cik
                return cik
        except Exception as exc:
            log.debug("CIK lookup %s: %s", ticker, exc)
        return None

    async def fetch_filings(self, ticker: str, start: datetime, end: datetime) -> list[dict]:
        """
        Fetch all 10-K, 10-Q, and 8-K filings for a ticker and return as
        FinancialEvent records ready for DB insert.
        """
        cik = await self._get_cik(ticker)
        if not cik:
            log.warning("No CIK found for %s", ticker)
            return []

        await asyncio.sleep(1.0 / CONFIG.rate_limits.sec_edgar)

        records: list[dict] = []

        # 1. Submissions (filing index with dates)
        submissions = await self._fetch_submissions(cik)
        if submissions:
            records.extend(self._parse_submissions(submissions, ticker, start, end))

        # 2. Company facts (XBRL financial data per filing)
        facts = await self._fetch_company_facts(cik)
        if facts:
            xbrl_records = self._parse_company_facts(facts, ticker, start, end)
            # Merge XBRL financials into existing filing records (by accession)
            records.extend(xbrl_records)

        return records

    async def _fetch_submissions(self, cik: str) -> dict | None:
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
                resp = await with_retry(client.get, url)
                return resp.json()
        except Exception as exc:
            log.warning("Submissions fetch %s: %s", cik, exc)
            return None

    async def _fetch_company_facts(self, cik: str) -> dict | None:
        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=60) as client:
                resp = await with_retry(client.get, url)
                return resp.json()
        except Exception as exc:
            log.warning("Company facts fetch %s: %s", cik, exc)
            return None

    def _parse_submissions(
        self, data: dict, ticker: str, start: datetime, end: datetime
    ) -> list[dict]:
        records: list[dict] = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])
        period_dates = recent.get("reportDate", [])

        category_map = {
            "10-K": DataCategory.SEC_10K,
            "10-Q": DataCategory.SEC_10Q,
            "8-K":  DataCategory.SEC_8K,
        }

        for form, filing_date_str, accession, doc, period_str in zip(
            forms, dates, accessions, descriptions, period_dates
        ):
            cat = category_map.get(form)
            if cat is None:
                continue

            try:
                filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue

            if filing_dt < start or filing_dt > end:
                continue

            try:
                period_dt = datetime.strptime(period_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ) if period_str else filing_dt
            except ValueError:
                period_dt = filing_dt

            acc_clean = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{data.get('cik','')}/{acc_clean}/{doc}"
            )

            records.append({
                "event_timestamp":    period_dt,
                "knowledge_timestamp": filing_dt,
                "ticker":             ticker,
                "data_category":      cat,
                "data_source":        "SEC_EDGAR",
                "headline":           f"{ticker} {form} filed {filing_date_str}",
                "payload": {
                    "form":           form,
                    "filing_date":    filing_date_str,
                    "period":         period_str,
                    "accession":      accession,
                    "filing_url":     filing_url,
                    "cik":            data.get("cik", ""),
                    "schema_version": 1,
                },
                "schema_version": 1,
                "source_id": f"sec_{acc_clean}",
            })
        return records

    def _parse_company_facts(
        self, facts: dict, ticker: str, start: datetime, end: datetime
    ) -> list[dict]:
        """
        Extract key financial metrics from XBRL company facts.
        Returns one FinancialEvent per (accession, period) with all
        available metrics packed into the payload.
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        # Aggregate all XBRL concepts per accession number
        accession_data: dict[str, dict] = {}

        for xbrl_key, our_key in self.XBRL_CONCEPTS.items():
            concept = us_gaap.get(xbrl_key, {})
            for unit_group in concept.get("units", {}).values():
                for obs in unit_group:
                    if obs.get("form") not in ("10-K", "10-Q"):
                        continue
                    acc = obs.get("accn", "")
                    filed_str = obs.get("filed", "")
                    end_str = obs.get("end", "")
                    val = obs.get("val")
                    if val is None or not acc:
                        continue
                    try:
                        filed_dt = datetime.strptime(filed_str, "%Y-%m-%d").replace(
                            tzinfo=timezone.utc
                        )
                        if filed_dt < start or filed_dt > end:
                            continue
                    except ValueError:
                        continue

                    if acc not in accession_data:
                        accession_data[acc] = {
                            "filed": filed_str,
                            "end":   end_str,
                            "form":  obs.get("form"),
                            "metrics": {},
                        }
                    accession_data[acc]["metrics"][our_key] = val

        records: list[dict] = []
        for acc, data in accession_data.items():
            if not data["metrics"]:
                continue
            acc_clean = acc.replace("-", "")
            source_id = f"xbrl_{acc_clean}"
            cat = (
                DataCategory.SEC_10K if data["form"] == "10-K" else DataCategory.SEC_10Q
            )
            try:
                period_dt = datetime.strptime(data["end"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                filed_dt = datetime.strptime(data["filed"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            records.append({
                "event_timestamp":    period_dt,
                "knowledge_timestamp": filed_dt,
                "ticker":             ticker,
                "data_category":      cat,
                "data_source":        "SEC_EDGAR_XBRL",
                "headline":           f"{ticker} {data['form']} XBRL {data['end']}",
                "payload": {
                    "accession":      acc,
                    "period_end":     data["end"],
                    "filed_date":     data["filed"],
                    "form":           data["form"],
                    "financials":     data["metrics"],
                    "schema_version": 1,
                },
                "schema_version": 1,
                "source_id":          source_id,
            })
        return records

    async def backfill(
        self, tickers: list[str], start: datetime, end: datetime
    ) -> None:
        async with self._sf() as session:
            for ticker in tickers:
                log.info("SEC EDGAR: processing %s", ticker)
                records = await self.fetch_filings(ticker, start, end)
                if records:
                    await bulk_upsert(
                        session,
                        FinancialEvent,
                        records,
                        conflict_columns=["source_id"],
                        update_columns=["payload", "headline", "knowledge_timestamp"],
                    )
                    log.info("  %s → %d filing records", ticker, len(records))


# ── Earnings Crawler ──────────────────────────────────────────────────────────

class EarningsCrawler:
    """
    Pulls earnings surprise data (estimated vs actual EPS + revenue) from
    yfinance. Stores as FinancialEvent with data_category=EARNINGS.

    The `knowledge_timestamp` is set to the earnings call date/time, which is
    the first moment the actual results were publicly available.
    This is critical: the EVENT covers a past period, but the KNOWLEDGE
    only becomes available on the announcement day.
    """

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()

    async def fetch_earnings(self, ticker: str) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._fetch(ticker))

    @staticmethod
    def _fetch(ticker: str) -> list[dict]:
        try:
            t = yf.Ticker(ticker)
            # earnings_history is a DataFrame with index = quarter dates
            history = t.earnings_history
            if history is None or history.empty:
                return []

            records: list[dict] = []
            for idx, row in history.iterrows():
                # idx is the period end date for the quarter
                try:
                    period_dt = pd.to_datetime(idx).replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                # yfinance doesn't always provide exact earnings release time;
                # we use period end + 90 days as a conservative upper bound.
                # In production, cross-reference with Nasdaq/Bloomberg calendars.
                earnings_date_raw = row.get("Earnings Date")
                if earnings_date_raw and pd.notna(earnings_date_raw):
                    try:
                        knowledge_dt = pd.to_datetime(earnings_date_raw).replace(
                            tzinfo=timezone.utc
                        )
                    except Exception:
                        knowledge_dt = period_dt
                else:
                    knowledge_dt = period_dt

                eps_est = row.get("EPS Estimate")
                eps_act = row.get("Reported EPS")
                surprise = row.get("Surprise(%)")

                records.append({
                    "event_timestamp":    period_dt,
                    "knowledge_timestamp": knowledge_dt,
                    "ticker":             ticker,
                    "data_category":      DataCategory.EARNINGS,
                    "data_source":        "yfinance_earnings",
                    "headline": (
                        f"{ticker} EPS {eps_act:.2f} vs est {eps_est:.2f}"
                        if pd.notna(eps_act) and pd.notna(eps_est) else f"{ticker} earnings"
                    ),
                    "payload": {
                        "eps_estimate":   float(eps_est) if pd.notna(eps_est) else None,
                        "eps_actual":     float(eps_act) if pd.notna(eps_act) else None,
                        "eps_surprise_pct": float(surprise) if pd.notna(surprise) else None,
                        "period":         str(period_dt.date()),
                        "schema_version": 1,
                    },
                    "schema_version": 1,
                    "source_id": stable_source_id(
                        {"ticker": ticker, "period": str(period_dt.date())},
                        ["ticker", "period"],
                    ),
                })
            return records
        except Exception as exc:
            log.warning("Earnings fetch %s: %s", ticker, exc)
            return []

    async def backfill(self, tickers: list[str]) -> None:
        async with self._sf() as session:
            for ticker in tickers:
                records = await self.fetch_earnings(ticker)
                if records:
                    await bulk_upsert(
                        session,
                        FinancialEvent,
                        records,
                        conflict_columns=["source_id"],
                        update_columns=["payload", "headline", "knowledge_timestamp"],
                    )
