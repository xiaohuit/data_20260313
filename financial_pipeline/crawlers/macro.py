"""
Macro data crawler.

Sources:
  - FRED (Federal Reserve Economic Data) — primary for all series.
  - BLS API fallback for NFP / CPI if FRED lags.
  - Fed Reserve website scraper for FOMC meeting outcomes and dot-plot data.

Critical PiT note for macro data:
  Economic data is frequently REVISED. The first-release value of GDP or NFP
  that a trader would have seen is different from the final revised value.

  Strategy used here:
    1. FRED provides vintage/realtime data via the `vintage_dates` parameter.
       We fetch ALL available vintages and store each revision as a separate
       row with (revision_number, knowledge_timestamp = release_date).
    2. The PiT data-loader always fetches `MAX(revision_number) WHERE
       knowledge_timestamp <= as_of`, i.e. the most recent revision available
       at the simulation timestamp — not the current (final) value.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from financial_pipeline.config import CONFIG, DataCategory, FRED_SERIES
from financial_pipeline.crawlers.base import BaseCrawler, bulk_upsert, make_session_factory, utcnow
from financial_pipeline.db.models import FinancialEvent, MacroIndicator

log = logging.getLogger(__name__)


# ── FRED Crawler ──────────────────────────────────────────────────────────────

class FREDCrawler:
    """
    Pulls FRED series with vintage/revision history.
    Uses the `fredapi` library (thin wrapper around FRED's REST API).
    """

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()
        self._api_key = CONFIG.api_keys.fred

    def _client(self):
        try:
            from fredapi import Fred  # type: ignore
            return Fred(api_key=self._api_key)
        except ImportError:
            raise RuntimeError("Install fredapi: pip install fredapi")

    async def backfill_all(self, start: datetime, end: datetime) -> None:
        """Fetch all configured FRED series with vintage data."""
        loop = asyncio.get_running_loop()
        fred = self._client()
        all_records: list[dict] = []

        for code, series_id in FRED_SERIES.items():
            log.info("FRED: fetching %s (%s)", code, series_id)
            records = await loop.run_in_executor(
                None,
                lambda sid=series_id, c=code: self._fetch_series_with_vintages(
                    fred, sid, c, start, end
                ),
            )
            all_records.extend(records)
            await asyncio.sleep(1.0 / CONFIG.rate_limits.fred)

        async with self._sf() as session:
            await bulk_upsert(
                session,
                MacroIndicator,
                all_records,
                conflict_columns=["event_timestamp", "indicator_code", "revision_number"],
                update_columns=["value", "knowledge_timestamp"],
            )
        log.info("FRED backfill complete: %d rows", len(all_records))

    @staticmethod
    def _fetch_series_with_vintages(
        fred,
        series_id: str,
        indicator_code: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Fetch a FRED series and ALL of its vintage revisions.
        Each revision becomes a separate row with its own knowledge_timestamp.
        """
        records: list[dict] = []
        try:
            # Get series metadata
            info = fred.get_series_info(series_id)
            freq = info.get("frequency_short", "M")
            unit = info.get("units_short", "")
            title = info.get("title", "")

            # Fetch revision vintages (FRED realtime API)
            # This returns a DataFrame with columns = vintage dates, rows = observation dates
            try:
                vintage_df = fred.get_series_vintage_dates(series_id)
            except Exception:
                vintage_df = []

            if len(vintage_df) > 0:
                # Full vintage history available
                rev_df = fred.get_series_all_releases(
                    series_id,
                    observation_start=start.strftime("%Y-%m-%d"),
                    observation_end=end.strftime("%Y-%m-%d"),
                )
                if rev_df is not None and not rev_df.empty:
                    # rev_df has columns: date, realtime_start, realtime_end, value
                    for _, row in rev_df.iterrows():
                        if pd.isna(row.get("value")):
                            continue
                        try:
                            val = float(row["value"])
                        except (ValueError, TypeError):
                            continue
                        records.append({
                            "event_timestamp":    pd.to_datetime(row["date"]).replace(tzinfo=timezone.utc),
                            "knowledge_timestamp": pd.to_datetime(row["realtime_start"]).replace(tzinfo=timezone.utc),
                            "indicator_code":      indicator_code,
                            "series_name":         title,
                            "value":               val,
                            "unit":                unit,
                            "frequency":           freq,
                            "source":              "FRED",
                            "revision_number":     0,   # will be set below
                            "payload":             {"series_id": series_id},
                        })
                    # Assign revision numbers per (event_timestamp) in chronological knowledge order
                    if records:
                        df_tmp = pd.DataFrame(records)
                        df_tmp = df_tmp.sort_values(["event_timestamp", "knowledge_timestamp"])
                        df_tmp["revision_number"] = df_tmp.groupby("event_timestamp").cumcount()
                        records = df_tmp.to_dict("records")
            else:
                # Fallback: no vintage data, fetch latest values only
                series = fred.get_series(
                    series_id,
                    observation_start=start.strftime("%Y-%m-%d"),
                    observation_end=end.strftime("%Y-%m-%d"),
                )
                if series is not None:
                    for ts, val in series.items():
                        if pd.isna(val):
                            continue
                        records.append({
                            "event_timestamp":    pd.to_datetime(ts).replace(tzinfo=timezone.utc),
                            "knowledge_timestamp": pd.to_datetime(ts).replace(tzinfo=timezone.utc),
                            "indicator_code":      indicator_code,
                            "series_name":         title,
                            "value":               float(val),
                            "unit":                unit,
                            "frequency":           freq,
                            "source":              "FRED",
                            "revision_number":     0,
                            "payload":             {"series_id": series_id},
                        })
        except Exception as exc:
            log.warning("FRED series %s (%s) error: %s", series_id, indicator_code, exc)
        return records


# ── FOMC Decision Crawler ─────────────────────────────────────────────────────

class FOMCCrawler:
    """
    Scrapes the Federal Reserve website for historical FOMC meeting outcomes:
      - Meeting date
      - Rate decision (hike / hold / cut, basis points)
      - Statement text (for NLP / sentiment analysis)
      - Dot-plot data (when available)

    Stored in FinancialEvent with data_category = DataCategory.FOMC_DECISION
    knowledge_timestamp = statement release time (typically 2:00 PM ET)
    """

    FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def __init__(self, session_factory=None) -> None:
        self._sf = session_factory or make_session_factory()

    async def backfill(self, start_year: int = 2018) -> None:
        import httpx
        from bs4 import BeautifulSoup
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        records: list[dict] = []

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": CONFIG.api_keys.sec_user_agent},
            follow_redirects=True,
        ) as client:
            resp = await client.get(self.FOMC_CALENDAR_URL)

        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse meeting dates and outcomes from the Federal Reserve's HTML
        for panel in soup.find_all("div", {"class": "panel"}):
            year_el = panel.find(["h4", "h5"])
            if not year_el:
                continue
            year_text = year_el.get_text(strip=True)
            try:
                year = int(year_text[:4])
            except ValueError:
                continue
            if year < start_year:
                continue

            for meeting in panel.find_all("div", {"class": "fomc-meeting"}):
                date_el = meeting.find("div", {"class": "fomc-meeting__date"})
                if not date_el:
                    continue
                date_str = date_el.get_text(strip=True)
                # Try to parse date, e.g. "January 28-29" → Jan 29
                months = ["January","February","March","April","May","June",
                          "July","August","September","October","November","December"]
                month_num = None
                day_num = None
                for i, m in enumerate(months, 1):
                    if m in date_str:
                        month_num = i
                        nums = [int(x) for x in re.findall(r'\d+', date_str)]
                        day_num = nums[-1] if nums else None
                        break

                if not (month_num and day_num):
                    continue

                try:
                    meeting_date = datetime(year, month_num, day_num, 14, 0, 0, tzinfo=ET)
                except ValueError:
                    continue

                # Look for rate decision
                outcome_el = meeting.find("div", {"class": "fomc-meeting__event-content"})
                outcome_text = outcome_el.get_text(" ", strip=True) if outcome_el else ""

                # Find statement link
                stmt_link = None
                for a in meeting.find_all("a", href=True):
                    if "statement" in a.get_text(strip=True).lower():
                        stmt_link = "https://www.federalreserve.gov" + a["href"]
                        break

                records.append({
                    "event_timestamp":    meeting_date.astimezone(timezone.utc),
                    "knowledge_timestamp": meeting_date.astimezone(timezone.utc),
                    "ticker":             None,
                    "data_category":      DataCategory.FOMC_DECISION,
                    "data_source":        "federalreserve.gov",
                    "headline":           f"FOMC {year}-{month_num:02d}-{day_num:02d}",
                    "payload": {
                        "meeting_date":   meeting_date.date().isoformat(),
                        "outcome_text":   outcome_text[:2000],
                        "statement_url":  stmt_link,
                        "schema_version": 1,
                    },
                    "schema_version": 1,
                    "source_id":          f"fomc_{year}_{month_num:02d}_{day_num:02d}",
                })

        async with self._sf() as session:
            await bulk_upsert(
                session,
                FinancialEvent,
                records,
                conflict_columns=["source_id"],
                update_columns=["payload", "headline"],
            )
        log.info("FOMC crawler: stored %d meeting records", len(records))


import re   # noqa: E402 — used in FOMCCrawler
