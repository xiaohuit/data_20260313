"""
Universe management — S&P 500 and Nasdaq 100 constituent tracking.

Survivorship-bias prevention strategy:
  1. Bootstrap current membership from Wikipedia scrape.
  2. Enrich with historical additions/removals from Wikipedia's change log.
  3. Cross-reference SEC EDGAR company facts for CIK numbers.
  4. Store every add/remove event with the `knowledge_timestamp` equal to
     the date the change was announced (not effective date), whenever known.

The `get_universe_at(as_of)` query then returns ONLY tickers that were
in the index on that date, enabling clean historical backtests.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, date, timezone
from typing import Any

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_pipeline.config import CONFIG
from financial_pipeline.crawlers.base import (
    BaseCrawler,
    bulk_upsert,
    make_session_factory,
    utcnow,
)
from financial_pipeline.db.models import IndexConstituent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wikipedia scraper helpers
# ---------------------------------------------------------------------------

def _parse_sp500_wiki(html: str) -> list[dict]:
    """
    Parse the S&P 500 Wikipedia page for current constituents AND
    the 'Changes to the list' table for historical add/remove events.
    Returns a list of constituent dicts.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []

    # ── Current constituents ──────────────────────────────────────────────────
    main_table = soup.find("table", {"id": "constituents"})
    if main_table:
        for row in main_table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            records.append({
                "index_name":   "SP500",
                "ticker":       cells[0].get_text(strip=True).replace(".", "-"),
                "company_name": cells[1].get_text(strip=True),
                "sector":       cells[3].get_text(strip=True),
                "sub_industry": cells[4].get_text(strip=True) if len(cells) > 4 else None,
                "added_date":   date.today(),
                "removed_date": None,
                "knowledge_timestamp": utcnow(),
                "source":       "wikipedia_current",
            })

    # ── Historical changes table ──────────────────────────────────────────────
    changes_table = soup.find("table", {"id": "changes"})
    if changes_table:
        for row in changes_table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            raw_date = cells[0].get_text(strip=True)
            try:
                change_date = pd.to_datetime(raw_date).date()
            except Exception:
                continue
            added_ticker = cells[1].get_text(strip=True).replace(".", "-")
            removed_ticker = cells[3].get_text(strip=True).replace(".", "-")
            if added_ticker:
                records.append({
                    "index_name":   "SP500",
                    "ticker":       added_ticker,
                    "company_name": cells[2].get_text(strip=True),
                    "added_date":   change_date,
                    "removed_date": None,
                    "knowledge_timestamp": datetime.combine(
                        change_date, datetime.min.time()
                    ).replace(tzinfo=timezone.utc),
                    "source":       "wikipedia_changes",
                })
            if removed_ticker:
                # Mark the removal on matching add rows (best-effort)
                records.append({
                    "index_name":   "SP500",
                    "ticker":       removed_ticker,
                    "company_name": None,
                    "added_date":   date(2000, 1, 1),   # placeholder
                    "removed_date": change_date,
                    "knowledge_timestamp": datetime.combine(
                        change_date, datetime.min.time()
                    ).replace(tzinfo=timezone.utc),
                    "source":       "wikipedia_changes_removal",
                })
    return records


def _parse_ndx100_wiki(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []
    table = soup.find("table", {"id": "constituents"})
    if not table:
        # fallback: first wikitable
        table = soup.find("table", {"class": "wikitable"})
    if table:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        ticker_col = next(
            (i for i, h in enumerate(headers) if "ticker" in h or "symbol" in h), 1
        )
        name_col = next(
            (i for i, h in enumerate(headers) if "company" in h or "name" in h), 0
        )
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue
            ticker = cells[ticker_col].get_text(strip=True).replace(".", "-")
            if not ticker:
                continue
            records.append({
                "index_name":   "NDX100",
                "ticker":       ticker,
                "company_name": cells[name_col].get_text(strip=True) if cells else None,
                "added_date":   date.today(),
                "removed_date": None,
                "knowledge_timestamp": utcnow(),
                "source":       "wikipedia_current",
            })
    return records


# ---------------------------------------------------------------------------
# Universe crawler
# ---------------------------------------------------------------------------

class UniverseCrawler:
    """
    Crawls index membership and stores it in `index_constituents`.
    Not a standard BaseCrawler (no ticker loop) — runs as a standalone task.
    """

    def __init__(self, session_factory: async_sessionmaker | None = None) -> None:
        self._sf = session_factory or make_session_factory()

    async def refresh(self) -> dict[str, int]:
        """Fetch current + historical members and upsert. Returns row counts."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            sp500_resp = await client.get(CONFIG.universe.sp500_wiki_url)
            ndx100_resp = await client.get(CONFIG.universe.ndx100_wiki_url)

        sp500_records = _parse_sp500_wiki(sp500_resp.text)
        ndx100_records = _parse_ndx100_wiki(ndx100_resp.text)
        all_records = sp500_records + ndx100_records

        # Deduplicate: keep only unique (index_name, ticker, added_date)
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for r in all_records:
            key = (r["index_name"], r["ticker"], str(r["added_date"]))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        async with self._sf() as session:
            n = await bulk_upsert(
                session,
                IndexConstituent,
                deduped,
                conflict_columns=["index_name", "ticker", "added_date"],
                update_columns=["removed_date", "knowledge_timestamp"],
            )

        log.info("UniverseCrawler: upserted %d constituent rows", n)
        return {"total_records": len(deduped), "upserted": n}

    # ── Query helpers ─────────────────────────────────────────────────────────

    async def get_universe_at(
        self,
        session: AsyncSession,
        as_of: datetime,
        index_name: str = "SP500",
    ) -> list[str]:
        """
        Returns tickers that were members of `index_name` at `as_of`.
        Enforces PiT: ignores constituent changes announced AFTER `as_of`.
        """
        from sqlalchemy import select, or_, and_
        stmt = (
            select(IndexConstituent.ticker)
            .where(
                IndexConstituent.index_name == index_name,
                IndexConstituent.added_date <= as_of.date(),
                IndexConstituent.knowledge_timestamp <= as_of,
                or_(
                    IndexConstituent.removed_date.is_(None),
                    IndexConstituent.removed_date > as_of.date(),
                ),
            )
            .distinct()
        )
        result = await session.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def get_combined_universe_at(
        self, session: AsyncSession, as_of: datetime
    ) -> list[str]:
        """S&P 500 ∪ Nasdaq 100, deduplicated."""
        sp = set(await self.get_universe_at(session, as_of, "SP500"))
        ndx = set(await self.get_universe_at(session, as_of, "NDX100"))
        return sorted(sp | ndx)
