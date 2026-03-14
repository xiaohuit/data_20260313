"""
Point-in-Time (PiT) Historical Data Loader.

This is the single most important module in the pipeline.
Its ONLY job is to answer:

    "Given a simulated current timestamp `as_of`, what data would a trader
     have had access to?"

It enforces the PiT guarantee by ALWAYS filtering:
    WHERE knowledge_timestamp <= :as_of

This prevents ALL forms of look-ahead bias:
  - Restated financials that weren't available yet.
  - Revised macro prints (first NFP release vs final).
  - Late insider/Congress disclosures.
  - 13F filings that came in after quarter-end.
  - Index re-additions after delistings.

Usage (in backtester):
    loader = PiTDataLoader()
    async for snapshot in loader.walk(
        start=datetime(2020, 1, 1, tz=utc),
        end=datetime(2024, 1, 1, tz=utc),
        frequency="1W",
    ):
        # snapshot.as_of  →  the simulated "now"
        # snapshot.prices →  pd.DataFrame of OHLCV known at that moment
        # snapshot.macro  →  dict of latest macro values
        # snapshot.events →  list of FinancialEvent payloads
        decision = ai_trader.decide(snapshot)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pandas as pd
from sqlalchemy import select, text, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_pipeline.config import CONFIG, DataCategory
from financial_pipeline.crawlers.base import make_session_factory
from financial_pipeline.crawlers.universe import UniverseCrawler
from financial_pipeline.db.models import (
    FinancialEvent,
    IndexConstituent,
    MacroIndicator,
    MarketOHLCV,
    TechnicalIndicator,
)

log = logging.getLogger(__name__)


# ── Snapshot dataclass (what the AI trader sees) ──────────────────────────────

@dataclass
class MarketSnapshot:
    """
    A complete, look-ahead-free view of the world at a given `as_of` moment.
    Everything in this object was publicly available on or before `as_of`.
    """
    as_of: datetime

    # ── Market ─────────────────────────────────────────────────────────────
    # Wide DataFrame: rows=dates (lookback), cols=tickers, values=adj_close.
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Same shape as prices but for raw (unadjusted) close.
    prices_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    # volume[ticker] → Series
    volumes: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Multi-index DataFrame: (date, ticker) → indicator values
    indicators: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Universe ────────────────────────────────────────────────────────────
    # Tickers in-index AT as_of (survivorship-bias free)
    universe: list[str] = field(default_factory=list)

    # ── Fundamentals ────────────────────────────────────────────────────────
    # Latest XBRL financials per ticker known at as_of
    # Dict[ticker, Dict[metric, value]]
    fundamentals: dict[str, dict] = field(default_factory=dict)

    # ── Macro ───────────────────────────────────────────────────────────────
    # Dict[indicator_code, latest_value_at_as_of]
    macro: dict[str, float] = field(default_factory=dict)
    # Full macro series (lookback) per indicator code
    macro_series: dict[str, pd.Series] = field(default_factory=dict)

    # ── Alternative / Sentiment ─────────────────────────────────────────────
    # Insider trades in the lookback window, known at as_of
    insider_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Most recent 13F position for each famous investor per ticker
    institutional_holdings: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Congress trades in lookback window known at as_of
    congress_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Recent earnings surprises
    earnings: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── FOMC ────────────────────────────────────────────────────────────────
    fomc_events: list[dict] = field(default_factory=list)

    def to_flat_dict(self) -> dict:
        """Serialize snapshot to a flat dict suitable for ML feature extraction."""
        out: dict = {"as_of": self.as_of.isoformat(), "universe": self.universe}
        out["macro"] = self.macro
        out["fundamentals"] = self.fundamentals
        if not self.prices.empty:
            out["latest_prices"] = self.prices.iloc[-1].to_dict()
        return out


# ── Resampling helpers ────────────────────────────────────────────────────────

FREQ_ALIASES = {
    "1D": "B",    # business day
    "1W": "W-FRI",
    "1M": "MS",
    "1Q": "QS",
}


def _resample_ohlcv(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame from daily bars to the target frequency.
    df must have columns: open, high, low, close, adj_close, volume
    indexed by datetime.
    """
    alias = FREQ_ALIASES.get(frequency, frequency)
    agg = {
        "open":      "first",
        "high":      "max",
        "low":       "min",
        "close":     "last",
        "adj_close": "last",
        "volume":    "sum",
    }
    available_agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample(alias).agg(available_agg).dropna(how="all")


# ── Main PiT Loader ───────────────────────────────────────────────────────────

class PiTDataLoader:
    """
    Async, PiT-safe data loader.

    All public methods accept `as_of: datetime` and guarantee that ONLY
    data with `knowledge_timestamp <= as_of` is returned.

    The `walk()` generator is the primary interface for backtesting:
    it yields successive MarketSnapshots at each simulation step.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker | None = None,
        default_lookback: timedelta = timedelta(days=365),
    ) -> None:
        self._sf = session_factory or make_session_factory()
        self._universe_crawler = UniverseCrawler(session_factory=self._sf)
        self._default_lookback = default_lookback

    # ── Backtesting walk ──────────────────────────────────────────────────────

    async def walk(
        self,
        start: datetime,
        end: datetime,
        frequency: str = "1W",
        lookback: timedelta | None = None,
        index_name: str = "SP500",
        include_indicators: bool = True,
        include_alternatives: bool = True,
        include_fundamentals: bool = True,
        include_macro: bool = True,
    ) -> AsyncIterator[MarketSnapshot]:
        """
        Yield a MarketSnapshot at each step between `start` and `end`.

        `frequency` controls the step size:
            '1D' = daily, '1W' = weekly, '1M' = monthly, '1Q' = quarterly

        `lookback` controls how far back each snapshot's price history extends.
        """
        lb = lookback or self._default_lookback
        step_map = {"1D": timedelta(days=1), "1W": timedelta(weeks=1),
                    "1M": timedelta(days=30), "1Q": timedelta(days=91)}
        step = step_map.get(frequency, timedelta(weeks=1))

        current = start
        while current <= end:
            snapshot = await self.build_snapshot(
                as_of=current,
                lookback=lb,
                frequency=frequency,
                index_name=index_name,
                include_indicators=include_indicators,
                include_alternatives=include_alternatives,
                include_fundamentals=include_fundamentals,
                include_macro=include_macro,
            )
            yield snapshot
            current += step

    # ── Snapshot builder ──────────────────────────────────────────────────────

    async def build_snapshot(
        self,
        as_of: datetime,
        lookback: timedelta = timedelta(days=365),
        frequency: str = "1D",
        index_name: str = "SP500",
        include_indicators: bool = True,
        include_alternatives: bool = True,
        include_fundamentals: bool = True,
        include_macro: bool = True,
    ) -> MarketSnapshot:
        """
        Build a complete PiT snapshot for the given `as_of` timestamp.
        All data fetches run concurrently where possible.
        """
        lookback_start = as_of - lookback

        async with self._sf() as session:
            # 1. Universe (PiT — which tickers were IN the index at as_of?)
            universe = await self._universe_crawler.get_universe_at(
                session, as_of, index_name
            )

            # 2. Prices + indicators run in parallel
            tasks: list = [
                self._load_prices(session, universe, as_of, lookback_start, frequency),
            ]
            if include_indicators:
                tasks.append(
                    self._load_indicators(session, universe, as_of, lookback_start)
                )
            if include_fundamentals:
                tasks.append(
                    self._load_fundamentals(session, universe, as_of)
                )
            if include_macro:
                tasks.append(
                    self._load_macro(session, as_of, lookback_start)
                )
            if include_alternatives:
                tasks.extend([
                    self._load_insider_trades(session, universe, as_of, lookback_start),
                    self._load_congress_trades(session, universe, as_of, lookback_start),
                    self._load_earnings(session, universe, as_of, lookback_start),
                    self._load_fomc(session, as_of, lookback_start),
                ])

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results safely
        def _safe(r, default):
            if isinstance(r, Exception):
                log.warning("Snapshot fetch error: %s", r)
                return default
            return r

        idx = 0
        prices_dfs = _safe(results[idx], (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
        idx += 1
        prices_adj, prices_raw, volumes = prices_dfs

        indicators_df = pd.DataFrame()
        if include_indicators:
            indicators_df = _safe(results[idx], pd.DataFrame()); idx += 1

        fundamentals = {}
        if include_fundamentals:
            fundamentals = _safe(results[idx], {}); idx += 1

        macro, macro_series = {}, {}
        if include_macro:
            macro, macro_series = _safe(results[idx], ({}, {})); idx += 1

        insider_df = pd.DataFrame()
        congress_df = pd.DataFrame()
        earnings_df = pd.DataFrame()
        fomc_events: list[dict] = []
        if include_alternatives:
            insider_df  = _safe(results[idx], pd.DataFrame()); idx += 1
            congress_df = _safe(results[idx], pd.DataFrame()); idx += 1
            earnings_df = _safe(results[idx], pd.DataFrame()); idx += 1
            fomc_events = _safe(results[idx], []); idx += 1

        return MarketSnapshot(
            as_of=as_of,
            universe=universe,
            prices=prices_adj,
            prices_raw=prices_raw,
            volumes=volumes,
            indicators=indicators_df,
            fundamentals=fundamentals,
            macro=macro,
            macro_series=macro_series,
            insider_trades=insider_df,
            congress_trades=congress_df,
            earnings=earnings_df,
            fomc_events=fomc_events,
        )

    # ── Individual data loaders ───────────────────────────────────────────────

    async def _load_prices(
        self,
        session: AsyncSession,
        tickers: list[str],
        as_of: datetime,
        lookback_start: datetime,
        frequency: str = "1D",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load adjusted close prices, raw close, and volume for all tickers.
        Returns three wide DataFrames: (adj_close, raw_close, volume).

        PiT guarantee: `knowledge_timestamp <= as_of`
        """
        if not tickers:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        stmt = (
            select(
                MarketOHLCV.event_timestamp,
                MarketOHLCV.ticker,
                MarketOHLCV.adj_close,
                MarketOHLCV.close,
                MarketOHLCV.volume,
            )
            .where(
                and_(
                    MarketOHLCV.ticker.in_(tickers),
                    MarketOHLCV.frequency == "1D",
                    MarketOHLCV.event_timestamp >= lookback_start,
                    MarketOHLCV.event_timestamp <= as_of,
                    MarketOHLCV.knowledge_timestamp <= as_of,   # ← PiT gate
                )
            )
            .order_by(MarketOHLCV.event_timestamp)
        )

        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], utc=True)

        # Pivot to wide format (date × ticker)
        adj_wide = df.pivot(index="date", columns="ticker", values="adj_close")
        raw_wide = df.pivot(index="date", columns="ticker", values="close")
        vol_wide = df.pivot(index="date", columns="ticker", values="volume")

        # Resample if needed
        if frequency != "1D":
            adj_wide = adj_wide.resample(FREQ_ALIASES.get(frequency, frequency)).last()
            raw_wide = raw_wide.resample(FREQ_ALIASES.get(frequency, frequency)).last()
            vol_wide = vol_wide.resample(FREQ_ALIASES.get(frequency, frequency)).sum()

        adj_wide.columns.name = None
        raw_wide.columns.name = None
        vol_wide.columns.name = None

        return adj_wide, raw_wide, vol_wide

    async def _load_indicators(
        self,
        session: AsyncSession,
        tickers: list[str],
        as_of: datetime,
        lookback_start: datetime,
    ) -> pd.DataFrame:
        """
        Load technical indicators as a multi-index DataFrame.
        Index: (date, ticker), Columns: indicator names.
        PiT: knowledge_timestamp <= as_of.
        """
        if not tickers:
            return pd.DataFrame()

        stmt = (
            select(
                TechnicalIndicator.event_timestamp,
                TechnicalIndicator.ticker,
                TechnicalIndicator.indicator_name,
                TechnicalIndicator.value,
            )
            .where(
                and_(
                    TechnicalIndicator.ticker.in_(tickers),
                    TechnicalIndicator.event_timestamp >= lookback_start,
                    TechnicalIndicator.event_timestamp <= as_of,
                    TechnicalIndicator.knowledge_timestamp <= as_of,
                )
            )
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "ticker", "indicator", "value"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        pivot = df.pivot_table(
            index=["date", "ticker"], columns="indicator", values="value", aggfunc="last"
        )
        pivot.columns.name = None
        return pivot

    async def _load_fundamentals(
        self,
        session: AsyncSession,
        tickers: list[str],
        as_of: datetime,
    ) -> dict[str, dict]:
        """
        Load the MOST RECENT fundamental snapshot per ticker (10-K or 10-Q).
        Returns Dict[ticker → Dict[metric, value]].
        PiT: knowledge_timestamp <= as_of (i.e. filing must have been made by as_of).
        """
        if not tickers:
            return {}

        # Use a window function to get the latest filing per ticker
        stmt = text("""
            WITH ranked AS (
                SELECT
                    ticker,
                    payload,
                    knowledge_timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker
                        ORDER BY knowledge_timestamp DESC
                    ) AS rn
                FROM financial_events
                WHERE ticker = ANY(:tickers)
                  AND data_category IN ('SEC_10K', 'SEC_10Q')
                  AND data_source = 'SEC_EDGAR_XBRL'
                  AND knowledge_timestamp <= :as_of
            )
            SELECT ticker, payload FROM ranked WHERE rn = 1
        """)
        result = await session.execute(
            stmt, {"tickers": tickers, "as_of": as_of}
        )
        rows = result.fetchall()
        out: dict[str, dict] = {}
        for ticker, payload in rows:
            if payload and "financials" in payload:
                out[ticker] = payload["financials"]
        return out

    async def _load_macro(
        self,
        session: AsyncSession,
        as_of: datetime,
        lookback_start: datetime,
    ) -> tuple[dict[str, float], dict[str, pd.Series]]:
        """
        Load macro indicators. Returns:
          (latest_snapshot, full_series_dict)

        For each indicator, selects the LATEST revision available at as_of
        (handles vintage/revision data correctly — prevents look-ahead from
        revised economic prints).
        """
        # Latest value per indicator (most recent revision at as_of)
        latest_stmt = text("""
            WITH ranked AS (
                SELECT
                    indicator_code,
                    event_timestamp,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY indicator_code, event_timestamp
                        ORDER BY revision_number DESC
                    ) AS rn
                FROM macro_indicators
                WHERE knowledge_timestamp <= :as_of
            )
            SELECT indicator_code, event_timestamp, value
            FROM ranked
            WHERE rn = 1
            ORDER BY indicator_code, event_timestamp DESC
        """)
        result = await session.execute(latest_stmt, {"as_of": as_of})
        rows = result.fetchall()

        latest_snapshot: dict[str, float] = {}
        series_data: dict[str, list[tuple]] = {}

        for code, ts, val in rows:
            if val is None:
                continue
            if code not in latest_snapshot:   # first row = most recent
                latest_snapshot[code] = float(val)
            if code not in series_data:
                series_data[code] = []
            if ts >= lookback_start:
                series_data[code].append((ts, float(val)))

        macro_series: dict[str, pd.Series] = {}
        for code, points in series_data.items():
            if points:
                idx, vals = zip(*points)
                macro_series[code] = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

        return latest_snapshot, macro_series

    async def _load_insider_trades(
        self,
        session: AsyncSession,
        tickers: list[str],
        as_of: datetime,
        lookback_start: datetime,
    ) -> pd.DataFrame:
        """
        Insider trades DISCLOSED (knowledge_timestamp) in the lookback window.
        PiT: only trades whose Form 4 was filed on or before as_of.
        """
        if not tickers:
            return pd.DataFrame()

        stmt = (
            select(
                FinancialEvent.event_timestamp,
                FinancialEvent.knowledge_timestamp,
                FinancialEvent.ticker,
                FinancialEvent.payload,
            )
            .where(
                and_(
                    FinancialEvent.ticker.in_(tickers),
                    FinancialEvent.data_category == DataCategory.INSIDER_TRADE,
                    FinancialEvent.knowledge_timestamp >= lookback_start,
                    FinancialEvent.knowledge_timestamp <= as_of,
                )
            )
            .order_by(FinancialEvent.knowledge_timestamp.desc())
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()

        records = []
        for trade_dt, disclose_dt, ticker, payload in rows:
            rec = {"trade_date": trade_dt, "disclosure_date": disclose_dt, "ticker": ticker}
            if payload:
                rec.update({
                    "insider_name": payload.get("insider_name"),
                    "title":        payload.get("title"),
                    "trade_type":   payload.get("trade_type"),
                    "price":        payload.get("price"),
                    "qty":          payload.get("qty"),
                    "value":        payload.get("value"),
                })
            records.append(rec)
        return pd.DataFrame(records)

    async def _load_congress_trades(
        self,
        session: AsyncSession,
        tickers: list[str],
        as_of: datetime,
        lookback_start: datetime,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        stmt = (
            select(
                FinancialEvent.event_timestamp,
                FinancialEvent.knowledge_timestamp,
                FinancialEvent.ticker,
                FinancialEvent.payload,
            )
            .where(
                and_(
                    FinancialEvent.ticker.in_(tickers),
                    FinancialEvent.data_category == DataCategory.CONGRESS_TRADE,
                    FinancialEvent.knowledge_timestamp >= lookback_start,
                    FinancialEvent.knowledge_timestamp <= as_of,
                )
            )
            .order_by(FinancialEvent.knowledge_timestamp.desc())
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()

        records = []
        for trade_dt, disclose_dt, ticker, payload in rows:
            rec = {"trade_date": trade_dt, "disclosure_date": disclose_dt, "ticker": ticker}
            if payload:
                rec.update({
                    "politician":    payload.get("politician"),
                    "party":         payload.get("party"),
                    "chamber":       payload.get("chamber"),
                    "tx_type":       payload.get("transaction_type"),
                    "amount_min":    payload.get("amount_min"),
                    "amount_max":    payload.get("amount_max"),
                    "delay_days":    payload.get("delay_days"),
                })
            records.append(rec)
        return pd.DataFrame(records)

    async def _load_earnings(
        self,
        session: AsyncSession,
        tickers: list[str],
        as_of: datetime,
        lookback_start: datetime,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        stmt = (
            select(
                FinancialEvent.event_timestamp,
                FinancialEvent.knowledge_timestamp,
                FinancialEvent.ticker,
                FinancialEvent.payload,
            )
            .where(
                and_(
                    FinancialEvent.ticker.in_(tickers),
                    FinancialEvent.data_category == DataCategory.EARNINGS,
                    FinancialEvent.knowledge_timestamp >= lookback_start,
                    FinancialEvent.knowledge_timestamp <= as_of,
                )
            )
            .order_by(FinancialEvent.knowledge_timestamp.desc())
        )
        result = await session.execute(stmt)
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()

        records = []
        for period_dt, announce_dt, ticker, payload in rows:
            rec = {"period_end": period_dt, "announced": announce_dt, "ticker": ticker}
            if payload:
                rec.update({
                    "eps_estimate":     payload.get("eps_estimate"),
                    "eps_actual":       payload.get("eps_actual"),
                    "eps_surprise_pct": payload.get("eps_surprise_pct"),
                })
            records.append(rec)
        return pd.DataFrame(records)

    async def _load_fomc(
        self,
        session: AsyncSession,
        as_of: datetime,
        lookback_start: datetime,
    ) -> list[dict]:
        stmt = (
            select(FinancialEvent.event_timestamp, FinancialEvent.payload)
            .where(
                and_(
                    FinancialEvent.data_category == DataCategory.FOMC_DECISION,
                    FinancialEvent.knowledge_timestamp >= lookback_start,
                    FinancialEvent.knowledge_timestamp <= as_of,
                )
            )
            .order_by(FinancialEvent.event_timestamp.desc())
            .limit(8)   # last 8 FOMC meetings
        )
        result = await session.execute(stmt)
        return [
            {"meeting_date": str(ts), **(payload or {})}
            for ts, payload in result.fetchall()
        ]

    # ── Convenience methods ───────────────────────────────────────────────────

    async def get_returns(
        self,
        tickers: list[str],
        as_of: datetime,
        lookback: timedelta = timedelta(days=252),
        frequency: str = "1D",
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of percentage returns for each ticker,
        aligned to `frequency`, PiT-safe at `as_of`.
        """
        async with self._sf() as session:
            adj, _, _ = await self._load_prices(
                session, tickers, as_of, as_of - lookback, frequency
            )
        if adj.empty:
            return pd.DataFrame()
        return adj.pct_change().dropna(how="all")

    async def get_correlation_matrix(
        self,
        tickers: list[str],
        as_of: datetime,
        lookback: timedelta = timedelta(days=252),
    ) -> pd.DataFrame:
        """Rolling correlation matrix known at `as_of`."""
        returns = await self.get_returns(tickers, as_of, lookback, "1D")
        if returns.empty:
            return pd.DataFrame()
        return returns.corr()

    async def evaluate_trade(
        self,
        ticker: str,
        entry_date: datetime,
        exit_date: datetime,
        direction: str = "LONG",
    ) -> dict:
        """
        Calculate realized P&L for a trade defined by entry/exit dates.
        Uses adj_close for return calculation (splits/dividends accounted for).
        Returns GAIN/LOSS as percentage and dollar terms (per share).
        """
        async with self._sf() as session:
            entry_price_stmt = (
                select(MarketOHLCV.adj_close)
                .where(
                    and_(
                        MarketOHLCV.ticker == ticker,
                        MarketOHLCV.frequency == "1D",
                        MarketOHLCV.event_timestamp >= entry_date,
                        MarketOHLCV.knowledge_timestamp <= entry_date,
                    )
                )
                .order_by(MarketOHLCV.event_timestamp)
                .limit(1)
            )
            exit_price_stmt = (
                select(MarketOHLCV.adj_close)
                .where(
                    and_(
                        MarketOHLCV.ticker == ticker,
                        MarketOHLCV.frequency == "1D",
                        MarketOHLCV.event_timestamp >= exit_date,
                        MarketOHLCV.knowledge_timestamp <= exit_date,
                    )
                )
                .order_by(MarketOHLCV.event_timestamp)
                .limit(1)
            )
            entry_result = await session.execute(entry_price_stmt)
            exit_result = await session.execute(exit_price_stmt)

        entry_row = entry_result.fetchone()
        exit_row = exit_result.fetchone()

        if not entry_row or not exit_row:
            return {"error": "price not found", "ticker": ticker}

        entry_price = float(entry_row[0])
        exit_price = float(exit_row[0])
        raw_return = (exit_price - entry_price) / entry_price
        pnl_pct = raw_return if direction == "LONG" else -raw_return
        pnl_per_share = (exit_price - entry_price) if direction == "LONG" else -(exit_price - entry_price)

        return {
            "ticker":          ticker,
            "direction":       direction,
            "entry_date":      entry_date.isoformat(),
            "exit_date":       exit_date.isoformat(),
            "entry_price":     entry_price,
            "exit_price":      exit_price,
            "pnl_pct":         round(pnl_pct * 100, 4),
            "pnl_per_share":   round(pnl_per_share, 4),
            "result":          "GAIN" if pnl_pct > 0 else "LOSS",
            "holding_days":    (exit_date - entry_date).days,
        }
