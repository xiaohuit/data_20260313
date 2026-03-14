"""
SQLAlchemy 2.0 ORM models.

Point-in-Time (PiT) design principles enforced here:
  - event_timestamp  : the period/moment the DATA ITSELF covers.
  - knowledge_timestamp: the moment this data became PUBLICLY AVAILABLE.
  - ingestion_timestamp: when OUR SYSTEM stored it (for auditing).

The backtesting data-loader always filters `knowledge_timestamp <= as_of`
to guarantee zero look-ahead bias.

Schema flexibility strategy:
  - Typed tables for high-frequency, well-structured data (OHLCV, macro).
  - `financial_events` as an EAV-style flexible table with a JSONB payload
    for any new data category. Adding a new source requires zero DDL changes.
  - `schema_version` column on flexible tables to support forward-migration
    of the payload shape without breaking old readers.
"""

from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── Utility mixin ────────────────────────────────────────────────────────────

class PiTMixin:
    """
    Every model that carries time-stamped data inherits this mixin.
    Guarantees the three PiT columns are always present and indexed.
    """
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ── Universe / Survivorship Bias Prevention ──────────────────────────────────

class IndexConstituent(Base):
    """
    Tracks every addition and deletion from SP500 and NDX100 over time.

    To reconstruct the universe AS OF a given date:
        SELECT ticker FROM index_constituents
        WHERE index_name = 'SP500'
          AND added_date <= :as_of_date
          AND (removed_date IS NULL OR removed_date > :as_of_date)
    """
    __tablename__ = "index_constituents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(20), nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    sub_industry: Mapped[str | None] = mapped_column(String(150))
    cik: Mapped[str | None] = mapped_column(String(20))          # SEC CIK
    figi: Mapped[str | None] = mapped_column(String(20))         # OpenFIGI

    added_date: Mapped[date] = mapped_column(Date, nullable=False)
    removed_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # knowledge_timestamp of the announcement (e.g. when S&P press-released it)
    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    source: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        UniqueConstraint("index_name", "ticker", "added_date", name="uq_constituent"),
        Index("ix_constituent_lookup", "index_name", "added_date", "removed_date"),
    )


# ── Market OHLCV ─────────────────────────────────────────────────────────────

class MarketOHLCV(Base, PiTMixin):
    """
    Stores raw price bars at any frequency (1D, 1H, 15m …).

    TimescaleDB converts this to a hypertable partitioned by event_timestamp.
    Corporate-action-adjusted prices live in adjusted_close / adjusted_open etc.

    NOTE: For intraday data, knowledge_timestamp == event_timestamp (price is
    known as the bar closes). For daily bars the knowledge_timestamp is the
    market-close time of that day in the exchange's timezone.
    """
    __tablename__ = "market_ohlcv"

    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, primary_key=True)
    frequency: Mapped[str] = mapped_column(String(10), nullable=False, primary_key=True)
    # e.g. '1D', '1H', '15T', '1W'

    open: Mapped[float | None] = mapped_column(Numeric(18, 6))
    high: Mapped[float | None] = mapped_column(Numeric(18, 6))
    low: Mapped[float | None] = mapped_column(Numeric(18, 6))
    close: Mapped[float | None] = mapped_column(Numeric(18, 6))
    volume: Mapped[int | None] = mapped_column(BigInteger)
    vwap: Mapped[float | None] = mapped_column(Numeric(18, 6))

    # Adjusted for splits and dividends (computed by source)
    adj_close: Mapped[float | None] = mapped_column(Numeric(18, 6))
    adj_open: Mapped[float | None] = mapped_column(Numeric(18, 6))
    adj_high: Mapped[float | None] = mapped_column(Numeric(18, 6))
    adj_low: Mapped[float | None] = mapped_column(Numeric(18, 6))

    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    source: Mapped[str | None] = mapped_column(String(50))

    __table_args__ = (
        Index("ix_ohlcv_ticker_ts", "ticker", "event_timestamp"),
        Index("ix_ohlcv_knowledge", "knowledge_timestamp"),
    )


# ── Technical Indicators ──────────────────────────────────────────────────────

class TechnicalIndicator(Base):
    """
    Pre-computed indicator values.

    Parameters (e.g. period, std-devs) are stored in a JSONB column so that
    SMA_20 and SMA_50 are distinct rows without schema changes.

    knowledge_timestamp == event_timestamp (indicator is knowable the moment
    the bar closes).
    """
    __tablename__ = "technical_indicators"

    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, primary_key=True)
    indicator_name: Mapped[str] = mapped_column(String(50), nullable=False, primary_key=True)
    # e.g. "SMA", "EMA", "RSI", "MACD", "BBANDS", "ATR"

    value: Mapped[float | None] = mapped_column(Numeric(24, 8))
    # For multi-value indicators (MACD line + signal + histogram), use payload
    parameters: Mapped[dict | None] = mapped_column(JSONB)
    # e.g. {"period": 20} or {"fast": 12, "slow": 26, "signal": 9}
    payload: Mapped[dict | None] = mapped_column(JSONB)
    # e.g. {"macd": 0.12, "signal": 0.08, "histogram": 0.04}

    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_ti_ticker_ts", "ticker", "event_timestamp"),
    )


# ── Options Chain ─────────────────────────────────────────────────────────────

class OptionsChain(Base):
    """
    End-of-day options snapshot.

    knowledge_timestamp = market close of event_timestamp date.
    Greeks are sourced from the exchange where available; otherwise computed
    by the pipeline using Black-Scholes-Merton.
    """
    __tablename__ = "options_chain"

    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, primary_key=True)
    expiration_date: Mapped[date] = mapped_column(Date, nullable=False, primary_key=True)
    strike: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, primary_key=True)
    option_type: Mapped[str] = mapped_column(String(1), nullable=False, primary_key=True)
    # 'C' = call, 'P' = put

    bid: Mapped[float | None] = mapped_column(Numeric(18, 4))
    ask: Mapped[float | None] = mapped_column(Numeric(18, 4))
    last: Mapped[float | None] = mapped_column(Numeric(18, 4))
    volume: Mapped[int | None] = mapped_column(Integer)
    open_interest: Mapped[int | None] = mapped_column(Integer)
    implied_volatility: Mapped[float | None] = mapped_column(Numeric(12, 8))
    delta: Mapped[float | None] = mapped_column(Numeric(10, 8))
    gamma: Mapped[float | None] = mapped_column(Numeric(10, 8))
    theta: Mapped[float | None] = mapped_column(Numeric(10, 8))
    vega: Mapped[float | None] = mapped_column(Numeric(10, 8))
    rho: Mapped[float | None] = mapped_column(Numeric(10, 8))

    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_options_ticker_ts", "ticker", "event_timestamp"),
    )


# ── Macro Indicators ──────────────────────────────────────────────────────────

class MacroIndicator(Base):
    """
    Macroeconomic time-series with revision support.

    The SAME event_timestamp + indicator_code can have multiple rows with
    increasing revision_number. The data-loader always selects the latest
    revision whose knowledge_timestamp <= as_of (i.e. real-time vintage data).

    This prevents look-ahead bias from economic data revisions (e.g. NFP
    first-print vs final revision).
    """
    __tablename__ = "macro_indicators"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    indicator_code: Mapped[str] = mapped_column(String(50), nullable=False)
    series_name: Mapped[str | None] = mapped_column(String(200))
    value: Mapped[float | None] = mapped_column(Numeric(20, 8))
    unit: Mapped[str | None] = mapped_column(String(50))
    frequency: Mapped[str | None] = mapped_column(String(20))   # 'M', 'Q', 'D'
    source: Mapped[str | None] = mapped_column(String(50))
    revision_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "event_timestamp", "indicator_code", "revision_number",
            name="uq_macro_revision"
        ),
        Index("ix_macro_code_ts", "indicator_code", "event_timestamp"),
        Index("ix_macro_knowledge", "knowledge_timestamp"),
    )


# ── Flexible Financial Events (EAV + JSONB) ───────────────────────────────────

class FinancialEvent(Base):
    """
    Schema-flexible table for ALL non-market data.

    Covers: SEC 10-K/10-Q/8-K, earnings, insider trades, 13F portfolio changes,
    Congressional trades, FOMC decisions, corporate news, M&A, etc.

    New data sources require ZERO DDL changes — just a new data_category string
    and a well-documented payload schema in the source's crawler class.

    `schema_version` lets the data-loader handle payload shape evolution
    gracefully (e.g. v1 had field X, v2 renamed it to Y).

    `ticker` is NULL for macro/geo events that aren't ticker-specific.
    """
    __tablename__ = "financial_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # PiT columns
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    knowledge_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ingestion_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Discriminators
    ticker: Mapped[str | None] = mapped_column(String(20), nullable=True)
    data_category: Mapped[str] = mapped_column(String(60), nullable=False)
    data_source: Mapped[str] = mapped_column(String(100), nullable=False)

    # Human-readable summary (searchable)
    headline: Mapped[str | None] = mapped_column(Text)

    # The full structured payload — shape varies by data_category
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Allows payload migration without breaking old readers
    schema_version: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)

    # Deduplication key (e.g. SEC accession number, filing URL hash)
    source_id: Mapped[str | None] = mapped_column(String(200), unique=True)

    __table_args__ = (
        Index("ix_fe_ticker_cat_ts", "ticker", "data_category", "event_timestamp"),
        Index("ix_fe_knowledge", "knowledge_timestamp"),
        Index("ix_fe_category_ts", "data_category", "event_timestamp"),
        # GIN index on payload for fast JSONB key searches
        Index("ix_fe_payload_gin", "payload", postgresql_using="gin"),
    )
