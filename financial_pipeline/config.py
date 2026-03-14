"""
Central configuration for the financial data pipeline.
All secrets are loaded from environment variables; this file only holds defaults and structure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import ClassVar


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DBConfig:
    """TimescaleDB connection settings."""
    url: str = field(
        default_factory=lambda: os.environ.get(
            "TIMESCALE_URL",
            "postgresql+asyncpg://pipeline:pipeline@localhost:5432/findata",
        )
    )
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RedisConfig:
    url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    rate_limit_db: int = 1
    cache_ttl_seconds: int = 300


# ---------------------------------------------------------------------------
# API keys (all read from env; no hard-coded secrets)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class APIKeys:
    fred: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))
    alpha_vantage: str = field(default_factory=lambda: os.environ.get("ALPHA_VANTAGE_KEY", ""))
    polygon: str = field(default_factory=lambda: os.environ.get("POLYGON_API_KEY", ""))
    sec_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "SEC_USER_AGENT", "FinancialPipeline research@example.com"
        )
    )


# ---------------------------------------------------------------------------
# Rate limits  (requests per second per source)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateLimits:
    yfinance: float = 2.0
    fred: float = 5.0
    sec_edgar: float = 10.0       # SEC asks ≤10 req/s
    open_insider: float = 0.5
    polygon: float = 5.0
    alpha_vantage: float = 0.2    # 5 req/min on free tier
    congress_trades: float = 0.5


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseConfig:
    indices: tuple[str, ...] = ("SP500", "NDX100")
    # Wikipedia URLs used to bootstrap current constituents
    sp500_wiki_url: str = (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    ndx100_wiki_url: str = (
        "https://en.wikipedia.org/wiki/Nasdaq-100"
    )
    # How far back to seed historical membership from SEC 8-K filings
    history_years: int = 6


# ---------------------------------------------------------------------------
# Data categories — used as discriminators in the flexible events table
# ---------------------------------------------------------------------------

class DataCategory:
    # Market
    OHLCV_DAILY: ClassVar[str] = "OHLCV_DAILY"
    OHLCV_INTRADAY: ClassVar[str] = "OHLCV_INTRADAY"
    OPTIONS_CHAIN: ClassVar[str] = "OPTIONS_CHAIN"
    TECHNICAL_INDICATOR: ClassVar[str] = "TECHNICAL_INDICATOR"
    # Fundamentals
    SEC_10K: ClassVar[str] = "SEC_10K"
    SEC_10Q: ClassVar[str] = "SEC_10Q"
    SEC_8K: ClassVar[str] = "SEC_8K"
    EARNINGS: ClassVar[str] = "EARNINGS"
    # Alternative / sentiment
    INSIDER_TRADE: ClassVar[str] = "INSIDER_TRADE"
    PORTFOLIO_13F: ClassVar[str] = "PORTFOLIO_13F"
    CONGRESS_TRADE: ClassVar[str] = "CONGRESS_TRADE"
    # Macro
    MACRO_SERIES: ClassVar[str] = "MACRO_SERIES"
    FOMC_DECISION: ClassVar[str] = "FOMC_DECISION"


# ---------------------------------------------------------------------------
# Macro series codes → FRED series IDs
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "CPI_YOY":        "CPIAUCSL",
    "CORE_CPI":       "CPILFESL",
    "PCE":            "PCE",
    "CORE_PCE":       "PCEPILFE",
    "PPI":            "PPIACO",
    "FED_FUNDS_RATE": "FEDFUNDS",
    "GDP":            "GDP",
    "GDP_GROWTH":     "A191RL1Q225SBEA",
    "NFP":            "PAYEMS",
    "UNEMPLOYMENT":   "UNRATE",
    "RETAIL_SALES":   "RSAFS",
    "PMI_MFG":        "MANEMP",          # proxy; ISM not on FRED freely
    "CONSUMER_CONF":  "UMCSENT",         # Michigan Consumer Sentiment
    "CRUDE_OIL_WTI":  "DCOILWTICO",
    "NATURAL_GAS":    "DHHNGSP",
    "SUPPLY_CHAIN":   "GSCPI",           # NY Fed Global Supply Chain Pressure
    "10Y_YIELD":      "GS10",
    "2Y_YIELD":       "GS2",
    "YIELD_CURVE":    "T10Y2Y",
    "DXY":            "DTWEXBGS",        # USD broad index
    "VIX":            "VIXCLS",
}

# ---------------------------------------------------------------------------
# Retry / resilience
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResilienceConfig:
    max_retries: int = 5
    backoff_base: float = 2.0            # exponential back-off base (seconds)
    backoff_max: float = 120.0
    circuit_breaker_threshold: int = 10  # failures before opening circuit
    circuit_breaker_timeout: timedelta = timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Top-level config singleton
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    db: DBConfig = field(default_factory=DBConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api_keys: APIKeys = field(default_factory=APIKeys)
    rate_limits: RateLimits = field(default_factory=RateLimits)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)

    # Storage paths for Parquet cold archive
    parquet_root: str = field(
        default_factory=lambda: os.environ.get("PARQUET_ROOT", "./data/parquet")
    )


CONFIG = PipelineConfig()
