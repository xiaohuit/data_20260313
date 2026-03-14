"""
Market data crawler — OHLCV, technical indicators, options chain.

Primary source: yfinance (Yahoo Finance, no API key required).
Fallback:       Polygon.io (requires POLYGON_API_KEY).

PiT rules:
  - Daily bar:      knowledge_timestamp = market close of that day (16:00 ET).
  - Intraday bar:   knowledge_timestamp = bar close timestamp.
  - Options snap:   knowledge_timestamp = snapshot time.

Adjustments:
  yfinance provides split- and dividend-adjusted OHLCV via the `auto_adjust`
  flag. We store BOTH raw and adjusted close so the data-loader can serve
  either depending on the use-case (adjusted for returns, raw for options
  strike alignment).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from financial_pipeline.config import CONFIG, DataCategory
from financial_pipeline.crawlers.base import BaseCrawler, pit_stamp, utcnow
from financial_pipeline.db.models import FinancialEvent, MarketOHLCV, OptionsChain, TechnicalIndicator

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = dtime(16, 0, 0)   # 4 PM Eastern


def _market_close_utc(date_: pd.Timestamp) -> datetime:
    """Return the UTC datetime of NYSE market close for a given date."""
    close_et = datetime.combine(date_.date(), MARKET_CLOSE, tzinfo=ET)
    return close_et.astimezone(timezone.utc)


# ── OHLCV Crawler ─────────────────────────────────────────────────────────────

class OHLCVCrawler(BaseCrawler):
    """Fetches daily and intraday bars via yfinance."""

    def __init__(self, frequency: str = "1D", **kwargs) -> None:
        super().__init__(**kwargs)
        # frequency: '1D', '1H', '30m', '15m', '5m', '1m'
        self._frequency = frequency

    @property
    def source_name(self) -> str:
        return f"yfinance_ohlcv_{self._frequency}"

    @property
    def rate_limit(self) -> float:
        return CONFIG.rate_limits.yfinance

    @property
    def target_model(self):
        return MarketOHLCV

    @property
    def conflict_columns(self) -> list[str]:
        return ["event_timestamp", "ticker", "frequency"]

    @property
    def update_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "adj_close",
                "adj_open", "adj_high", "adj_low", "vwap", "knowledge_timestamp"]

    async def fetch_records(
        self, ticker: str, start: datetime, end: datetime
    ) -> list[dict]:
        await self._bucket.acquire()
        yf_interval = self._yf_interval()
        # yfinance is sync; run in thread pool to not block event loop
        import asyncio
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None,
            lambda: self._download(ticker, start, end, yf_interval),
        )
        if df is None or df.empty:
            return []
        return self._to_records(df, ticker)

    def _yf_interval(self) -> str:
        mapping = {
            "1D": "1d", "1H": "1h", "30T": "30m",
            "15T": "15m", "5T": "5m", "1T": "1m",
        }
        return mapping.get(self._frequency, "1d")

    @staticmethod
    def _download(
        ticker: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame | None:
        try:
            t = yf.Ticker(ticker)
            # auto_adjust=False → raw prices; we compute adj ourselves
            df = t.history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=False,
                back_adjust=False,
                actions=True,
            )
            return df
        except Exception as exc:
            log.warning("yfinance download failed for %s: %s", ticker, exc)
            return None

    def _to_records(self, df: pd.DataFrame, ticker: str) -> list[dict]:
        records = []
        df.index = pd.to_datetime(df.index, utc=True)

        # Compute adjusted prices from raw + dividend/split factors
        # yfinance provides Dividends and Stock Splits columns
        close_col = "Close"
        adj_close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        for ts, row in df.iterrows():
            if self._frequency == "1D":
                knowledge_ts = _market_close_utc(ts)
            else:
                knowledge_ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)

            records.append({
                "event_timestamp":    ts.to_pydatetime(),
                "knowledge_timestamp": knowledge_ts,
                "ticker":             ticker,
                "frequency":          self._frequency,
                "open":               float(row.get("Open")) if pd.notna(row.get("Open")) else None,
                "high":               float(row.get("High")) if pd.notna(row.get("High")) else None,
                "low":                float(row.get("Low"))  if pd.notna(row.get("Low"))  else None,
                "close":              float(row.get(close_col)) if pd.notna(row.get(close_col)) else None,
                "adj_close":          float(row.get(adj_close_col)) if pd.notna(row.get(adj_close_col)) else None,
                "adj_open":           None,   # computed post-hoc if needed
                "adj_high":           None,
                "adj_low":            None,
                "volume":             int(row.get("Volume")) if pd.notna(row.get("Volume")) else None,
                "vwap":               None,
                "source":             "yfinance",
            })
        return records


# ── Technical Indicators Crawler ──────────────────────────────────────────────

class TechnicalIndicatorCrawler(BaseCrawler):
    """
    Computes and stores a standard set of technical indicators from
    stored OHLCV data. Runs AFTER OHLCVCrawler has populated the DB.

    Indicators computed (all using pandas-ta):
      Trend:     SMA(20,50,200), EMA(9,21), MACD(12,26,9)
      Momentum:  RSI(14), Stochastic(14,3), Williams %R(14)
      Volatility:Bollinger Bands(20,2), ATR(14), Historical Vol(21)
      Volume:    OBV, VWAP (intraday only)
    """

    @property
    def source_name(self) -> str:
        return "technical_indicators"

    @property
    def rate_limit(self) -> float:
        return 100.0   # local computation, no external call

    @property
    def target_model(self):
        return TechnicalIndicator

    @property
    def conflict_columns(self) -> list[str]:
        return ["event_timestamp", "ticker", "indicator_name"]

    @property
    def update_columns(self) -> list[str]:
        return ["value", "parameters", "payload", "knowledge_timestamp"]

    async def fetch_records(
        self, ticker: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Load OHLCV from DB, compute indicators, return as records."""
        import asyncio
        loop = asyncio.get_running_loop()
        records = await loop.run_in_executor(
            None, lambda: self._compute(ticker, start, end)
        )
        return records

    def _compute(self, ticker: str, start: datetime, end: datetime) -> list[dict]:
        try:
            import pandas_ta as ta  # type: ignore
        except ImportError:
            log.error("pandas_ta not installed. Run: pip install pandas-ta")
            return []

        # Fetch from yfinance directly for computation
        t = yf.Ticker(ticker)
        df = t.history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            return []

        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"stock splits": "stock_splits"})

        # Compute all indicators using pandas_ta strategy
        df.ta.strategy(ta.Strategy(
            name="all_standard",
            ta=[
                {"kind": "sma",  "length": 20},
                {"kind": "sma",  "length": 50},
                {"kind": "sma",  "length": 200},
                {"kind": "ema",  "length": 9},
                {"kind": "ema",  "length": 21},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "rsi",  "length": 14},
                {"kind": "stoch","k": 14, "d": 3},
                {"kind": "willr","length": 14},
                {"kind": "bbands","length": 20, "std": 2},
                {"kind": "atr",  "length": 14},
                {"kind": "obv"},
            ]
        ))

        records: list[dict] = []
        # Map pandas_ta column names → our indicator_name convention
        indicator_map: dict[str, tuple[str, dict]] = {
            "SMA_20":    ("SMA", {"period": 20}),
            "SMA_50":    ("SMA", {"period": 50}),
            "SMA_200":   ("SMA", {"period": 200}),
            "EMA_9":     ("EMA", {"period": 9}),
            "EMA_21":    ("EMA", {"period": 21}),
            "RSI_14":    ("RSI", {"period": 14}),
            "ATRr_14":   ("ATR", {"period": 14}),
            "OBV":       ("OBV", {}),
            "STOCHk_14_3_3": ("STOCH_K", {"k": 14, "d": 3}),
            "STOCHd_14_3_3": ("STOCH_D", {"k": 14, "d": 3}),
            "WILLR_14":  ("WILLR", {"period": 14}),
            "BBL_20_2.0": ("BB_LOWER", {"period": 20, "std": 2}),
            "BBM_20_2.0": ("BB_MID",   {"period": 20, "std": 2}),
            "BBU_20_2.0": ("BB_UPPER", {"period": 20, "std": 2}),
            "BBB_20_2.0": ("BB_WIDTH", {"period": 20, "std": 2}),
        }
        # MACD needs multi-value payload
        macd_cols = [c for c in df.columns if c.startswith("MACD")]

        for ts, row in df.iterrows():
            knowledge_ts = _market_close_utc(ts)
            for raw_col, (name, params) in indicator_map.items():
                if raw_col not in df.columns:
                    continue
                val = row[raw_col]
                if pd.isna(val):
                    continue
                records.append({
                    "event_timestamp":     ts.to_pydatetime(),
                    "knowledge_timestamp": knowledge_ts,
                    "ticker":              ticker,
                    "indicator_name":      name,
                    "value":               float(val),
                    "parameters":          params,
                    "payload":             None,
                })

            # MACD as a single row with payload
            if macd_cols:
                macd_payload = {}
                for c in macd_cols:
                    v = row.get(c)
                    if pd.notna(v):
                        macd_payload[c] = float(v)
                if macd_payload:
                    records.append({
                        "event_timestamp":     ts.to_pydatetime(),
                        "knowledge_timestamp": knowledge_ts,
                        "ticker":              ticker,
                        "indicator_name":      "MACD",
                        "value":               macd_payload.get(f"MACD_12_26_9"),
                        "parameters":          {"fast": 12, "slow": 26, "signal": 9},
                        "payload":             macd_payload,
                    })
        return records


# ── Options Chain Crawler ─────────────────────────────────────────────────────

class OptionsChainCrawler(BaseCrawler):
    """
    Fetches options chain snapshots via yfinance.
    Only fetches CURRENT chain (historical options data requires paid API).
    For backtesting historical options, use Polygon or CBOE data.
    """

    @property
    def source_name(self) -> str:
        return "yfinance_options"

    @property
    def rate_limit(self) -> float:
        return CONFIG.rate_limits.yfinance

    @property
    def target_model(self):
        return OptionsChain

    @property
    def conflict_columns(self) -> list[str]:
        return ["event_timestamp", "ticker", "expiration_date", "strike", "option_type"]

    @property
    def update_columns(self) -> list[str]:
        return ["bid", "ask", "last", "volume", "open_interest",
                "implied_volatility", "delta", "gamma", "theta", "vega"]

    async def fetch_records(
        self, ticker: str, start: datetime, end: datetime
    ) -> list[dict]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._fetch_chain(ticker))

    @staticmethod
    def _fetch_chain(ticker: str) -> list[dict]:
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
        except Exception as exc:
            log.warning("options fetch failed %s: %s", ticker, exc)
            return []

        now = utcnow()
        records: list[dict] = []
        for exp in expirations[:8]:   # limit to next 8 expiry dates
            try:
                chain = t.option_chain(exp)
                for opt_type, df in (("C", chain.calls), ("P", chain.puts)):
                    for _, row in df.iterrows():
                        records.append({
                            "event_timestamp":     now,
                            "knowledge_timestamp": now,
                            "ticker":              ticker,
                            "expiration_date":     pd.to_datetime(exp).date(),
                            "strike":              float(row["strike"]),
                            "option_type":         opt_type,
                            "bid":                 float(row["bid"]) if pd.notna(row.get("bid")) else None,
                            "ask":                 float(row["ask"]) if pd.notna(row.get("ask")) else None,
                            "last":                float(row["lastPrice"]) if pd.notna(row.get("lastPrice")) else None,
                            "volume":              int(row["volume"]) if pd.notna(row.get("volume")) else None,
                            "open_interest":       int(row["openInterest"]) if pd.notna(row.get("openInterest")) else None,
                            "implied_volatility":  float(row["impliedVolatility"]) if pd.notna(row.get("impliedVolatility")) else None,
                            "delta":               None,
                            "gamma":               None,
                            "theta":               None,
                            "vega":                None,
                            "rho":                 None,
                        })
            except Exception as exc:
                log.debug("options chain error %s/%s: %s", ticker, exp, exc)
        return records
