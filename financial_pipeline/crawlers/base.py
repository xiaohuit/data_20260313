"""
Abstract base crawler.

Provides:
  - Async HTTP client with per-source token-bucket rate limiting
  - Exponential back-off with jitter
  - Circuit breaker (open after N consecutive failures, auto-recovers)
  - PiT timestamp injection (knowledge_timestamp stamping)
  - Idempotent bulk upsert to TimescaleDB
  - Checkpoint/resume so long backfills survive restarts
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_pipeline.config import CONFIG, ResilienceConfig

log = logging.getLogger(__name__)


# ── Token-bucket rate limiter ────────────────────────────────────────────────

class TokenBucket:
    """
    Thread-safe async token bucket.
    `rate` tokens added per second; bucket capacity == rate (1-second burst).
    """
    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens += (now - self._last) * self._rate
            self._tokens = min(self._tokens, self._rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# ── Circuit breaker ──────────────────────────────────────────────────────────

class CircuitState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(self, cfg: ResilienceConfig) -> None:
        self._threshold = cfg.circuit_breaker_threshold
        self._timeout = cfg.circuit_breaker_timeout.total_seconds()
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None
        self._state = CircuitState.CLOSED

    def record_success(self) -> None:
        self._failures.clear()
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failures.append(now)
        # Prune old failures outside the window
        while self._failures and now - self._failures[0] > self._timeout:
            self._failures.popleft()
        if len(self._failures) >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            log.warning("Circuit OPENED after %d failures", len(self._failures))

    def is_open(self) -> bool:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - (self._opened_at or 0) > self._timeout:
                self._state = CircuitState.HALF_OPEN
                log.info("Circuit moved to HALF_OPEN")
                return False
            return True
        return False


# ── Retry decorator ──────────────────────────────────────────────────────────

async def with_retry(
    coro_fn,
    *args,
    cfg: ResilienceConfig = CONFIG.resilience,
    circuit: CircuitBreaker | None = None,
    **kwargs,
) -> Any:
    for attempt in range(cfg.max_retries + 1):
        if circuit and circuit.is_open():
            raise RuntimeError("Circuit is OPEN — skipping request")
        try:
            result = await coro_fn(*args, **kwargs)
            if circuit:
                circuit.record_success()
            return result
        except (httpx.HTTPStatusError, httpx.TransportError, asyncio.TimeoutError) as exc:
            if circuit:
                circuit.record_failure()
            if attempt == cfg.max_retries:
                raise
            wait = min(cfg.backoff_base ** attempt + random.uniform(0, 1), cfg.backoff_max)
            log.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs …",
                attempt + 1, cfg.max_retries, exc, wait,
            )
            await asyncio.sleep(wait)


# ── PiT utilities ────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def pit_stamp(
    records: list[dict],
    knowledge_timestamp: datetime | None = None,
) -> list[dict]:
    """
    Inject knowledge_timestamp into a list of record dicts.
    If not supplied, defaults to the current UTC time (live ingestion).
    """
    kt = knowledge_timestamp or utcnow()
    for r in records:
        r.setdefault("knowledge_timestamp", kt)
        r.setdefault("ingestion_timestamp", utcnow())
    return records


def stable_source_id(data: dict, keys: list[str]) -> str:
    """Deterministic dedup key from selected fields."""
    raw = json.dumps({k: str(data.get(k, "")) for k in sorted(keys)}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── Database session factory ─────────────────────────────────────────────────

def make_session_factory(db_url: str | None = None) -> async_sessionmaker[AsyncSession]:
    url = db_url or CONFIG.db.url
    engine = create_async_engine(
        url,
        pool_size=CONFIG.db.pool_size,
        max_overflow=CONFIG.db.max_overflow,
        echo=CONFIG.db.echo,
    )
    return async_sessionmaker(engine, expire_on_commit=False)


# ── Bulk upsert helper ───────────────────────────────────────────────────────

async def bulk_upsert(
    session: AsyncSession,
    table,           # SQLAlchemy Table or mapped class
    records: list[dict],
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
) -> int:
    """
    PostgreSQL INSERT … ON CONFLICT DO UPDATE.
    Returns number of rows written.
    """
    if not records:
        return 0

    tbl = table.__table__ if hasattr(table, "__table__") else table
    stmt = pg_insert(tbl).values(records)

    if update_columns:
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_columns,
            set_={col: stmt.excluded[col] for col in update_columns},
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)

    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount


# ── Checkpoint store (lightweight, Redis-backed) ─────────────────────────────

class CheckpointStore:
    """
    Persists crawler progress (last successfully processed date / cursor)
    to Redis so long backfills can resume after restarts.
    Falls back to in-memory if Redis is unavailable.
    """
    def __init__(self, namespace: str) -> None:
        self._ns = namespace
        self._mem: dict[str, str] = {}
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis  # type: ignore
                self._redis = aioredis.from_url(CONFIG.redis.url)
            except Exception:
                pass
        return self._redis

    async def get(self, key: str) -> str | None:
        r = await self._get_redis()
        if r:
            val = await r.get(f"{self._ns}:{key}")
            return val.decode() if val else None
        return self._mem.get(key)

    async def set(self, key: str, value: str) -> None:
        r = await self._get_redis()
        if r:
            await r.set(f"{self._ns}:{key}", value)
        else:
            self._mem[key] = value


# ── Abstract base crawler ────────────────────────────────────────────────────

class BaseCrawler(ABC):
    """
    All crawlers inherit from this class.

    Subclasses implement:
      - `source_name` property
      - `rate_limit` property (requests/sec)
      - `fetch_records(ticker, start, end)` → list[dict]
      - `target_model` property  →  SQLAlchemy model class
      - `conflict_columns` property
      - `update_columns` property
    """

    def __init__(self, session_factory: async_sessionmaker | None = None) -> None:
        self._session_factory = session_factory or make_session_factory()
        self._bucket = TokenBucket(self.rate_limit)
        self._circuit = CircuitBreaker(CONFIG.resilience)
        self._checkpoints = CheckpointStore(self.source_name)
        self._http: httpx.AsyncClient | None = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @property
    @abstractmethod
    def rate_limit(self) -> float: ...

    @abstractmethod
    async def fetch_records(
        self, ticker: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Fetch raw records and return as list of dicts ready for DB insert."""
        ...

    @property
    @abstractmethod
    def target_model(self): ...

    @property
    @abstractmethod
    def conflict_columns(self) -> list[str]: ...

    @property
    def update_columns(self) -> list[str]:
        """Columns to update on conflict. Override if needed."""
        return []

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    @asynccontextmanager
    async def http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        headers = {
            "User-Agent": CONFIG.api_keys.sec_user_agent,
            "Accept-Encoding": "gzip",
        }
        async with httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
            follow_redirects=True,
        ) as client:
            yield client

    async def get(self, url: str, **kwargs) -> httpx.Response:
        await self._bucket.acquire()
        async with self.http_client() as client:
            resp = await with_retry(
                client.get, url, circuit=self._circuit, **kwargs
            )
            resp.raise_for_status()
            return resp

    # ── Main entry points ─────────────────────────────────────────────────────

    async def backfill(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        batch_size: int = 50,
    ) -> None:
        """
        Backfill historical data for a list of tickers.
        Checkpoints progress so it can resume after interruption.
        """
        log.info("[%s] Starting backfill for %d tickers", self.source_name, len(tickers))
        async with self._session_factory() as session:
            for ticker in tickers:
                ck_key = f"{ticker}:{start.date()}:{end.date()}"
                done = await self._checkpoints.get(ck_key)
                if done == "1":
                    log.debug("[%s] %s already backfilled, skipping", self.source_name, ticker)
                    continue
                try:
                    records = await self.fetch_records(ticker, start, end)
                    if records:
                        await bulk_upsert(
                            session,
                            self.target_model,
                            records,
                            self.conflict_columns,
                            self.update_columns,
                        )
                    await self._checkpoints.set(ck_key, "1")
                    log.info("[%s] %s → %d rows", self.source_name, ticker, len(records))
                except Exception as exc:
                    log.error("[%s] %s failed: %s", self.source_name, ticker, exc)

    async def ingest_live(self, tickers: list[str]) -> None:
        """Ingest the most recent available data (called by scheduler)."""
        now = utcnow()
        async with self._session_factory() as session:
            for ticker in tickers:
                try:
                    records = await self.fetch_records(ticker, start=now, end=now)
                    if records:
                        await bulk_upsert(
                            session,
                            self.target_model,
                            records,
                            self.conflict_columns,
                            self.update_columns,
                        )
                except Exception as exc:
                    log.error("[%s] live ingest %s: %s", self.source_name, ticker, exc)
