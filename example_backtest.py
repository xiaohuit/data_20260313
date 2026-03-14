"""
Example: run a walk-forward backtest using the PiT data loader.

This script demonstrates the correct usage pattern for the AI trader:
  1. Walk forward through time in weekly steps.
  2. At each step, get a clean PiT snapshot (zero look-ahead).
  3. Make a simulated trade decision.
  4. Evaluate the trade outcome at the next step.

Run after bootstrap:
    python example_backtest.py
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from financial_pipeline.loader.pit_loader import PiTDataLoader, MarketSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UTC = timezone.utc


# ── Toy momentum strategy ─────────────────────────────────────────────────────

def momentum_signal(snapshot: MarketSnapshot, lookback_weeks: int = 12) -> dict[str, str]:
    """
    Simple cross-sectional momentum: buy top-decile, sell bottom-decile
    by 12-week return. Returns {ticker: 'LONG'|'SHORT'|'FLAT'}.
    """
    prices = snapshot.prices   # adj_close wide DF
    if prices.shape[0] < lookback_weeks:
        return {}

    returns = prices.iloc[-1] / prices.iloc[-lookback_weeks] - 1
    returns = returns.dropna()
    if returns.empty:
        return {}

    n = max(1, len(returns) // 10)
    top = set(returns.nlargest(n).index)
    bottom = set(returns.nsmallest(n).index)

    return {t: ("LONG" if t in top else "SHORT" if t in bottom else "FLAT")
            for t in returns.index}


# ── Backtest loop ─────────────────────────────────────────────────────────────

async def run_backtest(
    start_year: int = 2020,
    end_year: int = 2024,
    frequency: str = "1W",
) -> list[dict]:
    loader = PiTDataLoader(default_lookback=timedelta(days=365))

    start = datetime(start_year, 1, 1, tzinfo=UTC)
    end   = datetime(end_year,   1, 1, tzinfo=UTC)

    trades: list[dict] = []
    pending_exits: list[dict] = []   # [{ticker, direction, entry_date}]

    prev_signals: dict[str, str] = {}

    async for snapshot in loader.walk(
        start=start,
        end=end,
        frequency=frequency,
        lookback=timedelta(days=365),
        include_macro=True,
        include_alternatives=True,
        include_fundamentals=True,
    ):
        as_of = snapshot.as_of
        log.info("--- as_of: %s | universe: %d tickers ---", as_of.date(), len(snapshot.universe))

        # ── Evaluate pending exits ────────────────────────────────────────────
        for pending in pending_exits:
            result = await loader.evaluate_trade(
                ticker=pending["ticker"],
                entry_date=pending["entry_date"],
                exit_date=as_of,
                direction=pending["direction"],
            )
            trades.append(result)

        # ── Generate new signals ──────────────────────────────────────────────
        signals = momentum_signal(snapshot)

        # ── Log macro context ─────────────────────────────────────────────────
        if snapshot.macro:
            key_macro = {k: round(v, 3) for k, v in snapshot.macro.items()
                         if k in ("FED_FUNDS_RATE", "CPI_YOY", "GDP_GROWTH", "VIX")}
            log.info("  Macro: %s", key_macro)

        # ── Identify signal flips → new entries ───────────────────────────────
        new_entries: list[dict] = []
        for ticker, new_sig in signals.items():
            old_sig = prev_signals.get(ticker, "FLAT")
            if old_sig != new_sig and new_sig in ("LONG", "SHORT"):
                new_entries.append({
                    "ticker":     ticker,
                    "direction":  new_sig,
                    "entry_date": as_of,
                })

        pending_exits = new_entries   # will be evaluated next step
        prev_signals = signals

    # ── Final summary ─────────────────────────────────────────────────────────
    completed = [t for t in trades if "pnl_pct" in t]
    if completed:
        gains = [t["pnl_pct"] for t in completed if t["result"] == "GAIN"]
        losses = [t["pnl_pct"] for t in completed if t["result"] == "LOSS"]
        total_pnl = sum(t["pnl_pct"] for t in completed)
        win_rate = len(gains) / len(completed) * 100

        log.info("=" * 60)
        log.info("BACKTEST RESULTS (%d–%d, %s)", start_year, end_year, frequency)
        log.info("  Trades:   %d  |  Win Rate: %.1f%%", len(completed), win_rate)
        log.info("  Total PnL: %.2f%%", total_pnl)
        log.info("  Avg Gain: %.2f%%  |  Avg Loss: %.2f%%",
                 sum(gains)/len(gains) if gains else 0,
                 sum(losses)/len(losses) if losses else 0)
        log.info("=" * 60)

    return trades


if __name__ == "__main__":
    results = asyncio.run(run_backtest(start_year=2020, end_year=2024, frequency="1W"))
    print(f"\nCompleted {len(results)} trades.")
