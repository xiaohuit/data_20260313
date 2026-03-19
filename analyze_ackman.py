"""
Bill Ackman / Pershing Square Capital Management  13-F Performance Analyzer
===========================================================================

Outputs a machine-readable JSON schema (ackman_analysis.json) plus a
human-readable text report.

Schema layout: see trader_eval_lib.py

Usage:
    python analyze_ackman.py [--start 2015] [--copycat] [--no-cache]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trader_eval_lib import (
    get_13f_filings,
    build_portfolio_history,
    add_holding_quarter,
    load_prices,
    compute_performance,
    build_json_schema,
    save_json,
    print_report,
)

# ── Ackman / Pershing Square-specific constants ───────────────────────────────

ACKMAN_CIK     = "0001336528"
MANAGER_NAME   = "Bill Ackman"
MANAGER_ENTITY = "Pershing Square Capital Management"
MANAGER_STRAT  = "concentrated/activist"

MANAGER_META = {
    "name":     MANAGER_NAME,
    "entity":   MANAGER_ENTITY,
    "cik":      ACKMAN_CIK,
    "strategy": MANAGER_STRAT,
}

# Numeric CIK (no leading zeros, no "CIK" prefix) used for EDGAR URLs
_CIK_NUMERIC = "1336528"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bill Ackman / Pershing Square 13-F performance analyzer"
    )
    parser.add_argument("--start",    type=int, default=2015)
    parser.add_argument("--copycat",  action="store_true",
                        help="Use filing-date entry prices instead of quarter-end")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download from SEC (ignore local cache)")
    args = parser.parse_args()
    use_cache = not args.no_cache

    print("Step 1/4  Fetching 13-F filing list…")
    filings = get_13f_filings(cik=ACKMAN_CIK, use_cache=use_cache)
    if filings.empty:
        print("ERROR: no filings found"); sys.exit(1)
    print(f"  {len(filings)} filings  "
          f"({filings['period_end'].min()} → {filings['period_end'].max()})")

    print("\nStep 2/4  Downloading & parsing holdings…")
    portfolio = build_portfolio_history(
        filings, cik_numeric=_CIK_NUMERIC,
        start_year=args.start, use_cache=use_cache,
    )
    if portfolio.empty:
        print("ERROR: no holdings parsed"); sys.exit(1)
    portfolio = add_holding_quarter(portfolio)

    tickers = list(set(
        [t for t in portfolio["ticker"].dropna().unique().tolist() if t] + ["SPY"]
    ))
    print(f"\nStep 3/4  Loading prices for {len(tickers)} tickers…")
    prices = load_prices(tickers)
    if prices.empty:
        print("ERROR: no price data"); sys.exit(1)
    print(f"  {len(prices):,} rows  /  {prices['ticker'].nunique()} tickers")

    print("\nStep 4/4  Computing performance…")
    # Ackman runs a genuinely concentrated book (5–12 positions is normal),
    # so use a lower confidential_threshold than the default of 10.
    results = compute_performance(
        portfolio, prices, copycat=args.copycat, confidential_threshold=5
    )

    schema = build_json_schema(
        results,
        manager_meta=MANAGER_META,
        start_year=args.start,
        copycat=args.copycat,
    )

    # Save outputs
    json_path = Path("ackman_analysis.json")
    csv_path  = Path("ackman_analysis.csv")
    save_json(schema, json_path)
    results["positions"].to_csv(csv_path, index=False)
    results["holding_periods"].to_csv(Path("ackman_holding_periods.csv"), index=False)
    results["quarterly_returns"].to_csv(Path("ackman_quarterly.csv"), index=False)

    print_report(results, schema, manager_meta=MANAGER_META, copycat=args.copycat)
    print(f"\nOutputs written:")
    print(f"  {json_path}                  (full machine-readable schema)")
    print(f"  {csv_path}             (flat position-level detail)")
    print(f"  ackman_holding_periods.csv   (full hold duration per ticker)")
    print(f"  ackman_quarterly.csv         (quarterly portfolio returns)")


if __name__ == "__main__":
    main()
