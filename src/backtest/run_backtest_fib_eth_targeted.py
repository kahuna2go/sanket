"""ETH fib 0.745 — targeted sweep to find >100 trades/year + GO.

Tests session-only filter (untested so far) and RVOL thresholds below 1.2.

Usage:
  python -m src.backtest.run_backtest_fib_eth_targeted
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.run_backtest_fib import (
    FibConfig,
    _compute_bias_fib,
    _run_simulation_fib,
    _print_fib_table,
)
from src.backtest.fetch_history import load_cache

CONFIGS = [
    # Reference: known results
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=0.0, session_filter=False, tp1_frac=0.5, label="dev=2% no filters (ref)"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.2, session_filter=False, tp1_frac=0.5, label="dev=2% RVOL≥1.2 (ref, NO-GO)"),
    # Session-only (untested)
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=0.0, session_filter=True, tp1_frac=0.5, label="dev=2% + Session only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=0.0, session_filter=True, tp1_frac=0.0, label="dev=2% + Session only TP2only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.12, rvol_min=0.0, session_filter=True, tp1_frac=0.5, label="dev=2% zone=0.12 + Session only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.12, rvol_min=0.0, session_filter=True, tp1_frac=0.0, label="dev=2% zone=0.12 + Session TP2only"),
    # Looser RVOL thresholds
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.0, session_filter=False, tp1_frac=0.5, label="dev=2% RVOL≥1.0"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.1, session_filter=False, tp1_frac=0.5, label="dev=2% RVOL≥1.1"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.0, session_filter=True,  tp1_frac=0.5, label="dev=2% RVOL≥1.0 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.0, session_filter=True,  tp1_frac=0.0, label="dev=2% RVOL≥1.0 + Session TP2only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.1, session_filter=True,  tp1_frac=0.5, label="dev=2% RVOL≥1.1 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.1, session_filter=True,  tp1_frac=0.0, label="dev=2% RVOL≥1.1 + Session TP2only"),
]


async def main():
    candles_1h = load_cache("ETH", "1h") or []
    candles_5m = load_cache("ETH", "5m") or []
    if not candles_1h or not candles_5m:
        print("Missing ETH candles — run with --fetch first")
        return

    bias_list = _compute_bias_fib(candles_1h, 2.0)

    all_stats = [
        (cfg, _run_simulation_fib(candles_5m, bias_list, cfg, debug=False))
        for cfg in CONFIGS
    ]
    _print_fib_table("ETH", candles_5m, all_stats, entry_tf="5m")


if __name__ == "__main__":
    asyncio.run(main())
