"""TP split comparison for SOL using ZigZag market structure.

Mirrors run_backtest_tp_test.py but uses the ZZ approach (dev=2% — best SOL config).
Run after run_backtest_tp_test.py to compare both structure methods side by side.

Usage:
  python -m src.backtest.run_backtest_zz_tp_test
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.run_backtest_zz import (
    ZZConfig,
    _compute_bias_zz,
    _run_simulation_zz,
    _print_zz_table,
)
from src.backtest.fetch_history import load_cache

DEV = 2.0  # best-performing deviation for SOL

TP_CONFIGS = [
    ZZConfig(deviation_pct=DEV, rvol_min=0.0, tp1_frac=0.5, label="50/50  no filters"),
    ZZConfig(deviation_pct=DEV, rvol_min=0.0, tp1_frac=0.7, label="70/30  no filters"),
    ZZConfig(deviation_pct=DEV, rvol_min=0.0, tp1_frac=0.0, label="TP2only no filters"),
    ZZConfig(deviation_pct=DEV, rvol_min=1.2, tp1_frac=0.5, label="50/50  + RVOL≥1.2"),
    ZZConfig(deviation_pct=DEV, rvol_min=1.2, tp1_frac=0.7, label="70/30  + RVOL≥1.2"),
    ZZConfig(deviation_pct=DEV, rvol_min=1.2, tp1_frac=0.0, label="TP2only + RVOL≥1.2"),
    ZZConfig(deviation_pct=DEV, rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="50/50  + RVOL≥1.2 + Session"),
    ZZConfig(deviation_pct=DEV, rvol_min=1.2, session_filter=True, tp1_frac=0.7, label="70/30  + RVOL≥1.2 + Session"),
    ZZConfig(deviation_pct=DEV, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="TP2only + RVOL≥1.2 + Session"),
]


async def main():
    candles_1h = load_cache("SOL", "1h") or []
    candles_5m = load_cache("SOL", "5m") or []
    if not candles_1h or not candles_5m:
        print("Missing SOL candles — run with --fetch first")
        return

    bias_list = _compute_bias_zz(candles_1h, DEV)

    all_stats = [
        (cfg, _run_simulation_zz(candles_5m, bias_list, cfg, debug=False))
        for cfg in TP_CONFIGS
    ]
    _print_zz_table("SOL", candles_5m, all_stats, entry_tf="5m")


if __name__ == "__main__":
    asyncio.run(main())
