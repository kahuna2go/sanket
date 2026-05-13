"""VA Bounce entry + tight SL (last HL/LH ± 0.05 × range) — ETH.

Usage:
  python -m src.backtest.run_backtest_zz_tightsl_eth
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

CONFIGS = [
    # dev=2% — baseline VA-SL vs tight SL
    ZZConfig(deviation_pct=2.0, rvol_min=0.0, tp1_frac=0.5, tight_sl=False, label="dev=2% VA-SL  50/50  no filters"),
    ZZConfig(deviation_pct=2.0, rvol_min=0.0, tp1_frac=0.5, tight_sl=True,  label="dev=2% TightSL 50/50  no filters"),
    ZZConfig(deviation_pct=2.0, rvol_min=0.0, tp1_frac=0.0, tight_sl=True,  label="dev=2% TightSL TP2only no filters"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, tp1_frac=0.5, tight_sl=False, label="dev=2% VA-SL  50/50  + RVOL≥1.2"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, tp1_frac=0.5, tight_sl=True,  label="dev=2% TightSL 50/50  + RVOL≥1.2"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, tp1_frac=0.0, tight_sl=True,  label="dev=2% TightSL TP2only + RVOL≥1.2"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, session_filter=True, tp1_frac=0.5, tight_sl=False, label="dev=2% VA-SL  50/50  + RVOL≥1.2 + Session"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, session_filter=True, tp1_frac=0.5, tight_sl=True,  label="dev=2% TightSL 50/50  + RVOL≥1.2 + Session"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, session_filter=True, tp1_frac=0.0, tight_sl=False, label="dev=2% VA-SL  TP2only + RVOL≥1.2 + Session"),
    ZZConfig(deviation_pct=2.0, rvol_min=1.2, session_filter=True, tp1_frac=0.0, tight_sl=True,  label="dev=2% TightSL TP2only + RVOL≥1.2 + Session"),
    # dev=3%
    ZZConfig(deviation_pct=3.0, rvol_min=0.0, tp1_frac=0.5, tight_sl=False, label="dev=3% VA-SL  50/50  no filters"),
    ZZConfig(deviation_pct=3.0, rvol_min=0.0, tp1_frac=0.5, tight_sl=True,  label="dev=3% TightSL 50/50  no filters"),
    ZZConfig(deviation_pct=3.0, rvol_min=1.2, tp1_frac=0.5, tight_sl=True,  label="dev=3% TightSL 50/50  + RVOL≥1.2"),
    ZZConfig(deviation_pct=3.0, rvol_min=1.2, session_filter=True, tp1_frac=0.5, tight_sl=True,  label="dev=3% TightSL 50/50  + RVOL≥1.2 + Session"),
    ZZConfig(deviation_pct=3.0, rvol_min=1.2, session_filter=True, tp1_frac=0.0, tight_sl=True,  label="dev=3% TightSL TP2only + RVOL≥1.2 + Session"),
]


async def main():
    candles_1h = load_cache("ETH", "1h") or []
    candles_5m = load_cache("ETH", "5m") or []
    if not candles_1h or not candles_5m:
        print("Missing ETH candles — run with --fetch first")
        return

    dev_groups: dict[float, list] = {}
    for cfg in CONFIGS:
        dev_groups.setdefault(cfg.deviation_pct, []).append(cfg)

    all_stats = []
    for dev_pct, cfgs in sorted(dev_groups.items()):
        bias_list = _compute_bias_zz(candles_1h, dev_pct)
        for cfg in cfgs:
            stats = _run_simulation_zz(candles_5m, bias_list, cfg, debug=False)
            all_stats.append((cfg, stats))

    _print_zz_table("ETH", candles_5m, all_stats, entry_tf="5m")


if __name__ == "__main__":
    asyncio.run(main())
