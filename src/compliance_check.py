"""Compliance check — verifies live exchange state matches what the strategies
believe they're managing, and surfaces any fill-mismatch warnings they logged.

Catches the class of bug found 2026-07-06: ORB's tracked position size desynced
from the actual exchange fill, leading to an unprotected, unintended short.

Two checks:
  1. Live reconciliation — for each traded asset, does an open position have a
     protective reduce-only order that can actually close it? Are there
     orphaned reduce-only orders with no position behind them? Does the
     protective order's size roughly match the position?
  2. Log scan — recent "requested vs actual fill" / "leverage cap" / "wrong
     side" warnings the strategies already emit when something looks off.

Usage:
  python -m src.compliance_check
"""

import asyncio
import pathlib
import re
import sys

from src.trading.hyperliquid_api import HyperliquidAPI

ASSETS = ["SOL", "ETH", "xyz:SP500"]
_LOG_DIR = pathlib.Path(__file__).parent.parent / "logs"
_STRATEGY_LOGS = ("orb", "smc", "smoby")
_LOOKBACK_LINES = 500

_SIZE_MISMATCH_TOLERANCE = 0.20  # protective order may differ from position by up to 20%

_WARNING_PATTERNS = [re.compile(p) for p in (
    r"vs requested",
    r"exceeds leverage cap",
    r"wrong side of entry",
    r"could not infer outcome",
    r"entry failed",
)]


async def _check_asset(hl: HyperliquidAPI, asset: str) -> tuple[float, list[str]]:
    issues: list[str] = []
    state = await hl.get_user_state()
    pos = next((p for p in state["positions"] if hl._coin_matches(p.get("coin", ""), asset)), None)
    szi = float(pos.get("szi", 0)) if pos else 0.0

    orders = await hl.get_open_orders()
    reduce_orders = [
        o for o in orders
        if hl._coin_matches(o.get("coin", ""), asset) and o.get("reduceOnly")
    ]

    if abs(szi) > 0.001:
        # Short needs a buy to close; long needs a sell.
        needed_side = "B" if szi < 0 else "A"
        closing = [o for o in reduce_orders if o.get("side") == needed_side]
        if not closing:
            have = [o.get("side") for o in reduce_orders]
            issues.append(
                f"OPEN {szi:.4f} with NO protective order able to close it "
                f"(need side={needed_side}, resting sides={have})"
            )
        else:
            for o in closing:
                osz = float(o.get("sz", 0) or 0)
                if osz > 0 and abs(osz - abs(szi)) / abs(szi) > _SIZE_MISMATCH_TOLERANCE:
                    issues.append(
                        f"protective order size {osz:.4f} differs from position "
                        f"{abs(szi):.4f} by more than {_SIZE_MISMATCH_TOLERANCE:.0%}"
                    )
    elif reduce_orders:
        issues.append(f"{len(reduce_orders)} orphaned reduce-only order(s) with no open position")

    return szi, issues


def _scan_log_warnings(name: str) -> list[str]:
    path = _LOG_DIR / f"{name}.log"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-_LOOKBACK_LINES:]
    return [ln.strip() for ln in lines if any(p.search(ln) for p in _WARNING_PATTERNS)]


async def main() -> int:
    hl = HyperliquidAPI()
    hl.register_perp_dexs(["xyz"])

    print("=== Live position / protective order check ===")
    all_ok = True
    for asset in ASSETS:
        szi, issues = await _check_asset(hl, asset)
        if issues:
            all_ok = False
            print(f"[FAIL] {asset:12} position={szi:.4f}")
            for issue in issues:
                print(f"        - {issue}")
        else:
            print(f"[OK]   {asset:12} position={szi:.4f}")

    print()
    print(f"=== Strategy warnings in the last {_LOOKBACK_LINES} log lines ===")
    any_warnings = False
    for name in _STRATEGY_LOGS:
        for line in _scan_log_warnings(name):
            any_warnings = True
            print(f"[{name}] {line}")
    if not any_warnings:
        print("none")
    else:
        all_ok = False

    print()
    print("RESULT:", "ALL CLEAR" if all_ok else "ISSUES FOUND — see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
