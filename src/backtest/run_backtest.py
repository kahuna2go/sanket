"""Momentum Breakout backtest — matches the live system prompt entry rules exactly.

Strategy:
  4h bias bullish:  EMA20 > EMA50, MACD histogram > 0, ADX > 25
  4h bias bearish:  EMA20 < EMA50, MACD histogram < 0, ADX > 25
  ADX < 25 on 4h → no signal regardless of other conditions

  5m long entry:  close > prev bar high, OBV rising, RSI 50–70
  5m short entry: close < prev bar low,  OBV falling, RSI 30–50
  Entry price: close of signal bar
  TP: entry + 1.5 × ATR14 (long) / entry − 1.5 × ATR14 (short)
  SL: entry − 0.75 × ATR14 (long) / entry + 0.75 × ATR14 (short)
  One position at a time. SL hit → −1R, TP hit → +2R.

Go/no-go: win_rate > 0.38 AND total_trades >= 200.

Usage:
  python -m src.backtest.run_backtest --assets BTC ETH SOL
  python -m src.backtest.run_backtest --assets BTC --fetch   # re-fetch before running
"""

import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.indicators.local_indicators import (
    ema, adx as adx_fn, rsi as rsi_fn, obv as obv_fn, atr as atr_fn,
    compute_all, latest,
)
from src.backtest.fetch_history import load_cache, fetch_all, save_cache, cache_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_4h_bias(candles_4h: list) -> list[dict]:
    """Return per-4h-candle bias: 'bull', 'bear', or None (ADX < 25 or mixed)."""
    if len(candles_4h) < 51:
        return []

    closes = [c["close"] for c in candles_4h]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    adx_vals = adx_fn(candles_4h)

    # MACD histogram
    from src.indicators.local_indicators import macd as macd_fn
    macd_data = macd_fn(candles_4h)
    hist = macd_data["histogram"]

    results = []
    for i in range(len(candles_4h)):
        e20 = ema20[i]
        e50 = ema50[i]
        adx_v = adx_vals[i]
        h = hist[i]
        t = candles_4h[i]["t"]

        if any(v is None for v in (e20, e50, adx_v, h)):
            results.append({"t": t, "bias": None})
            continue
        if adx_v < 25:
            results.append({"t": t, "bias": None})
            continue
        if e20 > e50 and h > 0:
            results.append({"t": t, "bias": "bull"})
        elif e20 < e50 and h < 0:
            results.append({"t": t, "bias": "bear"})
        else:
            results.append({"t": t, "bias": None})

    return results


def _get_4h_bias_at(bias_list: list[dict], ts_5m: int) -> str | None:
    """Return the last closed 4h bias at or before ts_5m (no look-ahead)."""
    result = None
    for b in bias_list:
        if b["t"] <= ts_5m:
            result = b["bias"]
        else:
            break
    return result


def _run_simulation(candles_5m: list, bias_list: list[dict]) -> dict:
    """Simulate trades on 5m bars. Returns trade stats dict."""
    if len(candles_5m) < 20:
        return {}

    rsi_vals = rsi_fn(candles_5m, 14)
    obv_vals = obv_fn(candles_5m)
    atr_vals = atr_fn(candles_5m, 14)

    trades = []
    in_trade = False
    tp = sl = entry = 0.0
    direction = None

    for i in range(1, len(candles_5m)):
        bar = candles_5m[i]
        prev = candles_5m[i - 1]
        ts = bar["t"]

        # Resolve open trade first
        if in_trade:
            high = bar["high"]
            low = bar["low"]
            if direction == "long":
                if low <= sl:
                    trades.append(-1.0)
                    in_trade = False
                elif high >= tp:
                    trades.append(2.0)
                    in_trade = False
            else:  # short
                if high >= sl:
                    trades.append(-1.0)
                    in_trade = False
                elif low <= tp:
                    trades.append(2.0)
                    in_trade = False
            continue  # one position at a time

        bias = _get_4h_bias_at(bias_list, ts)
        if bias is None:
            continue

        rsi_v = rsi_vals[i]
        obv_v = obv_vals[i]
        obv_prev = obv_vals[i - 1]
        atr_v = atr_vals[i]

        if any(v is None for v in (rsi_v, atr_v)):
            continue
        if atr_v == 0:
            continue

        close = bar["close"]

        if bias == "bull" and close > prev["high"] and obv_v > obv_prev and 50 <= rsi_v <= 70:
            entry = close
            tp = entry + 1.5 * atr_v
            sl = entry - 0.75 * atr_v
            direction = "long"
            in_trade = True

        elif bias == "bear" and close < prev["low"] and obv_v < obv_prev and 30 <= rsi_v <= 50:
            entry = close
            tp = entry - 1.5 * atr_v
            sl = entry + 0.75 * atr_v
            direction = "short"
            in_trade = True

    if not trades:
        return {"trades": 0}

    wins = sum(1 for r in trades if r > 0)
    total_r = sum(trades)
    max_dd = _max_drawdown(trades)

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "total_r": total_r,
        "avg_r": total_r / len(trades),
        "max_dd_r": max_dd,
        "expectancy": total_r / len(trades),
    }


def _max_drawdown(results: list[float]) -> float:
    """Maximum drawdown in R from a list of trade results."""
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for r in results:
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return -max_dd  # return as negative


def _print_result(asset: str, stats: dict, candles_5m: list):
    if not stats or stats.get("trades", 0) == 0:
        print(f"\nAsset:        {asset}")
        print("              No trades generated — check data or signal params.")
        return

    from datetime import datetime, timezone
    def _dt(ts_ms):
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period_start = _dt(candles_5m[0]["t"]) if candles_5m else "?"
    period_end = _dt(candles_5m[-1]["t"]) if candles_5m else "?"

    n = stats["trades"]
    wr = stats["win_rate"]
    conclusive = n >= 200
    go = wr > 0.38 and conclusive

    print(f"\nAsset:        {asset}")
    print(f"Period:       {period_start} → {period_end}")
    print(f"Total trades: {n}{'' if conclusive else '  ⚠ < 200 — INCONCLUSIVE'}")
    print(f"Win rate:     {wr * 100:.1f}%")
    print(f"Total R:      {stats['total_r']:+.1f}")
    print(f"Avg R/trade:  {stats['avg_r']:+.3f}")
    print(f"Max DD (R):   {stats['max_dd_r']:.1f}")
    print(f"Expectancy:   {stats['expectancy']:+.3f}R")
    if not conclusive:
        print("Verdict:      INCONCLUSIVE (need ≥ 200 trades)")
    elif go:
        print("Verdict:      GO ✓ (win rate > 38%, ≥ 200 trades)")
    else:
        print("Verdict:      NO-GO ✗ (win rate ≤ 38%)")


async def run_asset(asset: str, years: int, fetch: bool):
    from src.trading.hyperliquid_api import HyperliquidAPI
    hl = None

    for interval in ("5m", "4h"):
        cached = load_cache(asset, interval)
        if cached is None or fetch:
            if hl is None:
                hl = HyperliquidAPI()
                await hl.get_meta_and_ctxs()
            print(f"Fetching {asset} {interval}…", end=" ", flush=True)
            candles = await fetch_all(hl, asset, interval, years)
            save_cache(asset, interval, candles)
            print(f"{len(candles)} bars")

    candles_5m = load_cache(asset, "5m") or []
    candles_4h = load_cache(asset, "4h") or []

    if not candles_5m or not candles_4h:
        print(f"{asset}: missing candle data — run fetch_history.py first")
        return

    bias_list = _compute_4h_bias(candles_4h)
    stats = _run_simulation(candles_5m, bias_list)
    _print_result(asset, stats, candles_5m)


async def main_async(assets: list[str], years: int, fetch: bool):
    for asset in assets:
        await run_asset(asset, years, fetch)


def main():
    parser = argparse.ArgumentParser(description="Momentum Breakout backtest")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--fetch", action="store_true", help="Re-fetch candles before running")
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.years, args.fetch))


if __name__ == "__main__":
    main()
