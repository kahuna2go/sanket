"""Momentum Breakout backtest — matches the live system prompt entry rules exactly.

Base strategy:
  4h bias bullish:  EMA20 > EMA50, MACD histogram > 0, ADX > 25
  4h bias bearish:  EMA20 < EMA50, MACD histogram < 0, ADX > 25
  ADX < 25 on 4h → no signal

  5m long entry:  close > prev bar high, OBV rising, RSI 50–70
  5m short entry: close < prev bar low,  OBV falling, RSI 30–50
  TP: entry ± 1.5×ATR14.  SL: entry ∓ 0.75×ATR14.  R:R = 2:1

Optional filters tested:
  --volume-filter   signal bar volume > 20-bar vol SMA (eliminates thin breakouts)
  --tight-rsi       RSI range 55–65 long / 35–45 short (tighter momentum window)
  --rr3             3:1 R:R — TP = 2.25×ATR, SL = 0.75×ATR

By default runs all 5 combinations and prints a comparison table.

Go/no-go: win_rate > 0.38 AND total_trades >= 200.

Usage:
  python -m src.backtest.run_backtest --assets BTC ETH SOL
  python -m src.backtest.run_backtest --assets BTC --fetch
"""

import argparse
import asyncio
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.indicators.local_indicators import (
    ema, adx as adx_fn, rsi as rsi_fn, obv as obv_fn, atr as atr_fn, sma,
)
from src.backtest.fetch_history import load_cache, fetch_all, save_cache


WIN_RATE_THRESHOLD = 0.38
MIN_TRADES = 200


@dataclass
class SimConfig:
    volume_filter: bool = False   # require bar volume > 20-bar vol SMA
    tight_rsi: bool = False       # 55-65 long / 35-45 short instead of 50-70 / 30-50
    rr3: bool = False             # 3:1 R:R: TP = 2.25×ATR, SL = 0.75×ATR (payout +3/−1)
    label: str = "Baseline (2:1)"


ALL_CONFIGS = [
    SimConfig(label="Baseline (2:1)"),
    SimConfig(volume_filter=True, label="+ Volume filter"),
    SimConfig(tight_rsi=True, label="+ Tight RSI"),
    SimConfig(volume_filter=True, tight_rsi=True, label="+ Volume + Tight RSI"),
    SimConfig(volume_filter=True, tight_rsi=True, rr3=True, label="+ Volume + Tight RSI + 3:1 R:R"),
]


# ---------------------------------------------------------------------------
# 4h bias
# ---------------------------------------------------------------------------

def _compute_4h_bias(candles_4h: list) -> list[dict]:
    if len(candles_4h) < 51:
        return []

    closes = [c["close"] for c in candles_4h]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    adx_vals = adx_fn(candles_4h)

    from src.indicators.local_indicators import macd as macd_fn
    hist = macd_fn(candles_4h)["histogram"]

    results = []
    for i, bar in enumerate(candles_4h):
        e20, e50, adx_v, h = ema20[i], ema50[i], adx_vals[i], hist[i]
        if any(v is None for v in (e20, e50, adx_v, h)):
            results.append({"t": bar["t"], "bias": None})
        elif adx_v < 25:
            results.append({"t": bar["t"], "bias": None})
        elif e20 > e50 and h > 0:
            results.append({"t": bar["t"], "bias": "bull"})
        elif e20 < e50 and h < 0:
            results.append({"t": bar["t"], "bias": "bear"})
        else:
            results.append({"t": bar["t"], "bias": None})

    return results


def _get_4h_bias_at(bias_list: list[dict], ts: int) -> str | None:
    result = None
    for b in bias_list:
        if b["t"] <= ts:
            result = b["bias"]
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_simulation(candles_5m: list, bias_list: list[dict], cfg: SimConfig) -> dict:
    if len(candles_5m) < 21:
        return {}

    rsi_vals = rsi_fn(candles_5m, 14)
    obv_vals = obv_fn(candles_5m)
    atr_vals = atr_fn(candles_5m, 14)

    volumes = [c["volume"] for c in candles_5m]
    vol_sma = sma(volumes, 20) if cfg.volume_filter else [None] * len(volumes)

    rsi_long_lo  = 55 if cfg.tight_rsi else 50
    rsi_long_hi  = 65 if cfg.tight_rsi else 70
    rsi_short_lo = 35 if cfg.tight_rsi else 30
    rsi_short_hi = 45 if cfg.tight_rsi else 50

    tp_mult = 2.25 if cfg.rr3 else 1.5
    sl_mult = 0.75
    win_payout = 3.0 if cfg.rr3 else 2.0

    trades: list[float] = []
    in_trade = False
    tp = sl = entry = 0.0
    direction = None

    for i in range(1, len(candles_5m)):
        bar  = candles_5m[i]
        prev = candles_5m[i - 1]

        if in_trade:
            if direction == "long":
                if bar["low"] <= sl:
                    trades.append(-1.0)
                    in_trade = False
                elif bar["high"] >= tp:
                    trades.append(win_payout)
                    in_trade = False
            else:
                if bar["high"] >= sl:
                    trades.append(-1.0)
                    in_trade = False
                elif bar["low"] <= tp:
                    trades.append(win_payout)
                    in_trade = False
            continue

        bias = _get_4h_bias_at(bias_list, bar["t"])
        if bias is None:
            continue

        rsi_v = rsi_vals[i]
        obv_v = obv_vals[i]
        atr_v = atr_vals[i]

        if rsi_v is None or atr_v is None or atr_v == 0:
            continue

        if cfg.volume_filter:
            v_sma = vol_sma[i]
            if v_sma is None or bar["volume"] <= v_sma:
                continue

        close = bar["close"]

        if bias == "bull" and close > prev["high"] and obv_v > obv_vals[i-1] \
                and rsi_long_lo <= rsi_v <= rsi_long_hi:
            entry, direction, in_trade = close, "long", True
            tp = entry + tp_mult * atr_v
            sl = entry - sl_mult * atr_v

        elif bias == "bear" and close < prev["low"] and obv_v < obv_vals[i-1] \
                and rsi_short_lo <= rsi_v <= rsi_short_hi:
            entry, direction, in_trade = close, "short", True
            tp = entry - tp_mult * atr_v
            sl = entry + sl_mult * atr_v

    if not trades:
        return {"trades": 0}

    wins = sum(1 for r in trades if r > 0)
    total_r = sum(trades)
    peak = cum = max_dd = 0.0
    for r in trades:
        cum += r
        if cum > peak:
            peak = cum
        if peak - cum > max_dd:
            max_dd = peak - cum

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "total_r": total_r,
        "avg_r": total_r / len(trades),
        "max_dd_r": -max_dd,
        "expectancy": total_r / len(trades),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _verdict(stats: dict) -> str:
    n  = stats.get("trades", 0)
    wr = stats.get("win_rate", 0)
    if n < MIN_TRADES:
        return f"INCONCLUSIVE (<{MIN_TRADES} trades)"
    return "GO ✓" if wr > WIN_RATE_THRESHOLD else "NO-GO ✗"


def _print_table(asset: str, candles_5m: list, all_stats: list[tuple[SimConfig, dict]]):
    from datetime import datetime, timezone
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = (
        f"{_dt(candles_5m[0]['t'])} → {_dt(candles_5m[-1]['t'])}"
        if candles_5m else "?"
    )

    print(f"\n{'='*80}")
    print(f"Asset: {asset}   Period: {period}")
    print(f"{'='*80}")
    print(f"{'Config':<38} {'Trades':>7} {'Win%':>6} {'TotalR':>8} {'AvgR':>7} {'MaxDD':>7} {'Verdict'}")
    print(f"{'-'*38} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*14}")

    for cfg, s in all_stats:
        if not s or s.get("trades", 0) == 0:
            print(f"{cfg.label:<38} {'—':>7}")
            continue
        print(
            f"{cfg.label:<38} "
            f"{s['trades']:>7} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{s['total_r']:>+8.1f} "
            f"{s['avg_r']:>+7.3f} "
            f"{s['max_dd_r']:>7.1f} "
            f"{_verdict(s)}"
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_asset(asset: str, years: int, fetch: bool, configs: list[SimConfig]):
    from src.trading.hyperliquid_api import HyperliquidAPI
    hl = None

    for interval in ("5m", "4h"):
        cached = load_cache(asset, interval)
        if cached is None or fetch:
            if hl is None:
                hl = HyperliquidAPI()
                await hl.get_meta_and_ctxs()
            print(f"Fetching {asset} {interval}…", end=" ", flush=True)
            candles, source = await fetch_all(hl, asset, interval, years)
            save_cache(asset, interval, candles)
            print(f"{len(candles)} bars [{source}]")

    candles_5m = load_cache(asset, "5m") or []
    candles_4h = load_cache(asset, "4h") or []

    if not candles_5m or not candles_4h:
        print(f"{asset}: missing candle data — run fetch_history.py first")
        return

    bias_list = _compute_4h_bias(candles_4h)
    all_stats = [(cfg, _run_simulation(candles_5m, bias_list, cfg)) for cfg in configs]
    _print_table(asset, candles_5m, all_stats)


async def main_async(assets: list[str], years: int, fetch: bool, configs: list[SimConfig]):
    for asset in assets:
        await run_asset(asset, years, fetch, configs)


def main():
    parser = argparse.ArgumentParser(description="Momentum Breakout backtest")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--volume-filter", action="store_true", help="Run only with volume filter")
    parser.add_argument("--tight-rsi", action="store_true", help="Run only with tight RSI")
    parser.add_argument("--rr3", action="store_true", help="Run only with 3:1 R:R")
    args = parser.parse_args()

    # If any specific flag given, run just that single config; otherwise run all combos
    if args.volume_filter or args.tight_rsi or args.rr3:
        configs = [SimConfig(
            volume_filter=args.volume_filter,
            tight_rsi=args.tight_rsi,
            rr3=args.rr3,
            label="Custom config",
        )]
    else:
        configs = ALL_CONFIGS

    asyncio.run(main_async(args.assets, args.years, args.fetch, configs))


if __name__ == "__main__":
    main()
