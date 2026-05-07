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
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.indicators.local_indicators import (
    ema, adx as adx_fn, rsi as rsi_fn, obv as obv_fn, atr as atr_fn, sma,
    bbands as bbands_fn,
)
from src.backtest.fetch_history import load_cache, fetch_all, save_cache


MIN_TRADES = 200

_VIENNA_TZ = ZoneInfo("Europe/Vienna")
_LONDON_START = 8 + 30 / 60  # 08:30
_LONDON_END   = 11.5          # 11:30
_NY_START     = 16.0          # 16:00
_NY_END       = 20.0          # 20:00


def _in_session(ts_ms: int) -> bool:
    hf = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    hf = hf.hour + hf.minute / 60
    return (_LONDON_START <= hf < _LONDON_END) or (_NY_START <= hf < _NY_END)


@dataclass
class SimConfig:
    volume_filter: bool = False   # require bar volume > 20-bar vol SMA
    tight_rsi: bool = False       # 55-65 long / 35-45 short instead of 50-70 / 30-50
    rr3: bool = False             # 3:1 R:R: TP = 2.25×ATR, SL = 0.75×ATR (payout +3/−1)
    session_filter: bool = False  # London 08:30-11:30 and NY 16:00-20:00 Vienna only
    label: str = "Baseline (2:1)"


ALL_CONFIGS = [
    SimConfig(rr3=True, label="Baseline (3:1)"),
    SimConfig(rr3=True, session_filter=True, label="+ Session filter"),
    SimConfig(rr3=True, session_filter=True, volume_filter=True, label="+ Session + Volume"),
    SimConfig(rr3=True, session_filter=True, tight_rsi=True, label="+ Session + Tight RSI"),
    SimConfig(rr3=True, session_filter=True, volume_filter=True, tight_rsi=True, label="+ Session + Volume + Tight RSI"),
    SimConfig(label="Reference: old 2:1 R:R"),
]


@dataclass
class GoldSimConfig:
    dxy_filter: bool = False
    label: str = "Baseline (4:1)"


GOLD_CONFIGS = [
    GoldSimConfig(label="Baseline (4:1)"),
    GoldSimConfig(dxy_filter=True, label="+ DXY filter (no long when USD rising)"),
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

        if cfg.session_filter and not _in_session(bar["t"]):
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

def _verdict(stats: dict, win_payout: float = 3.0) -> str:
    n  = stats.get("trades", 0)
    wr = stats.get("win_rate", 0)
    if n < MIN_TRADES:
        return f"INCONCLUSIVE (<{MIN_TRADES} trades)"
    # require 5pp above breakeven to cover fees and slippage
    threshold = 1.0 / (1.0 + win_payout) + 0.05
    return "GO ✓" if wr >= threshold else "NO-GO ✗"


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
        win_payout = 3.0 if cfg.rr3 else 2.0
        print(
            f"{cfg.label:<38} "
            f"{s['trades']:>7} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{s['total_r']:>+8.1f} "
            f"{s['avg_r']:>+7.3f} "
            f"{s['max_dd_r']:>7.1f} "
            f"{_verdict(s, win_payout)}"
        )


# ---------------------------------------------------------------------------
# Gold Range Breakout
# ---------------------------------------------------------------------------

def _compute_gold_4h_signals(candles_4h: list) -> list[dict]:
    """Compute per-4h-bar Gold setup signals: squeeze + ADX rising."""
    if len(candles_4h) < 30:
        return []

    bb = bbands_fn(candles_4h, period=20)
    adx_vals = adx_fn(candles_4h)

    widths: list[float | None] = []
    for u, m, lo in zip(bb["upper"], bb["middle"], bb["lower"]):
        if u is None or m is None or lo is None or m == 0:
            widths.append(None)
        else:
            widths.append((u - lo) / m)

    results = []
    for i, bar in enumerate(candles_4h):
        if widths[i] is None or adx_vals[i] is None:
            results.append({"t": bar["t"], "squeeze": False, "adx_rising": False})
            continue

        valid_window = [widths[j] for j in range(max(0, i - 7), i + 1) if widths[j] is not None]
        squeeze = len(valid_window) == 8 and widths[i] == min(valid_window)

        adx_5_ago = adx_vals[i - 5] if i >= 5 else None
        adx_rising = adx_5_ago is not None and adx_vals[i] > adx_5_ago

        results.append({"t": bar["t"], "squeeze": squeeze, "adx_rising": adx_rising})

    return results


def _get_gold_signal_at(signals: list[dict], ts: int) -> tuple[bool, bool]:
    result = (False, False)
    for s in signals:
        if s["t"] <= ts:
            result = (s["squeeze"], s["adx_rising"])
        else:
            break
    return result


def _fetch_dxy_rising_by_date(start_ms: int, end_ms: int) -> dict:
    """Return {date_str: bool (rising)} per calendar day. Empty dict on failure."""
    try:
        import yfinance as yf
        from datetime import datetime, timezone, timedelta
        start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        end = (datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        hist = yf.Ticker("DX-Y.NYB").history(start=start, end=end)
        if hist.empty:
            return {}
        closes = hist["Close"].tolist()
        dates = [str(ts.date()) for ts in hist.index]
        k = 2.0 / 6
        ema_val = closes[0]
        result = {}
        for date, c in zip(dates, closes):
            ema_val = c * k + ema_val * (1 - k)
            result[date] = c > ema_val
        return result
    except Exception as e:
        print(f"  DXY history fetch failed: {e}")
        return {}


def _run_gold_simulation(candles_5m: list, gold_4h_signals: list[dict],
                         cfg: GoldSimConfig, dxy_data: dict) -> dict:
    if len(candles_5m) < 25:
        return {}

    atr_vals = atr_fn(candles_5m, 14)

    TP_MULT = 2.0
    SL_MULT = 0.5
    WIN_PAYOUT = TP_MULT / SL_MULT  # 4.0

    trades: list[float] = []
    in_trade = False
    tp = sl = entry = 0.0
    direction = None

    from datetime import datetime, timezone

    for i in range(20, len(candles_5m)):
        bar = candles_5m[i]

        if in_trade:
            if direction == "long":
                if bar["low"] <= sl:
                    trades.append(-1.0)
                    in_trade = False
                elif bar["high"] >= tp:
                    trades.append(WIN_PAYOUT)
                    in_trade = False
            else:
                if bar["high"] >= sl:
                    trades.append(-1.0)
                    in_trade = False
                elif bar["low"] <= tp:
                    trades.append(WIN_PAYOUT)
                    in_trade = False
            continue

        squeeze, adx_rising = _get_gold_signal_at(gold_4h_signals, bar["t"])
        if not (squeeze and adx_rising):
            continue

        atr_v = atr_vals[i]
        if atr_v is None or atr_v == 0:
            continue

        window = candles_5m[i - 20:i]
        range_high = max(c["high"] for c in window)
        range_low = min(c["low"] for c in window)
        close = bar["close"]

        bar_date = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        dxy_rising = dxy_data.get(bar_date, False) if cfg.dxy_filter else False

        if not dxy_rising and close > range_high + 0.3 * atr_v:
            entry, direction, in_trade = close, "long", True
            tp = entry + TP_MULT * atr_v
            sl = entry - SL_MULT * atr_v
        elif close < range_low - 0.3 * atr_v:
            entry, direction, in_trade = close, "short", True
            tp = entry - TP_MULT * atr_v
            sl = entry + SL_MULT * atr_v

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


def _print_gold_table(asset: str, candles_5m: list, all_stats: list):
    from datetime import datetime, timezone

    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = (
        f"{_dt(candles_5m[0]['t'])} → {_dt(candles_5m[-1]['t'])}"
        if candles_5m else "?"
    )

    print(f"\n{'='*80}")
    print(f"Asset: {asset} (Range Breakout)   Period: {period}")
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


async def run_gold_asset(asset: str, years: int, fetch: bool):
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

    if not candles_4h:
        print(f"{asset}: missing 4h candle data — run fetch_history.py first")
        return

    gold_4h_signals = _compute_gold_4h_signals(candles_4h)

    ref = candles_5m or candles_4h
    print(f"Fetching DXY history for DXY-filter comparison…", end=" ", flush=True)
    dxy_data = _fetch_dxy_rising_by_date(ref[0]["t"], ref[-1]["t"])
    print(f"{len(dxy_data)} trading days")

    all_stats = [
        (cfg, _run_gold_simulation(candles_5m, gold_4h_signals, cfg, dxy_data))
        for cfg in GOLD_CONFIGS
    ]
    _print_gold_table(asset, candles_5m, all_stats)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_asset(asset: str, years: int, fetch: bool, configs: list[SimConfig]):
    from src.trading.hyperliquid_api import HyperliquidAPI
    hl = None

    for interval in ("5m", "1h"):
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
    candles_1h = load_cache(asset, "1h") or []

    if not candles_5m or not candles_1h:
        print(f"{asset}: missing candle data — run fetch_history.py first")
        return

    bias_list = _compute_4h_bias(candles_1h)
    all_stats = [(cfg, _run_simulation(candles_5m, bias_list, cfg)) for cfg in configs]
    _print_table(asset, candles_5m, all_stats)


async def main_async(assets: list[str], years: int, fetch: bool, configs: list[SimConfig]):
    for asset in assets:
        if asset == "xyz:GOLD":
            await run_gold_asset(asset, years, fetch)
        else:
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
