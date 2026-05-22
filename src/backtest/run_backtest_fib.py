"""ZigZag Market Structure + Fibonacci 0.745 retracement entry backtest.

Strategy:
  1h bias: zz_structure.trend = HH_HL (long) or LH_LL (short),
           swing_count >= 2.

  5m entry — no confirmation bar, no volume profile:
    Long:  bar.low  <= last_swing_high - (fib_center - fib_half_zone) * swing_range
           AND bar.close > last_swing_low   (not broken through swing low)
    Short: bar.high >= last_swing_low  + (fib_center - fib_half_zone) * swing_range
           AND bar.close < last_swing_high  (not broken through swing high)

  Entry at close of the touching bar.
  SL:  last_swing_low  - 0.4 × swing_range (long)
       last_swing_high + 0.4 × swing_range (short)
  TP1: last_swing_high (long) / last_swing_low (short)  — tp1_frac of position
  TP2: 127.2% fib extension — remainder

Usage:
  python -m src.backtest.run_backtest_fib --assets BTC ETH SOL
  python -m src.backtest.run_backtest_fib --assets SOL
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
    zz_structure   as zz_struct_fn,
    rvol           as rvol_fn,
)
from src.backtest.fetch_history import load_cache, fetch_all, save_cache

MIN_TRADES    = 20
STRUCT_WINDOW = 200

_VIENNA_TZ = ZoneInfo("Europe/Vienna")
_LONDON_START, _LONDON_END = 8 + 30 / 60, 11.5
_NY_START,     _NY_END     = 16.0, 20.0


def _in_session(ts_ms: int) -> bool:
    hf = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    hf = hf.hour + hf.minute / 60
    return (_LONDON_START <= hf < _LONDON_END) or (_NY_START <= hf < _NY_END)


@dataclass
class FibConfig:
    deviation_pct:   float = 2.0
    fib_center:      float = 0.745  # retracement level from the swing extreme
    fib_zone:        float = 0.08   # total zone width (±0.04 around center)
    rvol_min:        float = 0.0    # 5m RVOL filter (intraday)
    rvol_1h_min:     float = 0.0    # 1h RVOL filter (more appropriate for 4h-scale structure)
    session_filter:  bool  = False
    min_rr:          float = 1.5
    tp1_frac:        float = 0.5
    swing_count_min: int   = 2      # min confirmed ZZ pairs (≥3 = stronger 4h-style structure)
    label:           str   = "Baseline"


ALL_FIB_CONFIGS = [
    # dev=1.0% — tight 1h scope (intraday pivots, hours lookback)
    FibConfig(deviation_pct=1.0, fib_zone=0.08, rvol_min=0.0, tp1_frac=0.5, label="dev=1% zone=0.08 no filters"),
    FibConfig(deviation_pct=1.0, fib_zone=0.08, rvol_min=1.2, tp1_frac=0.5, label="dev=1% zone=0.08 + RVOL≥1.2"),
    FibConfig(deviation_pct=1.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=1% zone=0.08 + RVOL≥1.2 + Session"),
    FibConfig(deviation_pct=1.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=1% zone=0.08 + RVOL≥1.2 + Session TP2only"),
    # dev=1.5% — standard 1h scope per research
    FibConfig(deviation_pct=1.5, fib_zone=0.08, rvol_min=0.0, tp1_frac=0.5, label="dev=1.5% zone=0.08 no filters"),
    FibConfig(deviation_pct=1.5, fib_zone=0.08, rvol_min=1.2, tp1_frac=0.5, label="dev=1.5% zone=0.08 + RVOL≥1.2"),
    FibConfig(deviation_pct=1.5, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=1.5% zone=0.08 + RVOL≥1.2 + Session"),
    FibConfig(deviation_pct=1.5, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=1.5% zone=0.08 + RVOL≥1.2 + Session TP2only"),
    # dev=2.0% — current ETH live setting / upper 1h range
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=0.0, tp1_frac=0.5, label="dev=2% zone=0.08 no filters"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.2, tp1_frac=0.5, label="dev=2% zone=0.08 + RVOL≥1.2"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=2% zone=0.08 + RVOL≥1.2 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=2% zone=0.08 + RVOL≥1.2 + Session TP2only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.12, rvol_min=0.0, tp1_frac=0.5, label="dev=2% zone=0.12 no filters"),
    FibConfig(deviation_pct=2.0, fib_zone=0.12, rvol_min=1.2, tp1_frac=0.5, label="dev=2% zone=0.12 + RVOL≥1.2"),
    FibConfig(deviation_pct=2.0, fib_zone=0.12, rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=2% zone=0.12 + RVOL≥1.2 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.12, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=2% zone=0.12 + RVOL≥1.2 + Session TP2only"),
    # dev=3.0% — current SOL live setting / 4h-scale on 1h bars
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_min=0.0, tp1_frac=0.5, label="dev=3% no filters"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_min=1.2, tp1_frac=0.5, label="dev=3% + RVOL5m≥1.2"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=3% + RVOL5m≥1.2 + Session [LIVE SOL]"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=3% + RVOL5m≥1.2 + Session TP2only"),
    # -- 4h-appropriate filter alternatives for dev=3% --
    FibConfig(deviation_pct=3.0, fib_zone=0.08, session_filter=True, tp1_frac=0.5, label="dev=3% + Session only (no RVOL)"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, session_filter=True, tp1_frac=0.0, label="dev=3% + Session only TP2only"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_1h_min=1.2, tp1_frac=0.5, label="dev=3% + RVOL1h≥1.2"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=3% + RVOL1h≥1.2 + Session"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=3% + RVOL1h≥1.2 + Session TP2only"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, swing_count_min=3, tp1_frac=0.5, label="dev=3% + swingcount≥3"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, swing_count_min=3, session_filter=True, tp1_frac=0.5, label="dev=3% + swingcount≥3 + Session"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, swing_count_min=3, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=3% + swingcount≥3 + RVOL1h≥1.2 + Session"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, swing_count_min=3, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=3% + swingcount≥3 + RVOL1h≥1.2 + Session TP2only"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, min_rr=2.0, tp1_frac=0.5, label="dev=3% + minRR=2.0 no filters"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, min_rr=2.0, session_filter=True, tp1_frac=0.5, label="dev=3% + minRR=2.0 + Session"),
    FibConfig(deviation_pct=3.0, fib_zone=0.08, min_rr=2.0, session_filter=True, tp1_frac=0.0, label="dev=3% + minRR=2.0 + Session TP2only"),
    # -- 4h-appropriate filter alternatives for dev=2% (ETH live) --
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=2% + RVOL5m≥1.2 + Session TP2only [LIVE ETH]"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, session_filter=True, tp1_frac=0.5, label="dev=2% + Session only (no RVOL)"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, session_filter=True, tp1_frac=0.0, label="dev=2% + Session only TP2only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_1h_min=1.2, tp1_frac=0.5, label="dev=2% + RVOL1h≥1.2"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=2% + RVOL1h≥1.2 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=2% + RVOL1h≥1.2 + Session TP2only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, swing_count_min=3, tp1_frac=0.5, label="dev=2% + swingcount≥3"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, swing_count_min=3, session_filter=True, tp1_frac=0.5, label="dev=2% + swingcount≥3 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, swing_count_min=3, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.5, label="dev=2% + swingcount≥3 + RVOL1h≥1.2 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, swing_count_min=3, rvol_1h_min=1.2, session_filter=True, tp1_frac=0.0, label="dev=2% + swingcount≥3 + RVOL1h≥1.2 + Session TP2only"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, min_rr=2.0, tp1_frac=0.5, label="dev=2% + minRR=2.0 no filters"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, min_rr=2.0, session_filter=True, tp1_frac=0.5, label="dev=2% + minRR=2.0 + Session"),
    FibConfig(deviation_pct=2.0, fib_zone=0.08, min_rr=2.0, session_filter=True, tp1_frac=0.0, label="dev=2% + minRR=2.0 + Session TP2only"),
]


# ---------------------------------------------------------------------------
# 1h bias pre-computation  (no VP needed)
# ---------------------------------------------------------------------------

def _compute_bias_fib(candles_1h: list, deviation_pct: float) -> list[dict]:
    """Rolling ZigZag bias per 1h bar — carries struct levels for fib entry.

    Stores swing_count and rvol_1h in each entry so the simulation can apply
    swing_count_min and rvol_1h_min filters without recomputing the bias.
    """
    if len(candles_1h) < 10:
        return []

    print(f"  Pre-computing 1h ZZ bias (dev={deviation_pct}%) for {len(candles_1h)} bars…", flush=True)
    rvol_1h_series = rvol_fn(candles_1h, 20)

    results = []
    for i, bar in enumerate(candles_1h):
        window = candles_1h[max(0, i - STRUCT_WINDOW + 1):i + 1]
        struct = zz_struct_fn(window, deviation_pct=deviation_pct, current_price=bar["close"])
        r1h = rvol_1h_series[i]

        if struct is None or struct["trend"] == "mixed":
            results.append({"t": bar["t"], "bias": None, "structure": None, "swing_count": 0, "rvol_1h": r1h})
            continue

        sc = struct.get("swing_count", 0)
        if sc < 2:
            results.append({"t": bar["t"], "bias": None, "structure": None, "swing_count": sc, "rvol_1h": r1h})
            continue

        bias = "bull" if struct["trend"] == "HH_HL" else "bear"
        results.append({"t": bar["t"], "bias": bias, "structure": struct, "swing_count": sc, "rvol_1h": r1h})

    return results


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_simulation_fib(candles_5m: list, bias_list: list[dict], cfg: FibConfig,
                         debug: bool = False) -> dict:
    if len(candles_5m) < 21 or not bias_list:
        return {}

    rvol_vals  = rvol_fn(candles_5m, 20)
    fib_half   = cfg.fib_zone / 2
    fib_top    = cfg.fib_center - fib_half   # less retracement — zone upper price
    fib_bottom = cfg.fib_center + fib_half   # more retracement — zone lower price

    trades: list[float] = []

    in_trade   = False
    direction  = None
    sl = tp1 = tp2 = entry_price = 0.0
    win_r1 = win_r2 = 0.0
    tp1_hit   = False
    accrued_r = 0.0

    d_touches = d_filter_fail = d_rr_fail = d_opened = 0

    h4_ptr = 0

    for i in range(3, len(candles_5m)):
        bar   = candles_5m[i]
        ts    = bar["t"]
        close = bar["close"]

        while h4_ptr + 1 < len(bias_list) and bias_list[h4_ptr + 1]["t"] <= ts:
            h4_ptr += 1

        h4     = bias_list[h4_ptr]
        bias   = h4["bias"]
        struct = h4["structure"]

        # ── Manage open trade ──────────────────────────────────────────────
        if in_trade:
            if direction == "long":
                if cfg.tp1_frac == 0.0:
                    if bar["low"] <= sl:
                        trades.append(-1.0); in_trade = False
                    elif bar["high"] >= tp2:
                        trades.append(win_r2); in_trade = False
                elif not tp1_hit:
                    if bar["low"] <= sl:
                        trades.append(-1.0); in_trade = False
                    elif bar["high"] >= tp1:
                        tp1_hit   = True
                        accrued_r = cfg.tp1_frac * win_r1
                        sl        = entry_price
                else:
                    if bar["low"] <= sl:
                        trades.append(accrued_r); in_trade = False
                    elif bar["high"] >= tp2:
                        trades.append(accrued_r + (1 - cfg.tp1_frac) * win_r2); in_trade = False
            else:  # short
                if cfg.tp1_frac == 0.0:
                    if bar["high"] >= sl:
                        trades.append(-1.0); in_trade = False
                    elif bar["low"] <= tp2:
                        trades.append(win_r2); in_trade = False
                elif not tp1_hit:
                    if bar["high"] >= sl:
                        trades.append(-1.0); in_trade = False
                    elif bar["low"] <= tp1:
                        tp1_hit   = True
                        accrued_r = cfg.tp1_frac * win_r1
                        sl        = entry_price
                else:
                    if bar["high"] >= sl:
                        trades.append(accrued_r); in_trade = False
                    elif bar["low"] <= tp2:
                        trades.append(accrued_r + (1 - cfg.tp1_frac) * win_r2); in_trade = False
            continue

        if bias is None or struct is None:
            continue

        if h4.get("swing_count", 0) < cfg.swing_count_min:
            continue

        if cfg.rvol_1h_min > 0:
            r1h = h4.get("rvol_1h")
            if r1h is None or r1h < cfg.rvol_1h_min:
                continue

        if cfg.session_filter and not _in_session(ts):
            continue

        rvol_v     = rvol_vals[i]
        swing_high = struct["last_swing_high"]
        swing_low  = struct["last_swing_low"]
        swing_rng  = struct["swing_range"]

        # ── Detect fib zone touch ──────────────────────────────────────────
        if bias == "bull":
            zone_price_top = swing_high - fib_top    * swing_rng  # higher price (less retrace)
            zone_price_bot = swing_high - fib_bottom * swing_rng  # lower price  (more retrace)

            if not (bar["low"] <= zone_price_top and close > swing_low):
                continue

            d_touches += 1
            if cfg.rvol_min > 0 and (rvol_v is None or rvol_v < cfg.rvol_min):
                d_filter_fail += 1; continue

            sl_price = swing_low - 0.05 * swing_rng   # 1.05 fib extension below swing low
            tp1_p    = swing_high
            tp2_p    = struct["tp_speculative_long"]
            if tp2_p <= tp1_p:
                tp2_p = tp1_p

            risk = close - sl_price
            if risk <= 0:
                d_rr_fail += 1; continue
            wr1 = (tp1_p - close) / risk
            wr2 = (tp2_p - close) / risk
            if wr1 < cfg.min_rr:
                d_rr_fail += 1; continue

            entry_price = close
            sl, tp1, tp2 = sl_price, tp1_p, tp2_p
            win_r1, win_r2 = wr1, wr2
            direction, in_trade, tp1_hit, accrued_r = "long", True, False, 0.0
            d_opened += 1

        elif bias == "bear":
            zone_price_bot = swing_low + fib_top    * swing_rng
            zone_price_top = swing_low + fib_bottom * swing_rng

            if not (bar["high"] >= zone_price_bot and close < swing_high):
                continue

            d_touches += 1
            if cfg.rvol_min > 0 and (rvol_v is None or rvol_v < cfg.rvol_min):
                d_filter_fail += 1; continue

            sl_price = swing_high + 0.05 * swing_rng  # 1.05 fib extension above swing high
            tp1_p    = swing_low
            tp2_p    = struct["tp_speculative_short"]
            if tp2_p >= tp1_p:
                tp2_p = tp1_p

            risk = sl_price - close
            if risk <= 0:
                d_rr_fail += 1; continue
            wr1 = (close - tp1_p) / risk
            wr2 = (close - tp2_p) / risk
            if wr1 < cfg.min_rr:
                d_rr_fail += 1; continue

            entry_price = close
            sl, tp1, tp2 = sl_price, tp1_p, tp2_p
            win_r1, win_r2 = wr1, wr2
            direction, in_trade, tp1_hit, accrued_r = "short", True, False, 0.0
            d_opened += 1

    if debug:
        print(f"    [debug] touches={d_touches}  filter_fail={d_filter_fail}"
              f"  rr_fail={d_rr_fail}  opened={d_opened}  closed={len(trades)}")

    if not trades:
        return {"trades": 0}

    wins      = sum(1 for r in trades if r > 0)
    total_r   = sum(trades)
    win_rs    = [r for r in trades if r > 0]
    avg_win_r = sum(win_rs) / len(win_rs) if win_rs else 0.0

    peak = cum = max_dd = 0.0
    for r in trades:
        cum   += r
        peak   = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        "trades":    len(trades),
        "wins":      wins,
        "win_rate":  wins / len(trades),
        "avg_win_r": avg_win_r,
        "total_r":   total_r,
        "avg_r":     total_r / len(trades),
        "max_dd_r":  -max_dd,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _verdict(stats: dict) -> str:
    n = stats.get("trades", 0)
    if n < MIN_TRADES:
        return f"INCONCLUSIVE (<{MIN_TRADES})"
    wr    = stats.get("win_rate", 0)
    avg_w = stats.get("avg_win_r", 2.0)
    be_wr = 1.0 / (1.0 + avg_w) if avg_w > 0 else 1.0
    return "GO ✓" if wr >= be_wr + 0.05 else "NO-GO ✗"


def _print_fib_table(asset: str, candles: list, all_stats: list, entry_tf: str = "5m"):
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}" if candles else "?"

    print(f"\n{'='*96}")
    print(f"Asset: {asset}   Period: {period}")
    print(f"Strategy: 1h ZigZag MS + Fib 0.745 retracement entry  [entry: {entry_tf}]")
    print(f"{'='*96}")
    print(f"{'Config':<50} {'Trades':>7} {'Win%':>6} {'AvgWinR':>8} {'TotalR':>8} {'AvgR':>7} {'MaxDD':>7}  Verdict")
    print(f"{'-'*50} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*7}  {'-'*14}")

    for cfg, s in all_stats:
        if not s or s.get("trades", 0) == 0:
            print(f"{cfg.label:<50} {'—':>7}")
            continue
        print(
            f"{cfg.label:<50} "
            f"{s['trades']:>7} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{s['avg_win_r']:>8.2f} "
            f"{s['total_r']:>+8.1f} "
            f"{s['avg_r']:>+7.3f} "
            f"{s['max_dd_r']:>7.1f}  "
            f"{_verdict(s)}"
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_fib_asset(asset: str, years: int, fetch: bool, entry_tf: str = "5m"):
    from src.trading.hyperliquid_api import HyperliquidAPI
    hl = None

    intervals = ("1h",) if entry_tf == "1h" else ("5m", "1h")
    for interval in intervals:
        cached = load_cache(asset, interval)
        if cached is None or fetch:
            if hl is None:
                hl = HyperliquidAPI()
                await hl.get_meta_and_ctxs()
            print(f"Fetching {asset} {interval}…", end=" ", flush=True)
            candles, source = await fetch_all(hl, asset, interval, years)
            save_cache(asset, interval, candles)
            print(f"{len(candles)} bars [{source}]")

    candles_1h = load_cache(asset, "1h") or []
    if not candles_1h:
        print(f"{asset}: missing 1h candle data — run with --fetch first")
        return

    dev_groups: dict[float, list[FibConfig]] = {}
    for cfg in ALL_FIB_CONFIGS:
        dev_groups.setdefault(cfg.deviation_pct, []).append(cfg)

    all_stats: list[tuple] = []

    for dev_pct, cfgs in sorted(dev_groups.items()):
        bias_list = _compute_bias_fib(candles_1h, dev_pct)

        if entry_tf == "1h":
            _1h_ms    = 3_600_000
            sim_bias  = [{**b, "t": b["t"] + _1h_ms} for b in bias_list]
            entry_candles = candles_1h
        else:
            candles_5m = load_cache(asset, "5m") or []
            if not candles_5m:
                print(f"{asset}: missing 5m candle data — run with --fetch first")
                return
            sim_bias      = bias_list
            entry_candles = candles_5m

        for i, cfg in enumerate(cfgs):
            stats = _run_simulation_fib(entry_candles, sim_bias, cfg, debug=(i == 0))
            all_stats.append((cfg, stats))

    _print_fib_table(asset, entry_candles, all_stats, entry_tf=entry_tf)


async def main_async(assets: list[str], years: int, fetch: bool, entry_tf: str):
    for asset in assets:
        await run_fib_asset(asset, years, fetch, entry_tf=entry_tf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets",   nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--years",    type=int,  default=2)
    parser.add_argument("--fetch",    action="store_true")
    parser.add_argument("--entry-tf", default="5m", choices=["5m", "1h"])
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.years, args.fetch, entry_tf=args.entry_tf))


if __name__ == "__main__":
    main()
