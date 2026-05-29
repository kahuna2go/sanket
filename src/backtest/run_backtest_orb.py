"""Opening Range Breakout (ORB) backtest for S&P 500 Perp (or any session-based asset).

Strategy:
  Pre-session (15:00 CET):
    - Compute 4H 21-EMA bias (Bullish / Bearish / Neutral based on price vs EMA and slope).
    - Funding filter: skipped in backtest (no historical funding data available).

  Opening Range (15:30–15:45 CET):
    - ORH = highest high of the 3 × 5m candles in this window.
    - ORL = lowest low of the 3 × 5m candles in this window.

  Breakout watch (15:45–17:30 CET):
    - Enter long  if 5m close > ORH AND bias = Bullish.
    - Enter short if 5m close < ORL AND bias = Bearish.
    - First valid breakout only; skip the day if none by 17:30.

  Exit (tp_mode="range"):
    - TP1: 50% at entry ± 0.5 × range (SL moves to breakeven after TP1).
    - TP2: 50% at entry ± 1.0 × range.
    - SL:  ORL − sl_buffer × range (long) / ORH + sl_buffer × range (short).
    - Time stop: 20:00 CET — close any open position at market.

  Exit (tp_mode="fixed_rr"):
    - TP1: 50% at entry ± 2 × sl_dist (2:1 R:R).
    - TP2: 50% at entry ± 3 × sl_dist (3:1 R:R).
    - SL and time stop same as above.

Go/no-go criteria: win_rate >= breakeven + 5%  AND  trades >= 20.

Usage:
  python -m src.backtest.run_backtest_orb --asset SPX --fetch
  python -m src.backtest.run_backtest_orb --asset SPX --sl-buffer 0.05
  python -m src.backtest.run_backtest_orb --asset SPX --tp-mode fixed_rr
"""

import argparse
import asyncio
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.fetch_history import load_cache, fetch_all, save_cache

_VIENNA = ZoneInfo("Europe/Vienna")

# CET session boundaries (decimal hours, Vienna local time)
_OR_START   = 15.5          # 15:30
_OR_END     = 15.75         # 15:45
_WATCH_END  = 17.5          # 17:30
_TIME_STOP  = 20.0          # 20:00
_PRESESSION = 15.0          # 15:00 — bias evaluation point

# Weekends: S&P futures don't trade meaningfully Sat/Sun; skip.
_SKIP_WEEKDAYS = {5, 6}     # Saturday=5, Sunday=6


def _vhour(ts_ms: int) -> float:
    """Return Vienna local time as decimal hours."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA)
    return dt.hour + dt.minute / 60 + dt.second / 3600


def _vdate(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA).date()


def _weekday(ts_ms: int) -> int:
    return _vdate(ts_ms).weekday()


# ---------------------------------------------------------------------------
# 4H EMA bias
# ---------------------------------------------------------------------------

def _ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _compute_4h_bias(candles_4h: list, day: date, ema_period: int, slope_threshold: float) -> str:
    """Return 'bull', 'bear', or 'neutral' using all 4H candles up to 15:00 CET on `day`."""
    # Cutoff: 15:00 CET on the given day
    cutoff_ts = datetime(day.year, day.month, day.day, 15, 0, 0,
                         tzinfo=_VIENNA).timestamp() * 1000

    eligible = [c for c in candles_4h if c["t"] < cutoff_ts]
    if len(eligible) < ema_period + 2:
        return "neutral"

    closes = [c["close"] for c in eligible]
    ema_vals = _ema(closes, ema_period)
    if len(ema_vals) < 2:
        return "neutral"

    last_price = closes[-1]
    last_ema   = ema_vals[-1]
    prev_ema   = ema_vals[-2]
    slope      = (last_ema - prev_ema) / prev_ema if prev_ema else 0.0

    if last_price > last_ema and slope > slope_threshold:
        return "bull"
    if last_price < last_ema and slope < -slope_threshold:
        return "bear"
    return "neutral"


# ---------------------------------------------------------------------------
# Per-day simulation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    day:        date
    direction:  str          # 'long' or 'short'
    entry_px:   float
    tp1_px:     float
    tp2_px:     float
    sl_px:      float
    or_range:   float
    exit_px:    float  = 0.0
    exit_reason: str   = ""
    r_multiple: float  = 0.0


def _simulate_day(
    day: date,
    day_5m: list,        # 5m candles for this day, chronological
    bias: str,
    sl_buffer: float,
    min_range_pts: float,
    no_bias: bool = False,
    tp_mode: str = "range",
    entry_mode: str = "breakout",
    sl_mode: str = "or_extreme",
) -> Trade | None:
    """Return a Trade if an entry was taken, else None."""
    if not no_bias and bias == "neutral":
        return None

    # --- Build Opening Range ---
    or_bars = [c for c in day_5m
               if _OR_START <= _vhour(c["t"]) < _OR_END]
    if len(or_bars) < 2:
        return None  # incomplete OR window

    orh = max(c["high"]  for c in or_bars)
    orl = min(c["low"]   for c in or_bars)
    or_range = orh - orl

    if or_range < min_range_pts:
        return None  # range too tight — skip

    # --- Breakout watch: find initial breakout ---
    watch_bars = [c for c in day_5m
                  if _OR_END <= _vhour(c["t"]) < _WATCH_END]

    breakout_bar_t = None
    direction      = None
    for bar in watch_bars:
        if no_bias:
            if bar["close"] > orh:
                direction, breakout_bar_t = "long",  bar["t"]; break
            if bar["close"] < orl:
                direction, breakout_bar_t = "short", bar["t"]; break
        else:
            if bias == "bull" and bar["close"] > orh:
                direction, breakout_bar_t = "long",  bar["t"]; break
            if bias == "bear" and bar["close"] < orl:
                direction, breakout_bar_t = "short", bar["t"]; break

    if breakout_bar_t is None:
        return None  # no breakout before 17:30

    # --- Determine entry price and bar ---
    retest_bar_extreme = None  # low (long) or high (short) of the retest candle
    if entry_mode == "retest":
        entry_px     = None
        entry_bar_t  = None
        post_breakout = [b for b in watch_bars if b["t"] > breakout_bar_t]
        for bar in post_breakout:
            if direction == "long" and bar["low"] <= orh:
                entry_px, entry_bar_t = orh, bar["t"]
                retest_bar_extreme = bar["low"]
                break
            if direction == "short" and bar["high"] >= orl:
                entry_px, entry_bar_t = orl, bar["t"]
                retest_bar_extreme = bar["high"]
                break
        if entry_px is None:
            return None  # no retest within watch window
    else:  # "breakout" — enter at close of breakout bar
        entry_bar_t = breakout_bar_t
        bo_bar      = next(b for b in watch_bars if b["t"] == breakout_bar_t)
        entry_px    = bo_bar["close"]

    # --- Levels ---
    buf = sl_buffer * or_range
    if sl_mode == "retest_low" and retest_bar_extreme is not None:
        # Anchor SL to the retest candle's extreme, with a small buffer
        if direction == "long":
            sl_px = retest_bar_extreme - buf
        else:
            sl_px = retest_bar_extreme + buf
    else:  # "or_extreme" — SL outside the opposite OR edge
        if direction == "long":
            sl_px = orl - buf
        else:
            sl_px = orh + buf

    sl_dist = abs(entry_px - sl_px)

    if tp_mode == "fixed_rr":
        if direction == "long":
            tp1_px = entry_px + 2.0 * sl_dist
            tp2_px = entry_px + 3.0 * sl_dist
        else:
            tp1_px = entry_px - 2.0 * sl_dist
            tp2_px = entry_px - 3.0 * sl_dist
    else:  # "range"
        if direction == "long":
            tp1_px = entry_px + 0.5 * or_range
            tp2_px = entry_px + 1.0 * or_range
        else:
            tp1_px = entry_px - 0.5 * or_range
            tp2_px = entry_px - 1.0 * or_range

    # --- Forward simulate bar-by-bar after entry bar ---
    post_entry = [c for c in day_5m if c["t"] > entry_bar_t]
    tp1_hit   = False
    be_sl     = False   # SL moved to breakeven after TP1
    trail_max = None    # highest (long) / lowest (short) price seen since TP1; trail mode only

    for bar in post_entry:
        vh = _vhour(bar["t"])

        # Time stop
        if vh >= _TIME_STOP:
            exit_px = bar["open"]  # approximation: exit at next bar open
            r = (exit_px - entry_px) / sl_dist if direction == "long" \
                else (entry_px - exit_px) / sl_dist
            if tp1_hit:
                r = (r + 1.0) / 2  # average with the locked-in TP1 R
            return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                         or_range, exit_px, "time_stop", r)

        active_sl = entry_px if be_sl else sl_px  # breakeven SL after TP1

        if direction == "long":
            # SL / trail check (low touches) — use trail_max from previous bar
            if tp_mode == "trail" and tp1_hit:
                trail_sl = trail_max - 0.5 * or_range
                if bar["low"] <= trail_sl:
                    exit_px = trail_sl
                    r = (1.0 + (trail_sl - entry_px) / sl_dist) / 2
                    return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                                 or_range, exit_px, "trail", r)
            elif bar["low"] <= active_sl:
                exit_px = active_sl
                r_leg = (exit_px - entry_px) / sl_dist
                r = (r_leg + 1.0) / 2 if tp1_hit else r_leg
                return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                             or_range, exit_px, "sl", r)
            # TP1
            if not tp1_hit and bar["high"] >= tp1_px:
                tp1_hit = True
                be_sl   = True
                if tp_mode == "trail":
                    trail_max = tp1_px  # trail starts at TP1 level
            # TP2 (fixed target, non-trail only)
            if tp_mode != "trail" and tp1_hit and bar["high"] >= tp2_px:
                exit_px = tp2_px
                r = (1.0 + (tp2_px - entry_px) / sl_dist) / 2
                return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                             or_range, exit_px, "tp2", r)
            # Advance trail high at end of bar
            if tp_mode == "trail" and tp1_hit:
                trail_max = max(trail_max, bar["high"])

        else:  # short
            if tp_mode == "trail" and tp1_hit:
                trail_sl = trail_max + 0.5 * or_range
                if bar["high"] >= trail_sl:
                    exit_px = trail_sl
                    r = (1.0 + (entry_px - trail_sl) / sl_dist) / 2
                    return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                                 or_range, exit_px, "trail", r)
            elif bar["high"] >= active_sl:
                exit_px = active_sl
                r_leg = (entry_px - exit_px) / sl_dist
                r = (r_leg + 1.0) / 2 if tp1_hit else r_leg
                return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                             or_range, exit_px, "sl", r)
            if not tp1_hit and bar["low"] <= tp1_px:
                tp1_hit = True
                be_sl   = True
                if tp_mode == "trail":
                    trail_max = tp1_px
            if tp_mode != "trail" and tp1_hit and bar["low"] <= tp2_px:
                exit_px = tp2_px
                r = (1.0 + (entry_px - tp2_px) / sl_dist) / 2
                return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                             or_range, exit_px, "tp2", r)
            if tp_mode == "trail" and tp1_hit:
                trail_max = min(trail_max, bar["low"])

    # End of day data — close whatever is open
    if post_entry:
        exit_px = post_entry[-1]["close"]
        r_leg = (exit_px - entry_px) / sl_dist if direction == "long" \
                else (entry_px - exit_px) / sl_dist
        r = (r_leg + 1.0) / 2 if tp1_hit else r_leg
        return Trade(day, direction, entry_px, tp1_px, tp2_px, sl_px,
                     or_range, exit_px, "eod", r)

    return None


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

@dataclass
class ORBConfig:
    sl_buffer:         float = 0.10   # fraction of OR range added outside opposite edge
    min_range_pts:     float = 0.0    # minimum OR size in points (0 = no filter)
    ema_period:        int   = 21
    slope_threshold:   float = 0.0005 # minimum |slope| fraction for non-neutral bias
    tp_mode:           str   = "range"      # "range" or "fixed_rr"
    entry_mode:        str   = "breakout"   # "breakout" or "retest"
    sl_mode:           str   = "or_extreme" # "or_extreme" or "retest_low"
    label:             str   = "Baseline"


_BASE_PARAMS = [
    dict(sl_buffer=0.10, slope_threshold=0.0002, label="SL=10% / slope≥0.02%"),
    dict(sl_buffer=0.05, slope_threshold=0.0002, label="SL=5%  / slope≥0.02%"),
    dict(sl_buffer=0.10, slope_threshold=0.0010, label="SL=10% / slope≥0.10%"),
    dict(sl_buffer=0.05, slope_threshold=0.0010, label="SL=5%  / slope≥0.10%"),
    dict(sl_buffer=0.10, slope_threshold=0.0000, label="SL=10% / no slope filter"),
    dict(sl_buffer=0.05, slope_threshold=0.0000, label="SL=5%  / no slope filter"),
]

ALL_ORB_CONFIGS = (
    [ORBConfig(**{**p, "label": p["label"] + " [range/breakout/or_extreme]"},    tp_mode="range",    entry_mode="breakout", sl_mode="or_extreme") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [range/retest/or_extreme]"},      tp_mode="range",    entry_mode="retest",   sl_mode="or_extreme") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [range/retest/retest_low]"},      tp_mode="range",    entry_mode="retest",   sl_mode="retest_low") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [trail/breakout/or_extreme]"},    tp_mode="trail",    entry_mode="breakout", sl_mode="or_extreme") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [trail/retest/or_extreme]"},      tp_mode="trail",    entry_mode="retest",   sl_mode="or_extreme") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [trail/retest/retest_low]"},      tp_mode="trail",    entry_mode="retest",   sl_mode="retest_low") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [fixed_rr/breakout/or_extreme]"}, tp_mode="fixed_rr", entry_mode="breakout", sl_mode="or_extreme") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [fixed_rr/retest/or_extreme]"},   tp_mode="fixed_rr", entry_mode="retest",   sl_mode="or_extreme") for p in _BASE_PARAMS] +
    [ORBConfig(**{**p, "label": p["label"] + " [fixed_rr/retest/retest_low]"},   tp_mode="fixed_rr", entry_mode="retest",   sl_mode="retest_low") for p in _BASE_PARAMS]
)


def _run_config(cfg: ORBConfig, candles_5m: list, candles_4h: list, no_bias: bool = False) -> list[Trade]:
    # Group 5m candles by Vienna date
    days: dict[date, list] = {}
    for c in candles_5m:
        d = _vdate(c["t"])
        if _weekday(c["t"]) in _SKIP_WEEKDAYS:
            continue
        days.setdefault(d, []).append(c)

    trades = []
    for day in sorted(days):
        bias = _compute_4h_bias(candles_4h, day, cfg.ema_period, cfg.slope_threshold)
        trade = _simulate_day(day, days[day], bias, cfg.sl_buffer, cfg.min_range_pts, no_bias, cfg.tp_mode, cfg.entry_mode, cfg.sl_mode)
        if trade:
            trades.append(trade)
    return trades


def _print_results(cfg: ORBConfig, trades: list[Trade]):
    if not trades:
        print(f"  {cfg.label:<35} — no trades")
        return

    wins    = [t for t in trades if t.r_multiple > 0]
    win_pct = 100 * len(wins) / len(trades)
    avg_win = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
    total_r = sum(t.r_multiple for t in trades)
    avg_r   = total_r / len(trades)

    # Simplified max drawdown (R-based)
    peak, trough, max_dd = 0.0, 0.0, 0.0
    cumr = 0.0
    for t in trades:
        cumr += t.r_multiple
        if cumr > peak:
            peak = cumr
            trough = cumr
        else:
            trough = min(trough, cumr)
        max_dd = min(max_dd, trough - peak)

    # Breakeven win rate for 1R risk / ~1R avg win
    avg_win_r = avg_win if avg_win > 0 else 1.0
    be_wr = 100 / (1 + avg_win_r)
    go_nogo = "GO ✓" if (win_pct >= be_wr + 5 and len(trades) >= 20) else "NO-GO ✗"

    exits = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
    exits_str = " ".join(f"{k}={v}" for k, v in sorted(exits.items()))

    print(
        f"  {cfg.label:<35} | {len(trades):>4} trades | "
        f"win={win_pct:5.1f}% | avgWinR={avg_win:.2f} | "
        f"totalR={total_r:+.1f} | avgR={avg_r:+.3f} | "
        f"maxDD={max_dd:.1f}R | {go_nogo} | {exits_str}"
    )


def _print_breakout_funnel(candles_5m: list, candles_4h: list) -> None:
    """Print a day-by-day funnel: OR days → raw breakouts → bias-aligned setups."""
    days: dict[date, list] = {}
    for c in candles_5m:
        if _weekday(c["t"]) in _SKIP_WEEKDAYS:
            continue
        days.setdefault(_vdate(c["t"]), []).append(c)

    total = valid_or = 0
    bias_counts = {"bull": 0, "bear": 0, "neutral": 0}
    raw_long = raw_short = 0
    aligned_long = aligned_short = 0
    filtered_long = filtered_short = 0  # had raw breakout but wrong/neutral bias

    for day in sorted(days):
        total += 1
        day_5m = days[day]
        or_bars = [c for c in day_5m if _OR_START <= _vhour(c["t"]) < _OR_END]
        if len(or_bars) < 2:
            continue
        orh = max(c["high"] for c in or_bars)
        orl = min(c["low"]  for c in or_bars)
        if orh <= orl:
            continue
        valid_or += 1

        bias = _compute_4h_bias(candles_4h, day, 21, 0.0002)
        bias_counts[bias] = bias_counts.get(bias, 0) + 1

        watch_bars = [c for c in day_5m if _OR_END <= _vhour(c["t"]) < _WATCH_END]
        broke_long = broke_short = False
        for bar in watch_bars:
            if not broke_long and bar["close"] > orh:
                broke_long = True
            if not broke_short and bar["close"] < orl:
                broke_short = True

        if broke_long:
            raw_long += 1
            if bias == "bull":
                aligned_long += 1
            else:
                filtered_long += 1
        if broke_short:
            raw_short += 1
            if bias == "bear":
                aligned_short += 1
            else:
                filtered_short += 1

    raw_total     = raw_long + raw_short
    aligned_total = aligned_long + aligned_short
    pct = lambda n, d: f"{100*n/d:.1f}%" if d else "n/a"

    print(f"\n  BREAKOUT FUNNEL DIAGNOSTIC")
    print(f"  {'─'*60}")
    print(f"  Trading days total          : {total}")
    print(f"  Days with valid OR (≥2 bars): {valid_or}  ({pct(valid_or, total)} of days)")
    print(f"  Bias — bull={bias_counts['bull']}  bear={bias_counts['bear']}  neutral={bias_counts['neutral']}"
          f"  ({pct(bias_counts['bull'], valid_or)} / {pct(bias_counts['bear'], valid_or)} / {pct(bias_counts['neutral'], valid_or)})")
    print(f"  Raw breakouts (5m close)    : {raw_total}  ({pct(raw_total, valid_or)} of OR days)"
          f"  — long={raw_long}  short={raw_short}")
    print(f"  Bias-aligned (taken)        : {aligned_total}  ({pct(aligned_total, raw_total)} of raw breakouts)"
          f"  — long={aligned_long}  short={aligned_short}")
    print(f"  Filtered out by bias        : {filtered_long + filtered_short}"
          f"  — long filtered={filtered_long} (bias≠bull)  short filtered={filtered_short} (bias≠bear)")


async def main_async(asset: str, fetch: bool, years: int, single_config: ORBConfig | None, no_bias: bool = False, tp_mode_filter: str | None = None, entry_mode_filter: str | None = None, sl_mode_filter: str | None = None):
    hl = None
    for interval in ("5m", "4h"):
        cached = load_cache(asset, interval)
        if cached and not fetch:
            print(f"{asset} {interval}: {len(cached)} bars (cached)")
        else:
            if hl is None:
                from src.trading.hyperliquid_api import HyperliquidAPI
                hl = HyperliquidAPI()
                await hl.get_meta_and_ctxs()
                if ":" in asset:
                    dex = asset.split(":")[0]
                    await hl.get_meta_and_ctxs(dex=dex)
                    hl.register_perp_dexs([dex])
                    print(f"Registered HIP-3 dex: {dex}")
            print(f"Fetching {asset} {interval} ({years}y)…", end=" ", flush=True)
            candles, source = await fetch_all(hl, asset, interval, years)
            save_cache(asset, interval, candles)
            print(f"{len(candles)} bars [{source}]")

    candles_5m = load_cache(asset, "5m") or []
    candles_4h = load_cache(asset, "4h") or []

    if not candles_5m:
        print(f"ERROR: no 5m candles for {asset}. Run with --fetch or check asset name.")
        return

    print(f"\n{'='*110}")
    print(f"ORB Backtest — {asset} | {len(candles_5m)} × 5m bars | {len(candles_4h)} × 4h bars")
    print(f"{'='*110}")

    _print_breakout_funnel(candles_5m, candles_4h)
    print()

    if single_config:
        configs = [single_config]
    else:
        configs = ALL_ORB_CONFIGS
        if tp_mode_filter:
            configs = [c for c in configs if c.tp_mode == tp_mode_filter]
        if entry_mode_filter:
            configs = [c for c in configs if c.entry_mode == entry_mode_filter]
        if sl_mode_filter:
            configs = [c for c in configs if c.sl_mode == sl_mode_filter]
    current_group = None
    for cfg in configs:
        group = (cfg.entry_mode, cfg.sl_mode, cfg.tp_mode)
        if group != current_group:
            current_group = group
            entry_lbl = "BREAKOUT entry" if cfg.entry_mode == "breakout" else "RETEST entry  "
            sl_lbl    = "SL=OR extreme+buf" if cfg.sl_mode == "or_extreme" else "SL=retest low+buf"
            tp_lbl    = ("TP=range (0.5R/1.0R)" if cfg.tp_mode == "range"
                         else "TP=trail (50%@TP1, trail 0.5×range)" if cfg.tp_mode == "trail"
                         else "TP=fixed R:R (2×/3×SL)")
            print(f"\n  {entry_lbl}  |  {sl_lbl}  |  {tp_lbl}")
            print(f"  {'-'*105}")
        trades = _run_config(cfg, candles_5m, candles_4h, no_bias)
        _print_results(cfg, trades)

    print(f"{'='*110}\n")


def main():
    parser = argparse.ArgumentParser(description="ORB backtest for Hyperliquid S&P 500 Perp")
    parser.add_argument("--asset",     default="SPX", help="Asset ticker (e.g. SPX, SP500)")
    parser.add_argument("--fetch",     action="store_true", help="Re-fetch candle data")
    parser.add_argument("--years",     type=int,   default=1)
    parser.add_argument("--sl-buffer", type=float, default=None,
                        help="Run single config with this SL buffer fraction (e.g. 0.10)")
    parser.add_argument("--slope",     type=float, default=None,
                        help="EMA slope threshold for single-config run (e.g. 0.0005)")
    parser.add_argument("--no-bias",   action="store_true",
                        help="Ignore 4H EMA bias filter; enter any ORB breakout regardless of direction")
    parser.add_argument("--tp-mode",    default=None, choices=["range", "fixed_rr", "trail"],
                        help="TP mode: 'range' (default), 'fixed_rr' (2:1/3:1 off SL), or 'trail' (50%@TP1 then trail 0.5×range)")
    parser.add_argument("--entry-mode", default=None, choices=["breakout", "retest"],
                        help="Entry mode: 'breakout' (default) or 'retest' (wait for ORH/ORL touch)")
    parser.add_argument("--sl-mode",    default=None, choices=["or_extreme", "retest_low"],
                        help="SL mode: 'or_extreme' (default, SL outside OR) or 'retest_low' (SL below retest candle)")
    args = parser.parse_args()

    single = None
    if args.sl_buffer is not None:
        tp_mode    = args.tp_mode    or "range"
        entry_mode = args.entry_mode or "breakout"
        sl_mode    = args.sl_mode    or "or_extreme"
        single = ORBConfig(
            sl_buffer=args.sl_buffer,
            slope_threshold=args.slope if args.slope is not None else 0.0005,
            tp_mode=tp_mode,
            entry_mode=entry_mode,
            sl_mode=sl_mode,
            label=f"SL={args.sl_buffer:.0%} / slope={args.slope or 0.0005:.4f} [{tp_mode}/{entry_mode}/{sl_mode}]",
        )
    asyncio.run(main_async(args.asset, args.fetch, args.years, single, no_bias=args.no_bias, tp_mode_filter=args.tp_mode, entry_mode_filter=args.entry_mode, sl_mode_filter=args.sl_mode))


if __name__ == "__main__":
    main()
