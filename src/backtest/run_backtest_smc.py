"""SOL SMC Scalping Backtest.

Strategy (Smart Money Concepts):
  1. 1H bias  (optional filter): bullish (HH/HL, swing_count ≥ 2) or bearish (LH/LL)
  2. 5M sweep: bar wicks beyond a recently confirmed swing low/high AND closes back inside
  3. 5M CHoCH: within 12 bars of the sweep, price closes beyond the pre-sweep swing high/low
  4. 5M FVG:   identify the most recent Fair Value Gap in the displacement candles
  5. Entry:    price retraces into the FVG (long: bar.low ≤ fvg_hi → enter at fvg_hi)
  6. SL:       below the sweep wick (bull) / above the sweep wick (bear)
  7. TP:       entry ± 3 × risk  (fixed 1:3 R:R)

Swing detection: lookback = 5 bars on each side (confirmed with 5-bar delay, no lookahead).
Sweep lookback: only sweep swings confirmed within the last 20 bars.
CHoCH timeout:  12 / 24 / 48 bars (tested across configs).

Config variations:
  Baseline / + Session (London 08-10 + NY 13:30-15:30 UTC) / + 1H Bias / + Both

Usage:
  python -m src.backtest.run_backtest_smc --assets SOL
  python -m src.backtest.run_backtest_smc --assets SOL --fetch --years 2
"""

import argparse
import asyncio
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.backtest.fetch_history import load_cache, fetch_all, save_cache
from src.indicators.local_indicators import swing_structure

MIN_TRADES    = 20
TP_R          = 3.0
SL_R          = 1.0
SWEEP_LOOKBACK = 20   # max bars back for a swing to still be "recent"
BIAS_WINDOW    = 50   # 1H bars used for rolling bias computation

_UTC = timezone.utc

# Session windows in UTC hours (start, end)
_SESSION_WINDOWS_UTC: list[tuple[float, float]] = [
    (8.0,  10.0),   # London open
    (13.5, 15.5),   # NY open  (13:30–15:30)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_session_utc(ts_ms: int) -> bool:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=_UTC)
    hf = dt.hour + dt.minute / 60
    return any(s <= hf < e for s, e in _SESSION_WINDOWS_UTC)


def _resample_1h(candles_5m: list[dict]) -> list[dict]:
    result = []
    group: list[dict] = []
    for c in candles_5m:
        group.append(c)
        if len(group) == 12:
            result.append({
                "t":      group[0]["t"],
                "open":   group[0]["open"],
                "high":   max(g["high"]   for g in group),
                "low":    min(g["low"]    for g in group),
                "close":  group[-1]["close"],
                "volume": sum(g["volume"] for g in group),
            })
            group = []
    return result


def _compute_1h_bias(candles_1h: list[dict]) -> list[str | None]:
    """Compute 1H bias for each bar using a rolling window ending before bar j.

    bias[j] = bias known at the START of 1H bar j (no lookahead).
    Returns "bull", "bear", or None (mixed / insufficient data).
    """
    result: list[str | None] = []
    for j in range(len(candles_1h)):
        if j < BIAS_WINDOW:
            result.append(None)
            continue
        s = swing_structure(candles_1h[j - BIAS_WINDOW:j], lookback=3)
        if s is None:
            result.append(None)
        elif s["trend"] == "HH_HL" and s["swing_count"] >= 2:
            result.append("bull")
        elif s["trend"] == "LH_LL" and s["swing_count"] >= 2:
            result.append("bear")
        else:
            result.append(None)
    return result


def _align_bias_to_5m(bias_1h: list[str | None], n_5m: int) -> list[str | None]:
    """Map 1H bias index j to all 5M bars k in [j*12, (j+1)*12)."""
    out: list[str | None] = []
    for k in range(n_5m):
        j = k // 12
        out.append(bias_1h[j] if j < len(bias_1h) else None)
    return out


# ---------------------------------------------------------------------------
# Swing detection (precomputed, no lookahead via SWING_LOOKBACK delay)
# ---------------------------------------------------------------------------

def _find_swings(candles: list[dict], lookback: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (swing_lows, swing_highs) as (index, price) pairs.

    A swing at index j is confirmed in the simulation at bar j + lookback.
    """
    n = len(candles)
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    sl: list[tuple[int, float]] = []
    sh: list[tuple[int, float]] = []
    for j in range(lookback, n - lookback):
        neighbors_l = lows[j - lookback:j] + lows[j + 1:j + lookback + 1]
        neighbors_h = highs[j - lookback:j] + highs[j + 1:j + lookback + 1]
        if lows[j]  < min(neighbors_l):
            sl.append((j, lows[j]))
        if highs[j] > max(neighbors_h):
            sh.append((j, highs[j]))
    return sl, sh


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------

def _find_bullish_fvg(bars: list[dict]) -> tuple[float, float] | None:
    """Scan backward for the most recent bullish FVG.

    Bullish FVG: bars[i].high < bars[i+2].low  (gap between them).
    Returns (fvg_hi, fvg_lo) = (bars[i+2].low, bars[i].high), or None.
    """
    for i in range(len(bars) - 3, -1, -1):
        if bars[i]["high"] < bars[i + 2]["low"]:
            return bars[i + 2]["low"], bars[i]["high"]
    return None


def _find_bearish_fvg(bars: list[dict]) -> tuple[float, float] | None:
    """Scan backward for the most recent bearish FVG.

    Bearish FVG: bars[i].low > bars[i+2].high  (gap between them).
    Returns (fvg_hi, fvg_lo) = (bars[i].low, bars[i+2].high), or None.
    """
    for i in range(len(bars) - 3, -1, -1):
        if bars[i]["low"] > bars[i + 2]["high"]:
            return bars[i]["low"], bars[i + 2]["high"]
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SmcConfig:
    session_filter: bool = False
    bias_filter:    bool = False
    choch_timeout:  int  = 48
    swing_lookback: int  = 5
    label:          str  = "Baseline"


def _make_configs() -> list["SmcConfig"]:
    # Fix timeout at 48b (best from prior run); vary swing lookback 3/5/7
    configs = []
    for lb in [3, 5, 7]:
        suffix = f"SL{lb}"
        configs += [
            SmcConfig(session_filter=False, bias_filter=False, swing_lookback=lb,
                      label=f"Baseline / {suffix}"),
            SmcConfig(session_filter=True,  bias_filter=False, swing_lookback=lb,
                      label=f"+ Session / {suffix}"),
            SmcConfig(session_filter=False, bias_filter=True,  swing_lookback=lb,
                      label=f"+ 1H Bias / {suffix}"),
            SmcConfig(session_filter=True,  bias_filter=True,  swing_lookback=lb,
                      label=f"+ Session + Bias / {suffix}"),
        ]
    return configs


ALL_SMC_CONFIGS = _make_configs()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_simulation(
    candles_5m: list[dict],
    bias_5m: list[str | None],
    cfg: SmcConfig,
    debug: bool = False,
) -> dict:
    n = len(candles_5m)
    if n < 100:
        return {}

    lb = cfg.swing_lookback
    swing_lows, swing_highs = _find_swings(candles_5m, lb)

    trades: list[float] = []

    # State machine
    state = "IDLE"
    sweep_type      = ""
    sweep_price     = 0.0   # bull: sweep low; bear: sweep high
    sweep_bar_idx   = 0
    choch_deadline  = 0
    choch_target    = 0.0   # CHoCH fires when price closes beyond this level
    fvg_hi = fvg_lo = 0.0   # bull: entry at fvg_hi; bear: entry at fvg_lo
    trade_type      = ""
    trade_entry     = trade_sl = trade_tp = 0.0

    # Pointers into pre-sorted swing lists (advanced as bars confirm)
    sl_ptr = sh_ptr = 0
    visible_swing_lows:  list[tuple[int, float]] = []
    visible_swing_highs: list[tuple[int, float]] = []

    d_sweeps = d_timeout = d_no_fvg = d_inval = d_opened = 0

    for i in range(n):
        bar = candles_5m[i]

        # Advance visible swings: swing at j confirmed at bar j + lb
        while sl_ptr < len(swing_lows)  and swing_lows[sl_ptr][0]  + lb <= i:
            visible_swing_lows.append(swing_lows[sl_ptr])
            sl_ptr += 1
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] + lb <= i:
            visible_swing_highs.append(swing_highs[sh_ptr])
            sh_ptr += 1

        # ── IN_TRADE ──────────────────────────────────────────────────────────
        if state == "IN_TRADE":
            if trade_type == "long":
                if bar["low"] <= trade_sl:
                    trades.append(-SL_R); state = "IDLE"
                elif bar["high"] >= trade_tp:
                    trades.append(TP_R);  state = "IDLE"
            else:
                if bar["high"] >= trade_sl:
                    trades.append(-SL_R); state = "IDLE"
                elif bar["low"] <= trade_tp:
                    trades.append(TP_R);  state = "IDLE"
            continue

        # ── FVG_WAIT ──────────────────────────────────────────────────────────
        if state == "FVG_WAIT":
            if sweep_type == "bull":
                if bar["low"] < fvg_lo:
                    # Price blew through the FVG bottom — setup invalidated
                    d_inval += 1; state = "IDLE"
                elif bar["low"] <= fvg_hi:
                    # Price enters FVG from above → long entry at fvg_hi
                    entry = fvg_hi
                    risk  = entry - sweep_price
                    if risk > 0:
                        trade_entry = entry
                        trade_sl    = sweep_price
                        trade_tp    = entry + TP_R * risk
                        trade_type  = "long"
                        state       = "IN_TRADE"
                        d_opened   += 1
                    else:
                        state = "IDLE"
            else:  # bear
                if bar["high"] > fvg_hi:
                    d_inval += 1; state = "IDLE"
                elif bar["high"] >= fvg_lo:
                    # Price enters FVG from below → short entry at fvg_lo
                    entry = fvg_lo
                    risk  = sweep_price - entry
                    if risk > 0:
                        trade_entry = entry
                        trade_sl    = sweep_price
                        trade_tp    = entry - TP_R * risk
                        trade_type  = "short"
                        state       = "IN_TRADE"
                        d_opened   += 1
                    else:
                        state = "IDLE"
            continue

        # ── SWEPT ─────────────────────────────────────────────────────────────
        if state == "SWEPT":
            if i > choch_deadline:
                d_timeout += 1; state = "IDLE"
            elif sweep_type == "bull" and bar["close"] > choch_target:
                disp = candles_5m[sweep_bar_idx:i + 1]
                fvg = _find_bullish_fvg(disp)
                if fvg:
                    fvg_hi, fvg_lo = fvg
                    state = "FVG_WAIT"
                else:
                    d_no_fvg += 1; state = "IDLE"
            elif sweep_type == "bear" and bar["close"] < choch_target:
                disp = candles_5m[sweep_bar_idx:i + 1]
                fvg = _find_bearish_fvg(disp)
                if fvg:
                    fvg_hi, fvg_lo = fvg
                    state = "FVG_WAIT"
                else:
                    d_no_fvg += 1; state = "IDLE"
            continue

        # ── IDLE: detect sweeps ───────────────────────────────────────────────
        recent_lows  = [s for s in visible_swing_lows  if s[0] >= i - SWEEP_LOOKBACK]
        recent_highs = [s for s in visible_swing_highs if s[0] >= i - SWEEP_LOOKBACK]

        bias = bias_5m[i]

        # Bullish sweep
        if recent_lows and (not cfg.bias_filter or bias == "bull"):
            _, sl_price = recent_lows[-1]
            if bar["low"] < sl_price and bar["close"] > sl_price:
                if not cfg.session_filter or _in_session_utc(bar["t"]):
                    highs_before = [s for s in visible_swing_highs if s[0] < i]
                    if highs_before:
                        state          = "SWEPT"
                        sweep_type     = "bull"
                        sweep_price    = bar["low"]
                        sweep_bar_idx  = i
                        choch_deadline = i + cfg.choch_timeout
                        choch_target   = highs_before[-1][1]
                        d_sweeps      += 1
                        continue

        # Bearish sweep
        if recent_highs and (not cfg.bias_filter or bias == "bear"):
            _, sh_price = recent_highs[-1]
            if bar["high"] > sh_price and bar["close"] < sh_price:
                if not cfg.session_filter or _in_session_utc(bar["t"]):
                    lows_before = [s for s in visible_swing_lows if s[0] < i]
                    if lows_before:
                        state          = "SWEPT"
                        sweep_type     = "bear"
                        sweep_price    = bar["high"]
                        sweep_bar_idx  = i
                        choch_deadline = i + cfg.choch_timeout
                        choch_target   = lows_before[-1][1]
                        d_sweeps      += 1

    # Close any still-open trade at end of dataset
    if state == "IN_TRADE":
        last_close = candles_5m[-1]["close"]
        if trade_type == "long":
            risk = trade_entry - trade_sl
            r = (last_close - trade_entry) / risk if risk > 0 else 0.0
        else:
            risk = trade_sl - trade_entry
            r = (trade_entry - last_close) / risk if risk > 0 else 0.0
        trades.append(r)

    if debug:
        print(f"    [debug] sweeps={d_sweeps}  choch_timeout={d_timeout}"
              f"  no_fvg={d_no_fvg}  fvg_inval={d_inval}  opened={d_opened}  closed={len(trades)}")

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
    avg_w = stats.get("avg_win_r", TP_R)
    be_wr = 1.0 / (1.0 + avg_w) if avg_w > 0 else 1.0
    return "GO ✓" if wr >= be_wr + 0.05 else "NO-GO ✗"


def _dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=_UTC).strftime("%Y-%m-%d")


_STRATEGY_LABEL = "SMC Scalping: Sweep + CHoCH + FVG Fill  [1H bias, 5M entry, TP=3×risk]"
_RESULTS_FILE = (
    pathlib.Path(__file__).parent.parent.parent / "docs" / "results" / "backtest_results_smc.md"
)


def _print_table(asset: str, candles: list, all_stats: list):
    period = f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}" if candles else "?"
    print(f"\n{'='*94}")
    print(f"Asset: {asset}   Period: {period}")
    print(f"Strategy: {_STRATEGY_LABEL}")
    print(f"{'='*94}")
    print(f"{'Config':<38} {'Trades':>7} {'Win%':>6} {'AvgWinR':>8} {'TotalR':>8} {'AvgR':>7} {'MaxDD':>7}  Verdict")
    print(f"{'-'*38} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*7}  {'-'*14}")
    for cfg, s in all_stats:
        if not s or s.get("trades", 0) == 0:
            print(f"{cfg.label:<38} {'—':>7}")
            continue
        print(
            f"{cfg.label:<38} "
            f"{s['trades']:>7} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{s['avg_win_r']:>8.2f} "
            f"{s['total_r']:>+8.1f} "
            f"{s['avg_r']:>+7.3f} "
            f"{s['max_dd_r']:>7.1f}  "
            f"{_verdict(s)}"
        )


def _append_results_md(asset: str, candles: list, all_stats: list):
    period   = f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}" if candles else "?"
    run_date = datetime.now(tz=_UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "\n---\n\n",
        f"**Asset:** {asset}  |  **Period:** {period}  |  **Run:** {run_date}  |  **Entry TF:** 5m\n\n",
        f"**Strategy:** {_STRATEGY_LABEL}\n\n",
        "| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |\n",
        "|--------|--------|------|---------|--------|------|-------|---------|\n",
    ]
    for cfg, s in all_stats:
        if not s or s.get("trades", 0) == 0:
            lines.append(f"| {cfg.label} | — | | | | | | |\n")
            continue
        lines.append(
            f"| {cfg.label} "
            f"| {s['trades']} "
            f"| {s['win_rate']*100:.1f}% "
            f"| {s['avg_win_r']:.2f} "
            f"| {s['total_r']:+.1f} "
            f"| {s['avg_r']:+.3f} "
            f"| {s['max_dd_r']:.1f}R "
            f"| {_verdict(s)} |\n"
        )
    header_needed = not _RESULTS_FILE.exists() or _RESULTS_FILE.stat().st_size == 0
    with open(_RESULTS_FILE, "a") as f:
        if header_needed:
            f.write("# Backtest Results Log — SMC Scalping\n\n")
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_smc_asset(asset: str, years: int, fetch: bool):
    from src.trading.hyperliquid_api import HyperliquidAPI

    cached = load_cache(asset, "5m")
    if cached is None or fetch:
        hl = HyperliquidAPI()
        await hl.get_meta_and_ctxs()
        print(f"Fetching {asset} 5m…", end=" ", flush=True)
        candles_5m, source = await fetch_all(hl, asset, "5m", years)
        save_cache(asset, "5m", candles_5m)
        print(f"{len(candles_5m)} bars [{source}]")
    else:
        candles_5m = cached

    if not candles_5m:
        print(f"{asset}: no 5m data available")
        return

    candles_1h = _resample_1h(candles_5m)
    print(f"  Resampled {len(candles_5m)} × 5m → {len(candles_1h)} × 1h bars", flush=True)

    bias_1h = _compute_1h_bias(candles_1h)
    bias_5m = _align_bias_to_5m(bias_1h, len(candles_5m))

    all_stats = [
        (cfg, _run_simulation(candles_5m, bias_5m, cfg, debug=(i == 0)))
        for i, cfg in enumerate(ALL_SMC_CONFIGS)
    ]
    _print_table(asset, candles_5m, all_stats)
    _append_results_md(asset, candles_5m, all_stats)


async def main_async(assets: list[str], years: int, fetch: bool):
    for asset in assets:
        await run_smc_asset(asset, years, fetch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["SOL"])
    parser.add_argument("--years",  type=int,  default=2)
    parser.add_argument("--fetch",  action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.years, args.fetch))


if __name__ == "__main__":
    main()
