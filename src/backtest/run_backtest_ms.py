"""Market Structure + Volume Profile Value Area backtest.

Strategy:
  1h bias: swing_structure.trend = HH_HL (long) or LH_LL (short),
           swing_count >= 2  (at least hh-hl-hh / ll-lh-ll — no ADX/RSI gate).

  5m signal (Value Area Bounce):
    Long:  bar.low  <= VAL + 30% × VA_width  AND  bar.close > VAL
    Short: bar.high >= VAH - 30% × VA_width  AND  bar.close < VAH

  5m confirmation (one bar later):
    Long:  next bar also closes above VAL
    Short: next bar also closes below VAH
    → Bias change between signal and confirmation cancels the setup.

  SL:   VAL - 0.15 × VA_width (long) / VAH + 0.15 × VA_width (short)
  TP1:  VAH (long) / VAL (short)  — 50 % of position exits here
        SL moved to breakeven after TP1 hit
  TP2:  127.2 % Fibonacci extension of last 1h swing — remaining 50 % exits here
        Falls back to TP1 if no valid speculative target exists.
  R:R:  min_rr applies to TP1 leg only (TP2 is always further).

  Optional filter: RVOL >= threshold on signal bar.
  Optional filter: session (London open / NY open, Vienna time).

Go/no-go: win_rate >= breakeven_win_rate + 5%  AND  trades >= 20.

Usage:
  python -m src.backtest.run_backtest_ms --assets BTC ETH SOL
  python -m src.backtest.run_backtest_ms --assets SOL --fetch --years 2
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
    swing_structure as swing_struct_fn,
    volume_profile  as vol_profile_fn,
    rvol            as rvol_fn,
)
from src.backtest.fetch_history import load_cache, fetch_all, save_cache

MIN_TRADES       = 20
SL_BUFFER_PCT    = 0.15   # fraction of VA_width below VAL (long) / above VAH (short)
ZONE_PCT         = 0.30   # lower/upper entry zone = 30 % of VA_width from the edge
STRUCT_WINDOW    = 200    # rolling 1h bars for swing_structure  (~8.3 days)
VP_WINDOW        = 20     # rolling 1h bars for volume profile   (~20 hours)

_VIENNA_TZ = ZoneInfo("Europe/Vienna")
_LONDON_START, _LONDON_END = 8 + 30 / 60, 11.5
_NY_START,     _NY_END     = 16.0, 20.0


def _in_session(ts_ms: int) -> bool:
    hf = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    hf = hf.hour + hf.minute / 60
    return (_LONDON_START <= hf < _LONDON_END) or (_NY_START <= hf < _NY_END)


@dataclass
class MSConfig:
    rvol_min:       float = 0.0   # 0.0 = disabled
    session_filter: bool  = False
    min_rr:         float = 1.5
    tp1_frac:       float = 0.5   # fraction closed at TP1; 0.0 = skip TP1, run full size to TP2
    label:          str   = "Baseline"


ALL_MS_CONFIGS = [
    MSConfig(rvol_min=0.0, session_filter=False, label="No filters"),
    MSConfig(rvol_min=1.2, session_filter=False, label="+ RVOL ≥ 1.2"),
    MSConfig(rvol_min=1.5, session_filter=False, label="+ RVOL ≥ 1.5"),
    MSConfig(rvol_min=0.0, session_filter=True,  label="+ Session"),
    MSConfig(rvol_min=1.2, session_filter=True,  label="+ RVOL ≥ 1.2 + Session"),
    MSConfig(rvol_min=1.5, session_filter=True,  label="+ RVOL ≥ 1.5 + Session"),
]

# Preset: TP split comparison (SOL)
_MS_TP_TEST_CONFIGS = [
    MSConfig(rvol_min=0.0, tp1_frac=0.5, label="50/50  no filters"),
    MSConfig(rvol_min=0.0, tp1_frac=0.7, label="70/30  no filters"),
    MSConfig(rvol_min=0.0, tp1_frac=0.0, label="TP2only no filters"),
    MSConfig(rvol_min=1.2, tp1_frac=0.5, label="50/50  + RVOL≥1.2"),
    MSConfig(rvol_min=1.2, tp1_frac=0.7, label="70/30  + RVOL≥1.2"),
    MSConfig(rvol_min=1.2, tp1_frac=0.0, label="TP2only + RVOL≥1.2"),
    MSConfig(rvol_min=1.2, session_filter=True, tp1_frac=0.5, label="50/50  + RVOL≥1.2 + Session"),
    MSConfig(rvol_min=1.2, session_filter=True, tp1_frac=0.7, label="70/30  + RVOL≥1.2 + Session"),
    MSConfig(rvol_min=1.2, session_filter=True, tp1_frac=0.0, label="TP2only + RVOL≥1.2 + Session"),
]

MS_PRESETS: dict[str, list[MSConfig]] = {
    "default": ALL_MS_CONFIGS,
    "tp-test": _MS_TP_TEST_CONFIGS,
}


# ---------------------------------------------------------------------------
# 1h bias pre-computation
# ---------------------------------------------------------------------------

def _compute_bias(candles_1h: list) -> list[dict]:
    """Rolling bias per 1h bar — no lookahead, O(n) via capped window.

    swing_structure window: last STRUCT_WINDOW bars (~8 days).
    volume_profile  window: last VP_WINDOW     bars (~20 hours).
    Bias requires HH_HL or LH_LL with swing_count >= 2.
    """
    if len(candles_1h) < 10:
        return []

    print(f"  Pre-computing 1h bias for {len(candles_1h)} bars…", flush=True)
    results = []
    for i, bar in enumerate(candles_1h):
        struct = swing_struct_fn(
            candles_1h[max(0, i - STRUCT_WINDOW + 1):i + 1],
            current_price=bar["close"],
        )
        vp = vol_profile_fn(candles_1h[max(0, i - VP_WINDOW + 1):i + 1])

        if struct is None or struct["trend"] == "mixed" or vp is None:
            results.append({"t": bar["t"], "bias": None, "structure": struct, "vp": vp})
            continue

        if struct.get("swing_count", 0) < 2:
            results.append({"t": bar["t"], "bias": None, "structure": struct, "vp": vp})
            continue

        bias = "bull" if struct["trend"] == "HH_HL" else "bear"
        results.append({"t": bar["t"], "bias": bias, "structure": struct, "vp": vp})

    return results


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_simulation_ms(candles_5m: list, bias_list: list[dict], cfg: MSConfig,
                       debug: bool = False) -> dict:
    if len(candles_5m) < 21 or not bias_list:
        return {}

    rvol_vals = rvol_fn(candles_5m, 20)

    trades: list[float] = []

    # Open trade state
    in_trade   = False
    direction  = None
    sl         = tp1 = tp2 = entry_price = 0.0
    win_r1     = win_r2 = 0.0
    tp1_hit    = False
    accrued_r  = 0.0

    # Pending confirmation state  (signal bar detected, waiting for next bar)
    pending_dir    = None      # "long" | "short" | None
    pending_sl     = 0.0
    pending_tp1    = pending_tp2 = 0.0
    pending_vp_val = pending_vp_vah = 0.0
    pending_bias   = None

    d_bounces = d_filter_fail = d_rr_fail = d_confirmed = d_opened = 0

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
        vp     = h4["vp"]

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
                        sl        = entry_price          # move SL to breakeven
                else:
                    if bar["low"] <= sl:                 # stopped at breakeven
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

        # ── Check confirmation bar ─────────────────────────────────────────
        if pending_dir is not None:
            if bias != pending_bias:          # bias flipped — cancel
                pending_dir = None
                # fall through to check for fresh signal on this bar
            elif pending_dir == "long":
                if close > pending_vp_val:    # confirmed
                    risk  = close - pending_sl
                    if risk > 0:
                        wr1 = (pending_tp1 - close) / risk
                        wr2 = (pending_tp2 - close) / risk
                        if wr1 >= cfg.min_rr:
                            d_confirmed += 1
                            entry_price = close
                            sl, tp1, tp2 = pending_sl, pending_tp1, pending_tp2
                            win_r1, win_r2 = wr1, wr2
                            direction, in_trade, tp1_hit, accrued_r = "long", True, False, 0.0
                            pending_dir = None
                            d_opened += 1
                            continue
                pending_dir = None            # confirmation failed or R:R broke
                continue
            else:  # pending short
                if close < pending_vp_vah:    # confirmed
                    risk = pending_sl - close
                    if risk > 0:
                        wr1 = (close - pending_tp1) / risk
                        wr2 = (close - pending_tp2) / risk
                        if wr1 >= cfg.min_rr:
                            d_confirmed += 1
                            entry_price = close
                            sl, tp1, tp2 = pending_sl, pending_tp1, pending_tp2
                            win_r1, win_r2 = wr1, wr2
                            direction, in_trade, tp1_hit, accrued_r = "short", True, False, 0.0
                            pending_dir = None
                            d_opened += 1
                            continue
                pending_dir = None
                continue

        if bias is None or vp is None or struct is None:
            continue

        if cfg.session_filter and not _in_session(ts):
            continue

        rvol_v   = rvol_vals[i]
        va_width = vp["vah"] - vp["val"]
        if va_width <= 0:
            continue

        # ── Detect signal bar ──────────────────────────────────────────────
        if bias == "bull":
            if not (bar["low"] <= vp["val"] + ZONE_PCT * va_width and close > vp["val"]):
                continue

            d_bounces += 1
            if cfg.rvol_min > 0 and (rvol_v is None or rvol_v < cfg.rvol_min):
                d_filter_fail += 1; continue

            sl_price = vp["val"] - SL_BUFFER_PCT * va_width
            tp1_p    = vp["vah"]
            tp2_raw  = struct.get("tp_speculative_long")
            tp2_p    = tp2_raw if (tp2_raw and tp2_raw > tp1_p) else tp1_p

            risk = close - sl_price
            if risk <= 0 or (tp1_p - close) / risk < cfg.min_rr:
                d_rr_fail += 1; continue

            pending_dir    = "long"
            pending_sl     = sl_price
            pending_tp1    = tp1_p
            pending_tp2    = tp2_p
            pending_vp_val = vp["val"]
            pending_vp_vah = vp["vah"]
            pending_bias   = bias

        elif bias == "bear":
            if not (bar["high"] >= vp["vah"] - ZONE_PCT * va_width and close < vp["vah"]):
                continue

            d_bounces += 1
            if cfg.rvol_min > 0 and (rvol_v is None or rvol_v < cfg.rvol_min):
                d_filter_fail += 1; continue

            sl_price = vp["vah"] + SL_BUFFER_PCT * va_width
            tp1_p    = vp["val"]
            tp2_raw  = struct.get("tp_speculative_short")
            tp2_p    = tp2_raw if (tp2_raw and tp2_raw < tp1_p) else tp1_p

            risk = sl_price - close
            if risk <= 0 or (close - tp1_p) / risk < cfg.min_rr:
                d_rr_fail += 1; continue

            pending_dir    = "short"
            pending_sl     = sl_price
            pending_tp1    = tp1_p
            pending_tp2    = tp2_p
            pending_vp_val = vp["val"]
            pending_vp_vah = vp["vah"]
            pending_bias   = bias

    if debug:
        print(f"    [debug] bounces={d_bounces}  filter_fail={d_filter_fail}"
              f"  rr_fail={d_rr_fail}  confirmed={d_confirmed}"
              f"  opened={d_opened}  closed={len(trades)}")

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


def _print_ms_table(asset: str, candles: list, all_stats: list, entry_tf: str = "5m"):
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = (
        f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}"
        if candles else "?"
    )

    print(f"\n{'='*90}")
    print(f"Asset: {asset}   Period: {period}")
    print(f"Strategy: 1h MS (swing_count≥2) + VA Bounce + Confirmation bar + Partial exit  [entry: {entry_tf}]")
    print(f"{'='*90}")
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


# ---------------------------------------------------------------------------
# Markdown file output
# ---------------------------------------------------------------------------

_RESULTS_FILE = pathlib.Path(__file__).parent.parent.parent / "docs" / "results" / "backtest_results_ms.md"


_STRATEGY_LABEL = "1h MS (swing_count≥2) + VA Bounce + Confirmation bar + Partial exit"


def _append_results_md(asset: str, candles: list, all_stats: list, entry_tf: str):
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = (
        f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}"
        if candles else "?"
    )
    run_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"\n---\n\n",
        f"**Asset:** {asset}  |  **Period:** {period}  |  **Run:** {run_date}  |  **Entry TF:** {entry_tf}\n\n",
        f"**Strategy:** {_STRATEGY_LABEL}\n\n",
        f"| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |\n",
        f"|--------|--------|------|---------|--------|------|-------|---------|\n",
    ]
    for cfg, s in all_stats:
        if not s or s.get("trades", 0) == 0:
            lines.append(f"| {cfg.label} | — | | | | | | |\n")
            continue
        verdict = _verdict(s)
        lines.append(
            f"| {cfg.label} "
            f"| {s['trades']} "
            f"| {s['win_rate']*100:.1f}% "
            f"| {s['avg_win_r']:.2f} "
            f"| {s['total_r']:+.1f} "
            f"| {s['avg_r']:+.3f} "
            f"| {s['max_dd_r']:.1f}% "
            f"| {verdict} |\n"
        )

    header_needed = not _RESULTS_FILE.exists() or _RESULTS_FILE.stat().st_size == 0
    with open(_RESULTS_FILE, "a") as f:
        if header_needed:
            f.write("# Backtest Results Log\n\n")
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_ms_asset(asset: str, years: int, fetch: bool, entry_tf: str = "5m",
                       preset: str = "default"):
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

    bias_list = _compute_bias(candles_1h)

    if entry_tf == "1h":
        _1h_ms = 3_600_000
        bias_shifted = [
            {**b, "t": b["t"] + _1h_ms} for b in bias_list
        ]
        entry_candles = candles_1h
        sim_bias      = bias_shifted
    else:
        candles_5m = load_cache(asset, "5m") or []
        if not candles_5m:
            print(f"{asset}: missing 5m candle data — run with --fetch first")
            return
        entry_candles = candles_5m
        sim_bias      = bias_list

    configs = MS_PRESETS[preset]
    all_stats = [
        (cfg, _run_simulation_ms(entry_candles, sim_bias, cfg, debug=(i == 0)))
        for i, cfg in enumerate(configs)
    ]
    _print_ms_table(asset, entry_candles, all_stats, entry_tf=entry_tf)
    _append_results_md(asset, entry_candles, all_stats, entry_tf=entry_tf)


async def main_async(assets: list[str], years: int, fetch: bool, entry_tf: str,
                     preset: str):
    for asset in assets:
        await run_ms_asset(asset, years, fetch, entry_tf=entry_tf, preset=preset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets",   nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--years",    type=int,  default=2)
    parser.add_argument("--fetch",    action="store_true")
    parser.add_argument("--entry-tf", default="5m", choices=["5m", "1h"],
                        help="Candle timeframe for entry signals (default: 5m)")
    parser.add_argument("--preset",   default="default",
                        choices=list(MS_PRESETS),
                        help="Config preset (default: default)")
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.years, args.fetch,
                           entry_tf=args.entry_tf, preset=args.preset))


if __name__ == "__main__":
    main()
