"""50/50 Momentum + Mean Reversion Blend backtest.

Combines two entry signals:

1. Momentum (4h bias):
   - 4h: EMA20 > EMA50, MACD > 0, ADX > 25 (bull) OR reverse (bear)
   - 5m: close > prev high, OBV rising, RSI 50-70 (long) OR reverse (short)

2. Mean Reversion (1h structure + VA bounce):
   - 1h: swing_structure.trend = HH_HL (long) or LH_LL (short), swing_count >= 2
   - 5m: VA bounce + confirmation bar

Entry:
  Either momentum OR mean reversion signal fires on 5m → enter
  Same TP/SL logic as MS backtest (VA-based)
  Track which signal triggered each trade

TP:  VAH (long) / VAL (short) — 50% exits, SL → breakeven
TP2: 127.2% Fibonacci extension — remaining 50% exits
SL:  VAL - 0.15×VA_width (long) / VAH + 0.15×VA_width (short)
R:R: min_rr = 1.5 on TP1 leg

Usage:
  python -m backtest.run_backtest_blend --assets SOL
  python -m backtest.run_backtest_blend --assets SOL --fetch
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
    ema, adx as adx_fn, rsi as rsi_fn, obv as obv_fn, atr as atr_fn,
)
from backtest.fetch_history import load_cache, fetch_all, save_cache

MIN_TRADES       = 20
SL_BUFFER_PCT    = 0.15
ZONE_PCT         = 0.30
STRUCT_WINDOW    = 200
VP_WINDOW        = 20

_VIENNA_TZ = ZoneInfo("Europe/Vienna")
_LONDON_START, _LONDON_END = 8 + 30 / 60, 11.5
_NY_START,     _NY_END     = 16.0, 20.0


def _in_session(ts_ms: int) -> bool:
    hf = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    hf = hf.hour + hf.minute / 60
    return (_LONDON_START <= hf < _LONDON_END) or (_NY_START <= hf < _NY_END)


@dataclass
class BlendConfig:
    rvol_min:       float = 1.2   # mean reversion filter
    session_filter: bool  = False
    regime_filter:  bool  = False  # ADX>25 → momentum only; ADX<20 → MR only; 20-25 → skip
    min_rr:         float = 1.5
    label:          str   = "Baseline"


ALL_BLEND_CONFIGS = [
    BlendConfig(rvol_min=0.0, session_filter=False, label="No filters"),
    BlendConfig(rvol_min=1.2, session_filter=False, label="+ RVOL ≥ 1.2"),
    BlendConfig(rvol_min=0.0, session_filter=True,  label="+ Session"),
    BlendConfig(rvol_min=1.2, session_filter=True,  label="+ RVOL ≥ 1.2 + Session"),
    BlendConfig(rvol_min=0.0, regime_filter=True,   label="+ Regime"),
    BlendConfig(rvol_min=1.2, regime_filter=True,   label="+ RVOL ≥ 1.2 + Regime"),
    BlendConfig(rvol_min=1.2, session_filter=True, regime_filter=True, label="+ RVOL ≥ 1.2 + Sess + Regime"),
]


# ---------------------------------------------------------------------------
# 1h MS bias (mean reversion)
# ---------------------------------------------------------------------------

def _compute_ms_bias(candles_1h: list) -> list[dict]:
    """Rolling 1h bias for mean reversion structure."""
    if len(candles_1h) < 10:
        return []

    print(f"  Pre-computing 1h MS bias for {len(candles_1h)} bars…", flush=True)
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
# 4h momentum bias
# ---------------------------------------------------------------------------

def _compute_momentum_bias(candles_4h: list) -> list[dict]:
    """Rolling 4h bias for momentum (EMA/MACD/ADX)."""
    if len(candles_4h) < 51:
        return []

    print(f"  Pre-computing 4h momentum bias for {len(candles_4h)} bars…", flush=True)
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
            results.append({"t": bar["t"], "bias": None, "adx": None})
        elif adx_v < 25:
            results.append({"t": bar["t"], "bias": None, "adx": adx_v})
        elif e20 > e50 and h > 0:
            results.append({"t": bar["t"], "bias": "bull", "adx": adx_v})
        elif e20 < e50 and h < 0:
            results.append({"t": bar["t"], "bias": "bear", "adx": adx_v})
        else:
            results.append({"t": bar["t"], "bias": None, "adx": adx_v})

    return results


def _get_bias_at(bias_list: list[dict], ts: int) -> str | None:
    result = None
    for b in bias_list:
        if b["t"] <= ts:
            result = b.get("bias")
        else:
            break
    return result


def _get_adx_at(bias_list: list[dict], ts: int) -> float | None:
    result = None
    for b in bias_list:
        if b["t"] <= ts:
            result = b.get("adx")
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_blend_simulation(candles_5m: list, ms_bias_list: list[dict],
                           momentum_bias_list: list[dict], cfg: BlendConfig) -> dict:
    """Run 5m backtest with both momentum and mean reversion signals."""
    if len(candles_5m) < 21 or not ms_bias_list:
        return {}

    rvol_vals = rvol_fn(candles_5m, 20)
    rsi_vals = rsi_fn(candles_5m, 14)
    obv_vals = obv_fn(candles_5m)
    atr_vals = atr_fn(candles_5m, 14)

    trades: list[tuple[float, str]] = []  # (return, signal_type)

    # Trade state
    in_trade = False
    direction = None
    sl = tp1 = tp2 = entry_price = 0.0
    win_r1 = win_r2 = 0.0
    tp1_hit = False
    accrued_r = 0.0

    # Pending confirmation (signal bar detected, waiting for confirmation)
    pending_dir = None
    pending_sl = 0.0
    pending_tp1 = pending_tp2 = 0.0
    pending_vp_val = pending_vp_vah = 0.0
    pending_bias = None
    pending_signal_type = None  # "momentum" or "mean_reversion"

    ms_ptr = mom_ptr = 0

    for i in range(3, len(candles_5m)):
        bar = candles_5m[i]
        ts = bar["t"]
        prev = candles_5m[i - 1]
        close = bar["close"]

        # Advance pointers
        while ms_ptr + 1 < len(ms_bias_list) and ms_bias_list[ms_ptr + 1]["t"] <= ts:
            ms_ptr += 1
        while mom_ptr + 1 < len(momentum_bias_list) and momentum_bias_list[mom_ptr + 1]["t"] <= ts:
            mom_ptr += 1

        ms_bias_data = ms_bias_list[ms_ptr] if ms_ptr < len(ms_bias_list) else {}
        mom_bias = _get_bias_at(momentum_bias_list, ts)

        # ── Manage open trade ──────────────────────────────────────────────
        if in_trade:
            if direction == "long":
                if not tp1_hit:
                    if bar["low"] <= sl:
                        trades.append((-1.0, pending_signal_type))
                        in_trade = False
                    elif bar["high"] >= tp1:
                        tp1_hit = True
                        accrued_r = 0.5 * win_r1
                        sl = entry_price
                else:
                    if bar["low"] <= sl:
                        trades.append((accrued_r, pending_signal_type))
                        in_trade = False
                    elif bar["high"] >= tp2:
                        trades.append((accrued_r + 0.5 * win_r2, pending_signal_type))
                        in_trade = False
            else:  # short
                if not tp1_hit:
                    if bar["high"] >= sl:
                        trades.append((-1.0, pending_signal_type))
                        in_trade = False
                    elif bar["low"] <= tp1:
                        tp1_hit = True
                        accrued_r = 0.5 * win_r1
                        sl = entry_price
                else:
                    if bar["high"] >= sl:
                        trades.append((accrued_r, pending_signal_type))
                        in_trade = False
                    elif bar["low"] <= tp2:
                        trades.append((accrued_r + 0.5 * win_r2, pending_signal_type))
                        in_trade = False
            continue

        # ── Check confirmation bar ─────────────────────────────────────────
        if pending_dir is not None:
            ms_bias = ms_bias_data.get("bias")
            if ms_bias != pending_bias and pending_signal_type == "mean_reversion":
                pending_dir = None
            elif mom_bias != pending_bias and pending_signal_type == "momentum":
                pending_dir = None
            elif pending_dir == "long":
                confirm_ok = (
                    (pending_signal_type == "mean_reversion" and close > pending_vp_val) or
                    (pending_signal_type == "momentum")  # momentum doesn't need confirmation
                )
                if confirm_ok:
                    risk = close - pending_sl
                    if risk > 0:
                        wr1 = (pending_tp1 - close) / risk
                        wr2 = (pending_tp2 - close) / risk
                        if wr1 >= cfg.min_rr:
                            entry_price = close
                            sl, tp1, tp2 = pending_sl, pending_tp1, pending_tp2
                            win_r1, win_r2 = wr1, wr2
                            direction, in_trade, tp1_hit, accrued_r = "long", True, False, 0.0
                            pending_dir = None
                            continue
                pending_dir = None
                continue
            else:  # pending short
                confirm_ok = (
                    (pending_signal_type == "mean_reversion" and close < pending_vp_vah) or
                    (pending_signal_type == "momentum")
                )
                if confirm_ok:
                    risk = pending_sl - close
                    if risk > 0:
                        wr1 = (close - pending_tp1) / risk
                        wr2 = (close - pending_tp2) / risk
                        if wr1 >= cfg.min_rr:
                            entry_price = close
                            sl, tp1, tp2 = pending_sl, pending_tp1, pending_tp2
                            win_r1, win_r2 = wr1, wr2
                            direction, in_trade, tp1_hit, accrued_r = "short", True, False, 0.0
                            pending_dir = None
                            continue
                pending_dir = None
                continue

        if cfg.session_filter and not _in_session(ts):
            continue

        adx_now = _get_adx_at(momentum_bias_list, ts) if cfg.regime_filter else None
        # regime_filter: ADX>25 → trending (momentum only); ADX<20 → ranging (MR only); 20-25 → skip
        if cfg.regime_filter:
            if adx_now is None or 20 <= adx_now <= 25:
                continue
            allow_momentum = adx_now > 25
            allow_mr       = adx_now < 20
        else:
            allow_momentum = allow_mr = True

        # ── Check momentum signal ──────────────────────────────────────────
        rsi_v = rsi_vals[i]
        obv_v = obv_vals[i]
        atr_v = atr_vals[i]

        if allow_momentum and rsi_v is not None and obv_v is not None and atr_v is not None and atr_v > 0:
            if mom_bias == "bull" and close > prev["high"] and obv_v > obv_vals[i-1] \
                    and 50 <= rsi_v <= 70:
                pending_dir = "long"
                pending_bias = "bull"
                pending_signal_type = "momentum"
                pending_sl = close - 0.75 * atr_v
                pending_tp1 = close + 2.25 * atr_v
                pending_tp2 = pending_tp1
                pending_vp_val = pending_vp_vah = 0

            elif mom_bias == "bear" and close < prev["low"] and obv_v < obv_vals[i-1] \
                    and 30 <= rsi_v <= 50:
                pending_dir = "short"
                pending_bias = "bear"
                pending_signal_type = "momentum"
                pending_sl = close + 0.75 * atr_v
                pending_tp1 = close - 2.25 * atr_v
                pending_tp2 = pending_tp1
                pending_vp_val = pending_vp_vah = 0

        if not allow_mr:
            continue

        # ── Check mean reversion signal (VA bounce) ─────────────────────
        ms_bias = ms_bias_data.get("bias")
        vp = ms_bias_data.get("vp")
        struct = ms_bias_data.get("structure")

        if ms_bias is None or vp is None or struct is None:
            continue

        va_width = vp["vah"] - vp["val"]
        if va_width <= 0:
            continue

        rvol_v = rvol_vals[i]

        if ms_bias == "bull":
            if not (bar["low"] <= vp["val"] + ZONE_PCT * va_width and close > vp["val"]):
                continue

            if cfg.rvol_min > 0 and (rvol_v is None or rvol_v < cfg.rvol_min):
                continue

            sl_price = vp["val"] - SL_BUFFER_PCT * va_width
            tp1_p = vp["vah"]
            tp2_raw = struct.get("tp_speculative_long")
            tp2_p = tp2_raw if (tp2_raw and tp2_raw > tp1_p) else tp1_p

            risk = close - sl_price
            if risk <= 0 or (tp1_p - close) / risk < cfg.min_rr:
                continue

            pending_dir = "long"
            pending_sl = sl_price
            pending_tp1 = tp1_p
            pending_tp2 = tp2_p
            pending_vp_val = vp["val"]
            pending_vp_vah = vp["vah"]
            pending_bias = ms_bias
            pending_signal_type = "mean_reversion"

        elif ms_bias == "bear":
            if not (bar["high"] >= vp["vah"] - ZONE_PCT * va_width and close < vp["vah"]):
                continue

            if cfg.rvol_min > 0 and (rvol_v is None or rvol_v < cfg.rvol_min):
                continue

            sl_price = vp["vah"] + SL_BUFFER_PCT * va_width
            tp1_p = vp["val"]
            tp2_raw = struct.get("tp_speculative_short")
            tp2_p = tp2_raw if (tp2_raw and tp2_raw < tp1_p) else tp1_p

            risk = sl_price - close
            if risk <= 0 or (close - tp1_p) / risk < cfg.min_rr:
                continue

            pending_dir = "short"
            pending_sl = sl_price
            pending_tp1 = tp1_p
            pending_tp2 = tp2_p
            pending_vp_val = vp["val"]
            pending_vp_vah = vp["vah"]
            pending_bias = ms_bias
            pending_signal_type = "mean_reversion"

    if not trades:
        return {"trades": 0}

    # Compute stats
    wins = sum(1 for r, _ in trades if r > 0)
    total_r = sum(r for r, _ in trades)
    win_rs = [r for r, _ in trades if r > 0]
    avg_win_r = sum(win_rs) / len(win_rs) if win_rs else 0.0

    momentum_trades = sum(1 for _, sig in trades if sig == "momentum")
    mr_trades = sum(1 for _, sig in trades if sig == "mean_reversion")

    peak = cum = max_dd = 0.0
    for r, _ in trades:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "avg_win_r": avg_win_r,
        "total_r": total_r,
        "avg_r": total_r / len(trades),
        "max_dd_r": -max_dd,
        "momentum_trades": momentum_trades,
        "mr_trades": mr_trades,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _verdict(stats: dict) -> str:
    n = stats.get("trades", 0)
    if n < MIN_TRADES:
        return f"INCONCLUSIVE (<{MIN_TRADES})"
    wr = stats.get("win_rate", 0)
    avg_w = stats.get("avg_win_r", 2.0)
    be_wr = 1.0 / (1.0 + avg_w) if avg_w > 0 else 1.0
    return "GO ✓" if wr >= be_wr + 0.05 else "NO-GO ✗"


def _print_blend_table(asset: str, candles: list, all_stats: list):
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = (
        f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}"
        if candles else "?"
    )

    print(f"\n{'='*100}")
    print(f"Asset: {asset}   Period: {period}")
    print(f"{'='*100}")
    print(f"{'Config':<35} {'Trades':>7} {'Win%':>6} {'Mom/MR':>10} "
          f"{'TotalR':>8} {'AvgR':>7} {'MaxDD':>7} {'Verdict'}")
    print(f"{'-'*35} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*14}")

    for cfg, s in all_stats:
        if not s or s.get("trades", 0) == 0:
            print(f"{cfg.label:<35} {'—':>7}")
            continue
        mom = s.get("momentum_trades", 0)
        mr = s.get("mr_trades", 0)
        mom_pct = 100 * mom / (mom + mr) if (mom + mr) > 0 else 0
        print(
            f"{cfg.label:<35} "
            f"{s['trades']:>7} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{mom:>3}/{mr:<3} ({mom_pct:>5.1f}%) "
            f"{s['total_r']:>+8.1f} "
            f"{s['avg_r']:>+7.3f} "
            f"{s['max_dd_r']:>7.1f} "
            f"{_verdict(s)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Blend momentum + mean reversion backtest")
    parser.add_argument("--assets", type=str, default="SOL", help="Asset symbols (space-separated)")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from Hyperliquid")
    parser.add_argument("--years", type=int, default=2, help="Years of history (with --fetch)")
    args = parser.parse_args()

    assets = args.assets.split()
    if not assets:
        assets = ["SOL"]

    print(f"\n{'='*100}")
    print(f"50/50 Momentum + Mean Reversion Blend Backtest")
    print(f"{'='*100}\n")

    for asset in assets:
        print(f"Asset: {asset}")
        symbol = asset  # Cache is stored as "SOL", "BTC", etc., not "SOLUSDT"

        # Load or fetch data
        from src.trading.hyperliquid_api import HyperliquidAPI
        hl = None

        for interval in ("5m", "1h", "4h"):
            cached = load_cache(symbol, interval)
            if cached is None or args.fetch:
                if hl is None:
                    hl = HyperliquidAPI()
                    await hl.get_meta_and_ctxs()
                print(f"  Fetching {symbol} {interval}…", end=" ", flush=True)
                candles, source = await fetch_all(hl, symbol, interval, args.years)
                save_cache(symbol, interval, candles)
                print(f"{len(candles)} bars [{source}]")

        candles_5m = load_cache(symbol, "5m") or []
        candles_1h = load_cache(symbol, "1h") or []
        candles_4h = load_cache(symbol, "4h") or []

        if not candles_5m:
            print(f"    ✗ No cached data for {symbol}. Use --fetch to download.")
            continue

        if len(candles_5m) < 100:
            print(f"    ✗ Insufficient 5m data ({len(candles_5m)} bars)")
            continue

        print(f"    ✓ {len(candles_5m)} 5m bars, {len(candles_1h)} 1h bars, {len(candles_4h)} 4h bars")

        # Compute bias
        ms_bias = _compute_ms_bias(candles_1h)
        momentum_bias = _compute_momentum_bias(candles_4h)

        # Run all configs
        all_stats = []
        for cfg in ALL_BLEND_CONFIGS:
            stats = _run_blend_simulation(candles_5m, ms_bias, momentum_bias, cfg)
            all_stats.append((cfg, stats))

        # Print results
        _print_blend_table(asset, candles_5m, all_stats)

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    asyncio.run(main())
