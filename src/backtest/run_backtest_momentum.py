"""SOL Momentum Scalper backtest (15m).

Strategy:
  EMA9/EMA21 crossover on 15m candles.
  Long:  EMA9 crosses above EMA21  AND  RSI(14) > 55  AND  RVOL >= threshold
  Short: EMA9 crosses below EMA21  AND  RSI(14) < 45  AND  RVOL >= threshold

  Entry: close of signal bar (market entry on next open is approximated as close).
  TP:    entry + 2 × ATR14  (long)  /  entry − 2 × ATR14  (short)
  SL:    entry − 1 × ATR14  (long)  /  entry + 1 × ATR14  (short)
  R:R:   2:1 fixed.  One trade at a time — no new signal while in trade.

Optional filters:
  RVOL >= threshold  (default disabled = 0.0)
  Session filter (London + NY open, Vienna time)

Go/no-go: win_rate >= breakeven_win_rate + 5%  AND  trades >= 20.
Breakeven WR for 2:1 R:R = 1/(1+2) ≈ 33.3% → need ≥ 38.3% to pass.

Usage:
  python -m src.backtest.run_backtest_momentum --assets SOL
  python -m src.backtest.run_backtest_momentum --assets SOL --fetch --years 2
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

from src.backtest.fetch_history import load_cache, fetch_all, save_cache

MIN_TRADES = 20
TP_R = 2.0   # TP at 2× risk
SL_R = 1.0   # SL at 1× risk

_VIENNA_TZ = ZoneInfo("Europe/Vienna")
_LONDON_START, _LONDON_END = 8 + 30 / 60, 11.5
_NY_START,     _NY_END     = 16.0, 20.0


def _in_session(ts_ms: int) -> bool:
    hf = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    hf = hf.hour + hf.minute / 60
    return (_LONDON_START <= hf < _LONDON_END) or (_NY_START <= hf < _NY_END)


# ---------------------------------------------------------------------------
# Indicator helpers (inline — no external deps)
# ---------------------------------------------------------------------------

def _ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average, None until enough data."""
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    k = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    sma = sum(values[:period]) / period
    result[period - 1] = sma
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Wilder-smoothed RSI. Returns None until period+1 bars available."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    # Initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        j = i - 1  # index into deltas/gains/losses (offset by 1)
        if i == period:
            pass  # already set above
        else:
            avg_gain = (avg_gain * (period - 1) + gains[j]) / period
            avg_loss = (avg_loss * (period - 1) + losses[j]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        result[i] = 100 - 100 / (1 + rs)
    return result


def _atr(candles: list[dict], period: int = 14) -> list[float | None]:
    """Average True Range (Wilder smoothing). None until period bars."""
    result: list[float | None] = [None] * len(candles)
    if len(candles) < period + 1:
        return result
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    # Initial ATR = SMA of first `period` TRs
    atr_val = sum(trs[:period]) / period
    result[period] = atr_val  # index period corresponds to trs[period-1]
    for i in range(period + 1, len(candles)):
        atr_val = (atr_val * (period - 1) + trs[i - 1]) / period
        result[i] = atr_val
    return result


def _rvol(candles: list[dict], period: int = 20) -> list[float | None]:
    """Relative volume = bar_volume / rolling_avg_volume. None until period bars."""
    volumes = [c["volume"] for c in candles]
    result: list[float | None] = [None] * len(candles)
    for i in range(period, len(candles)):
        avg = sum(volumes[i - period:i]) / period
        result[i] = volumes[i] / avg if avg > 0 else None
    return result


# ---------------------------------------------------------------------------
# 15m resampler
# ---------------------------------------------------------------------------

def _resample_15m(candles_5m: list[dict]) -> list[dict]:
    """Aggregate 5m candles into 15m bars (groups of 3 by aligned timestamp)."""
    result = []
    group: list[dict] = []
    for c in candles_5m:
        group.append(c)
        if len(group) == 3:
            result.append({
                "t":      group[0]["t"],
                "open":   group[0]["open"],
                "high":   max(g["high"]   for g in group),
                "low":    min(g["low"]    for g in group),
                "close":  group[-1]["close"],
                "volume": sum(g["volume"] for g in group),
            })
            group = []
    # drop any incomplete group at the end
    return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MomConfig:
    rvol_min:       float = 0.0
    session_filter: bool  = False
    label:          str   = "Baseline"


ALL_MOM_CONFIGS = [
    MomConfig(rvol_min=0.0, session_filter=False, label="No filters"),
    MomConfig(rvol_min=1.5, session_filter=False, label="+ RVOL ≥ 1.5"),
    MomConfig(rvol_min=2.0, session_filter=False, label="+ RVOL ≥ 2.0"),
    MomConfig(rvol_min=0.0, session_filter=True,  label="+ Session"),
    MomConfig(rvol_min=1.5, session_filter=True,  label="+ RVOL ≥ 1.5 + Session"),
    MomConfig(rvol_min=2.0, session_filter=True,  label="+ RVOL ≥ 2.0 + Session"),
]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _run_simulation_mom(candles_15m: list[dict], cfg: MomConfig,
                        debug: bool = False) -> dict:
    n = len(candles_15m)
    if n < 30:
        return {}

    closes  = [c["close"] for c in candles_15m]
    ema9    = _ema(closes, 9)
    ema21   = _ema(closes, 21)
    rsi14   = _rsi(closes, 14)
    atr14   = _atr(candles_15m, 14)
    rvol20  = _rvol(candles_15m, 20)

    trades: list[float] = []
    in_trade  = False
    direction = None
    tp = sl = 0.0
    risk_pts  = 0.0

    d_signals = d_rsi_fail = d_rvol_fail = d_opened = 0

    # Need at least bar 21 for EMA21 to be valid
    for i in range(22, n):
        bar = candles_15m[i]

        # Manage open trade on this bar
        if in_trade:
            if direction == "long":
                if bar["low"] <= sl:
                    trades.append(-SL_R)
                    in_trade = False
                elif bar["high"] >= tp:
                    trades.append(TP_R)
                    in_trade = False
            else:  # short
                if bar["high"] >= sl:
                    trades.append(-SL_R)
                    in_trade = False
                elif bar["low"] <= tp:
                    trades.append(TP_R)
                    in_trade = False
            continue

        # All indicators must be valid
        e9_prev, e9_cur  = ema9[i - 1],  ema9[i]
        e21_prev, e21_cur = ema21[i - 1], ema21[i]
        rsi_cur  = rsi14[i]
        atr_cur  = atr14[i]
        rvol_cur = rvol20[i]

        if any(v is None for v in [e9_prev, e9_cur, e21_prev, e21_cur, rsi_cur, atr_cur]):
            continue

        # Detect crossover on this bar (previous bar had opposite order)
        long_cross  = e9_prev < e21_prev and e9_cur >= e21_cur
        short_cross = e9_prev > e21_prev and e9_cur <= e21_cur

        if not long_cross and not short_cross:
            continue

        d_signals += 1

        # Session filter
        if cfg.session_filter and not _in_session(bar["t"]):
            continue

        entry = bar["close"]

        if long_cross:
            if rsi_cur <= 55:
                d_rsi_fail += 1
                continue
            if cfg.rvol_min > 0 and (rvol_cur is None or rvol_cur < cfg.rvol_min):
                d_rvol_fail += 1
                continue
            risk_pts  = atr_cur * SL_R
            tp = entry + atr_cur * TP_R
            sl = entry - risk_pts
            direction = "long"

        else:  # short_cross
            if rsi_cur >= 45:
                d_rsi_fail += 1
                continue
            if cfg.rvol_min > 0 and (rvol_cur is None or rvol_cur < cfg.rvol_min):
                d_rvol_fail += 1
                continue
            risk_pts  = atr_cur * SL_R
            tp = entry - atr_cur * TP_R
            sl = entry + risk_pts
            direction = "short"

        d_opened += 1
        in_trade = True

    if debug:
        print(f"    [debug] signals={d_signals}  rsi_fail={d_rsi_fail}"
              f"  rvol_fail={d_rvol_fail}  opened={d_opened}  closed={len(trades)}")

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


def _print_mom_table(asset: str, candles: list, all_stats: list):
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period = (
        f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}"
        if candles else "?"
    )

    print(f"\n{'='*90}")
    print(f"Asset: {asset}   Period: {period}")
    print(f"Strategy: EMA9/21 crossover + RSI(14) + RVOL  [entry: 15m, TP=2×ATR, SL=1×ATR]")
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
# Markdown output
# ---------------------------------------------------------------------------

_RESULTS_FILE = pathlib.Path(__file__).parent.parent.parent / "docs" / "results" / "backtest_results_momentum.md"

_STRATEGY_LABEL = "EMA9/21 crossover + RSI(14) + RVOL  [15m, TP=2×ATR, SL=1×ATR]"


def _append_results_md(asset: str, candles: list, all_stats: list):
    def _dt(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    period   = f"{_dt(candles[0]['t'])} → {_dt(candles[-1]['t'])}" if candles else "?"
    run_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"\n---\n\n",
        f"**Asset:** {asset}  |  **Period:** {period}  |  **Run:** {run_date}  |  **Entry TF:** 15m\n\n",
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
            f"| {s['max_dd_r']:.1f}R "
            f"| {verdict} |\n"
        )

    header_needed = not _RESULTS_FILE.exists() or _RESULTS_FILE.stat().st_size == 0
    with open(_RESULTS_FILE, "a") as f:
        if header_needed:
            f.write("# Backtest Results Log — Momentum Scalper\n\n")
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_momentum_asset(asset: str, years: int, fetch: bool):
    from src.trading.hyperliquid_api import HyperliquidAPI
    hl = None

    # We need 5m data to resample into 15m
    cached_5m = load_cache(asset, "5m")
    if cached_5m is None or fetch:
        if hl is None:
            hl = HyperliquidAPI()
            await hl.get_meta_and_ctxs()
        print(f"Fetching {asset} 5m…", end=" ", flush=True)
        candles_5m, source = await fetch_all(hl, asset, "5m", years)
        save_cache(asset, "5m", candles_5m)
        print(f"{len(candles_5m)} bars [{source}]")
    else:
        candles_5m = cached_5m

    if not candles_5m:
        print(f"{asset}: no 5m data available")
        return

    candles_15m = _resample_15m(candles_5m)
    print(f"  Resampled {len(candles_5m)} × 5m → {len(candles_15m)} × 15m bars", flush=True)

    all_stats = [
        (cfg, _run_simulation_mom(candles_15m, cfg, debug=(i == 0)))
        for i, cfg in enumerate(ALL_MOM_CONFIGS)
    ]
    _print_mom_table(asset, candles_15m, all_stats)
    _append_results_md(asset, candles_15m, all_stats)


async def main_async(assets: list[str], years: int, fetch: bool):
    for asset in assets:
        await run_momentum_asset(asset, years, fetch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["SOL"])
    parser.add_argument("--years",  type=int,  default=2)
    parser.add_argument("--fetch",  action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.years, args.fetch))


if __name__ == "__main__":
    main()
