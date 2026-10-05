"""VWAP strategy backtest for BTC (or any crypto with cached 5m/1h candles).

Three straightforward VWAP strategies:

  pullback (intraday, 15m bars, VWAP anchored 00:00 UTC):
    - Long:  previous `trend_bars` closes above VWAP, VWAP rising vs 4 bars ago,
             bar low touches VWAP and closes back above → enter at close.
    - Short: mirror.
    - SL = signal-bar extreme ∓ 0.5×ATR(14). TP = entry ± rr × risk.
    - Entries 06:00–20:00 UTC, max 2 trades/day, flat at end of UTC day.

  bands (intraday mean reversion, 15m bars):
    - Long:  previous close below VWAP − k·σ, current close back above it.
    - Short: mirror on the upper band.
    - SL = 4-bar extreme ∓ 0.25×ATR. TP = VWAP at entry. Skip if TP < min_rr × risk.
    - Same session window / end-of-day flat as pullback.

  rolling (multi-day trend, 4h bars, rolling N-day VWAP):
    - Long while close > rolling VWAP and VWAP rising vs 6 bars (1 day) ago.
    - Short mirror (unless long_only). Exit on a close back across VWAP.
    - Protective SL = entry ∓ sl_atr × ATR(14) at entry.

Costs: taker fee 0.0432%/side + 0.01%/side slippage. Longs on multi-day holds pay
funding at FUNDING_APR (conservative; shorts receive nothing). When SL and TP are both
inside one bar, SL is assumed. All results in R (risk = initial SL distance).

Candles are Binance spot (BTCUSDT) — VWAP uses spot volume as the weighting proxy.

Usage:
  python -m backtest.run_backtest_vwap
  python -m backtest.run_backtest_vwap --strategy pullback
"""

import argparse
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from backtest.fetch_history import load_cache

FEE_SIDE    = 0.000432
SLIP_SIDE   = 0.0001
COST_RT     = 2 * (FEE_SIDE + SLIP_SIDE)
FUNDING_APR = 0.10

SESSION_START_H = 6
SESSION_END_H   = 20
MAX_TRADES_DAY  = 2


@dataclass
class Trade:
    t: int          # entry timestamp ms
    side: int       # +1 long, -1 short
    r: float        # net R after costs


def _dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _resample(candles: list, minutes: int) -> list:
    """Aggregate candles into `minutes` buckets aligned to UTC epoch."""
    ms = minutes * 60_000
    out = []
    for c in candles:
        b = c["t"] - c["t"] % ms
        if out and out[-1]["t"] == b:
            o = out[-1]
            o["high"] = max(o["high"], c["high"])
            o["low"] = min(o["low"], c["low"])
            o["close"] = c["close"]
            o["volume"] += c["volume"]
        else:
            out.append({"t": b, "open": c["open"], "high": c["high"],
                        "low": c["low"], "close": c["close"], "volume": c["volume"]})
    return out


def _atr(bars: list, n: int = 14) -> list[float]:
    out, prev_atr = [], None
    for i, b in enumerate(bars):
        tr = b["high"] - b["low"] if i == 0 else max(
            b["high"] - b["low"], abs(b["high"] - bars[i-1]["close"]), abs(b["low"] - bars[i-1]["close"]))
        prev_atr = tr if prev_atr is None else (prev_atr * (n - 1) + tr) / n
        out.append(prev_atr)
    return out


def _session_vwap(bars: list) -> tuple[list[float], list[float]]:
    """VWAP and volume-weighted σ, reset at 00:00 UTC."""
    vwap, sd = [], []
    day, pv, pv2, v = None, 0.0, 0.0, 0.0
    for b in bars:
        d = _dt(b["t"]).date()
        if d != day:
            day, pv, pv2, v = d, 0.0, 0.0, 0.0
        tp = (b["high"] + b["low"] + b["close"]) / 3
        pv += tp * b["volume"]; pv2 += tp * tp * b["volume"]; v += b["volume"]
        m = pv / v if v else tp
        vwap.append(m)
        sd.append(math.sqrt(max(pv2 / v - m * m, 0.0)) if v else 0.0)
    return vwap, sd


def _rolling_vwap(bars: list, n: int) -> list[float]:
    out, pv, v = [], 0.0, 0.0
    for i, b in enumerate(bars):
        tp = (b["high"] + b["low"] + b["close"]) / 3
        pv += tp * b["volume"]; v += b["volume"]
        if i >= n:
            o = bars[i - n]
            otp = (o["high"] + o["low"] + o["close"]) / 3
            pv -= otp * o["volume"]; v -= o["volume"]
        out.append(pv / v if v else tp)
    return out


def _net_r(side: int, entry: float, exit_px: float, risk: float, hold_h: float = 0.0) -> float:
    gross = side * (exit_px - entry) / risk
    cost = COST_RT * entry / risk
    funding = FUNDING_APR * hold_h / 8760 * entry / risk if side > 0 else 0.0
    return gross - cost - funding


# ── Intraday engine ───────────────────────────────────────────────────────────

def _run_intraday(bars: list, signal_fn) -> list[Trade]:
    """Walk bars; `signal_fn(i)` returns (side, sl, tp) or None. One position at a time."""
    trades: list[Trade] = []
    pos = None              # (side, entry, sl, tp, t)
    day, n_today = None, 0
    for i in range(20, len(bars)):
        b = bars[i]
        dt = _dt(b["t"])
        if dt.date() != day:
            day, n_today = dt.date(), 0
        last_bar_of_day = i + 1 >= len(bars) or _dt(bars[i+1]["t"]).date() != dt.date()

        if pos:
            side, entry, sl, tp, t0 = pos
            risk = abs(entry - sl)
            hit_sl = b["low"] <= sl if side > 0 else b["high"] >= sl
            hit_tp = b["high"] >= tp if side > 0 else b["low"] <= tp
            if hit_sl:
                trades.append(Trade(t0, side, _net_r(side, entry, sl, risk))); pos = None
            elif hit_tp:
                trades.append(Trade(t0, side, _net_r(side, entry, tp, risk))); pos = None
            elif last_bar_of_day:
                trades.append(Trade(t0, side, _net_r(side, entry, b["close"], risk))); pos = None
            continue

        if not (SESSION_START_H <= dt.hour < SESSION_END_H) or n_today >= MAX_TRADES_DAY:
            continue
        sig = signal_fn(i)
        if sig:
            side, sl, tp = sig
            entry = b["close"]
            if side * (entry - sl) <= 0:
                continue
            pos = (side, entry, sl, tp, b["t"]); n_today += 1
    return trades


def run_pullback(bars: list, rr: float, trend_bars: int) -> list[Trade]:
    vwap, _ = _session_vwap(bars)
    atr = _atr(bars)

    def sig(i):
        b = bars[i]
        prev = range(i - trend_bars, i)
        if all(bars[j]["close"] > vwap[j] for j in prev) and vwap[i] > vwap[i-4] \
                and b["low"] <= vwap[i] < b["close"]:
            sl = b["low"] - 0.5 * atr[i]
            return 1, sl, b["close"] + rr * (b["close"] - sl)
        if all(bars[j]["close"] < vwap[j] for j in prev) and vwap[i] < vwap[i-4] \
                and b["high"] >= vwap[i] > b["close"]:
            sl = b["high"] + 0.5 * atr[i]
            return -1, sl, b["close"] - rr * (sl - b["close"])
        return None
    return _run_intraday(bars, sig)


def run_bands(bars: list, k: float, min_rr: float) -> list[Trade]:
    vwap, sd = _session_vwap(bars)
    atr = _atr(bars)

    def sig(i):
        b, p = bars[i], bars[i-1]
        lo_band_p, lo_band = vwap[i-1] - k * sd[i-1], vwap[i] - k * sd[i]
        hi_band_p, hi_band = vwap[i-1] + k * sd[i-1], vwap[i] + k * sd[i]
        if p["close"] < lo_band_p and b["close"] > lo_band:
            sl = min(x["low"] for x in bars[i-3:i+1]) - 0.25 * atr[i]
            if vwap[i] - b["close"] >= min_rr * (b["close"] - sl):
                return 1, sl, vwap[i]
        if p["close"] > hi_band_p and b["close"] < hi_band:
            sl = max(x["high"] for x in bars[i-3:i+1]) + 0.25 * atr[i]
            if b["close"] - vwap[i] >= min_rr * (sl - b["close"]):
                return -1, sl, vwap[i]
        return None
    return _run_intraday(bars, sig)


# ── Multi-day engine ──────────────────────────────────────────────────────────

def run_rolling(bars: list, days: int, sl_atr: float, long_only: bool) -> list[Trade]:
    n = days * 6
    vwap = _rolling_vwap(bars, n)
    atr = _atr(bars)
    trades: list[Trade] = []
    pos = None              # (side, entry, sl, t)
    for i in range(n + 6, len(bars)):
        b = bars[i]
        if pos:
            side, entry, sl, t0 = pos
            risk = abs(entry - sl)
            hold_h = (b["t"] - t0) / 3_600_000 + 4
            if (b["low"] <= sl) if side > 0 else (b["high"] >= sl):
                trades.append(Trade(t0, side, _net_r(side, entry, sl, risk, hold_h))); pos = None
            elif side * (b["close"] - vwap[i]) < 0:
                trades.append(Trade(t0, side, _net_r(side, entry, b["close"], risk, hold_h))); pos = None
            continue
        rising = vwap[i] > vwap[i-6]
        if b["close"] > vwap[i] and rising and bars[i-1]["close"] <= vwap[i-1]:
            pos = (1, b["close"], b["close"] - sl_atr * atr[i], b["t"])
        elif not long_only and b["close"] < vwap[i] and not rising and bars[i-1]["close"] >= vwap[i-1]:
            pos = (-1, b["close"], b["close"] + sl_atr * atr[i], b["t"])
    return trades


# ── Reporting ─────────────────────────────────────────────────────────────────

def _stats(trades: list[Trade]) -> dict:
    rs = [t.r for t in trades]
    eq = peak = dd = 0.0
    for r in rs:
        eq += r; peak = max(peak, eq); dd = min(dd, eq - peak)
    wins = sum(r for r in rs if r > 0); losses = -sum(r for r in rs if r < 0)
    months = max((trades[-1].t - trades[0].t) / (30.4 * 86_400_000), 1) if trades else 1
    return {
        "n": len(rs), "per_mo": len(rs) / months,
        "wr": 100 * sum(r > 0 for r in rs) / len(rs) if rs else 0,
        "avg": sum(rs) / len(rs) if rs else 0, "tot": sum(rs), "dd": dd,
        "pf": wins / losses if losses else float("inf"),
    }


def _by_year(trades: list[Trade]) -> dict[int, float]:
    out = defaultdict(float)
    for t in trades:
        out[_dt(t.t).year] += t.r
    return out


def _print(label: str, trades: list[Trade], years: list[int]):
    if not trades:
        print(f"  {label:<38} no trades"); return
    s = _stats(trades)
    yr = _by_year(trades)
    ls = [t.r for t in trades if t.side > 0]; ss = [t.r for t in trades if t.side < 0]
    print(f"  {label:<38}{s['n']:>5}{s['per_mo']:>6.1f}{s['wr']:>6.1f}%{s['avg']:>+7.3f}{s['tot']:>+8.1f}"
          f"{s['dd']:>+8.1f}{s['pf']:>6.2f}{sum(ls):>+8.1f}{sum(ss):>+8.1f}"
          + "".join(f"{yr.get(y, 0):>+8.1f}" for y in years))


def main():
    p = argparse.ArgumentParser(description="VWAP strategy backtest")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--strategy", choices=["pullback", "bands", "rolling"], default=None)
    a = p.parse_args()

    c5 = load_cache(a.asset, "5m"); c1h = load_cache(a.asset, "1h")
    if not c5 or not c1h:
        print(f"Missing cache — run: python -m backtest.fetch_history --assets {a.asset} --intervals 5m 1h --years 3")
        return
    b15 = _resample(c5, 15); b4h = _resample(c1h, 240)
    years = sorted({_dt(b["t"]).year for b in b15})
    print(f"\nVWAP backtest — {a.asset} | {_dt(b15[0]['t']):%Y-%m-%d} → {_dt(b15[-1]['t']):%Y-%m-%d}"
          f" | costs {COST_RT*100:.4f}% RT, funding {FUNDING_APR:.0%} APR on longs (multi-day)")
    hdr = (f"  {'config':<38}{'N':>5}{'/mo':>6}{'WR':>7}{'avgR':>7}{'totR':>8}{'maxDD':>8}{'PF':>6}"
           f"{'longR':>8}{'shortR':>8}" + "".join(f"{y:>8}" for y in years))

    if a.strategy in (None, "pullback"):
        print("\n PULLBACK (15m, session VWAP)\n" + hdr)
        for tb in (2, 4):
            for rr in (1.5, 2.0, 3.0):
                _print(f"trend_bars={tb} rr={rr}", run_pullback(b15, rr, tb), years)
    if a.strategy in (None, "bands"):
        print("\n BANDS (15m, session VWAP ± kσ fade → VWAP)\n" + hdr)
        for k in (1.5, 2.0, 2.5):
            for mr in (0.0, 1.0):
                _print(f"k={k} min_rr={mr}", run_bands(b15, k, mr), years)
    if a.strategy in (None, "rolling"):
        print("\n ROLLING (4h, N-day rolling VWAP trend)\n" + hdr)
        for d in (3, 5, 10):
            for sl in (2.0, 3.0):
                for lo in (False, True):
                    _print(f"days={d} sl_atr={sl} {'long-only' if lo else 'long+short'}",
                           run_rolling(b4h, d, sl, lo), years)
    print()


if __name__ == "__main__":
    main()
