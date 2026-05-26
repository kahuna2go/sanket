"""FVG Opening-Range Breakout backtest — SPY via Polygon.io (or yfinance fallback).

Data sources (in priority order):
  1. Polygon.io  — free account at polygon.io gives 2+ years of 5m data for SPY.
                   Set POLYGON_API_KEY in your .env file.
  2. yfinance    — fallback; only retains the last ~60 days of 5m data.

Strategy (5m candles; same logic applies if you swap in 1m data):
  1. Opening Range: first 5m candle of the regular session (09:30–09:35 EST).
     ORH = that candle's high, ORL = its low.

  2. FVG breakout (5m candles, first occurrence only):
     Bullish: 3-candle sequence [c0, c1, c2] where
              c0.high < c2.low (gap exists) AND c1 closes above ORH
              AND c0.high <= ORH (breakout happens through c1).
     Bearish: c0.low > c2.high AND c1 closes below ORL AND c0.low >= ORL.

  3. Retest: next candle that enters the FVG zone after the breakout.
     Invalidation: if price crosses fully back through the FVG zone, skip day.

  4. Engulfing confirmation:
     Long:  candle after retest closes above retest candle's high.
     Short: candle after retest closes below retest candle's low.

  5. Stop:   one tick (0.01 for SPY) beyond the retest candle's extreme.
     Target: fixed 3:1 risk-to-reward.
     Time stop: 16:00 EST.

Usage:
  python -m src.backtest.run_backtest_fvg --fetch          # needs POLYGON_API_KEY in .env
  python -m src.backtest.run_backtest_fvg --fetch --yf     # yfinance fallback (~60 days)
  python -m src.backtest.run_backtest_fvg                  # use cached data
"""

import argparse
import json
import pathlib
import ssl
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

_EASTERN    = ZoneInfo("America/New_York")
_CACHE_DIR  = pathlib.Path(__file__).parent / "cache"
_CACHE_FILE = _CACHE_DIR / "SPY_5m_fvg.json"

_TICK       = 0.01    # SPY minimum price increment
_RR         = 3.0     # fixed risk-to-reward
_OR_START_H = 9 + 30 / 60   # 09:30
_OR_END_H   = 9 + 35 / 60   # 09:35
_CLOSE_H    = 16.0           # 16:00
_SKIP_WD    = {5, 6}         # Saturday, Sunday


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _ehour(ts: pd.Timestamp) -> float:
    e = ts.astimezone(_EASTERN)
    return e.hour + e.minute / 60


def _edate(ts: pd.Timestamp) -> date:
    return ts.astimezone(_EASTERN).date()


def load_cache() -> list[dict] | None:
    if not _CACHE_FILE.exists():
        return None
    with open(_CACHE_FILE) as f:
        return json.load(f)


def save_cache(bars: list[dict]):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(bars, f)


def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


# ---------------------------------------------------------------------------
# Fetch: Polygon.io  (free tier — 5 req/min, 2-year history)
# ---------------------------------------------------------------------------

def fetch_polygon(api_key: str, years: int = 2) -> list[dict]:
    end   = datetime.now(_EASTERN).date()
    start = end - timedelta(days=years * 365 + 5)
    print(f"Fetching SPY 5m via Polygon.io  {start} → {end}…")

    bars: list[dict] = []
    seen: set = set()
    ctx = _ssl_ctx()

    # Polygon paginates via next_url; fetch until no more pages
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute"
        f"/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )

    page = 0
    while url:
        page += 1
        try:
            with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"\n  Polygon fetch error on page {page}: {e}")
            break

        status = data.get("status")
        if status == "ERROR":
            print(f"\n  Polygon API error: {data.get('error', data)}")
            break

        for r in data.get("results", []):
            t = int(r["t"])
            if t not in seen:
                seen.add(t)
                bars.append({
                    "t":      t,
                    "open":   float(r["o"]),
                    "high":   float(r["h"]),
                    "low":    float(r["l"]),
                    "close":  float(r["c"]),
                    "volume": float(r.get("v", 0)),
                })

        url = data.get("next_url")
        if url:
            url += f"&apiKey={api_key}"
            print(f"  Page {page}: {len(data.get('results', []))} bars, fetching next…", flush=True)
            time.sleep(12)   # free tier: 5 req/min → wait 12s between calls

    bars.sort(key=lambda b: b["t"])
    print(f"  Total: {len(bars)} bars fetched from Polygon.io")
    return bars


# ---------------------------------------------------------------------------
# Fetch: yfinance fallback (~60 days)
# ---------------------------------------------------------------------------

def fetch_yfinance(years: int = 2) -> list[dict]:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed — run: pip install yfinance")
        sys.exit(1)

    # yfinance only keeps 5m data for ~60 days; fetch what's available
    end   = datetime.now(_EASTERN).date()
    start = end - timedelta(days=58)
    print(f"Fetching SPY 5m via yfinance  {start} → {end}  (60-day limit applies)…")

    df = yf.download("SPY", start=str(start), end=str(end),
                     interval="5m", progress=False, auto_adjust=True)
    if df.empty:
        print("ERROR: no data returned from yfinance.")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    bars = []
    for ts, row in df.iterrows():
        if pd.isna(row["Close"]):
            continue
        bars.append({
            "t":      int(pd.Timestamp(ts).timestamp() * 1000),
            "open":   float(row["Open"]),
            "high":   float(row["High"]),
            "low":    float(row["Low"]),
            "close":  float(row["Close"]),
            "volume": float(row.get("Volume", 0)),
        })

    print(f"  {len(bars)} bars fetched (note: only ~60 days available via yfinance)")
    return bars


# ---------------------------------------------------------------------------
# Per-day simulation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    day:         date
    direction:   str
    orh:         float
    orl:         float
    fvg_lo:      float
    fvg_hi:      float
    entry_px:    float
    stop_px:     float
    target_px:   float
    exit_px:     float = 0.0
    exit_reason: str   = ""
    r_multiple:  float = 0.0


def _simulate_day(day: date, day_bars: list[dict]) -> Trade | None:
    bars = sorted(day_bars, key=lambda b: b["t"])

    # Locate the opening-range candle (starts at 09:30 EST)
    or_bar = None
    for b in bars:
        ts = pd.Timestamp(b["t"], unit="ms", tz="UTC")
        if abs(_ehour(ts) - _OR_START_H) < (1 / 60):
            or_bar = b
            break
    if or_bar is None:
        return None

    orh = or_bar["high"]
    orl = or_bar["low"]

    # Candles after the opening range, before session close
    post = [b for b in bars
            if _OR_END_H <= _ehour(pd.Timestamp(b["t"], unit="ms", tz="UTC")) < _CLOSE_H]

    if len(post) < 3:
        return None

    state      = "scanning"
    fvg_dir    = None
    fvg_lo     = fvg_hi = 0.0
    retest_bar = None
    entry_px   = stop_px = target_px = sl_dist = 0.0

    for i, bar in enumerate(post):
        h = _ehour(pd.Timestamp(bar["t"], unit="ms", tz="UTC"))

        if h >= _CLOSE_H:
            if state == "in_trade":
                exit_px = bar["open"]
                r = (exit_px - entry_px) / sl_dist if fvg_dir == "long" \
                    else (entry_px - exit_px) / sl_dist
                return Trade(day, fvg_dir, orh, orl, fvg_lo, fvg_hi,
                             entry_px, stop_px, target_px, exit_px, "time_stop", r)
            break

        if state == "scanning":
            if i < 2:
                continue
            c0, c1, c2 = post[i - 2], post[i - 1], post[i]

            # Bullish FVG breaking ORH
            if (c0["high"] < c2["low"]
                    and c1["close"] > orh
                    and c0["high"] <= orh):
                state   = "waiting_retest"
                fvg_dir = "long"
                fvg_lo  = c0["high"]
                fvg_hi  = c2["low"]

            # Bearish FVG breaking ORL
            elif (c0["low"] > c2["high"]
                    and c1["close"] < orl
                    and c0["low"] >= orl):
                state   = "waiting_retest"
                fvg_dir = "short"
                fvg_lo  = c2["high"]
                fvg_hi  = c0["low"]

        elif state == "waiting_retest":
            if fvg_dir == "long":
                if bar["low"] < fvg_lo:       # back below FVG entirely — invalid
                    return None
                if bar["low"] <= fvg_hi:      # touched FVG zone → retest
                    retest_bar = bar
                    state = "waiting_engulf"
            else:
                if bar["high"] > fvg_hi:      # back above FVG entirely — invalid
                    return None
                if bar["high"] >= fvg_lo:     # touched FVG zone → retest
                    retest_bar = bar
                    state = "waiting_engulf"

        elif state == "waiting_engulf":
            if fvg_dir == "long":
                if bar["close"] > retest_bar["high"]:   # bullish engulfing
                    entry_px  = bar["close"]
                    stop_px   = retest_bar["low"] - _TICK
                    sl_dist   = entry_px - stop_px
                    target_px = entry_px + _RR * sl_dist
                    state = "in_trade"
                elif bar["low"] < retest_bar["low"]:    # retest candle exceeded
                    return None
            else:
                if bar["close"] < retest_bar["low"]:    # bearish engulfing
                    entry_px  = bar["close"]
                    stop_px   = retest_bar["high"] + _TICK
                    sl_dist   = stop_px - entry_px
                    target_px = entry_px - _RR * sl_dist
                    state = "in_trade"
                elif bar["high"] > retest_bar["high"]:  # retest candle exceeded
                    return None

        elif state == "in_trade":
            if fvg_dir == "long":
                if bar["low"] <= stop_px:
                    return Trade(day, "long", orh, orl, fvg_lo, fvg_hi,
                                 entry_px, stop_px, target_px, stop_px, "sl", -1.0)
                if bar["high"] >= target_px:
                    return Trade(day, "long", orh, orl, fvg_lo, fvg_hi,
                                 entry_px, stop_px, target_px, target_px, "tp", _RR)
            else:
                if bar["high"] >= stop_px:
                    return Trade(day, "short", orh, orl, fvg_lo, fvg_hi,
                                 entry_px, stop_px, target_px, stop_px, "sl", -1.0)
                if bar["low"] <= target_px:
                    return Trade(day, "short", orh, orl, fvg_lo, fvg_hi,
                                 entry_px, stop_px, target_px, target_px, "tp", _RR)

    # End of day — close open position
    if state == "in_trade" and post:
        exit_px = post[-1]["close"]
        r = (exit_px - entry_px) / sl_dist if fvg_dir == "long" \
            else (entry_px - exit_px) / sl_dist
        return Trade(day, fvg_dir, orh, orl, fvg_lo, fvg_hi,
                     entry_px, stop_px, target_px, exit_px, "eod", r)

    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results(trades: list[Trade], source: str):
    if not trades:
        print("No trades found.")
        return

    wins    = [t for t in trades if t.r_multiple > 0]
    total_r = sum(t.r_multiple for t in trades)
    win_pct = 100 * len(wins) / len(trades)
    avg_r   = total_r / len(trades)
    be_wr   = 100 / (1 + _RR)   # 25% at 3:1

    peak = trough = max_dd = cumr = 0.0
    for t in trades:
        cumr += t.r_multiple
        if cumr > peak:
            peak = trough = cumr
        else:
            trough = min(trough, cumr)
            max_dd = min(max_dd, trough - peak)

    exits = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    go_nogo = "GO ✓" if (win_pct >= be_wr + 5 and len(trades) >= 20) else "NO-GO ✗"

    w = 82
    print(f"\n{'='*w}")
    print(f"  FVG ORB Backtest — SPY ({source})   |   5m candles   |   {_RR:.0f}:1 R:R")
    print(f"{'='*w}")
    print(f"  Trades      : {len(trades)}  ({len(wins)} wins / {len(trades)-len(wins)} losses)")
    print(f"  Win rate    : {win_pct:.1f}%  (breakeven {be_wr:.1f}%)")
    print(f"  Total R     : {total_r:+.2f}R")
    print(f"  Avg R/trade : {avg_r:+.3f}R")
    print(f"  Max drawdown: {max_dd:.1f}R")
    print(f"  Exits       : {' | '.join(f'{k}={v}' for k, v in sorted(exits.items()))}")
    print(f"  Verdict     : {go_nogo}")
    print(f"{'='*w}")

    by_year: dict[int, list[Trade]] = {}
    for t in trades:
        by_year.setdefault(t.day.year, []).append(t)
    print()
    for yr, yt in sorted(by_year.items()):
        yw   = [t for t in yt if t.r_multiple > 0]
        yr_r = sum(t.r_multiple for t in yt)
        print(f"  {yr}: {len(yt):>3} trades | win={100*len(yw)/len(yt):5.1f}% | totalR={yr_r:+.1f}R")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FVG ORB backtest — SPY")
    parser.add_argument("--fetch", action="store_true",
                        help="Re-fetch and cache data")
    parser.add_argument("--yf", action="store_true",
                        help="Use yfinance instead of Polygon.io (~60 days only)")
    parser.add_argument("--years", type=int, default=2,
                        help="Years of history to fetch via Polygon (default 2)")
    args = parser.parse_args()

    source = "yfinance" if args.yf else "Polygon.io"

    if args.fetch or not _CACHE_FILE.exists():
        if args.yf:
            bars = fetch_yfinance(args.years)
        else:
            import os
            api_key = os.environ.get("POLYGON_API_KEY", "")
            if not api_key:
                print(
                    "ERROR: POLYGON_API_KEY not set in .env\n"
                    "  → Sign up free at https://polygon.io (no credit card needed)\n"
                    "  → Add POLYGON_API_KEY=your_key to Sanket/.env\n"
                    "  → Or use --yf for yfinance fallback (~60 days only)"
                )
                sys.exit(1)
            bars = fetch_polygon(api_key, args.years)
        save_cache(bars)
    else:
        bars = load_cache()
        assert bars is not None
        print(f"Loaded {len(bars)} cached bars from {_CACHE_FILE}")
        # Infer source from data age
        if bars:
            oldest = datetime.fromtimestamp(bars[0]["t"] / 1000, tz=_EASTERN).date()
            days_back = (datetime.now(_EASTERN).date() - oldest).days
            source = "Polygon.io" if days_back > 90 else "yfinance"

    if not bars:
        print("ERROR: no data.")
        sys.exit(1)

    # Group bars by Eastern date, skip weekends
    days: dict[date, list] = {}
    for b in bars:
        ts = pd.Timestamp(b["t"], unit="ms", tz="UTC")
        d  = _edate(ts)
        if d.weekday() in _SKIP_WD:
            continue
        days.setdefault(d, []).append(b)

    trades: list[Trade] = []
    for day in sorted(days):
        t = _simulate_day(day, days[day])
        if t:
            trades.append(t)

    _print_results(trades, source)


if __name__ == "__main__":
    main()
