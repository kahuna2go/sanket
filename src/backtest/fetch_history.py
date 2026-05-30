"""Fetch and cache historical candle data for backtesting.

Caches to data/candles/{asset}_{interval}.json so repeated runs don't re-fetch.

Hyperliquid keeps limited history for high-frequency intervals:
  5m  → ~17 days     15m → ~52 days
  1h  → ~7 months    4h  → ~16 months

For intervals with insufficient Hyperliquid history, falls back to Binance
spot (USDT pair) which has full history back to 2019. Price tracks closely
enough for signal validation.

For HIP-3 assets (xyz:SP500, xyz:GOLD, etc.) with even less Hyperliquid
history, falls back to yfinance (SPY, GC=F, etc.). yfinance returns up to
~60 days of 5m data per request; the fetcher chunks into 50-day windows to
maximise coverage. 4h data is fetched as 1h and resampled.

Run directly:  python -m src.backtest.fetch_history --assets BTC ETH SOL --years 2
"""

import argparse
import asyncio
import json
import pathlib
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.trading.hyperliquid_api import HyperliquidAPI

CACHE_DIR = pathlib.Path("data/candles")

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000,
}

# Hyperliquid retains roughly this many bars per interval before history drops off
HL_RETENTION_BARS = {
    "1m": 1_000, "5m": 5_000, "15m": 5_000,
    "1h": 5_000, "4h": 5_000, "1d": 5_000,
}

# Binance symbol map for fallback
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "ARB": "ARBUSDT",
    "SUI": "SUIUSDT",
}

# Binance uses different interval strings for some TFs
BINANCE_INTERVAL = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "12h": "12h",
    "1d": "1d",
}

# yfinance tickers for HIP-3 assets that have no Binance equivalent
YFINANCE_SYMBOLS = {
    "xyz:SP500":  "SPY",   # SPDR S&P 500 ETF — liquid intraday proxy
    "xyz:GOLD":   "GC=F",  # Gold futures
    "xyz:OIL":    "CL=F",  # WTI crude futures
    "xyz:SILVER": "SI=F",  # Silver futures
}

# TradingView (tvdatafeed) symbols for HIP-3 assets — deeper 5m history than yfinance
TVDATAFEED_SYMBOLS = {
    "xyz:SP500": ("SPY", "AMEX"),
}

HL_BATCH = 5000
BINANCE_BATCH = 1000  # Binance max per request


def cache_path(asset: str, interval: str) -> pathlib.Path:
    safe = asset.replace(":", "_")
    return CACHE_DIR / f"{safe}_{interval}.json"


def load_cache(asset: str, interval: str) -> list | None:
    p = cache_path(asset, interval)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_cache(asset: str, interval: str, candles: list):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path(asset, interval), "w") as f:
        json.dump(candles, f)


def _hl_has_enough(asset: str, interval: str, years: int) -> bool:
    """Return True if Hyperliquid is likely to have enough history."""
    if ":" in asset:
        return False  # HIP-3 assets often have even less history
    interval_ms = INTERVAL_MS.get(interval, 300_000)
    needed_bars = int(years * 365 * 24 * 3600 * 1000 / interval_ms)
    retention = HL_RETENTION_BARS.get(interval, 5_000)
    return retention >= needed_bars * 0.9


async def fetch_hl(hl: HyperliquidAPI, asset: str, interval: str, years: int) -> list:
    """Page backwards through Hyperliquid candles."""
    interval_ms = INTERVAL_MS.get(interval, 300_000)
    end_ms = int(time.time() * 1000)
    target_start_ms = end_ms - years * 365 * 24 * 3600 * 1000
    batch_span_ms = HL_BATCH * interval_ms

    all_candles: list = []
    seen: set = set()
    cur_end = end_ms

    while cur_end > target_start_ms:
        cur_start = max(cur_end - batch_span_ms, target_start_ms)
        try:
            if ":" in asset:
                raw = await hl._retry(
                    lambda cs=cur_start, ce=cur_end: hl.info.post("/info", {
                        "type": "candleSnapshot",
                        "req": {"coin": asset, "interval": interval,
                                "startTime": cs, "endTime": ce},
                    })
                )
            else:
                raw = await hl._retry(
                    lambda cs=cur_start, ce=cur_end: hl.info.candles_snapshot(asset, interval, cs, ce)
                )
        except Exception as e:
            print(f"\n  HL fetch error: {e}", end="")
            break

        if not raw:
            break

        for c in raw:
            t = c.get("t")
            if t not in seen:
                seen.add(t)
                all_candles.append({
                    "t": t,
                    "open": float(c.get("o", 0)),
                    "high": float(c.get("h", 0)),
                    "low": float(c.get("l", 0)),
                    "close": float(c.get("c", 0)),
                    "volume": float(c.get("v", 0)),
                })

        cur_end = cur_start
        await asyncio.sleep(0.15)

    all_candles.sort(key=lambda c: c["t"])
    return all_candles


def _ssl_ctx() -> ssl.SSLContext:
    """Return an SSL context with certifi certs (fixes macOS LibreSSL issue)."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return ctx


def fetch_binance(asset: str, interval: str, years: int) -> list:
    """Fetch candles from Binance public API (no key required)."""
    symbol = BINANCE_SYMBOLS.get(asset)
    if not symbol:
        return []

    iv = BINANCE_INTERVAL.get(interval)
    if not iv:
        return []

    end_ms = int(time.time() * 1000)
    target_start_ms = end_ms - years * 365 * 24 * 3600 * 1000
    interval_ms = INTERVAL_MS.get(interval, 300_000)
    batch_span_ms = BINANCE_BATCH * interval_ms
    ctx = _ssl_ctx()

    all_candles: list = []
    seen: set = set()
    cur_start = target_start_ms

    while cur_start < end_ms:
        cur_end = min(cur_start + batch_span_ms, end_ms)
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval={iv}"
            f"&startTime={cur_start}&endTime={cur_end}&limit={BINANCE_BATCH}"
        )
        try:
            with urllib.request.urlopen(url, timeout=15, context=ctx) as resp:
                raw = json.loads(resp.read())
        except Exception as e:
            print(f"\n  Binance fetch error: {e}", end="")
            break

        if not raw:
            break

        for row in raw:
            t = int(row[0])
            if t not in seen:
                seen.add(t)
                all_candles.append({
                    "t": t,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                })

        last_t = int(raw[-1][0])
        if last_t <= cur_start:
            break
        cur_start = last_t + interval_ms
        time.sleep(0.1)  # Binance rate limit is generous but be polite

    all_candles.sort(key=lambda c: c["t"])
    return all_candles


def fetch_yfinance(asset: str, interval: str, years: int) -> list:
    """Fetch candles from yfinance.

    5m: yfinance hard-limits to the last 60 days regardless of the requested
    range. The fetcher chunks requests into 50-day windows to maximise coverage
    within that window; older chunks silently return no data.

    4h: fetched as 1h (yfinance max 730 days) and resampled.
    """
    try:
        import yfinance as yf
        import pandas as pd
        import io, contextlib
    except ImportError:
        print("\n  yfinance/pandas not installed — skipping", end="")
        return []

    symbol = YFINANCE_SYMBOLS.get(asset)
    if not symbol:
        return []

    end_dt = datetime.now(timezone.utc)

    def _to_candles(df, ts_col=None) -> list:
        rows = []
        for ts, row in df.iterrows():
            rows.append({
                "t":      int(ts.timestamp() * 1000),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return rows

    # 4h: yfinance 1h max is 730 days; resample to 4h
    if interval == "4h":
        start_dt = end_dt - timedelta(days=min(int(365 * years), 729))
        try:
            ticker = yf.Ticker(symbol)
            with contextlib.redirect_stderr(io.StringIO()):
                df = ticker.history(start=start_dt.date(), end=end_dt.date(), interval="1h", auto_adjust=True)
            if df.empty:
                return []
            df.index = pd.to_datetime(df.index, utc=True)
            df4 = df[["Open", "High", "Low", "Close", "Volume"]].resample("4h", closed="left", label="left").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna(subset=["Open"])
            candles = _to_candles(df4)
            candles.sort(key=lambda c: c["t"])
            return candles
        except Exception as e:
            print(f"\n  yfinance 4h fetch error: {e}", end="")
            return []

    if interval not in ("1m", "2m", "5m", "15m", "30m", "1h"):
        return []

    # 5m / other intraday: chunk into 50-day windows
    # Only the last 60 days will actually return data; older chunks come back empty.
    start_dt = end_dt - timedelta(days=int(365 * years))
    all_candles: list = []
    seen: set = set()
    chunk_days = 50
    cur_start = start_dt

    while cur_start < end_dt:
        cur_end = min(cur_start + timedelta(days=chunk_days), end_dt)
        try:
            ticker = yf.Ticker(symbol)
            with contextlib.redirect_stderr(io.StringIO()):
                df = ticker.history(
                    start=cur_start.date(),
                    end=cur_end.date(),
                    interval=interval,
                    auto_adjust=True,
                )
            if not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                for row in _to_candles(df):
                    if row["t"] not in seen:
                        seen.add(row["t"])
                        all_candles.append(row)
        except Exception:
            pass
        cur_start = cur_end
        time.sleep(0.2)

    all_candles.sort(key=lambda c: c["t"])
    return all_candles


def fetch_tvdatafeed(asset: str, interval: str) -> list:
    """Fetch candles via tvdatafeed (TradingView WebSocket scraper).

    Returns up to 5000 bars of 5m data (~3 months) without login.
    4h is fetched as 1h and resampled, same as yfinance path.
    SSL is patched to use certifi so it works on macOS.
    """
    sym = TVDATAFEED_SYMBOLS.get(asset)
    if not sym:
        return []

    try:
        import ssl, certifi, websocket, pandas as pd
        from tvDatafeed import TvDatafeed, Interval as TvInterval

        # Patch websocket SSL to use certifi CA bundle (required on macOS)
        _orig = websocket.create_connection
        def _patched(*args, **kwargs):
            kwargs.setdefault("sslopt", {})["ca_certs"] = certifi.where()
            return _orig(*args, **kwargs)
        websocket.create_connection = _patched
        import tvDatafeed.main as _tvm; _tvm.create_connection = _patched

    except ImportError:
        print("\n  tvdatafeed/pandas not installed — skipping", end="")
        return []

    ticker, exchange = sym
    tv_interval_map = {
        "5m":  TvInterval.in_5_minute,
        "1h":  TvInterval.in_1_hour,
        "4h":  TvInterval.in_4_hour,
    }

    import os
    username = os.environ.get("TV_USERNAME")
    password = os.environ.get("TV_PASSWORD")

    try:
        import io, contextlib, logging
        with contextlib.redirect_stderr(io.StringIO()):
            logging.disable(logging.WARNING)
            if username and password:
                tv = TvDatafeed(username=username, password=password)
            else:
                tv = TvDatafeed()
            logging.disable(logging.NOTSET)

        if interval == "4h":
            df = tv.get_hist(ticker, exchange, interval=TvInterval.in_1_hour, n_bars=20000)
            if df is None or df.empty:
                return []
            df4 = df[["open", "high", "low", "close", "volume"]].rename(
                columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
            ).resample("4h", closed="left", label="left").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna(subset=["Open"])
            rows = []
            for ts, row in df4.iterrows():
                rows.append({"t": int(ts.timestamp() * 1000), "open": float(row["Open"]),
                             "high": float(row["High"]), "low": float(row["Low"]),
                             "close": float(row["Close"]), "volume": float(row["Volume"])})
            rows.sort(key=lambda c: c["t"])
            return rows

        tv_iv = tv_interval_map.get(interval)
        if tv_iv is None:
            return []
        df = tv.get_hist(ticker, exchange, interval=tv_iv, n_bars=20000)
        if df is None or df.empty:
            return []
        rows = []
        for ts, row in df.iterrows():
            rows.append({"t": int(ts.timestamp() * 1000), "open": float(row["open"]),
                         "high": float(row["high"]), "low": float(row["low"]),
                         "close": float(row["close"]), "volume": float(row["volume"])})
        rows.sort(key=lambda c: c["t"])
        return rows

    except Exception as e:
        print(f"\n  tvdatafeed fetch error: {e}", end="")
        return []


def import_tv_csv(csv_path: str, interval: str) -> list:
    """Parse a TradingView 'Export chart data' CSV into the standard candle format.

    TradingView exports either Unix timestamps or datetime strings in the first
    column.  Datetime strings are parsed with timezone info when present; bare
    datetimes (no offset) are assumed UTC.  The 4h interval is accepted directly
    or resampled from 1h rows if needed.

    Expected columns (case-insensitive):
      time | open | high | low | close | volume  (Volume optional)
    """
    try:
        import csv as _csv
        from datetime import timezone as _tz
        from zoneinfo import ZoneInfo as _ZI
    except ImportError:
        print("csv module not available"); return []

    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        # Normalise header names
        col_map = {}
        for h in (reader.fieldnames or []):
            hn = h.strip().lower()
            if hn in ("time", "date", "datetime", "timestamp"):
                col_map["time"] = h
            elif hn == "open":
                col_map["open"] = h
            elif hn == "high":
                col_map["high"] = h
            elif hn == "low":
                col_map["low"] = h
            elif hn in ("close", "price"):
                col_map["close"] = h
            elif hn in ("volume", "vol"):
                col_map["volume"] = h

        if not all(k in col_map for k in ("time", "open", "high", "low", "close")):
            raise ValueError(f"Missing required columns. Found: {reader.fieldnames}")

        for row in reader:
            raw_t = row[col_map["time"]].strip()
            # Parse timestamp
            if raw_t.lstrip("-").isdigit():
                t_sec = int(raw_t)
                # TV sometimes exports in seconds, sometimes milliseconds
                t_ms = t_sec * 1000 if t_sec < 1e12 else t_sec
            else:
                # Try ISO / common date formats; use UTC if no tz info
                from dateutil import parser as _dp
                try:
                    dt = _dp.parse(raw_t)
                except Exception:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                t_ms = int(dt.timestamp() * 1000)

            try:
                rows.append({
                    "t":      t_ms,
                    "open":   float(row[col_map["open"]]),
                    "high":   float(row[col_map["high"]]),
                    "low":    float(row[col_map["low"]]),
                    "close":  float(row[col_map["close"]]),
                    "volume": float(row[col_map["volume"]]) if "volume" in col_map else 0.0,
                })
            except (ValueError, KeyError):
                continue

    rows.sort(key=lambda c: c["t"])
    # Deduplicate by timestamp
    seen: set = set()
    deduped = []
    for r in rows:
        if r["t"] not in seen:
            seen.add(r["t"])
            deduped.append(r)
    return deduped


async def fetch_all(hl: HyperliquidAPI, asset: str, interval: str, years: int) -> tuple[list, str]:
    """Fetch candle history, preferring yfinance for HIP-3 assets and Binance for crypto.

    Returns (candles, source) where source is 'hyperliquid', 'binance', or 'yfinance'.
    """
    # HIP-3 assets (xyz:*): try tvdatafeed first (deeper 5m history), then yfinance
    if ":" in asset and asset in TVDATAFEED_SYMBOLS:
        candles = fetch_tvdatafeed(asset, interval)
        if candles:
            return candles, "tvdatafeed"
        print(f"\n  tvdatafeed returned no data for {asset} {interval}, falling back to yfinance", end="")

    if ":" in asset and asset in YFINANCE_SYMBOLS:
        candles = fetch_yfinance(asset, interval, years)
        if candles:
            return candles, "yfinance"
        # yfinance failed — fall through to HL
        print(f"\n  yfinance returned no data for {asset} {interval}, falling back to Hyperliquid", end="")

    if _hl_has_enough(asset, interval, years):
        candles = await fetch_hl(hl, asset, interval, years)
        return candles, "hyperliquid"

    # Crypto assets with long history on Binance
    if asset in BINANCE_SYMBOLS and interval in BINANCE_INTERVAL and ":" not in asset:
        candles = fetch_binance(asset, interval, years)
        if candles:
            return candles, "binance"

    # Fall back to whatever HL has
    candles = await fetch_hl(hl, asset, interval, years)
    return candles, "hyperliquid (limited)"


async def main_async(assets: list[str], intervals: list[str], years: int, force: bool):
    hl = HyperliquidAPI()
    await hl.get_meta_and_ctxs()

    for asset in assets:
        for interval in intervals:
            p = cache_path(asset, interval)
            if p.exists() and not force:
                existing = load_cache(asset, interval)
                print(f"{asset} {interval}: cached ({len(existing)} bars) — skipping (use --force to re-fetch)")
                continue

            print(f"Fetching {asset} {interval} ({years}y)…", end=" ", flush=True)
            candles, source = await fetch_all(hl, asset, interval, years)
            save_cache(asset, interval, candles)
            print(f"{len(candles)} bars [{source}] → {p}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache candle history for backtesting")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--intervals", nargs="+", default=["5m", "4h"])
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    parser.add_argument(
        "--import-csv", metavar="FILE",
        help=(
            "Import a TradingView 'Export chart data' CSV into the cache. "
            "Requires --assets (single asset) and --intervals (single interval). "
            "Example: --import-csv ~/Downloads/SPX_5m.csv --assets xyz:SP500 --intervals 5m"
        ),
    )
    args = parser.parse_args()

    if args.import_csv:
        if len(args.assets) != 1 or len(args.intervals) != 1:
            print("--import-csv requires exactly one --assets and one --intervals value")
            return
        asset, interval = args.assets[0], args.intervals[0]
        print(f"Importing {args.import_csv} → {asset} {interval}…", end=" ", flush=True)
        candles = import_tv_csv(args.import_csv, interval)
        if not candles:
            print("ERROR: no candles parsed — check CSV format")
            return
        save_cache(asset, interval, candles)
        from datetime import datetime, timezone
        oldest = datetime.fromtimestamp(candles[0]["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        newest = datetime.fromtimestamp(candles[-1]["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"{len(candles)} bars | {oldest} → {newest} → {cache_path(asset, interval)}")
        return

    asyncio.run(main_async(args.assets, args.intervals, args.years, args.force))


if __name__ == "__main__":
    main()
