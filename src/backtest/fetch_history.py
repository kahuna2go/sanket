"""Fetch and cache historical candle data for backtesting.

Caches to data/candles/{asset}_{interval}.json so repeated runs don't re-fetch.

Hyperliquid keeps limited history for high-frequency intervals:
  5m  → ~17 days     15m → ~52 days
  1h  → ~7 months    4h  → ~16 months

For intervals with insufficient Hyperliquid history, falls back to Binance
spot (USDT pair) which has full history back to 2019. Price tracks closely
enough for signal validation.

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


async def fetch_all(hl: HyperliquidAPI, asset: str, interval: str, years: int) -> tuple[list, str]:
    """Fetch candle history, using Binance fallback when HL history is too short.

    Returns (candles, source) where source is 'hyperliquid' or 'binance'.
    """
    if _hl_has_enough(asset, interval, years):
        candles = await fetch_hl(hl, asset, interval, years)
        return candles, "hyperliquid"

    # Check if we need the fallback
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
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.intervals, args.years, args.force))


if __name__ == "__main__":
    main()
