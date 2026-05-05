"""Fetch and cache historical candle data for backtesting.

Caches to data/candles/{asset}_{interval}.json so repeated runs don't re-fetch.
Run directly:  python -m src.backtest.fetch_history --assets BTC ETH SOL --years 2
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

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

BATCH = 5000


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


async def fetch_all(hl: HyperliquidAPI, asset: str, interval: str, years: int = 2) -> list:
    """Page backwards through Hyperliquid candles covering `years` of history."""
    interval_ms = INTERVAL_MS.get(interval, 300_000)
    end_ms = int(time.time() * 1000)
    target_start_ms = end_ms - years * 365 * 24 * 3600 * 1000
    batch_span_ms = BATCH * interval_ms

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
            print(f"  Warning: fetch error at cur_end={cur_end}: {e}")
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
        await asyncio.sleep(0.2)  # stay well under rate limits

    all_candles.sort(key=lambda c: c["t"])
    return all_candles


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
            candles = await fetch_all(hl, asset, interval, years)
            save_cache(asset, interval, candles)
            print(f"{len(candles)} bars saved to {p}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache Hyperliquid candle history")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--intervals", nargs="+", default=["5m", "4h"])
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = parser.parse_args()
    asyncio.run(main_async(args.assets, args.intervals, args.years, args.force))


if __name__ == "__main__":
    main()
