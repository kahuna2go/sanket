"""Macro-level market filters fetched concurrently at the top of each trading cycle.

Returns a structured dict used to gate or modulate new position opens.
All fetches run via asyncio.gather — individual failures return safe defaults
so the trading loop is never blocked.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta


async def get_macro_context() -> dict:
    """Fetch all macro signals concurrently. Safe defaults on any failure."""
    fear_greed_result, high_impact_result, dxy_result = await asyncio.gather(
        _fetch_fear_greed(),
        _fetch_high_impact_event(),
        _fetch_dxy_trend(),
        return_exceptions=True,
    )

    fear_greed = fear_greed_result if isinstance(fear_greed_result, int) else 50
    high_impact = high_impact_result if isinstance(high_impact_result, bool) else False
    dxy_rising = dxy_result if isinstance(dxy_result, bool) else False

    if isinstance(fear_greed_result, Exception):
        logging.warning("Macro: Fear & Greed fetch failed: %s", fear_greed_result)
    if isinstance(high_impact_result, Exception):
        logging.warning("Macro: Finnhub calendar fetch failed: %s", high_impact_result)
    if isinstance(dxy_result, Exception):
        logging.warning("Macro: DXY fetch failed: %s", dxy_result)

    min_thesis = 4 if (fear_greed < 20 or fear_greed > 80) else 3

    return {
        "fear_greed": fear_greed,
        "dxy_rising": dxy_rising,
        "high_impact_event_imminent": high_impact,
        "block_new_opens": high_impact,
        "min_thesis_strength_to_open": min_thesis,
        "reduce_long_allocation": dxy_rising,
    }


async def _fetch_fear_greed() -> int:
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get("https://api.alternative.me/fng/") as resp:
            data = await resp.json(content_type=None)
            return int(data["data"][0]["value"])


async def _fetch_high_impact_event() -> bool:
    import aiohttp
    from src.config_loader import CONFIG
    key = CONFIG.get("finnhub_api_key")
    if not key:
        return False

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(minutes=60)
    from_dt = now.strftime("%Y-%m-%d")
    to_dt = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/calendar/economic"
        f"?from={from_dt}&to={to_dt}&token={key}"
    )

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            data = await resp.json(content_type=None)

    for event in data.get("economicCalendar", []):
        if event.get("impact") != "high":
            continue
        raw_time = event.get("time") or event.get("datetime") or ""
        try:
            event_dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if now <= event_dt <= cutoff:
                return True
        except (ValueError, TypeError):
            continue
    return False


async def _fetch_dxy_trend() -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _dxy_sync)


def _dxy_sync() -> bool:
    try:
        import yfinance as yf
    except ImportError:
        logging.warning("Macro: yfinance not installed — DXY check skipped")
        return False

    hist = yf.Ticker("DX-Y.NYB").history(period="15d")
    if hist.empty or len(hist) < 6:
        return False

    closes = hist["Close"].tolist()
    k = 2.0 / (5 + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return closes[-1] > ema
