"""Local technical indicator computation from OHLCV candle data.

Replaces external TAAPI dependency by computing indicators directly from
Hyperliquid candle snapshots. All functions accept lists of candle dicts
with keys: open, high, low, close, volume.
"""

from __future__ import annotations
import math


def _closes(candles: list[dict]) -> list[float]:
    return [c["close"] for c in candles]


def _highs(candles: list[dict]) -> list[float]:
    return [c["high"] for c in candles]


def _lows(candles: list[dict]) -> list[float]:
    return [c["low"] for c in candles]


def _volumes(candles: list[dict]) -> list[float]:
    return [c["volume"] for c in candles]


# ---------------------------------------------------------------------------
# EMA / SMA
# ---------------------------------------------------------------------------

def sma(values: list[float], period: int) -> list[float | None]:
    """Simple moving average. Returns list same length as values."""
    result: list[float | None] = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(values[i - period + 1: i + 1]) / period)
    return result


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average."""
    result: list[float | None] = []
    k = 2.0 / (period + 1)
    prev = None
    for i, v in enumerate(values):
        if i < period - 1:
            result.append(None)
        elif i == period - 1:
            prev = sum(values[:period]) / period
            result.append(prev)
        else:
            prev = v * k + prev * (1 - k)
            result.append(prev)
    return result


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(candles: list[dict], period: int = 14) -> list[float | None]:
    """Relative Strength Index using Wilder's smoothing."""
    closes = _closes(candles)
    if len(closes) < period + 1:
        return [None] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    result: list[float | None] = [None] * period  # first `period` values are None

    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(round(100.0 - (100.0 / (1.0 + rs)), 4))

    for i in range(period, len(deltas)):
        gain = max(deltas[i], 0)
        loss = abs(min(deltas[i], 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(round(100.0 - (100.0 / (1.0 + rs)), 4))

    return result


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(candles: list[dict], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line, signal line, and histogram.

    Returns:
        {"macd": [...], "signal": [...], "histogram": [...]}
    """
    closes = _closes(candles)
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd_line: list[float | None] = []
    for f, s in zip(ema_fast, ema_slow):
        if f is not None and s is not None:
            macd_line.append(round(f - s, 6))
        else:
            macd_line.append(None)

    # Signal line is EMA of MACD values (skip Nones at start)
    valid_macd = [v for v in macd_line if v is not None]
    signal_line_raw = ema(valid_macd, signal) if len(valid_macd) >= signal else [None] * len(valid_macd)

    # Reconstruct full-length signal
    signal_line: list[float | None] = [None] * (len(macd_line) - len(valid_macd))
    signal_line.extend(signal_line_raw)

    histogram: list[float | None] = []
    for m, s in zip(macd_line, signal_line):
        if m is not None and s is not None:
            histogram.append(round(m - s, 6))
        else:
            histogram.append(None)

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(candles: list[dict], period: int = 14) -> list[float | None]:
    """Average True Range."""
    if len(candles) < 2:
        return [None] * len(candles)

    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        prev_c = candles[i - 1]["close"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)

    result: list[float | None] = [None] * period  # first period values undefined
    if len(true_ranges) < period:
        return [None] * len(candles)

    avg = sum(true_ranges[:period]) / period
    result.append(round(avg, 6))

    for i in range(period, len(true_ranges)):
        avg = (avg * (period - 1) + true_ranges[i]) / period
        result.append(round(avg, 6))

    return result


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bbands(candles: list[dict], period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands: upper, middle (SMA), lower.

    Returns:
        {"upper": [...], "middle": [...], "lower": [...]}
    """
    closes = _closes(candles)
    middle = sma(closes, period)
    upper: list[float | None] = []
    lower: list[float | None] = []

    for i in range(len(closes)):
        if middle[i] is None:
            upper.append(None)
            lower.append(None)
        else:
            window = closes[i - period + 1: i + 1]
            mean = middle[i]
            variance = sum((x - mean) ** 2 for x in window) / period
            sd = math.sqrt(variance)
            upper.append(round(mean + std_dev * sd, 6))
            lower.append(round(mean - std_dev * sd, 6))

    return {"upper": upper, "middle": middle, "lower": lower}


def bbands_squeeze(candles: list[dict], period: int = 20, lookback: int = 8) -> dict:
    """Detect Bollinger Band squeeze: bands at their tightest point in `lookback` bars.

    Returns:
        {"squeeze": bool, "width": float | None, "expanding": bool}
    """
    data = bbands(candles, period)
    upper, middle, lower = data["upper"], data["middle"], data["lower"]

    widths: list[float | None] = []
    for u, m, lo in zip(upper, middle, lower):
        if u is None or m is None or lo is None or m == 0:
            widths.append(None)
        else:
            widths.append((u - lo) / m)

    valid = [w for w in widths if w is not None]
    if len(valid) < lookback + 1:
        return {"squeeze": False, "width": None, "expanding": False}

    current = valid[-1]
    window = valid[-lookback:]
    squeeze = current == min(window)
    expanding = current > valid[-2]

    return {"squeeze": squeeze, "width": round(current, 6), "expanding": expanding}


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------

def stoch_rsi(candles: list[dict], rsi_period: int = 14, stoch_period: int = 14,
              k_smooth: int = 3, d_smooth: int = 3) -> dict:
    """Stochastic RSI returning %K and %D lines.

    Returns:
        {"k": [...], "d": [...]}
    """
    rsi_vals = rsi(candles, rsi_period)
    valid_rsi = [v for v in rsi_vals if v is not None]

    stoch_k_raw: list[float | None] = []
    for i in range(len(valid_rsi)):
        if i < stoch_period - 1:
            stoch_k_raw.append(None)
        else:
            window = valid_rsi[i - stoch_period + 1: i + 1]
            lo = min(window)
            hi = max(window)
            if hi == lo:
                stoch_k_raw.append(50.0)
            else:
                stoch_k_raw.append(round((valid_rsi[i] - lo) / (hi - lo) * 100, 4))

    # Smooth %K
    valid_k = [v for v in stoch_k_raw if v is not None]
    k_line = sma(valid_k, k_smooth) if len(valid_k) >= k_smooth else [None] * len(valid_k)
    # %D is SMA of smoothed %K
    valid_k_smoothed = [v for v in k_line if v is not None]
    d_line = sma(valid_k_smoothed, d_smooth) if len(valid_k_smoothed) >= d_smooth else [None] * len(valid_k_smoothed)

    # Pad to original length
    pad = len(rsi_vals) - len(valid_rsi)
    full_k: list[float | None] = [None] * (pad + (len(valid_rsi) - len(valid_k)) + (len(valid_k) - len(k_line)))
    full_k.extend(k_line)
    full_d: list[float | None] = [None] * (len(rsi_vals) - len(d_line))
    full_d.extend(d_line)

    return {"k": full_k, "d": full_d}


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def adx(candles: list[dict], period: int = 14) -> list[float | None]:
    """Average Directional Index."""
    if len(candles) < period + 1:
        return [None] * len(candles)

    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []
    tr_list: list[float] = []

    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        prev_h = candles[i - 1]["high"]
        prev_l = candles[i - 1]["low"]
        prev_c = candles[i - 1]["close"]

        plus_dm = max(h - prev_h, 0) if (h - prev_h) > (prev_l - l) else 0
        minus_dm = max(prev_l - l, 0) if (prev_l - l) > (h - prev_h) else 0
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        tr_list.append(tr)

    if len(tr_list) < period:
        return [None] * len(candles)

    # Wilder smoothing
    atr_val = sum(tr_list[:period])
    plus_dm_smooth = sum(plus_dm_list[:period])
    minus_dm_smooth = sum(minus_dm_list[:period])

    dx_list: list[float] = []

    plus_di = (plus_dm_smooth / atr_val) * 100 if atr_val else 0
    minus_di = (minus_dm_smooth / atr_val) * 100 if atr_val else 0
    di_sum = plus_di + minus_di
    dx_list.append(abs(plus_di - minus_di) / di_sum * 100 if di_sum else 0)

    for i in range(period, len(tr_list)):
        atr_val = atr_val - (atr_val / period) + tr_list[i]
        plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm_list[i]
        minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm_list[i]

        plus_di = (plus_dm_smooth / atr_val) * 100 if atr_val else 0
        minus_di = (minus_dm_smooth / atr_val) * 100 if atr_val else 0
        di_sum = plus_di + minus_di
        dx_list.append(abs(plus_di - minus_di) / di_sum * 100 if di_sum else 0)

    # ADX is Wilder smoothed DX
    result: list[float | None] = [None] * (period * 2)
    if len(dx_list) >= period:
        adx_val = sum(dx_list[:period]) / period
        result.append(round(adx_val, 4))
        for i in range(period, len(dx_list)):
            adx_val = (adx_val * (period - 1) + dx_list[i]) / period
            result.append(round(adx_val, 4))

    # Pad to full candle length
    while len(result) < len(candles):
        result.insert(0, None)
    return result[:len(candles)]


# ---------------------------------------------------------------------------
# OBV (On-Balance Volume)
# ---------------------------------------------------------------------------

def obv(candles: list[dict]) -> list[float]:
    """On-Balance Volume."""
    closes = _closes(candles)
    volumes = _volumes(candles)
    result = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result.append(result[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            result.append(result[-1] - volumes[i])
        else:
            result.append(result[-1])
    return result


# ---------------------------------------------------------------------------
# VWAP (Volume Weighted Average Price)
# ---------------------------------------------------------------------------

def vwap(candles: list[dict]) -> list[float | None]:
    """Cumulative VWAP (resets not implemented — suitable for intraday)."""
    cum_vol = 0.0
    cum_tp_vol = 0.0
    result: list[float | None] = []
    for c in candles:
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        cum_vol += c["volume"]
        cum_tp_vol += tp * c["volume"]
        if cum_vol > 0:
            result.append(round(cum_tp_vol / cum_vol, 6))
        else:
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# Relative Volume
# ---------------------------------------------------------------------------

def rvol(candles: list[dict], period: int = 20) -> list[float | None]:
    """Relative Volume: current bar volume divided by N-bar average volume."""
    volumes = _volumes(candles)
    result: list[float | None] = []
    for i in range(len(volumes)):
        if i < period:
            result.append(None)
        else:
            avg = sum(volumes[i - period:i]) / period
            result.append(round(volumes[i] / avg, 4) if avg > 0 else None)
    return result


# ---------------------------------------------------------------------------
# Swing Structure (Market Structure: HH/HL or LH/LL)
# ---------------------------------------------------------------------------

def swing_structure(candles: list[dict], lookback: int = 3,
                    current_price: float | None = None) -> dict | None:
    """Detect market structure from swing highs and lows.

    Returns trend label, key levels, and pre-computed SL/TP levels, or None
    when there are insufficient candles or swings to determine structure.

    Buffers applied:
      SL: 0.4 × swing_range beyond the invalidation swing
      TP conservative: 0.25 × swing_range before the target level
      TP speculative:  127.2% Fibonacci extension (0.272 × swing_range beyond last swing)
    """
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    n = len(candles)

    if n < 2 * lookback + 1:
        return None

    if current_price is None:
        current_price = candles[-1]["close"]

    # Confirmed swing high: strictly greatest high within ±lookback bars
    # Confirmed swing low:  strictly lowest  low  within ±lookback bars
    swing_high_prices: list[float] = []
    swing_low_prices: list[float] = []

    for i in range(lookback, n - lookback):
        neighbors_h = [highs[j] for j in range(i - lookback, i + lookback + 1) if j != i]
        neighbors_l = [lows[j]  for j in range(i - lookback, i + lookback + 1) if j != i]
        if highs[i] > max(neighbors_h):
            swing_high_prices.append(highs[i])
        if lows[i] < min(neighbors_l):
            swing_low_prices.append(lows[i])

    if len(swing_high_prices) < 2 or len(swing_low_prices) < 2:
        return None

    # Trend: check last 3 confirmed swings for ascending/descending sequence
    def _direction(series: list[float]) -> str:
        recent = series[-3:] if len(series) >= 3 else series
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            return "ascending"
        if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            return "descending"
        return "mixed"

    h_dir = _direction(swing_high_prices)
    l_dir = _direction(swing_low_prices)

    if h_dir == "ascending" and l_dir == "ascending":
        trend = "HH_HL"
    elif h_dir == "descending" and l_dir == "descending":
        trend = "LH_LL"
    else:
        trend = "mixed"

    # Count consecutive swings confirming current trend
    swing_count = 0
    if trend == "HH_HL":
        for i in range(len(swing_high_prices) - 1, 0, -1):
            if swing_high_prices[i] > swing_high_prices[i - 1]:
                swing_count += 1
            else:
                break
    elif trend == "LH_LL":
        for i in range(len(swing_high_prices) - 1, 0, -1):
            if swing_high_prices[i] < swing_high_prices[i - 1]:
                swing_count += 1
            else:
                break

    last_swing_high = swing_high_prices[-1]
    last_swing_low = swing_low_prices[-1]
    swing_range = last_swing_high - last_swing_low

    if swing_range <= 0:
        return None

    # Nearest resistance above and support below current price
    above = sorted(h for h in swing_high_prices if h > current_price)
    below = sorted((l for l in swing_low_prices if l < current_price), reverse=True)
    next_resistance = above[0] if above else None
    next_support = below[0] if below else None

    # Break of structure relative to current price
    bos: str | None = None
    if current_price > last_swing_high:
        bos = "bullish"
    elif current_price < last_swing_low:
        bos = "bearish"

    sl_buf = 0.4 * swing_range
    tp_buf = 0.25 * swing_range
    fib = 0.272 * swing_range

    tp_cons_long  = (next_resistance - tp_buf) if next_resistance else (last_swing_high + fib)
    tp_cons_short = (next_support + tp_buf)    if next_support    else (last_swing_low  - fib)

    return {
        "trend": trend,
        "swing_count": swing_count,
        "last_swing_high": round(last_swing_high, 4),
        "last_swing_low": round(last_swing_low, 4),
        "swing_range": round(swing_range, 4),
        "next_resistance": round(next_resistance, 4) if next_resistance else None,
        "next_support": round(next_support, 4) if next_support else None,
        "bos": bos,
        "sl_long":  round(last_swing_low  - sl_buf, 4),
        "sl_short": round(last_swing_high + sl_buf, 4),
        "tp_conservative_long":  round(tp_cons_long,  4),
        "tp_conservative_short": round(tp_cons_short, 4),
        "tp_speculative_long":   round(last_swing_high + fib, 4),
        "tp_speculative_short":  round(last_swing_low  - fib, 4),
    }


# ---------------------------------------------------------------------------
# ZigZag + ZigZag Market Structure
# ---------------------------------------------------------------------------

def zigzag(candles: list[dict], deviation_pct: float = 3.0) -> list[dict]:
    """ZigZag indicator: alternating confirmed swing HIGH/LOW points.

    A new pivot is confirmed only when price reverses by at least deviation_pct%
    from the last pivot. The final point in the returned list is always the
    current forming (unconfirmed) pivot — drop it for structure analysis.

    Each point: {"type": "HIGH"|"LOW", "price": float, "index": int, "t": int}
    """
    if not candles:
        return []

    dev = deviation_pct / 100.0
    points: list[dict] = []

    last_price = candles[0]["high"]
    last_type  = "HIGH"
    last_idx   = 0

    for i in range(1, len(candles)):
        c = candles[i]
        if last_type == "HIGH":
            if c["high"] > last_price:
                last_price = c["high"]
                last_idx   = i
            elif c["low"] <= last_price * (1 - dev):
                points.append({"type": "HIGH", "price": last_price, "index": last_idx, "t": candles[last_idx]["t"]})
                last_price = c["low"]
                last_type  = "LOW"
                last_idx   = i
        else:
            if c["low"] < last_price:
                last_price = c["low"]
                last_idx   = i
            elif c["high"] >= last_price * (1 + dev):
                points.append({"type": "LOW", "price": last_price, "index": last_idx, "t": candles[last_idx]["t"]})
                last_price = c["high"]
                last_type  = "HIGH"
                last_idx   = i

    points.append({"type": last_type, "price": last_price, "index": last_idx, "t": candles[last_idx]["t"]})
    return points


def zz_structure(candles: list[dict], deviation_pct: float = 3.0,
                 current_price: float | None = None) -> dict | None:
    """Market structure from ZigZag pivots.

    Drop-in replacement for swing_structure — returns the same keys.
    Returns None when there are fewer than 4 confirmed pivots (2H + 2L minimum).
    swing_count counts consecutive confirming pairs from the end on both highs
    and lows; returns the minimum of the two streaks.
    """
    zz = zigzag(candles, deviation_pct)
    confirmed = zz[:-1]  # last point is still forming — exclude

    if len(confirmed) < 4:
        return None

    if current_price is None:
        current_price = candles[-1]["close"]

    highs = [p["price"] for p in confirmed if p["type"] == "HIGH"]
    lows  = [p["price"] for p in confirmed if p["type"] == "LOW"]

    if len(highs) < 2 or len(lows) < 2:
        return None

    def _direction(series: list[float]) -> str:
        recent = series[-3:] if len(series) >= 3 else series
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            return "ascending"
        if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            return "descending"
        return "mixed"

    h_dir = _direction(highs)
    l_dir = _direction(lows)

    if h_dir == "ascending" and l_dir == "ascending":
        trend = "HH_HL"
    elif h_dir == "descending" and l_dir == "descending":
        trend = "LH_LL"
    else:
        trend = "mixed"

    def _streak(series: list[float], ascending: bool) -> int:
        count = 0
        for i in range(len(series) - 1, 0, -1):
            if (ascending and series[i] > series[i - 1]) or \
               (not ascending and series[i] < series[i - 1]):
                count += 1
            else:
                break
        return count

    swing_count = 0
    if trend == "HH_HL":
        swing_count = min(_streak(highs, ascending=True), _streak(lows, ascending=True))
    elif trend == "LH_LL":
        swing_count = min(_streak(highs, ascending=False), _streak(lows, ascending=False))

    last_swing_high = highs[-1]
    last_swing_low  = lows[-1]
    swing_range = last_swing_high - last_swing_low

    if swing_range <= 0:
        return None

    above = sorted(h for h in highs if h > current_price)
    below = sorted((l for l in lows  if l < current_price), reverse=True)
    next_resistance = above[0] if above else None
    next_support    = below[0] if below else None

    bos: str | None = None
    if current_price > last_swing_high:
        bos = "bullish"
    elif current_price < last_swing_low:
        bos = "bearish"

    sl_buf = 0.4 * swing_range
    tp_buf = 0.25 * swing_range
    fib    = 0.272 * swing_range

    tp_cons_long  = (next_resistance - tp_buf) if next_resistance else (last_swing_high + fib)
    tp_cons_short = (next_support    + tp_buf) if next_support    else (last_swing_low  - fib)

    return {
        "trend":          trend,
        "swing_count":    swing_count,
        "last_swing_high": round(last_swing_high, 4),
        "last_swing_low":  round(last_swing_low,  4),
        "swing_range":     round(swing_range, 4),
        "next_resistance": round(next_resistance, 4) if next_resistance else None,
        "next_support":    round(next_support,    4) if next_support    else None,
        "bos": bos,
        "sl_long":  round(last_swing_low  - sl_buf, 4),
        "sl_short": round(last_swing_high + sl_buf, 4),
        "tp_conservative_long":  round(tp_cons_long,  4),
        "tp_conservative_short": round(tp_cons_short, 4),
        "tp_speculative_long":   round(last_swing_high + fib, 4),
        "tp_speculative_short":  round(last_swing_low  - fib, 4),
    }


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------

def volume_profile(candles: list[dict], bins: int = 50) -> dict | None:
    """Volume Profile: distributes each candle's volume proportionally across
    the price bins its high–low range covers.

    Returns POC (highest-volume price), VAH and VAL (the bounds of the Value
    Area — the contiguous region that contains 70 % of total volume, expanded
    outward from the POC), plus the raw value-area percentage achieved.
    Returns None when there are fewer than 2 candles or no volume.
    """
    if len(candles) < 2:
        return None

    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    price_min = min(lows)
    price_max = max(highs)

    if price_max <= price_min:
        return None

    bin_size = (price_max - price_min) / bins
    bin_vol  = [0.0] * bins

    for c in candles:
        candle_range = c["high"] - c["low"]
        vol = c["volume"]
        if candle_range <= 0:
            idx = min(int((c["close"] - price_min) / bin_size), bins - 1)
            bin_vol[idx] += vol
            continue
        for i in range(bins):
            bin_low  = price_min + i * bin_size
            bin_high = bin_low + bin_size
            overlap  = max(0.0, min(c["high"], bin_high) - max(c["low"], bin_low))
            if overlap > 0:
                bin_vol[i] += vol * (overlap / candle_range)

    total_vol = sum(bin_vol)
    if total_vol <= 0:
        return None

    poc_idx   = max(range(bins), key=lambda i: bin_vol[i])
    poc_price = price_min + (poc_idx + 0.5) * bin_size

    # Expand value area from POC until ≥ 70 % of volume is covered
    target = 0.70 * total_vol
    va_vol  = bin_vol[poc_idx]
    lo_idx  = poc_idx
    hi_idx  = poc_idx

    while va_vol < target:
        can_lo = lo_idx > 0
        can_hi = hi_idx < bins - 1
        if not can_lo and not can_hi:
            break
        vol_lo = bin_vol[lo_idx - 1] if can_lo else -1.0
        vol_hi = bin_vol[hi_idx + 1] if can_hi else -1.0
        if vol_hi >= vol_lo:
            hi_idx += 1
            va_vol += bin_vol[hi_idx]
        else:
            lo_idx -= 1
            va_vol += bin_vol[lo_idx]

    vah = price_min + (hi_idx + 1) * bin_size
    val = price_min + lo_idx * bin_size

    return {
        "poc": round(poc_price, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "value_area_pct": round(va_vol / total_vol * 100, 1),
    }


# ---------------------------------------------------------------------------
# High-level helper: compute all standard indicators for an asset
# ---------------------------------------------------------------------------

def compute_all(candles: list[dict], current_price: float | None = None) -> dict:
    """Compute a standard suite of indicators from candle data.

    Args:
        candles: List of OHLCV dicts from Hyperliquid.
        current_price: Live mid-price used for swing_structure level calculations.
                       Defaults to the last candle close when omitted.

    Returns:
        Dict with indicator names as keys and series/values as values.
    """
    if not candles:
        return {}

    closes = _closes(candles)
    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)
    rsi7_series = rsi(candles, 7)
    rsi14_series = rsi(candles, 14)
    macd_data = macd(candles)
    atr3_series = atr(candles, 3)
    atr14_series = atr(candles, 14)
    bbands_data = bbands(candles)
    adx_series = adx(candles)
    obv_series = obv(candles)
    vwap_series = vwap(candles)
    rvol_series  = rvol(candles)
    structure    = swing_structure(candles, current_price=current_price)
    vol_profile  = volume_profile(candles)

    return {
        "ema20": ema20_series,
        "ema50": ema50_series,
        "rsi7": rsi7_series,
        "rsi14": rsi14_series,
        "macd": macd_data["macd"],
        "macd_signal": macd_data["signal"],
        "macd_histogram": macd_data["histogram"],
        "atr3": atr3_series,
        "atr14": atr14_series,
        "bbands_upper": bbands_data["upper"],
        "bbands_middle": bbands_data["middle"],
        "bbands_lower": bbands_data["lower"],
        "adx": adx_series,
        "obv": obv_series,
        "vwap": vwap_series,
        "rvol": rvol_series,
        "swing_structure": structure,
        "volume_profile": vol_profile,
    }


def last_n(series: list, n: int = 10) -> list:
    """Return the last ``n`` non-None values from a series."""
    valid = [v for v in series if v is not None]
    return valid[-n:]


def latest(series: list):
    """Return the last non-None value from a series, or None."""
    for v in reversed(series):
        if v is not None:
            return v
    return None
