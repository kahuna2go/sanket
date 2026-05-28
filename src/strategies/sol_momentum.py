"""SOL Momentum Scalper — live strategy.

Entry: EMA9/21 crossover on 15m candles + RSI(14) confirmation + session filter.
  Long:  EMA9 crosses above EMA21  AND  RSI > 55
  Short: EMA9 crosses below EMA21  AND  RSI < 45
  Session filter: Asia (01:00–07:00 Vienna) + London (08:30–11:30) + NY open (16:00–20:00)

Exit: TP = entry ± 2×ATR14, SL = entry ∓ 1×ATR14  (2:1 R:R)
  Both placed on exchange as reduce-only trigger orders immediately after entry,
  so the position is protected even if the process restarts.

Risk: risk_pct × account_value ÷ ATR14 = SOL contract size (default 1.5%)
Leverage: 3x (set on exchange at startup)

Usage:
  STRATEGY=sol_momentum DRY_RUN=true python -m src.main --assets SOL --interval 15m
"""

import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.trading.hyperliquid_api import HyperliquidAPI

_VIENNA_TZ = ZoneInfo("Europe/Vienna")

# Asia open + London open + NY open (Vienna time) — backtested GO ✓ on SOL 2024–2026
_SESSION_WINDOWS = [(1.0, 7.0), (8 + 30 / 60, 11.5), (16.0, 20.0)]


def _in_session(ts_ms: int) -> bool:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    hf = dt.hour + dt.minute / 60
    return any(start <= hf < end for start, end in _SESSION_WINDOWS)


# ---------------------------------------------------------------------------
# Indicator helpers (duplicated from backtest so this module is self-contained)
# ---------------------------------------------------------------------------

def _ema(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        j = i - 1
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[j]) / period
            avg_loss = (avg_loss * (period - 1) + losses[j]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        result[i] = 100 - 100 / (1 + rs)
    return result


def _atr(candles: list[dict], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    if len(candles) < period + 1:
        return result
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_val = sum(trs[:period]) / period
    result[period] = atr_val
    for i in range(period + 1, len(candles)):
        atr_val = (atr_val * (period - 1) + trs[i - 1]) / period
        result[i] = atr_val
    return result


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class SolMomentum:
    ASSET         = "SOL"
    CANDLE_COUNT  = 80    # 80 completed 15m bars (~20h) — enough indicator warmup
    LOOP_INTERVAL = 60    # seconds between cycles
    LEVERAGE      = 3

    def __init__(self, hl: HyperliquidAPI, risk_pct: float = 0.015, dry_run: bool = False):
        self.hl       = hl
        self.risk_pct = risk_pct
        self.dry_run  = dry_run

        # Active trade state
        self._in_trade     = False
        self._direction: str | None = None
        self._entry_price  = 0.0
        self._tp_price     = 0.0
        self._sl_price     = 0.0
        self._size         = 0.0
        self._tp_oid       = None
        self._sl_oid       = None

        self._last_bar_ts  = 0   # timestamp of last processed completed bar

        # Session stats
        self._stats = {"trades": 0, "wins": 0, "losses": 0, "total_r": 0.0}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        logging.info(
            "[SolMomentum] Starting. risk=%.1f%% leverage=%dx dry_run=%s",
            self.risk_pct * 100, self.LEVERAGE, self.dry_run,
        )
        await self.hl.set_leverage(self.ASSET, self.LEVERAGE)
        while True:
            try:
                await self._cycle()
            except Exception as e:
                logging.error("[SolMomentum] cycle error: %s", e, exc_info=True)
            await asyncio.sleep(self.LOOP_INTERVAL)

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    async def _cycle(self):
        # Fetch and drop the last (still-forming) bar
        raw = await self.hl.get_candles(self.ASSET, "15m", self.CANDLE_COUNT)
        candles = raw[:-1] if len(raw) >= 2 else raw
        if len(candles) < 25:
            return

        latest_ts = candles[-1]["t"]

        if self._in_trade:
            await self._manage_trade()
            return

        # Only act on a freshly closed bar
        if latest_ts <= self._last_bar_ts:
            return
        self._last_bar_ts = latest_ts

        await self._check_signal(candles)

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    async def _manage_trade(self):
        """Detect if our position closed (TP or SL hit) and record outcome."""
        state = await self.hl.get_user_state()
        pos   = next((p for p in state["positions"] if p.get("coin") == self.ASSET), None)
        szi   = float(pos.get("szi", 0)) if pos else 0.0

        if abs(szi) > 0.001:
            return  # still open

        # Position gone — check which bracket order is still resting to infer outcome
        open_orders = await self.hl.get_open_orders()
        open_oids   = {o.get("oid") for o in open_orders
                       if self.hl._coin_matches(o.get("coin", ""), self.ASSET)}

        tp_resting = self._tp_oid is not None and self._tp_oid in open_oids
        sl_resting = self._sl_oid is not None and self._sl_oid in open_oids

        if sl_resting and not tp_resting:
            # TP filled → win; cancel hanging SL
            outcome, pnl_r = "win", TP_R
            if not self.dry_run:
                await self.hl.cancel_order(self.ASSET, self._sl_oid)
        elif tp_resting and not sl_resting:
            # SL filled → loss; cancel hanging TP
            outcome, pnl_r = "loss", -SL_R
            if not self.dry_run:
                await self.hl.cancel_order(self.ASSET, self._tp_oid)
        else:
            outcome, pnl_r = "unknown", 0.0
            if not self.dry_run:
                await self.hl.cancel_all_orders(self.ASSET)

        self._stats["trades"] += 1
        self._stats["total_r"] += pnl_r
        if pnl_r > 0:
            self._stats["wins"] += 1
        elif pnl_r < 0:
            self._stats["losses"] += 1

        logging.info(
            "[SolMomentum] CLOSED %s | %s | entry=%.4f TP=%.4f SL=%.4f | "
            "pnl=%.1fR | totals: %d trades / %d W / %d L / %.1fR",
            outcome.upper(), self._direction,
            self._entry_price, self._tp_price, self._sl_price, pnl_r,
            self._stats["trades"], self._stats["wins"],
            self._stats["losses"], self._stats["total_r"],
        )

        self._in_trade    = False
        self._direction   = None
        self._tp_oid      = None
        self._sl_oid      = None

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    async def _check_signal(self, candles: list[dict]):
        closes = [c["close"] for c in candles]
        ema9   = _ema(closes, 9)
        ema21  = _ema(closes, 21)
        rsi14  = _rsi(closes, 14)
        atr14  = _atr(candles, 14)

        i = len(candles) - 1
        e9_prev,  e9_cur  = ema9[i - 1],  ema9[i]
        e21_prev, e21_cur = ema21[i - 1], ema21[i]
        rsi_cur = rsi14[i]
        atr_cur = atr14[i]

        if any(v is None for v in [e9_prev, e9_cur, e21_prev, e21_cur, rsi_cur, atr_cur]):
            return

        in_session  = _in_session(candles[-1]["t"])
        long_cross  = e9_prev < e21_prev and e9_cur >= e21_cur
        short_cross = e9_prev > e21_prev and e9_cur <= e21_cur
        trend       = "▲" if e9_cur > e21_cur else "▼"

        logging.info(
            "[SolMomentum] EMA9=%.2f EMA21=%.2f RSI=%.1f ATR=%.2f %s session=%s",
            e9_cur, e21_cur, rsi_cur, atr_cur, trend, "ON" if in_session else "off",
        )

        if not long_cross and not short_cross:
            return

        if not in_session:
            logging.info("[SolMomentum] crossover outside session — skipped")
            return

        if long_cross and rsi_cur <= 55:
            logging.info("[SolMomentum] long cross RSI=%.1f ≤ 55 — skipped", rsi_cur)
            return
        if short_cross and rsi_cur >= 45:
            logging.info("[SolMomentum] short cross RSI=%.1f ≥ 45 — skipped", rsi_cur)
            return

        direction = "long" if long_cross else "short"

        # Size from risk budget
        state = await self.hl.get_user_state()
        total_value = state["total_value"]
        if total_value <= 0:
            logging.warning("[SolMomentum] account value 0 — skipping")
            return

        risk_usd = total_value * self.risk_pct
        size_sol = self.hl.round_size(self.ASSET, risk_usd / atr_cur)
        if size_sol <= 0:
            return

        entry = await self.hl.get_current_price(self.ASSET)
        if entry <= 0:
            return

        if direction == "long":
            tp = self.hl.round_price(entry + 2.0 * atr_cur)
            sl = self.hl.round_price(entry - 1.0 * atr_cur)
        else:
            tp = self.hl.round_price(entry - 2.0 * atr_cur)
            sl = self.hl.round_price(entry + 1.0 * atr_cur)

        logging.info(
            "[SolMomentum] SIGNAL %s | entry=~%.4f TP=%.4f SL=%.4f | "
            "size=%.4f SOL | risk=$%.2f (%.1f%% of $%.0f)",
            direction.upper(), entry, tp, sl, size_sol,
            risk_usd, self.risk_pct * 100, total_value,
        )

        if self.dry_run:
            logging.info("[SolMomentum] DRY RUN — order skipped")
            self._in_trade    = True
            self._direction   = direction
            self._entry_price = entry
            self._tp_price    = tp
            self._sl_price    = sl
            self._size        = size_sol
            return

        # Market entry
        if direction == "long":
            await self.hl.place_buy_order(self.ASSET, size_sol)
        else:
            await self.hl.place_sell_order(self.ASSET, size_sol)

        # Bracket orders placed on exchange — protect position if bot restarts
        is_long  = direction == "long"
        tp_resp  = await self.hl.place_take_profit(self.ASSET, is_long, size_sol, tp)
        sl_resp  = await self.hl.place_stop_loss(self.ASSET, is_long, size_sol, sl)

        tp_oids  = self.hl.extract_oids(tp_resp)
        sl_oids  = self.hl.extract_oids(sl_resp)

        self._in_trade    = True
        self._direction   = direction
        self._entry_price = entry
        self._tp_price    = tp
        self._sl_price    = sl
        self._size        = size_sol
        self._tp_oid      = tp_oids[0] if tp_oids else None
        self._sl_oid      = sl_oids[0] if sl_oids else None

        logging.info(
            "[SolMomentum] entry placed | tp_oid=%s sl_oid=%s",
            self._tp_oid, self._sl_oid,
        )


# R constants (match backtest)
TP_R = 2.0
SL_R = 1.0
