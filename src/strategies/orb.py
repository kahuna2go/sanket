"""ORB (Opening Range Breakout) strategy — xyz:SP500.

Fully rule-based. No LLM for trade decisions.
Haiku fires on phase transitions for brief market commentary (logging only).

Session: 15:00–20:00 CET
  15:00–15:30  bias eval (4H EMA21 + funding)
  15:30–15:45  OR formation (5m candles)
  15:45–17:30  breakout detection + retest entry
  20:00        time stop

Exit:
  TP1 (50% at ORH ± 0.5×range) → SL to breakeven → 0.5×range trail
  Time stop at 20:00 CET regardless
"""

import asyncio
import json
import logging
import pathlib
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

from src.trading.hyperliquid_api import HyperliquidAPI
from src.config_loader import CONFIG
from src.utils import trade_log

_VIENNA     = ZoneInfo("Europe/Vienna")
_ASSET      = "xyz:SP500"
_SL_BUF     = 0.05    # 5% of OR range buffer below retest low / above retest high
_FUND_THRESH = 0.0003  # 0.03% per 8h


def _vhour(ts_ms: int) -> float:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA)
    return dt.hour + dt.minute / 60 + dt.second / 3600


def _phase(hf: float) -> str:
    if hf < 15.0:    return "pre_session"
    if hf < 15.5:    return "pre_open"
    if hf < 15.75:   return "or_formation"
    if hf < 17.5:    return "breakout_watch"
    if hf < 20.0:    return "in_session"
    return "time_stop"


def _ema(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


class Orb:
    ASSET         = _ASSET
    LOOP_INTERVAL = 60  # seconds

    def __init__(self, hl: HyperliquidAPI, risk_pct: float = 0.015, dry_run: bool = False):
        self.hl       = hl
        self.risk_pct = risk_pct
        self.dry_run  = dry_run

        # --- daily ORB state (reset each morning) ---
        self._day:   date | None = None
        self._bias:  str  | None = None   # "bull" / "bear" / "neutral"
        self._bias_done          = False
        self._funding_ok_long    = True
        self._funding_ok_short   = True
        self._orh:  float | None = None
        self._orl:  float | None = None
        self._breakout_pending: str | None = None  # "long" / "short"
        self._retest_low:  float | None = None
        self._retest_high: float | None = None
        self._trade_taken        = False

        # --- active trade state ---
        self._in_trade     = False
        self._is_long      = False
        self._amount       = 0.0
        self._entry_px     = 0.0
        self._tp1          = 0.0
        self._or_range     = 0.0
        self._sl_price     = 0.0
        self._sl_oid       = None
        self._trail_active = False
        self._trail_max    = 0.0


    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        logging.info("[ORB] Starting. risk=%.1f%% dry_run=%s", self.risk_pct * 100, self.dry_run)
        # Pre-cache HIP-3 metadata so round_size works for xyz:SP500
        try:
            await self.hl.get_meta_and_ctxs(dex="xyz")
        except Exception as e:
            logging.warning("[ORB] HIP-3 meta pre-fetch failed: %s", e)
        while True:
            try:
                await self._cycle()
            except Exception as e:
                logging.error("[ORB] cycle error: %s", e, exc_info=True)
            await asyncio.sleep(self.LOOP_INTERVAL)

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    async def _cycle(self):
        now_v = datetime.now(timezone.utc).astimezone(_VIENNA)
        today = now_v.date()
        hf    = now_v.hour + now_v.minute / 60 + now_v.second / 3600

        if today.weekday() >= 5:  # no ORB on weekends
            return

        if today != self._day:
            self._reset_day(today)
            if hf >= 15.0:
                await self._warm_up(hf)

        # Time stop: force-close at 20:00 CET
        if hf >= 20.0:
            if self._in_trade:
                await self._time_stop()
            return

        # Only active inside the ORB window
        if hf < 15.0:
            return

        # If in trade: manage TP1 / trail; skip signal logic
        if self._in_trade:
            await self._manage_trade()
            return

        # One-time bias + funding evaluation (15:00+)
        if not self._bias_done:
            await self._eval_bias()

        # OR formation: collect 15:30–15:45 candles
        if self._orh is None and hf >= 15.5:
            await self._build_or()

        # Breakout + retest detection: 15:45–17:30 CET
        if 15.75 <= hf < 17.5 and not self._trade_taken and self._orh is not None:
            await self._check_breakout()

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _reset_day(self, today: date):
        self._day              = today
        self._bias             = None
        self._bias_done        = False
        self._funding_ok_long  = True
        self._funding_ok_short = True
        self._orh              = None
        self._orl              = None
        self._breakout_pending = None
        self._retest_low       = None
        self._retest_high      = None
        self._trade_taken      = False
        logging.info("[ORB] New day reset (%s)", today)

    # ------------------------------------------------------------------
    # Mid-session warm-up (called after _reset_day when hf >= 15.0)
    # ------------------------------------------------------------------

    async def _warm_up(self, hf: float):
        """Reconstruct intra-day state after a mid-session restart.

        Runs once per day, immediately after _reset_day, when the bot
        starts (or restarts) while the ORB session is already in progress.
        Restores enough state to decide whether a trade can still be taken
        and to continue managing any open position.
        """
        logging.info("[ORB] Warm-up: reconstructing state for %s (hf=%.2f)", self._day, hf)

        # 1. Bias + funding — always live, re-evaluate
        await self._eval_bias()

        # 2. OR levels — reconstruct from candle history if window has passed
        if hf >= 15.5:
            await self._build_or()

        # 3. Check trades.jsonl: was a trade already taken today?
        today_str = self._day.isoformat()
        log_path = pathlib.Path(__file__).parent.parent.parent / "trades.jsonl"
        try:
            if log_path.exists():
                with open(log_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("strategy") == "orb" and rec.get("ts", "").startswith(today_str):
                                self._trade_taken = True
                                logging.info("[ORB] Warm-up: trade already logged today — skipping entry")
                                break
                        except (json.JSONDecodeError, KeyError):
                            pass
        except Exception as e:
            logging.warning("[ORB] Warm-up: could not read trades.jsonl: %s", e)

        # 4. Check exchange for an open position — reconstruct active trade state
        try:
            state = await self.hl.get_user_state()
            short_name = self.ASSET.split(":", 1)[-1]
            pos = next(
                (p for p in state["positions"]
                 if p.get("coin") in (self.ASSET, short_name)),
                None,
            )
            if not pos or abs(float(pos.get("szi", 0) or 0)) < 0.001:
                return  # no open position — warm-up complete

            szi      = float(pos["szi"])
            is_long  = szi > 0
            amount   = abs(szi)
            entry_px = float(pos.get("entryPx", 0) or 0)

            if self._orh is None or self._orl is None:
                # OR not available yet (restart before 15:30) — mark in_trade to block new entries
                logging.warning("[ORB] Warm-up: open position found but OR unavailable — blocking new entries")
                self._in_trade    = True
                self._is_long     = is_long
                self._amount      = amount
                self._entry_px    = entry_px
                self._trade_taken = True
                return

            or_range = self._orh - self._orl
            tp1 = round(self._orh + 0.5 * or_range, 2) if is_long \
                else round(self._orl - 0.5 * or_range, 2)

            # Find SL order
            sl_price = self.hl.round_price(entry_px)  # fallback: treat entry as SL
            sl_oid   = None
            try:
                orders = await self.hl.get_open_orders()
                for o in orders:
                    if o.get("coin") in (self.ASSET, short_name) and "triggerPx" in o:
                        sl_price = float(o["triggerPx"])
                        sl_oid   = o.get("oid")
                        break
            except Exception as e:
                logging.warning("[ORB] Warm-up: could not read open orders: %s", e)

            self._set_trade(is_long, amount, entry_px, tp1, or_range, sl_price, sl_oid)

            # Infer trail state: SL at/beyond entry means TP1 was already hit
            trail_active = (sl_price >= entry_px) if is_long else (sl_price <= entry_px)
            if trail_active:
                # Reconstruct trail_max from SL: trail_sl = trail_max ± 0.5*range
                self._trail_max = sl_price + 0.5 * or_range if is_long \
                    else sl_price - 0.5 * or_range
            self._trail_active = trail_active

            logging.info(
                "[ORB] Warm-up: restored %s — entry=%.2f tp1=%.2f sl=%.2f trail=%s",
                "LONG" if is_long else "SHORT", entry_px, tp1, sl_price, trail_active,
            )
        except Exception as e:
            logging.warning("[ORB] Warm-up: position check failed: %s", e)

    # ------------------------------------------------------------------
    # Bias + OR formation
    # ------------------------------------------------------------------

    async def _eval_bias(self):
        try:
            candles = await self.hl.get_candles(self.ASSET, "4h", 50)
            if len(candles) >= 21:
                closes = [c["close"] for c in candles]
                ema21  = _ema(closes, 21)
                last_c = closes[-1]
                last_e = next((v for v in reversed(ema21) if v is not None), None)
                prev_e = next((v for v in reversed(ema21[:-1]) if v is not None), None)
                if last_e and prev_e:
                    slope = (last_e - prev_e) / prev_e
                    if last_c > last_e and slope > 0.0:
                        self._bias = "bull"
                    elif last_c < last_e and slope < 0.0:
                        self._bias = "bear"
                    else:
                        self._bias = "neutral"
            fund = await self.hl.get_funding_rate(self.ASSET)
            if fund is not None:
                self._funding_ok_long  = fund <= _FUND_THRESH
                self._funding_ok_short = fund >= -_FUND_THRESH
            self._bias_done = True
            logging.info("[ORB] bias=%s fund_ok_long=%s fund_ok_short=%s",
                         self._bias, self._funding_ok_long, self._funding_ok_short)
        except Exception as e:
            logging.error("[ORB] bias eval error: %s", e)

    async def _build_or(self):
        try:
            candles = await self.hl.get_candles(self.ASSET, "5m", 30)
            or_candles = [c for c in candles if c.get("t") and 15.5 <= _vhour(c["t"]) < 15.75]
            if or_candles:
                self._orh = max(c["high"] for c in or_candles)
                self._orl = min(c["low"]  for c in or_candles)
                logging.info("[ORB] OR formed: high=%.2f low=%.2f range=%.2f",
                             self._orh, self._orl, self._orh - self._orl)
        except Exception as e:
            logging.error("[ORB] OR build error: %s", e)

    # ------------------------------------------------------------------
    # Breakout + retest
    # ------------------------------------------------------------------

    async def _check_breakout(self):
        price = await self.hl.get_current_price(self.ASSET)
        if not price:
            return

        orh, orl = self._orh, self._orl
        bp = self._breakout_pending

        if bp is None:
            try:
                cs = await self.hl.get_candles(self.ASSET, "5m", 3)
                last_close = cs[-2]["close"] if len(cs) >= 2 else price
            except Exception:
                last_close = price

            if last_close > orh and self._bias == "bull" and self._funding_ok_long:
                self._breakout_pending = "long"
                logging.info("[ORB] Long breakout (close=%.2f > ORH=%.2f) — awaiting retest", last_close, orh)
            elif last_close < orl and self._bias == "bear" and self._funding_ok_short:
                self._breakout_pending = "short"
                logging.info("[ORB] Short breakout (close=%.2f < ORL=%.2f) — awaiting retest", last_close, orl)

        elif bp == "long":
            if price < orl:
                self._breakout_pending = None
                self._retest_low = None
                logging.info("[ORB] Long breakout failed (%.2f < ORL %.2f) — cleared", price, orl)
            elif price <= orh:
                try:
                    cs = await self.hl.get_candles(self.ASSET, "5m", 2)
                    self._retest_low = cs[-1]["low"] if cs else price
                except Exception:
                    self._retest_low = price
                logging.info("[ORB] Long retest @ %.2f (ORH=%.2f retest_low=%.2f) — entering",
                             price, orh, self._retest_low)
                await self._enter(is_long=True, current_price=price)

        else:  # bp == "short"
            if price > orh:
                self._breakout_pending = None
                self._retest_high = None
                logging.info("[ORB] Short breakout failed (%.2f > ORH %.2f) — cleared", price, orh)
            elif price >= orl:
                try:
                    cs = await self.hl.get_candles(self.ASSET, "5m", 2)
                    self._retest_high = cs[-1]["high"] if cs else price
                except Exception:
                    self._retest_high = price
                logging.info("[ORB] Short retest @ %.2f (ORL=%.2f retest_high=%.2f) — entering",
                             price, orl, self._retest_high)
                await self._enter(is_long=False, current_price=price)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def _enter(self, is_long: bool, current_price: float):
        orh, orl  = self._orh, self._orl
        or_range  = orh - orl

        if is_long:
            tp1      = round(orh + 0.5 * or_range, 2)
            sl_base  = self._retest_low if self._retest_low else orl
            sl_price = self.hl.round_price(round(sl_base - _SL_BUF * or_range, 2))
        else:
            tp1      = round(orl - 0.5 * or_range, 2)
            sl_base  = self._retest_high if self._retest_high else orh
            sl_price = self.hl.round_price(round(sl_base + _SL_BUF * or_range, 2))

        risk_per_unit = abs(current_price - sl_price)
        if risk_per_unit <= 0:
            logging.warning("[ORB] SL too close to entry — skipping")
            return

        state     = await self.hl.get_user_state()
        risk_usd  = state["total_value"] * self.risk_pct
        amount    = self.hl.round_size(self.ASSET, risk_usd / risk_per_unit)
        if amount <= 0:
            logging.warning("[ORB] Computed size is 0 — skipping")
            return

        direction = "LONG" if is_long else "SHORT"
        logging.info("[ORB] ENTRY %s %.4f @ %.2f  TP1=%.2f  SL=%.2f  risk=$%.0f",
                     direction, amount, current_price, tp1, sl_price, risk_usd)

        if self.dry_run:
            logging.info("[ORB] DRY RUN — order skipped")
            self._set_trade(is_long, amount, current_price, tp1, or_range, sl_price, None)
            return

        try:
            await (self.hl.place_buy_order(self.ASSET, amount)
                   if is_long else self.hl.place_sell_order(self.ASSET, amount))
            await asyncio.sleep(0.5)
            sl_order = await self.hl.place_stop_loss(self.ASSET, is_long, amount, sl_price)
            sl_oids  = self.hl.extract_oids(sl_order)
            self._set_trade(is_long, amount, current_price, tp1, or_range, sl_price,
                            sl_oids[0] if sl_oids else None)
        except Exception as e:
            logging.error("[ORB] entry failed: %s", e)

    def _set_trade(self, is_long, amount, entry_px, tp1, or_range, sl_price, sl_oid):
        self._in_trade     = True
        self._is_long      = is_long
        self._amount       = amount
        self._entry_px     = entry_px
        self._tp1          = tp1
        self._or_range     = or_range
        self._sl_price     = sl_price
        self._sl_oid       = sl_oid
        self._trail_active = False
        self._trail_max    = entry_px
        self._trade_taken  = True
        self._breakout_pending = None

    # ------------------------------------------------------------------
    # Trade management (TP1 / trail)
    # ------------------------------------------------------------------

    async def _manage_trade(self):
        # Detect external close (trail SL hit) when trail is active
        if self._trail_active:
            state = await self.hl.get_user_state()
            short_name = self.ASSET.split(":", 1)[-1]
            pos = next(
                (p for p in state["positions"]
                 if p.get("coin") in (self.ASSET, short_name)),
                None,
            )
            if abs(float(pos.get("szi", 0)) if pos else 0.0) < 0.001:
                logging.info("[ORB] Trail SL hit — position closed")
                trade_log.append({
                    "strategy": "orb", "asset": self.ASSET,
                    "dir": "long" if self._is_long else "short",
                    "entry": self._entry_px, "tp": None, "sl": self._sl_price,
                    "size": self._amount, "outcome": "trail_win", "pnl_r": None,
                })
                self._in_trade = False
                self._trail_active = False
                return

        price = await self.hl.get_current_price(self.ASSET)
        if not price:
            return

        if not self._trail_active:
            tp1_hit = (self._is_long and price >= self._tp1) or (not self._is_long and price <= self._tp1)
            if not tp1_hit:
                return

            # TP1: close 50%, move SL to breakeven, activate trail
            half = self.hl.round_size(self.ASSET, self._amount / 2)
            if half > 0 and not self.dry_run:
                try:
                    if self._is_long:
                        await self.hl.place_sell_order(self.ASSET, half)
                    else:
                        await self.hl.place_buy_order(self.ASSET, half)
                    self._amount -= half
                except Exception as e:
                    logging.error("[ORB] TP1 partial close failed: %s", e)

            be_sl = self.hl.round_price(self._entry_px)
            if not self.dry_run:
                try:
                    if self._sl_oid:
                        await self.hl.cancel_order(self.ASSET, self._sl_oid)
                    sl_order = await self.hl.place_stop_loss(self.ASSET, self._is_long, self._amount, be_sl)
                    oids = self.hl.extract_oids(sl_order)
                    self._sl_oid   = oids[0] if oids else None
                    self._sl_price = be_sl
                except Exception as e:
                    logging.error("[ORB] SL to BE failed: %s", e)

            self._trail_active = True
            self._trail_max    = price
            logging.info("[ORB] TP1 hit @ %.2f — 50%% closed, SL→BE=%.2f, range trail active", price, be_sl)
            trade_log.append({
                "strategy": "orb", "asset": self.ASSET,
                "dir": "long" if self._is_long else "short",
                "entry": self._entry_px, "tp": self._tp1, "sl": self._sl_price,
                "size": self._amount, "outcome": "tp1_partial", "pnl_r": 1.0,
            })

        else:
            # Range trail: advance SL as price moves in our favour
            new_max = max(self._trail_max, price) if self._is_long else min(self._trail_max, price)
            self._trail_max = new_max
            trail_sl = self.hl.round_price(
                new_max - 0.5 * self._or_range if self._is_long
                else new_max + 0.5 * self._or_range
            )
            moved = (self._is_long and trail_sl > self._sl_price) or \
                    (not self._is_long and trail_sl < self._sl_price)
            if not moved:
                return

            if not self.dry_run:
                try:
                    if self._sl_oid:
                        await self.hl.cancel_order(self.ASSET, self._sl_oid)
                    sl_order = await self.hl.place_stop_loss(self.ASSET, self._is_long, self._amount, trail_sl)
                    oids = self.hl.extract_oids(sl_order)
                    self._sl_oid   = oids[0] if oids else None
                    self._sl_price = trail_sl
                except Exception as e:
                    logging.error("[ORB] trail SL update failed: %s", e)

            logging.info("[ORB] Trail SL → %.2f (trail_max=%.2f)", trail_sl, new_max)

    # ------------------------------------------------------------------
    # Time stop
    # ------------------------------------------------------------------

    async def _time_stop(self):
        logging.info("[ORB] Time stop — closing position")
        if not self.dry_run:
            try:
                await self.hl.place_close_order(self.ASSET)
            except Exception as e:
                logging.error("[ORB] time stop close failed: %s", e)
        trade_log.append({
            "strategy": "orb", "asset": self.ASSET,
            "dir": "long" if self._is_long else "short",
            "entry": self._entry_px, "tp": self._tp1, "sl": self._sl_price,
            "size": self._amount, "outcome": "time_stop", "pnl_r": None,
        })
        self._in_trade     = False
        self._trail_active = False

