"""ORB (Opening Range Breakout) strategy — xyz:SP500.

Fully rule-based. No LLM for trade decisions.
Haiku fires on phase transitions for brief market commentary (logging only).

Session: 15:00–20:00 CET
  15:00–15:30  bias eval (4H EMA21 + funding)
  15:30–15:45  OR formation (5m candles)
  15:45–17:30  breakout detection + retest entry
  20:00        time stop

Entry: retest of ORH/ORL after a 5m close beyond it, in the 4H bias direction.
SL:    ORL − 10%×range (long) / ORH + 10%×range (short).
Exit (tp2_swing — backtest/run_backtest_orb.py):
  Hold full size until TP2 (ORH/ORL ± 1×range) is touched → SL to TP1
  (± 0.5×range) → trail behind confirmed 3-bar 5m swing lows/highs.
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
_SL_BUF     = 0.10    # 10% of OR range beyond the opposite OR edge
_SLOPE_MIN  = 0.0002  # min |4H EMA21 slope| (fraction) for a non-neutral bias
_FUND_THRESH = 0.0003  # 0.03% per 8h
_MAX_LEVERAGE = 5      # cap on entry notional as a multiple of account value; exchange max for xyz:SP500
RISK_USDC   = 50.0     # fixed dollar risk per trade


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

    def __init__(self, hl: HyperliquidAPI, dry_run: bool = False):
        self.hl       = hl
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
        self._trade_taken        = False

        # --- active trade state ---
        self._in_trade     = False
        self._is_long      = False
        self._amount       = 0.0
        self._entry_px     = 0.0
        self._tp1          = 0.0
        self._tp2          = 0.0
        self._tp2_hit      = False
        self._tp2_bar_t    = 0
        self._sl_price     = 0.0
        self._sl_oid       = None
        self._entry_time   = 0
        self._risk         = 0.0


    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        logging.info("[ORB] Starting. risk=$%.0f dry_run=%s", RISK_USDC, self.dry_run)
        # Pre-cache HIP-3 metadata so round_size works for xyz:SP500, and
        # register the dex so the SDK Info object populates name_to_coin for market_open.
        try:
            await self.hl.get_meta_and_ctxs(dex="xyz")
            self.hl.register_perp_dexs(["xyz"])
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

        # OR formation: collect 15:30–15:45 candles. Wait for the window to
        # actually close (15.75) — building at 15.5 would fetch a still-forming
        # candle for "now" (candleSnapshot includes the in-progress bar) and
        # lock in a range from a few seconds of price action instead of 15
        # minutes. Confirmed live 2026-07-10: locked OR range=2.00 at hf=15.51,
        # vs the window's true closed range of 14.00 once it had fully elapsed.
        if self._orh is None and hf >= 15.75:
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

        # 2. OR levels — reconstruct from candle history once the window has
        # actually closed (see _cycle for why 15.5 is too early).
        if hf >= 15.75:
            await self._build_or()

        # 3. Check trades.jsonl: was a trade already taken today?
        today_str = self._day.isoformat()
        log_path = pathlib.Path(__file__).parent.parent.parent / "data" / "trades.jsonl"
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
                self._entry_time  = int(datetime.now(timezone.utc).timestamp() * 1000)
                self._trade_taken = True
                return

            or_range = self._orh - self._orl
            tp1 = round(self._orh + 0.5 * or_range, 2) if is_long \
                else round(self._orl - 0.5 * or_range, 2)
            tp2 = round(self._orh + 1.0 * or_range, 2) if is_long \
                else round(self._orl - 1.0 * or_range, 2)

            # The only resting trigger order ORB keeps is the SL
            sl_price = self.hl.round_price(entry_px)  # fallback: treat entry as SL
            sl_oid   = None
            try:
                orders = await self.hl.get_open_orders()
                for o in orders:
                    if o.get("coin") in (self.ASSET, short_name) and "triggerPx" in o:
                        sl_price = float(o["triggerPx"])
                        sl_oid   = o.get("oid")
            except Exception as e:
                logging.warning("[ORB] Warm-up: could not read open orders: %s", e)

            self._set_trade(is_long, amount, entry_px, tp1, tp2, sl_price, sl_oid)

            # SL only moves into profit after TP2 is touched
            if (sl_price > entry_px) if is_long else (sl_price < entry_px):
                self._tp2_hit   = True
                self._tp2_bar_t = self._entry_time  # swings before the restart are unknown
                self._risk      = 0.0  # original SL is gone — only the trail SL is visible

            logging.info(
                "[ORB] Warm-up: restored %s — entry=%.2f tp2=%.2f sl=%.2f tp2_hit=%s",
                "LONG" if is_long else "SHORT", entry_px, tp2, sl_price, self._tp2_hit,
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
                    if last_c > last_e and slope > _SLOPE_MIN:
                        self._bias = "bull"
                    elif last_c < last_e and slope < -_SLOPE_MIN:
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
                logging.info("[ORB] Long breakout failed (%.2f < ORL %.2f) — cleared", price, orl)
            elif price <= orh:
                logging.info("[ORB] Long retest @ %.2f (ORH=%.2f) — entering", price, orh)
                await self._enter(is_long=True, current_price=price)

        else:  # bp == "short"
            if price > orh:
                self._breakout_pending = None
                logging.info("[ORB] Short breakout failed (%.2f > ORH %.2f) — cleared", price, orh)
            elif price >= orl:
                logging.info("[ORB] Short retest @ %.2f (ORL=%.2f) — entering", price, orl)
                await self._enter(is_long=False, current_price=price)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def _enter(self, is_long: bool, current_price: float):
        orh, orl  = self._orh, self._orl
        or_range  = orh - orl

        if is_long:
            tp1      = round(orh + 0.5 * or_range, 2)
            tp2      = round(orh + 1.0 * or_range, 2)
            sl_price = self.hl.round_price(round(orl - _SL_BUF * or_range, 2))
        else:
            tp1      = round(orl - 0.5 * or_range, 2)
            tp2      = round(orl - 1.0 * or_range, 2)
            sl_price = self.hl.round_price(round(orh + _SL_BUF * or_range, 2))

        risk_per_unit = abs(current_price - sl_price)
        if risk_per_unit <= 0:
            logging.warning("[ORB] SL too close to entry — skipping")
            return

        state     = await self.hl.get_user_state()
        amount    = self.hl.round_size(self.ASSET, RISK_USDC / risk_per_unit)
        if amount <= 0:
            logging.warning("[ORB] Computed size is 0 — skipping")
            return

        # Sanity cap: a tight SL relative to price (common for SP500's small OR
        # ranges) can blow the risk-based formula up to an unfillable notional.
        # Never request more than the account's configured cross leverage supports.
        max_amount = self.hl.round_size(self.ASSET, (state["total_value"] * _MAX_LEVERAGE) / current_price)
        if amount > max_amount:
            logging.warning("[ORB] Computed size %.4f exceeds leverage cap — clamping to %.4f",
                             amount, max_amount)
            amount = max_amount
        if amount <= 0:
            logging.warning("[ORB] Computed size is 0 after cap — skipping")
            return

        direction = "LONG" if is_long else "SHORT"
        logging.info("[ORB] ENTRY %s %.4f @ %.2f  TP1=%.2f  TP2=%.2f  SL=%.2f  risk=$%.0f",
                     direction, amount, current_price, tp1, tp2, sl_price, RISK_USDC)

        if self.dry_run:
            logging.info("[ORB] DRY RUN — order skipped")
            self._set_trade(is_long, amount, current_price, tp1, tp2, sl_price, None)
            return

        try:
            order  = await (self.hl.place_buy_order(self.ASSET, amount)
                             if is_long else self.hl.place_sell_order(self.ASSET, amount))
            filled = self.hl.extract_filled_size(order)
            if filled <= 0:
                logging.warning("[ORB] Entry order did not fill — skipping")
                return
            if filled != amount:
                logging.warning("[ORB] Entry filled %.4f vs requested %.4f — tracking actual fill",
                                 filled, amount)
            amount = filled
            await asyncio.sleep(0.5)
            sl_order = await self.hl.place_stop_loss(self.ASSET, is_long, amount, sl_price)
            sl_oids  = self.hl.extract_oids(sl_order)
            if not sl_oids:
                logging.critical("[ORB] Entry SL placement failed — position is UNPROTECTED")
            self._set_trade(is_long, amount, current_price, tp1, tp2, sl_price,
                            sl_oids[0] if sl_oids else None)
        except Exception as e:
            logging.error("[ORB] entry failed: %s", e)

    def _set_trade(self, is_long, amount, entry_px, tp1, tp2, sl_price, sl_oid):
        self._in_trade     = True
        self._is_long      = is_long
        self._amount       = amount
        self._entry_px     = entry_px
        self._tp1          = tp1
        self._tp2          = tp2
        self._tp2_hit      = False
        self._tp2_bar_t    = 0
        self._sl_price     = sl_price
        self._sl_oid       = sl_oid
        self._trade_taken  = True
        self._breakout_pending = None
        self._entry_time   = int(datetime.now(timezone.utc).timestamp() * 1000)
        self._risk         = abs(entry_px - sl_price)  # per unit, fixed at entry; _sl_price moves later

    # ------------------------------------------------------------------
    # Trade management (TP2 trigger / swing trail)
    # ------------------------------------------------------------------

    async def _place_protective_sl(self, target_sl: float) -> bool:
        """(Re)place the resting stop-loss at target_sl, replacing any existing one.

        Cancel-then-place is not atomic, so a rejected/failed placement can
        leave the position with no resting stop at all. Retries once before
        giving up so a single transient rejection doesn't leave it unprotected
        silently. Returns True only if the exchange confirmed a resting order.
        """
        if self.dry_run:
            self._sl_price = target_sl
            return True
        for attempt in range(2):
            if self._sl_oid:
                try:
                    await self.hl.cancel_order(self.ASSET, self._sl_oid)
                except Exception as e:
                    logging.warning("[ORB] SL cancel failed (attempt %d/2): %s", attempt + 1, e)
                self._sl_oid = None
            try:
                sl_order = await self.hl.place_stop_loss(self.ASSET, self._is_long, self._amount, target_sl)
                oids = self.hl.extract_oids(sl_order)
                if oids:
                    self._sl_oid   = oids[0]
                    self._sl_price = target_sl
                    return True
                logging.error("[ORB] SL placement returned no oid (attempt %d/2)", attempt + 1)
            except Exception as e:
                logging.error("[ORB] SL placement failed (attempt %d/2): %s", attempt + 1, e)
        logging.critical(
            "[ORB] Position UNPROTECTED — SL replacement failed after retry. amount=%.4f entry=%.2f",
            self._amount, self._entry_px,
        )
        return False

    async def _manage_trade(self):
        """tp2_swing exit (see backtest/run_backtest_orb.py): hold the full
        position until price touches TP2, then floor the SL at TP1 and trail
        it behind confirmed 3-bar swing lows/highs on completed 5m bars. The
        only exits are the resting SL and the 20:00 time stop."""
        if await self._position_closed():
            return
        if not self._tp2:
            return  # restored without OR levels — only the resting SL / time stop manage it

        cs = await self.hl.get_candles(self.ASSET, "5m", 60)
        entry_bar_t = self._entry_time - self._entry_time % 300_000
        since_entry = [c for c in cs if c["t"] > entry_bar_t]

        if not self._tp2_hit:
            hit = next((c for c in since_entry
                        if (c["high"] >= self._tp2 if self._is_long else c["low"] <= self._tp2)), None)
            if not hit:
                return
            self._tp2_hit   = True
            self._tp2_bar_t = hit["t"]
            if await self._place_protective_sl(self.hl.round_price(self._tp1)):
                logging.info("[ORB] TP2 %.2f touched — SL→TP1=%.2f, swing trail active", self._tp2, self._sl_price)
            else:
                logging.error("[ORB] TP2 touched, but SL→TP1 FAILED — position may be unprotected")
            return

        # Swing trail on completed bars (cs[-1] is still forming) since the TP2 bar
        done = [c for c in cs[:-1] if c["t"] >= self._tp2_bar_t]
        if self._is_long:
            swings = [done[i]["low"] for i in range(1, len(done) - 1)
                      if done[i]["low"] < done[i - 1]["low"] and done[i]["low"] < done[i + 1]["low"]]
            target = self.hl.round_price(max(swings)) if swings else None
            moved  = target is not None and target > self._sl_price
        else:
            swings = [done[i]["high"] for i in range(1, len(done) - 1)
                      if done[i]["high"] > done[i - 1]["high"] and done[i]["high"] > done[i + 1]["high"]]
            target = self.hl.round_price(min(swings)) if swings else None
            moved  = target is not None and target < self._sl_price
        if not moved:
            return
        if await self._place_protective_sl(target):
            logging.info("[ORB] Swing trail SL → %.2f", target)
        else:
            logging.error("[ORB] Swing trail SL update FAILED (target=%.2f) — position may be unprotected", target)

    async def _position_closed(self) -> bool:
        """Detect the SL closing the position; log the result. True if closed."""
        if self.dry_run:
            price = await self.hl.get_current_price(self.ASSET)
            if not price or (price > self._sl_price if self._is_long else price < self._sl_price):
                return False
            fills = [{"sz": self._amount, "px": self._sl_price}]
        else:
            state = await self.hl.get_user_state()
            pos = next((p for p in state["positions"] if self.hl._coin_matches(p.get("coin", ""), self.ASSET)), None)
            if pos and abs(float(pos.get("szi", 0))) >= 0.001:
                return False
            fills = await self._close_fills()
            if not fills:
                logging.warning("[ORB] Position flat but no close fills found yet — retrying next cycle")
                return False

        exit_px, size, pnl_r = self._fill_result(fills)
        won = (exit_px > self._entry_px) if self._is_long else (exit_px < self._entry_px)
        logging.info("[ORB] SL hit @ %.2f — position closed (%sR, tp2_hit=%s)", exit_px, pnl_r, self._tp2_hit)
        trade_log.append({
            "strategy": "orb", "asset": self.ASSET,
            "dir": "long" if self._is_long else "short",
            "entry": self._entry_px, "tp": self._tp2, "sl": self._sl_price,
            "size": size, "exit": exit_px, "outcome": "win" if won else "loss",
            "pnl_r": pnl_r, "tp2_hit": self._tp2_hit,
        })
        self._in_trade = False
        return True

    async def _close_fills(self) -> list[dict]:
        """Fills since entry that reduced this trade's position."""
        close_side = "A" if self._is_long else "B"
        return [
            f for f in await self.hl.get_recent_fills(limit=50)
            if self.hl._coin_matches(f.get("coin", ""), self.ASSET)
            and f.get("side") == close_side
            and (f.get("time") or 0) >= self._entry_time
        ]

    def _fill_result(self, fills: list[dict]) -> tuple[float, float, float | None]:
        """(VWAP exit price, total size, R per unit vs. the entry risk) for a set of close fills.

        R is None when the entry risk is unknown (warm-up restore after TP2).
        """
        size = sum(float(f["sz"]) for f in fills)
        exit_px = round(sum(float(f["px"]) * float(f["sz"]) for f in fills) / size, 2)
        pnl = (exit_px - self._entry_px) if self._is_long else (self._entry_px - exit_px)
        pnl_r = round(pnl / self._risk, 2) if self._risk > 0 else None
        return exit_px, round(size, 6), pnl_r

    # ------------------------------------------------------------------
    # Time stop
    # ------------------------------------------------------------------

    async def _time_stop(self):
        logging.info("[ORB] Time stop — closing position")
        exit_px, pnl_r = None, None
        if not self.dry_run:
            if self._sl_oid:
                try:
                    await self.hl.cancel_order(self.ASSET, self._sl_oid)
                except Exception as e:
                    logging.warning("[ORB] Time stop: SL order cancel failed: %s", e)
            try:
                await self.hl.place_close_order(self.ASSET)
                await asyncio.sleep(1)
                fills = await self._close_fills()
                if fills:
                    exit_px, _, pnl_r = self._fill_result(fills)
            except Exception as e:
                logging.error("[ORB] time stop close failed: %s", e)
        trade_log.append({
            "strategy": "orb", "asset": self.ASSET,
            "dir": "long" if self._is_long else "short",
            "entry": self._entry_px, "tp": self._tp2, "sl": self._sl_price,
            "size": self._amount, "exit": exit_px, "outcome": "time_stop", "pnl_r": pnl_r,
            "tp2_hit": self._tp2_hit,
        })
        self._in_trade = False

