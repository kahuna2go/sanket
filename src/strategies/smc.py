"""SMC Strategy — live implementation (SOL and ETH).

Entry: liquidity sweep + CHoCH + FVG fill on 5M candles.
  Long:  sweep below confirmed swing low → CHoCH above pre-sweep swing high
         → price retraces to midpoint of bullish FVG → enter long
  Short: mirror

Exit: TP = entry ± 3×risk, SL = entry ∓ risk (sweep wick).
  Bracket orders placed immediately after entry.

Risk: fixed 50 USDC per trade.
Mutual exclusion: checks for an existing position in the asset before entering.

Configs:
  SOL Candidate A — Narrow session (08-10 + 13:30-15:30 UTC)
  ETH Candidate B — Broad session (07-17 UTC)

Warm-up on startup: replays last 200 5M bars to reconstruct any in-flight
  setup so no setup is missed after a restart or connection drop.
"""

import asyncio
import logging
from datetime import datetime, timezone
from src.utils import trade_log

from src.trading.hyperliquid_api import HyperliquidAPI

_UTC = timezone.utc

# Shared params (both assets)
_SWING_LOOKBACK = 5
_SWEEP_LOOKBACK = 20   # bars
_CHOCH_TIMEOUT  = 48   # bars → 4h
_FVG_ENTRY      = "mid50"

RISK_USDC = 50.0
TP_R      = 3.0

_WARMUP_BARS = 200   # ~17h of 5M data

# SOL Candidate A — Narrow session
SOL_SESSION_WINDOWS: list[tuple[float, float]] = [
    (8.0,  10.0),   # London open
    (13.5, 15.5),   # NY open
]

# ETH Candidate B — Broad session 07-17 UTC
ETH_SESSION_WINDOWS: list[tuple[float, float]] = [
    (7.0, 17.0),
]


# ---------------------------------------------------------------------------
# Helpers (self-contained — no backtest imports)
# ---------------------------------------------------------------------------

def _in_session(ts_ms: int, windows: list[tuple[float, float]]) -> bool:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=_UTC)
    hf = dt.hour + dt.minute / 60
    return any(s <= hf < e for s, e in windows)


def _find_swings(candles: list[dict], lookback: int) -> tuple[list, list]:
    n = len(candles)
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    sl, sh = [], []
    for j in range(lookback, n - lookback):
        nl = lows[j-lookback:j]  + lows[j+1:j+lookback+1]
        nh = highs[j-lookback:j] + highs[j+1:j+lookback+1]
        if lows[j]  < min(nl): sl.append((j, lows[j]))
        if highs[j] > max(nh): sh.append((j, highs[j]))
    return sl, sh


def _find_bullish_fvg(bars: list[dict]) -> tuple[float, float] | None:
    for i in range(len(bars) - 3, -1, -1):
        if bars[i]["high"] < bars[i+2]["low"]:
            return bars[i+2]["low"], bars[i]["high"]   # (fvg_hi, fvg_lo)
    return None


def _find_bearish_fvg(bars: list[dict]) -> tuple[float, float] | None:
    for i in range(len(bars) - 3, -1, -1):
        if bars[i]["low"] > bars[i+2]["high"]:
            return bars[i]["low"], bars[i+2]["high"]   # (fvg_hi, fvg_lo)
    return None


def _reconstruct_state(candles: list[dict], session_windows: list[tuple[float, float]]) -> dict:
    """Replay candles through SMC state machine, return current state dict."""
    n = len(candles)
    lb = _SWING_LOOKBACK
    swing_lows, swing_highs = _find_swings(candles, lb)

    state       = "IDLE"
    sweep_type  = ""
    sweep_price = 0.0
    sweep_idx   = 0
    choch_dl    = 0
    choch_tgt   = 0.0
    fvg_hi = fvg_lo = fvg_start = 0

    sl_ptr = sh_ptr = 0
    vis_lows:  list[tuple[int, float]] = []
    vis_highs: list[tuple[int, float]] = []

    for i in range(n):
        bar = candles[i]
        while sl_ptr < len(swing_lows)  and swing_lows[sl_ptr][0]  + lb <= i:
            vis_lows.append(swing_lows[sl_ptr]);   sl_ptr += 1
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] + lb <= i:
            vis_highs.append(swing_highs[sh_ptr]); sh_ptr += 1

        if state == "IN_TRADE":
            continue

        if state == "FVG_WAIT":
            mid = (fvg_hi + fvg_lo) / 2
            trig = mid if _FVG_ENTRY == "mid50" else fvg_hi
            if sweep_type == "bull":
                if bar["low"] < fvg_lo:
                    state = "IDLE"
                elif bar["low"] <= trig:
                    state = "IN_TRADE"
            else:
                if bar["high"] > fvg_hi:
                    state = "IDLE"
                elif bar["high"] >= (mid if _FVG_ENTRY == "mid50" else fvg_lo):
                    state = "IN_TRADE"
            continue

        if state == "SWEPT":
            if i > choch_dl:
                state = "IDLE"
            elif sweep_type == "bull" and bar["close"] > choch_tgt:
                fvg = _find_bullish_fvg(candles[sweep_idx:i+1])
                if fvg:
                    fvg_hi, fvg_lo = fvg; fvg_start = i; state = "FVG_WAIT"
                else:
                    state = "IDLE"
            elif sweep_type == "bear" and bar["close"] < choch_tgt:
                fvg = _find_bearish_fvg(candles[sweep_idx:i+1])
                if fvg:
                    fvg_hi, fvg_lo = fvg; fvg_start = i; state = "FVG_WAIT"
                else:
                    state = "IDLE"
            continue

        # IDLE
        rec_lows  = [s for s in vis_lows  if s[0] >= i - _SWEEP_LOOKBACK]
        rec_highs = [s for s in vis_highs if s[0] >= i - _SWEEP_LOOKBACK]
        if rec_lows:
            _, sl_px = rec_lows[-1]
            if bar["low"] < sl_px and bar["close"] > sl_px and _in_session(bar["t"], session_windows):
                hb = [s for s in vis_highs if s[0] < i]
                if hb:
                    state = "SWEPT"; sweep_type = "bull"
                    sweep_price = bar["low"]; sweep_idx = i
                    choch_dl = i + _CHOCH_TIMEOUT; choch_tgt = hb[-1][1]
                    continue
        if rec_highs:
            _, sh_px = rec_highs[-1]
            if bar["high"] > sh_px and bar["close"] < sh_px and _in_session(bar["t"], session_windows):
                lb2 = [s for s in vis_lows if s[0] < i]
                if lb2:
                    state = "SWEPT"; sweep_type = "bear"
                    sweep_price = bar["high"]; sweep_idx = i
                    choch_dl = i + _CHOCH_TIMEOUT; choch_tgt = lb2[-1][1]

    bars_remaining_choch    = max(0, choch_dl - (n - 1))
    bars_remaining_fvg_wait = n - 1 - fvg_start if state == "FVG_WAIT" else 0

    return {
        "state":            state,
        "sweep_type":       sweep_type,
        "sweep_price":      sweep_price,
        "choch_target":     choch_tgt,
        "choch_bars_left":  bars_remaining_choch,
        "fvg_hi":           fvg_hi,
        "fvg_lo":           fvg_lo,
        "fvg_bars_elapsed": bars_remaining_fvg_wait,
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class Smc:
    CANDLE_COUNT  = 210    # enough for warm-up + indicator warmup
    LOOP_INTERVAL = 60     # seconds

    def __init__(
        self,
        hl: HyperliquidAPI,
        asset: str = "SOL",
        session_windows: list[tuple[float, float]] | None = None,
        dry_run: bool = False,
    ):
        self.hl               = hl
        self.ASSET            = asset
        self._session_windows = session_windows if session_windows is not None else SOL_SESSION_WINDOWS
        self.dry_run          = dry_run
        self._tag             = f"[SMC/{asset}]"

        # SMC state machine
        self._state        = "IDLE"
        self._sweep_type   = ""
        self._sweep_price  = 0.0
        self._choch_tgt    = 0.0
        self._choch_dl_ts: float = 0.0
        self._fvg_hi       = 0.0
        self._fvg_lo       = 0.0
        self._sweep_bar_ts: float = 0.0

        # Trade state
        self._in_trade   = False
        self._direction: str | None = None
        self._entry      = 0.0
        self._tp         = 0.0
        self._sl         = 0.0
        self._size       = 0.0
        self._tp_oid     = None
        self._sl_oid     = None

        self._last_bar_ts = 0
        self._stats = {"trades": 0, "wins": 0, "losses": 0, "total_r": 0.0}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        logging.info("%s Starting. risk=$%.0f dry_run=%s", self._tag, RISK_USDC, self.dry_run)
        await self._warm_up()
        while True:
            try:
                await self._cycle()
            except Exception as e:
                logging.error("%s cycle error: %s", self._tag, e, exc_info=True)
            await asyncio.sleep(self.LOOP_INTERVAL)

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    async def _warm_up(self):
        raw = await self.hl.get_candles(self.ASSET, "5m", self.CANDLE_COUNT)
        candles = raw[:-1] if len(raw) >= 2 else raw
        if len(candles) < _WARMUP_BARS:
            logging.info("%s warm-up: not enough bars (%d) — starting cold", self._tag, len(candles))
            return

        recent = candles[-_WARMUP_BARS:]
        s = _reconstruct_state(recent, self._session_windows)
        self._state       = s["state"]
        self._sweep_type  = s["sweep_type"]
        self._sweep_price = s["sweep_price"]
        self._choch_tgt   = s["choch_target"]
        self._fvg_hi      = s["fvg_hi"]
        self._fvg_lo      = s["fvg_lo"]

        now_ts = datetime.now(_UTC).timestamp()
        if self._state == "SWEPT":
            self._choch_dl_ts = now_ts + s["choch_bars_left"] * 5 * 60

        last_ts = datetime.fromtimestamp(recent[-1]["t"] / 1000, tz=_UTC).strftime("%H:%M UTC")
        logging.info(
            "%s warm-up done (last bar %s): state=%s sweep_type=%s choch_tgt=%.4f fvg=%.4f–%.4f",
            self._tag, last_ts, self._state, self._sweep_type,
            self._choch_tgt, self._fvg_lo, self._fvg_hi,
        )

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    async def _cycle(self):
        raw = await self.hl.get_candles(self.ASSET, "5m", 80)
        candles = raw[:-1] if len(raw) >= 2 else raw
        if len(candles) < 20:
            return

        latest_ts = candles[-1]["t"]

        if self._in_trade:
            await self._manage_trade()
            return

        if latest_ts <= self._last_bar_ts:
            return
        self._last_bar_ts = latest_ts

        await self._tick(candles)

    # ------------------------------------------------------------------
    # State machine tick (one completed 5M bar)
    # ------------------------------------------------------------------

    async def _tick(self, candles: list[dict]):
        bar = candles[-1]
        now_ts = datetime.now(_UTC).timestamp()

        swing_lows, swing_highs = _find_swings(candles, _SWING_LOOKBACK)
        n = len(candles)
        i = n - 1

        vis_lows  = [(j, p) for j, p in swing_lows  if j + _SWING_LOOKBACK <= i]
        vis_highs = [(j, p) for j, p in swing_highs if j + _SWING_LOOKBACK <= i]

        # ── FVG_WAIT ──
        if self._state == "FVG_WAIT":
            mid  = (self._fvg_hi + self._fvg_lo) / 2
            trig = mid   # mid50

            if self._sweep_type == "bull":
                if bar["low"] < self._fvg_lo:
                    logging.info("%s FVG invalidated (bull) — IDLE", self._tag)
                    self._state = "IDLE"
                elif bar["low"] <= trig:
                    await self._enter(trig, "long")
            else:
                if bar["high"] > self._fvg_hi:
                    logging.info("%s FVG invalidated (bear) — IDLE", self._tag)
                    self._state = "IDLE"
                elif bar["high"] >= trig:
                    await self._enter(trig, "short")
            return

        # ── SWEPT ──
        if self._state == "SWEPT":
            if now_ts > self._choch_dl_ts:
                logging.info("%s CHoCH timeout — IDLE", self._tag)
                self._state = "IDLE"
            elif self._sweep_type == "bull" and bar["close"] > self._choch_tgt:
                disp_start = max(0, n - 1 - _CHOCH_TIMEOUT)
                fvg = _find_bullish_fvg(candles[disp_start:])
                if fvg:
                    self._fvg_hi, self._fvg_lo = fvg
                    self._state = "FVG_WAIT"
                    logging.info("%s CHoCH confirmed (bull) — FVG %.4f–%.4f",
                                 self._tag, self._fvg_lo, self._fvg_hi)
                else:
                    logging.info("%s CHoCH confirmed but no FVG — IDLE", self._tag)
                    self._state = "IDLE"
            elif self._sweep_type == "bear" and bar["close"] < self._choch_tgt:
                disp_start = max(0, n - 1 - _CHOCH_TIMEOUT)
                fvg = _find_bearish_fvg(candles[disp_start:])
                if fvg:
                    self._fvg_hi, self._fvg_lo = fvg
                    self._state = "FVG_WAIT"
                    logging.info("%s CHoCH confirmed (bear) — FVG %.4f–%.4f",
                                 self._tag, self._fvg_lo, self._fvg_hi)
                else:
                    logging.info("%s CHoCH confirmed but no FVG — IDLE", self._tag)
                    self._state = "IDLE"
            return

        # ── IDLE: detect sweep ──
        if not _in_session(bar["t"], self._session_windows):
            return

        rec_lows  = [(j, p) for j, p in vis_lows  if j >= i - _SWEEP_LOOKBACK]
        rec_highs = [(j, p) for j, p in vis_highs if j >= i - _SWEEP_LOOKBACK]

        if rec_lows:
            _, sl_px = rec_lows[-1]
            if bar["low"] < sl_px and bar["close"] > sl_px:
                hb = [p for j, p in vis_highs if j < i]
                if hb:
                    self._state       = "SWEPT"
                    self._sweep_type  = "bull"
                    self._sweep_price = bar["low"]
                    self._sweep_bar_ts = bar["t"] / 1000
                    self._choch_dl_ts  = now_ts + _CHOCH_TIMEOUT * 5 * 60
                    self._choch_tgt    = hb[-1]
                    logging.info("%s Bull sweep — low=%.4f CHoCH target=%.4f",
                                 self._tag, self._sweep_price, self._choch_tgt)
                    return

        if rec_highs:
            _, sh_px = rec_highs[-1]
            if bar["high"] > sh_px and bar["close"] < sh_px:
                lb2 = [p for j, p in vis_lows if j < i]
                if lb2:
                    self._state       = "SWEPT"
                    self._sweep_type  = "bear"
                    self._sweep_price = bar["high"]
                    self._sweep_bar_ts = bar["t"] / 1000
                    self._choch_dl_ts  = now_ts + _CHOCH_TIMEOUT * 5 * 60
                    self._choch_tgt    = lb2[-1]
                    logging.info("%s Bear sweep — high=%.4f CHoCH target=%.4f",
                                 self._tag, self._sweep_price, self._choch_tgt)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def _enter(self, trigger: float, direction: str):
        user_state = await self.hl.get_user_state()
        for pos in user_state.get("positions", []):
            if pos.get("coin") == self.ASSET and abs(float(pos.get("szi", 0))) > 0.001:
                logging.info("%s position already open (%.4f) — skipping (first-signal-wins)",
                             self._tag, float(pos.get("szi", 0)))
                return

        sl   = self._sweep_price
        risk = abs(trigger - sl)
        if risk <= 0:
            self._state = "IDLE"
            return

        size = self.hl.round_size(self.ASSET, RISK_USDC / risk)
        if size <= 0:
            self._state = "IDLE"
            return

        is_long  = direction == "long"
        tp       = self.hl.round_price(trigger + TP_R * risk if is_long else trigger - TP_R * risk)
        sl       = self.hl.round_price(sl)
        entry_px = self.hl.round_price(trigger)

        logging.info("%s SIGNAL %s | entry=%.4f TP=%.4f SL=%.4f | size=%.4f %s | risk=$%.2f",
                     self._tag, direction.upper(), entry_px, tp, sl, size, self.ASSET, RISK_USDC)

        if self.dry_run:
            logging.info("%s DRY RUN — order skipped", self._tag)
            self._in_trade  = True
            self._direction = direction
            self._entry     = entry_px
            self._tp        = tp
            self._sl        = sl
            self._size      = size
            self._state     = "IDLE"
            return

        if is_long:
            await self.hl.place_buy_order(self.ASSET, size)
        else:
            await self.hl.place_sell_order(self.ASSET, size)

        await asyncio.sleep(0.5)

        tp_order = await self.hl.place_take_profit(self.ASSET, is_long, size, tp)
        sl_order = await self.hl.place_stop_loss(self.ASSET, is_long, size, sl)

        tp_oids = self.hl.extract_oids(tp_order)
        sl_oids = self.hl.extract_oids(sl_order)

        self._in_trade  = True
        self._direction = direction
        self._entry     = entry_px
        self._tp        = tp
        self._sl        = sl
        self._size      = size
        self._tp_oid    = tp_oids[0] if tp_oids else None
        self._sl_oid    = sl_oids[0] if sl_oids else None
        self._state     = "IDLE"

        logging.info("%s entry placed | tp_oid=%s sl_oid=%s", self._tag, self._tp_oid, self._sl_oid)

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    async def _manage_trade(self):
        user_state = await self.hl.get_user_state()
        pos = next(
            (p for p in user_state.get("positions", []) if p.get("coin") == self.ASSET),
            None,
        )
        szi = float(pos.get("szi", 0)) if pos else 0.0
        if abs(szi) > 0.001:
            return

        open_orders = await self.hl.get_open_orders()
        open_oids   = {o.get("oid") for o in open_orders
                       if self.hl._coin_matches(o.get("coin", ""), self.ASSET)}

        tp_alive = self._tp_oid is not None and self._tp_oid in open_oids
        sl_alive = self._sl_oid is not None and self._sl_oid in open_oids

        if sl_alive and not tp_alive:
            outcome, pnl_r = "win", TP_R
            if not self.dry_run:
                await self.hl.cancel_order(self.ASSET, self._sl_oid)
        elif tp_alive and not sl_alive:
            outcome, pnl_r = "loss", -1.0
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
            "%s CLOSED %s | %s | entry=%.4f TP=%.4f SL=%.4f | "
            "pnl=%.1fR | totals: %d trades / %d W / %d L / %.1fR",
            self._tag, outcome.upper(), self._direction,
            self._entry, self._tp, self._sl, pnl_r,
            self._stats["trades"], self._stats["wins"],
            self._stats["losses"], self._stats["total_r"],
        )

        trade_log.append({
            "strategy": "smc", "asset": self.ASSET,
            "dir": self._direction, "entry": self._entry,
            "tp": self._tp, "sl": self._sl,
            "size": self._size, "outcome": outcome, "pnl_r": pnl_r,
        })

        self._in_trade  = False
        self._direction = None
        self._tp_oid    = None
        self._sl_oid    = None
