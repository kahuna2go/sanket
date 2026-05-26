"""Funding rate scalper — trades around Hyperliquid 8h funding events.

Funding occurs at 00:00, 08:00, and 16:00 UTC.
Pre-funding (T-40m to T+0): enter in the direction pressure builds
  - positive rate (longs pay) → expect pre-funding dip → long on dip
  - negative rate (shorts pay) → expect pre-funding pump → short on pump
Post-funding (T+0 to T+10m): fade if price moved >0.5% from funding price.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

_FUNDING_HOURS_UTC = [0, 8, 16]


class FundingScalper:
    def __init__(self, hyperliquid, risk_mgr, cfg: dict, diary_path: str, dry_run: bool):
        self.hl = hyperliquid
        self.risk_mgr = risk_mgr
        self.cfg = cfg
        self.diary_path = diary_path
        self.dry_run = dry_run
        self.assets: list[str] = cfg.get("assets", ["xyz:SP500", "ETH", "SOL"])
        # Tracks open positions placed by this strategy: asset → trade dict
        self._active: dict[str, dict] = {}
        # Remembers price at each funding time to detect post-funding moves
        self._funding_price_snapshot: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_next_funding_time(self) -> datetime:
        """Return the next funding timestamp in UTC."""
        now = datetime.now(timezone.utc)
        for h in _FUNDING_HOURS_UTC:
            candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if candidate > now:
                return candidate
        tomorrow = (now + timedelta(days=1)).replace(hour=_FUNDING_HOURS_UTC[0],
                                                      minute=0, second=0, microsecond=0)
        return tomorrow

    def is_active_window(self) -> bool:
        """True if now is within T-pre_window to T+post_window of any funding event."""
        now = datetime.now(timezone.utc)
        pre = self.cfg.get("pre_funding_window", 40)
        post = self.cfg.get("post_funding_window", 10)
        for h in _FUNDING_HOURS_UTC:
            funding_dt = now.replace(hour=h, minute=0, second=0, microsecond=0)
            delta_min = (now - funding_dt).total_seconds() / 60.0
            if -pre <= delta_min <= post:
                return True
        return False

    async def get_funding_rates(self) -> dict[str, float]:
        """Fetch current 8h funding rates for all monitored assets."""
        rates: dict[str, float] = {}
        for asset in self.assets:
            rate = await self.hl.get_funding_rate(asset)
            rates[asset] = rate if rate is not None else 0.0
        return rates

    async def execute(self, account_state: dict) -> list[dict]:
        """Run one funding-scalper cycle. Returns list of actions taken."""
        if not self.cfg.get("enabled", True):
            return []

        now = datetime.now(timezone.utc)
        pre_min = self.cfg.get("pre_funding_window", 40)
        post_min = self.cfg.get("post_funding_window", 10)
        min_rate = self.cfg.get("min_funding_rate", 0.01) / 100.0  # % → decimal
        actions: list[dict] = []

        rates = await self.get_funding_rates()

        for asset, rate in rates.items():
            if abs(rate) < min_rate:
                continue
            if asset in self._active:
                continue  # already have a position from this strategy

            current_price = await self.hl.get_current_price(asset)
            if not current_price:
                continue

            for h in _FUNDING_HOURS_UTC:
                funding_dt = now.replace(hour=h, minute=0, second=0, microsecond=0)
                delta_min = (now - funding_dt).total_seconds() / 60.0

                if -pre_min < delta_min <= 0:
                    # Snapshot price at funding time boundary
                    self._funding_price_snapshot[asset] = current_price
                    action = await self._execute_pre_funding(
                        asset, rate, current_price, account_state
                    )
                    if action:
                        actions.append(action)
                    break

                elif 0 < delta_min <= post_min:
                    action = await self._execute_post_funding_fade(
                        asset, current_price, account_state
                    )
                    if action:
                        actions.append(action)
                    break

        return actions

    # ------------------------------------------------------------------
    # Sub-strategies
    # ------------------------------------------------------------------

    async def _execute_pre_funding(
        self, asset: str, rate: float, current_price: float, account_state: dict
    ) -> dict | None:
        """Enter in the funding-pressured direction; wait for dip/pump first."""
        is_long = rate > 0  # positive rate → longs pay → price dips → long the dip
        discount = self.cfg.get("entry_discount", 0.003)
        entry_target = self.hl.round_price(
            current_price * (1 - discount if is_long else 1 + discount)
        )
        tp_pct = discount * 1.5  # TP: 1.5× the expected dip recovery
        sl_pct = discount * 2.5  # SL: beyond 2.5× the expected move
        tp_price = self.hl.round_price(entry_target * (1 + tp_pct if is_long else 1 - tp_pct))
        sl_price = self.hl.round_price(entry_target * (1 - sl_pct if is_long else 1 + sl_pct))
        alloc = (
            self.cfg.get("position_size_usd", 500)
            * self.cfg.get("position_size_multiplier", 1.0)
        )
        reason = f"pre_funding: rate={rate:.4%}, {'long' if is_long else 'short'} on dip"

        # Price already at/past entry target → market order, otherwise limit
        use_limit = (is_long and current_price > entry_target) or (
            not is_long and current_price < entry_target
        )
        return await self._place_trade(
            asset, is_long, alloc, current_price, tp_price, sl_price,
            account_state, reason,
            limit_price=entry_target if use_limit else None,
        )

    async def _execute_post_funding_fade(
        self, asset: str, current_price: float, account_state: dict
    ) -> dict | None:
        """Mean-reversion trade if price moved > post_funding_threshold since funding."""
        threshold = self.cfg.get("post_funding_threshold", 0.005)
        snap = self._funding_price_snapshot.get(asset)
        if snap is None:
            return None
        move = (current_price - snap) / snap
        if abs(move) < threshold:
            return None

        is_long = move < 0  # price fell → fade down → long the recovery
        alloc = (
            self.cfg.get("position_size_usd", 500)
            * self.cfg.get("position_size_multiplier", 1.0)
        )
        reversion_target = abs(move) * 0.5  # target half-reversion
        stop_extend = abs(move) * 0.3       # stop if it extends 30% further
        tp_price = self.hl.round_price(
            current_price * (1 + reversion_target if is_long else 1 - reversion_target)
        )
        sl_price = self.hl.round_price(
            current_price * (1 - stop_extend if is_long else 1 + stop_extend)
        )
        reason = f"post_funding_fade: move={move:.3%}, {'long' if is_long else 'short'}"
        return await self._place_trade(
            asset, is_long, alloc, current_price, tp_price, sl_price, account_state, reason
        )

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def _place_trade(
        self,
        asset: str,
        is_long: bool,
        alloc_usd: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        account_state: dict,
        reason: str,
        limit_price: float | None = None,
    ) -> dict | None:
        trade_spec = {
            "asset": asset,
            "action": "buy" if is_long else "sell",
            "allocation_usd": alloc_usd,
            "current_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }
        ok, blocked_reason, trade_spec = self.risk_mgr.validate_trade(
            trade_spec, account_state, account_state.get("initial_value", 0)
        )
        if not ok:
            log.warning("[HYBRID:FUNDING] BLOCKED %s: %s", asset, blocked_reason)
            return None

        alloc_usd = trade_spec["allocation_usd"]
        amount = self.hl.round_size(asset, alloc_usd / entry_price)
        tp_price = trade_spec["tp_price"]
        sl_price = trade_spec["sl_price"]

        side = "LONG" if is_long else "SHORT"
        prefix = "[DRY_RUN] " if self.dry_run else ""
        log.info(
            "[HYBRID:FUNDING] %s%s %s @ %.4f  TP=%.4f  SL=%.4f | %s",
            prefix, side, asset, entry_price, tp_price, sl_price, reason,
        )

        if not self.dry_run:
            try:
                if limit_price is not None:
                    await self.hl.place_limit_with_tpsl(
                        asset, is_long, amount, limit_price, tp_price, sl_price
                    )
                else:
                    if is_long:
                        await self.hl.place_buy_order(asset, amount)
                    else:
                        await self.hl.place_sell_order(asset, amount)
                    await self.hl.place_take_profit(asset, is_long, amount, tp_price)
                    await self.hl.place_stop_loss(asset, is_long, amount, sl_price)
            except Exception as exc:
                log.error("[HYBRID:FUNDING] Order error for %s: %s", asset, exc)
                return None

        record = {
            "strategy": "funding",
            "asset": asset,
            "is_long": is_long,
            "amount": amount,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "reason": reason,
        }
        self._active[asset] = record
        self._write_diary(asset, "buy" if is_long else "sell", record)
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mark_closed(self, asset: str) -> None:
        """Remove asset from active tracking when the position closes."""
        self._active.pop(asset, None)
        self._funding_price_snapshot.pop(asset, None)

    def _write_diary(self, asset: str, action: str, record: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "action": action,
            "strategy": "hybrid:funding",
            **{k: v for k, v in record.items() if k not in ("asset",)},
        }
        try:
            with open(self.diary_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.error("[HYBRID:FUNDING] Diary write error: %s", exc)
