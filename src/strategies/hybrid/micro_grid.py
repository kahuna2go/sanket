"""Micro grid trading — limit-order grid around current price.

Operates only during peak hours: 13:00–17:00 UTC (US-EU overlap).
Outside peak hours all grid orders are cancelled.

Grid layout (symmetric, grid_levels total):
  - grid_levels/2 limit buys below current price
  - grid_levels/2 limit sells above current price
  - When a buy fills, a sell (take-profit) is placed at price + 1 spacing
  - When a sell fills, a buy (take-profit) is placed at price - 1 spacing

Each GridLevel tracks: price, oid, side ("buy"/"sell"), tp_price.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class GridLevel:
    asset: str
    side: str          # "buy" or "sell"
    price: float
    size: float
    tp_price: float
    oid: int | None = None
    filled: bool = False


class MicroGrid:
    def __init__(self, hyperliquid, risk_mgr, cfg: dict, diary_path: str, dry_run: bool):
        self.hl = hyperliquid
        self.risk_mgr = risk_mgr
        self.cfg = cfg
        self.diary_path = diary_path
        self.dry_run = dry_run
        # asset → list[GridLevel]
        self._grids: dict[str, list[GridLevel]] = {}
        self._open_oids: set[int] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_peak_hours(self) -> bool:
        """True during peak_hours_start <= UTC hour < peak_hours_end."""
        now = datetime.now(timezone.utc)
        start = self.cfg.get("peak_hours_start", 13)
        end = self.cfg.get("peak_hours_end", 17)
        return start <= now.hour < end

    def create_grid(self, asset: str, current_price: float) -> list[GridLevel]:
        """Build the ordered list of GridLevel objects for asset at current_price."""
        asset_cfg = self.cfg.get("assets", {}).get(asset, {})
        spacing_pct = asset_cfg.get("spacing_pct", 0.15) / 100.0
        levels = asset_cfg.get("grid_levels", 10)
        size = asset_cfg.get("position_size", 1.0)
        half = max(levels // 2, 1)

        grid: list[GridLevel] = []
        for i in range(1, half + 1):
            buy_px = self.hl.round_price(current_price * (1 - spacing_pct * i))
            tp_buy = self.hl.round_price(current_price * (1 - spacing_pct * (i - 1)))
            grid.append(GridLevel(
                asset=asset, side="buy", price=buy_px, size=size, tp_price=tp_buy
            ))
            sell_px = self.hl.round_price(current_price * (1 + spacing_pct * i))
            tp_sell = self.hl.round_price(current_price * (1 + spacing_pct * (i - 1)))
            grid.append(GridLevel(
                asset=asset, side="sell", price=sell_px, size=size, tp_price=tp_sell
            ))
        return grid

    async def run_cycle(self, account_state: dict) -> None:
        """One grid cycle: create grids for any ungridded assets, check triggers."""
        if not self.cfg.get("enabled", True):
            return
        if not self.is_peak_hours():
            if self._grids:
                await self.close_all_grids()
            return

        for asset, asset_cfg in self.cfg.get("assets", {}).items():
            if asset not in self._grids:
                current_price = await self.hl.get_current_price(asset)
                if not current_price:
                    continue
                grid = self.create_grid(asset, current_price)
                await self._place_grid(asset, grid, account_state)
                self._grids[asset] = grid
                log.info("[HYBRID:GRID] Created grid for %s at %.4f (%d levels)", asset, current_price, len(grid))

        await self._check_all_triggers()

    async def close_all_grids(self) -> None:
        """Cancel all resting grid orders for every asset."""
        log.info("[HYBRID:GRID] Closing all grids (outside peak hours or shutdown)")
        for asset in list(self._grids.keys()):
            await self._cancel_grid(asset)
        self._grids.clear()
        self._open_oids.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _place_grid(
        self, asset: str, grid: list[GridLevel], account_state: dict
    ) -> None:
        """Place limit orders for all grid levels that pass risk checks."""
        for level in grid:
            amount = self.hl.round_size(asset, level.size)
            alloc_usd = amount * level.price
            trade_spec = {
                "asset": asset,
                "action": "buy" if level.side == "buy" else "sell",
                "allocation_usd": alloc_usd,
                "current_price": level.price,
            }
            ok, reason, _ = self.risk_mgr.validate_trade(
                trade_spec, account_state, account_state.get("initial_value", 0)
            )
            if not ok:
                log.debug("[HYBRID:GRID] Level blocked %s @ %.4f: %s", asset, level.price, reason)
                continue

            prefix = "[DRY_RUN] " if self.dry_run else ""
            log.info(
                "[HYBRID:GRID] %sPlace %s %s @ %.4f  TP=%.4f",
                prefix, level.side.upper(), asset, level.price, level.tp_price,
            )
            if not self.dry_run:
                try:
                    if level.side == "buy":
                        result = await self.hl.place_limit_buy(asset, amount, level.price)
                    else:
                        result = await self.hl.place_limit_sell(asset, amount, level.price)
                    oids = self.hl.extract_oids(result)
                    if oids:
                        level.oid = oids[0]
                        self._open_oids.add(oids[0])
                except Exception as exc:
                    log.error("[HYBRID:GRID] Limit order error for %s @ %.4f: %s", asset, level.price, exc)

    async def _check_all_triggers(self) -> None:
        """Scan open orders; for any filled level place the TP order."""
        if self.dry_run:
            return
        try:
            open_orders = await self.hl.get_open_orders()
            live_oids = {o.get("oid") for o in open_orders if o.get("oid")}
        except Exception as exc:
            log.warning("[HYBRID:GRID] Open orders fetch error: %s", exc)
            return

        for asset, grid in self._grids.items():
            for level in grid:
                if level.filled or level.oid is None:
                    continue
                if level.oid not in live_oids:
                    # Order left the book → assume filled
                    level.filled = True
                    self._open_oids.discard(level.oid)
                    log.info(
                        "[HYBRID:GRID] Grid triggered: %s %s @ %.4f",
                        level.side.upper(), asset, level.price,
                    )
                    await self._place_tp(level)
                    self._write_diary(asset, level)

    async def _place_tp(self, level: GridLevel) -> None:
        """After a grid fill, place the take-profit at the next grid level."""
        tp_side_is_buy = level.side == "sell"  # filled sell → TP is a buy
        amount = self.hl.round_size(level.asset, level.size)
        tp_price = self.hl.round_price(level.tp_price)
        try:
            if tp_side_is_buy:
                result = await self.hl.place_limit_buy(level.asset, amount, tp_price)
            else:
                result = await self.hl.place_limit_sell(level.asset, amount, tp_price)
            oids = self.hl.extract_oids(result)
            log.info(
                "[HYBRID:GRID] TP placed %s %s @ %.4f  oid=%s",
                "BUY" if tp_side_is_buy else "SELL", level.asset, tp_price,
                oids[0] if oids else "?",
            )
        except Exception as exc:
            log.error("[HYBRID:GRID] TP placement error for %s: %s", level.asset, exc)

    async def _cancel_grid(self, asset: str) -> None:
        """Cancel all resting limit orders for asset that belong to this grid."""
        grid = self._grids.get(asset, [])
        for level in grid:
            if level.oid is not None and not level.filled:
                try:
                    await self.hl.cancel_order(asset, level.oid)
                except Exception as exc:
                    log.warning("[HYBRID:GRID] Cancel failed for %s oid=%s: %s", asset, level.oid, exc)
        log.info("[HYBRID:GRID] Grid cancelled for %s", asset)

    def _write_diary(self, asset: str, level: GridLevel) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": "hybrid:grid",
            "asset": asset,
            "action": "grid_fill",
            "side": level.side,
            "price": level.price,
            "size": level.size,
            "tp_price": level.tp_price,
        }
        try:
            with open(self.diary_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.error("[HYBRID:GRID] Diary write error: %s", exc)
