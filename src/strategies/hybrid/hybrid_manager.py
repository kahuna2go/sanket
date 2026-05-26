"""Hybrid strategy coordinator.

Priority queue:
  1. Funding window (T-40m to T+10m) → funding_scalper only
  2. Stat-arb signal present         → stat_arb
  3. Peak hours + no non-grid positions → micro_grid

Kill switch: pauses new entries if daily account drawdown > daily_loss_kill_switch_pct.
Existing TP/SL orders remain on the book during a pause.

Enable with STRATEGY=hybrid in .env.
Config lives in config/hybrid_config.yaml (relative to CWD).
"""

import asyncio
import json
import logging
import pathlib
from datetime import datetime, timezone, date

import yaml

from src.config_loader import CONFIG
from src.strategies.hybrid.funding_scalper import FundingScalper
from src.strategies.hybrid.stat_arb import StatArb
from src.strategies.hybrid.micro_grid import MicroGrid

log = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path("config/hybrid_config.yaml")


class HybridManager:
    def __init__(self, hyperliquid, risk_mgr):
        self.hl = hyperliquid
        self.risk_mgr = risk_mgr
        self.diary_path = "diary.jsonl"
        self.dry_run: bool = CONFIG.get("dry_run", False)

        cfg = self._load_config()
        self.funding_scalper = FundingScalper(
            hyperliquid, risk_mgr, cfg["funding_scalper"], self.diary_path, self.dry_run
        )
        self.stat_arb = StatArb(
            hyperliquid, risk_mgr, cfg["stat_arb"], self.diary_path, self.dry_run
        )
        self.micro_grid = MicroGrid(
            hyperliquid, risk_mgr, cfg["micro_grid"], self.diary_path, self.dry_run
        )

        self._kill_switch_pct: float = cfg.get("risk", {}).get("daily_loss_kill_switch_pct", 3.0)
        self._kill_switch_active: bool = False
        self._daily_high: float | None = None
        self._daily_high_date: date | None = None
        self._initial_value: float | None = None

        # Per-strategy metrics
        self.metrics: dict[str, dict] = {
            "funding": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "stat_arb": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "grid":     {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Priority-based execution loop. Runs indefinitely."""
        log.info("[HYBRID] Manager starting (dry_run=%s)", self.dry_run)
        if self.dry_run:
            log.warning("[HYBRID] DRY RUN MODE — no real orders will be placed")

        while True:
            try:
                await self._cycle()
            except Exception as exc:
                log.error("[HYBRID] Unhandled cycle error: %s", exc, exc_info=True)
            await asyncio.sleep(30)

    async def _cycle(self) -> None:
        account_state = await self.hl.get_user_state()
        account_value = float(account_state.get("total_value") or account_state["balance"])
        account_state["initial_value"] = self._initial_value or account_value

        if self._initial_value is None:
            self._initial_value = account_value

        self._update_kill_switch(account_value)
        if self._kill_switch_active:
            log.warning("[HYBRID] Kill switch ACTIVE (daily drawdown > %.1f%%) — no new entries", self._kill_switch_pct)
            return

        # Priority 1: Funding window — ONLY run funding scalper
        if self.funding_scalper.is_active_window():
            actions = await self.funding_scalper.execute(account_state)
            if actions:
                self.metrics["funding"]["trades"] += len(actions)
                log.info("[HYBRID:FUNDING] %d action(s) this cycle", len(actions))
            return

        # Priority 2: Stat arb signal
        signal = await self.stat_arb.check_signal()
        if signal:
            results = await self.stat_arb.execute_hedge_portfolio(signal, account_state)
            if results:
                self.metrics["stat_arb"]["trades"] += 1

        # Priority 3: Micro grid — only if no non-grid positions open
        if self.micro_grid.is_peak_hours() and not self._has_non_grid_positions():
            await self.micro_grid.run_cycle(account_state)
        elif not self.micro_grid.is_peak_hours() and self.micro_grid._grids:
            await self.micro_grid.close_all_grids()

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _update_kill_switch(self, account_value: float) -> None:
        today = datetime.now(timezone.utc).date()
        if self._daily_high_date != today:
            self._daily_high = account_value
            self._daily_high_date = today
            self._kill_switch_active = False
        elif account_value > (self._daily_high or 0.0):
            self._daily_high = account_value

        if self._kill_switch_active:
            return
        if self._daily_high and self._daily_high > 0:
            drawdown = (self._daily_high - account_value) / self._daily_high * 100.0
            if drawdown >= self._kill_switch_pct:
                self._kill_switch_active = True
                log.warning(
                    "[HYBRID] Kill switch triggered: %.2f%% drawdown (threshold %.1f%%)",
                    drawdown, self._kill_switch_pct,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_non_grid_positions(self) -> bool:
        """True if funding or stat_arb strategies have open positions."""
        return bool(self.funding_scalper._active) or (self.stat_arb._active is not None)

    @staticmethod
    def _load_config() -> dict:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Hybrid config not found at {_CONFIG_PATH.resolve()}. "
                "Copy config/hybrid_config.yaml from the repo."
            )
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def get_metrics(self) -> dict:
        """Return per-strategy performance metrics."""
        out = {}
        for name, m in self.metrics.items():
            trades = m["trades"]
            wins = m["wins"]
            out[name] = {
                "trades": trades,
                "win_rate": round(wins / trades, 3) if trades else None,
                "total_pnl": round(m["pnl"], 4),
            }
        return out
