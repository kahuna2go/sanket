"""Unit tests for the hybrid strategy components."""

import asyncio
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# FundingScalper tests
# ---------------------------------------------------------------------------

class TestFundingScalerTimings(unittest.TestCase):

    def _make_scalper(self, cfg=None):
        from src.strategies.hybrid.funding_scalper import FundingScalper
        hl = MagicMock()
        hl.round_price = lambda x: round(x, 2)
        hl.round_size = lambda asset, amt: round(amt, 4)
        risk = MagicMock()
        risk.validate_trade.return_value = (True, "", {"asset": "ETH", "action": "buy", "allocation_usd": 500, "tp_price": 2600.0, "sl_price": 2400.0})
        default_cfg = {
            "enabled": True,
            "assets": ["ETH", "SOL"],
            "min_funding_rate": 0.01,
            "pre_funding_window": 40,
            "post_funding_window": 10,
            "entry_discount": 0.003,
            "post_funding_threshold": 0.005,
            "position_size_usd": 500,
            "position_size_multiplier": 1.0,
        }
        if cfg:
            default_cfg.update(cfg)
        return FundingScalper(hl, risk, default_cfg, "test_diary.jsonl", dry_run=True)

    def test_get_next_funding_time_is_in_future(self):
        scalper = self._make_scalper()
        nft = scalper.get_next_funding_time()
        self.assertGreater(nft, datetime.now(timezone.utc))

    def test_get_next_funding_time_is_on_funding_hour(self):
        from src.strategies.hybrid.funding_scalper import _FUNDING_HOURS_UTC
        scalper = self._make_scalper()
        nft = scalper.get_next_funding_time()
        self.assertIn(nft.hour, _FUNDING_HOURS_UTC)
        self.assertEqual(nft.minute, 0)
        self.assertEqual(nft.second, 0)

    def test_is_active_window_at_funding_time(self):
        from src.strategies.hybrid import funding_scalper as fs_mod
        scalper = self._make_scalper()
        # Exactly at funding time (delta = 0) should be active
        now_at_funding = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)
        with patch.object(fs_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = now_at_funding
            result = scalper.is_active_window()
        self.assertTrue(result)

    def test_is_active_window_before_pre_window(self):
        from src.strategies.hybrid import funding_scalper as fs_mod
        scalper = self._make_scalper()
        # 50 minutes before funding — outside pre_funding_window=40
        now_early = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0) - timedelta(minutes=50)
        with patch.object(fs_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = now_early
            result = scalper.is_active_window()
        self.assertFalse(result)

    def test_is_active_window_in_pre_window(self):
        from src.strategies.hybrid import funding_scalper as fs_mod
        scalper = self._make_scalper()
        # 20 minutes before 08:00 funding → should be active
        now_pre = datetime.now(timezone.utc).replace(hour=7, minute=40, second=0, microsecond=0)
        with patch.object(fs_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = now_pre
            result = scalper.is_active_window()
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# StatArb tests
# ---------------------------------------------------------------------------

class TestStatArbCalculations(unittest.TestCase):

    def _make_stat_arb(self):
        from src.strategies.hybrid.stat_arb import StatArb
        hl = MagicMock()
        risk = MagicMock()
        cfg = {
            "enabled": True,
            "sol_asset": "SOL",
            "eth_asset": "ETH",
            "spx_ticker": "^GSPC",
            "lookback_periods": 120,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "stop_loss_z_score": 3.0,
            "check_interval": 60,
            "position_size": 1000,
        }
        return StatArb(hl, risk, cfg, "test_diary.jsonl", dry_run=True)

    def test_z_score_zero_for_flat_series(self):
        arb = self._make_stat_arb()
        spread = [10.0] * 120
        z = arb.calculate_z_score(spread, 120)
        self.assertAlmostEqual(z, 0.0, places=5)

    def test_z_score_positive_for_high_current(self):
        arb = self._make_stat_arb()
        # 119 values at 10, then spike to 20 → strongly positive z
        spread = [10.0] * 119 + [20.0]
        z = arb.calculate_z_score(spread, 120)
        self.assertGreater(z, 2.0)

    def test_z_score_negative_for_low_current(self):
        arb = self._make_stat_arb()
        spread = [10.0] * 119 + [0.0]
        z = arb.calculate_z_score(spread, 120)
        self.assertLess(z, -2.0)

    def test_z_score_respects_lookback(self):
        arb = self._make_stat_arb()
        # Only last 20 values matter; make them all same → z should be 0
        spread = list(range(100)) + [50.0] * 20
        z = arb.calculate_z_score(spread, 20)
        self.assertAlmostEqual(z, 0.0, places=5)

    def test_hedge_ratios_returns_three_series(self):
        arb = self._make_stat_arb()
        n = 60
        sol = [150.0 + i * 0.01 for i in range(n)]
        eth = [2500.0 + i * 0.1 for i in range(n)]
        spx = [5200.0 + i * 1.0 for i in range(n)]
        spread, b_eth, b_spx = arb.calculate_hedge_ratios(sol, eth, spx)
        self.assertEqual(len(spread), n)
        self.assertIsInstance(b_eth, float)
        self.assertIsInstance(b_spx, float)

    def test_hedge_ratios_raises_on_insufficient_data(self):
        arb = self._make_stat_arb()
        with self.assertRaises(ValueError):
            arb.calculate_hedge_ratios([1.0] * 5, [1.0] * 5, [1.0] * 5)


# ---------------------------------------------------------------------------
# MicroGrid tests
# ---------------------------------------------------------------------------

class TestMicroGridLevels(unittest.TestCase):

    def _make_grid(self, cfg=None):
        from src.strategies.hybrid.micro_grid import MicroGrid
        hl = MagicMock()
        hl.round_price = lambda x: round(x, 2)
        hl.round_size = lambda asset, amt: round(amt, 4)
        risk = MagicMock()
        risk.validate_trade.return_value = (True, "", {})
        default_cfg = {
            "enabled": True,
            "peak_hours_start": 13,
            "peak_hours_end": 17,
            "assets": {
                "ETH": {"spacing_pct": 0.15, "grid_levels": 10, "position_size": 0.5},
                "SOL": {"spacing_pct": 0.20, "grid_levels": 10, "position_size": 5.0},
            },
        }
        if cfg:
            default_cfg.update(cfg)
        return MicroGrid(hl, risk, default_cfg, "test_diary.jsonl", dry_run=True)

    def test_create_grid_level_count(self):
        grid = self._make_grid()
        levels = grid.create_grid("ETH", 2500.0)
        # 10 levels = 5 buy + 5 sell
        self.assertEqual(len(levels), 10)
        buys = [l for l in levels if l.side == "buy"]
        sells = [l for l in levels if l.side == "sell"]
        self.assertEqual(len(buys), 5)
        self.assertEqual(len(sells), 5)

    def test_buy_levels_below_current_price(self):
        grid = self._make_grid()
        levels = grid.create_grid("ETH", 2500.0)
        for l in levels:
            if l.side == "buy":
                self.assertLess(l.price, 2500.0)

    def test_sell_levels_above_current_price(self):
        grid = self._make_grid()
        levels = grid.create_grid("ETH", 2500.0)
        for l in levels:
            if l.side == "sell":
                self.assertGreater(l.price, 2500.0)

    def test_tp_price_is_closer_to_current_than_entry(self):
        grid = self._make_grid()
        levels = grid.create_grid("SOL", 150.0)
        for l in levels:
            if l.side == "buy":
                # TP (take profit) should be above entry price
                self.assertGreater(l.tp_price, l.price)
            else:
                # TP for sell should be below entry price
                self.assertLess(l.tp_price, l.price)

    def test_is_peak_hours(self):
        from src.strategies.hybrid import micro_grid as mg_mod
        grid = self._make_grid()
        # 14:30 UTC → inside peak hours
        peak_time = datetime(2026, 5, 26, 14, 30, tzinfo=timezone.utc)
        with patch.object(mg_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = peak_time
            self.assertTrue(grid.is_peak_hours())

    def test_not_peak_hours_at_night(self):
        from src.strategies.hybrid import micro_grid as mg_mod
        grid = self._make_grid()
        # 02:00 UTC → outside peak hours
        night_time = datetime(2026, 5, 26, 2, 0, tzinfo=timezone.utc)
        with patch.object(mg_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = night_time
            self.assertFalse(grid.is_peak_hours())


# ---------------------------------------------------------------------------
# HybridManager priority queue tests
# ---------------------------------------------------------------------------

class TestHybridManagerPriority(unittest.IsolatedAsyncioTestCase):

    def _build_manager(self):
        from src.strategies.hybrid.hybrid_manager import HybridManager
        hl = MagicMock()
        hl.get_user_state = AsyncMock(return_value={
            "total_value": 10000.0,
            "balance": 9000.0,
            "positions": [],
        })
        risk = MagicMock()
        with patch("src.strategies.hybrid.hybrid_manager.HybridManager._load_config") as mock_cfg:
            mock_cfg.return_value = {
                "funding_scalper": {
                    "enabled": True,
                    "assets": ["ETH"],
                    "min_funding_rate": 0.01,
                    "pre_funding_window": 40,
                    "post_funding_window": 10,
                    "entry_discount": 0.003,
                    "post_funding_threshold": 0.005,
                    "position_size_usd": 500,
                    "position_size_multiplier": 1.0,
                },
                "stat_arb": {
                    "enabled": True,
                    "sol_asset": "SOL",
                    "eth_asset": "ETH",
                    "spx_ticker": "^GSPC",
                    "lookback_periods": 120,
                    "entry_z_score": 2.0,
                    "exit_z_score": 0.5,
                    "stop_loss_z_score": 3.0,
                    "check_interval": 60,
                    "position_size": 1000,
                },
                "micro_grid": {
                    "enabled": True,
                    "peak_hours_start": 13,
                    "peak_hours_end": 17,
                    "assets": {},
                },
                "risk": {"daily_loss_kill_switch_pct": 3.0},
            }
            mgr = HybridManager(hl, risk)
        return mgr

    async def test_funding_window_blocks_stat_arb(self):
        mgr = self._build_manager()
        # Fund scalper says we're in the window; stat_arb should NOT be called
        mgr.funding_scalper.is_active_window = MagicMock(return_value=True)
        mgr.funding_scalper.execute = AsyncMock(return_value=[])
        mgr.stat_arb.check_signal = AsyncMock(return_value={"type": "enter", "direction": "long_sol", "z": 2.5, "b_eth": 0.06, "b_spx": 0.0001})
        mgr.stat_arb.execute_hedge_portfolio = AsyncMock(return_value=[])
        mgr.micro_grid.is_peak_hours = MagicMock(return_value=False)

        await mgr._cycle()

        mgr.funding_scalper.execute.assert_awaited_once()
        mgr.stat_arb.execute_hedge_portfolio.assert_not_awaited()

    async def test_stat_arb_runs_outside_funding_window(self):
        mgr = self._build_manager()
        mgr.funding_scalper.is_active_window = MagicMock(return_value=False)
        mgr.stat_arb.check_signal = AsyncMock(return_value={"type": "enter", "direction": "long_sol", "z": 2.5, "b_eth": 0.06, "b_spx": 0.0001})
        mgr.stat_arb.execute_hedge_portfolio = AsyncMock(return_value=[{"asset": "SOL"}])
        mgr.micro_grid.is_peak_hours = MagicMock(return_value=False)

        await mgr._cycle()

        mgr.stat_arb.check_signal.assert_awaited_once()
        mgr.stat_arb.execute_hedge_portfolio.assert_awaited_once()

    async def test_grid_blocked_when_non_grid_positions_exist(self):
        mgr = self._build_manager()
        mgr.funding_scalper.is_active_window = MagicMock(return_value=False)
        mgr.stat_arb.check_signal = AsyncMock(return_value=None)
        mgr.micro_grid.is_peak_hours = MagicMock(return_value=True)
        mgr.micro_grid.run_cycle = AsyncMock()
        # Simulate stat_arb having an open position
        mgr.stat_arb._active = {"direction": "long_sol"}

        await mgr._cycle()

        mgr.micro_grid.run_cycle.assert_not_awaited()

    async def test_kill_switch_blocks_new_entries(self):
        mgr = self._build_manager()
        mgr._kill_switch_active = True
        mgr._daily_high = 10000.0
        mgr._daily_high_date = datetime.now(timezone.utc).date()
        mgr.funding_scalper.execute = AsyncMock(return_value=[])
        mgr.stat_arb.check_signal = AsyncMock(return_value=None)

        await mgr._cycle()

        mgr.funding_scalper.execute.assert_not_awaited()
        mgr.stat_arb.check_signal.assert_not_awaited()

    def test_kill_switch_triggers_at_threshold(self):
        mgr = self._build_manager()
        mgr._daily_high = 10000.0
        mgr._daily_high_date = datetime.now(timezone.utc).date()
        # Drawdown of 3.5% should trigger kill switch (threshold = 3.0%)
        mgr._update_kill_switch(9650.0)
        self.assertTrue(mgr._kill_switch_active)

    def test_kill_switch_does_not_trigger_below_threshold(self):
        mgr = self._build_manager()
        mgr._daily_high = 10000.0
        mgr._daily_high_date = datetime.now(timezone.utc).date()
        # Drawdown of 2.5% should NOT trigger kill switch
        mgr._update_kill_switch(9750.0)
        self.assertFalse(mgr._kill_switch_active)


if __name__ == "__main__":
    unittest.main()
