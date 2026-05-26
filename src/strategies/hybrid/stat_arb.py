"""Statistical arbitrage — SOL vs (β_eth·ETH + β_spx·SPX) spread.

Regression: SOL_price = β0 + β_eth·ETH_price + β_spx·SPX_price + ε
Computed via 120-period (2h) rolling OLS on 1-min closes.

Entry:  |z-score| > entry_z_score (default 2.0)
Exit:   |z-score| < exit_z_score  (default 0.5)
Stop:   |z-score| > stop_z_score  (default 3.0)

SPX data comes from yfinance (^GSPC, 1-min bars).  The strategy is
inactive when SPX market is closed (weekends, pre/after market) or when
fewer than 50 SPX bars are available.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger(__name__)


def _ols_betas(y: list[float], x1: list[float], x2: list[float]) -> tuple[float, float, float]:
    """OLS: y = β0 + β1·x1 + β2·x2.  Returns (β0, β1, β2)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x1, x2])
    Y = np.array(y)
    betas, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return float(betas[0]), float(betas[1]), float(betas[2])


class StatArb:
    def __init__(self, hyperliquid, risk_mgr, cfg: dict, diary_path: str, dry_run: bool):
        self.hl = hyperliquid
        self.risk_mgr = risk_mgr
        self.cfg = cfg
        self.diary_path = diary_path
        self.dry_run = dry_run
        self.sol_asset: str = cfg.get("sol_asset", "SOL")
        self.eth_asset: str = cfg.get("eth_asset", "ETH")
        self.spx_ticker: str = cfg.get("spx_ticker", "^GSPC")
        self.lookback: int = cfg.get("lookback_periods", 120)
        self.entry_z: float = cfg.get("entry_z_score", 2.0)
        self.exit_z: float = cfg.get("exit_z_score", 0.5)
        self.stop_z: float = cfg.get("stop_loss_z_score", 3.0)
        self.position_size: float = cfg.get("position_size", 1000.0)  # USD notional on SOL
        # Active hedge portfolio: None when flat, dict when position open
        self._active: dict | None = None
        # Cached betas from last regression
        self._last_betas: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_ohlc_data(self, asset: str, timeframe: str, periods: int) -> list[dict]:
        """Fetch OHLCV candles from Hyperliquid. Returns list of dicts."""
        return await self.hl.get_candles(asset, timeframe, periods)

    def calculate_hedge_ratios(
        self, sol_closes: list[float], eth_closes: list[float], spx_closes: list[float]
    ) -> tuple[list[float], float, float]:
        """Run OLS regression and return (spread_series, β_eth, β_spx).

        spread = SOL - (β0 + β_eth·ETH + β_spx·SPX)
        """
        n = min(len(sol_closes), len(eth_closes), len(spx_closes))
        if n < 30:
            raise ValueError(f"Insufficient data for regression: {n} bars")
        sol = sol_closes[-n:]
        eth = eth_closes[-n:]
        spx = spx_closes[-n:]
        b0, b_eth, b_spx = _ols_betas(sol, eth, spx)
        spread = [s - (b0 + b_eth * e + b_spx * x) for s, e, x in zip(sol, eth, spx)]
        return spread, b_eth, b_spx

    def calculate_z_score(self, spread: list[float], lookback: int) -> float:
        """Z-score of the most recent spread value against the rolling window."""
        window = spread[-lookback:]
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / len(window)
        std = var ** 0.5
        if std == 0.0:
            return 0.0
        return (spread[-1] - mean) / std

    async def check_signal(self) -> dict | None:
        """Compute spread and z-score. Returns signal dict or None."""
        if not self.cfg.get("enabled", True):
            return None

        try:
            sol_candles, eth_candles, spx_closes = await asyncio.gather(
                self.fetch_ohlc_data(self.sol_asset, "1m", self.lookback + 10),
                self.fetch_ohlc_data(self.eth_asset, "1m", self.lookback + 10),
                self._fetch_spx_closes(self.lookback + 10),
            )
        except Exception as exc:
            log.warning("[HYBRID:STAT_ARB] Data fetch failed: %s", exc)
            return None

        if len(spx_closes) < 50:
            log.debug("[HYBRID:STAT_ARB] Skipping — SPX has only %d bars (market likely closed)", len(spx_closes))
            return None

        sol_closes = [c["close"] for c in sol_candles]
        eth_closes = [c["close"] for c in eth_candles]
        n = min(len(sol_closes), len(eth_closes), len(spx_closes))

        try:
            spread, b_eth, b_spx = self.calculate_hedge_ratios(
                sol_closes[-n:], eth_closes[-n:], spx_closes[-n:]
            )
        except ValueError as exc:
            log.debug("[HYBRID:STAT_ARB] Regression skipped: %s", exc)
            return None

        z = self.calculate_z_score(spread, self.lookback)
        self._last_betas = (0.0, b_eth, b_spx)

        log.debug("[HYBRID:STAT_ARB] z=%.3f  β_eth=%.4f  β_spx=%.6f", z, b_eth, b_spx)

        if self._active is not None:
            # Manage existing position
            if abs(z) < self.exit_z:
                return {"type": "exit", "z": z, "b_eth": b_eth, "b_spx": b_spx, "reason": "z_mean_reversion"}
            if abs(z) > self.stop_z:
                return {"type": "exit", "z": z, "b_eth": b_eth, "b_spx": b_spx, "reason": "z_stop_loss"}
            return None

        # New entry
        if z > self.entry_z:
            # SOL expensive vs hedge → short SOL, long ETH+SPX
            return {"type": "enter", "direction": "short_sol", "z": z, "b_eth": b_eth, "b_spx": b_spx}
        if z < -self.entry_z:
            # SOL cheap vs hedge → long SOL, short ETH+SPX
            return {"type": "enter", "direction": "long_sol", "z": z, "b_eth": b_eth, "b_spx": b_spx}
        return None

    async def execute_hedge_portfolio(self, signal: dict, account_state: dict) -> list[dict]:
        """Enter or exit the full 3-leg hedge position."""
        sig_type = signal["type"]
        if sig_type == "exit":
            return await self.close_hedge_portfolio(signal.get("reason", "signal"))

        direction = signal["direction"]  # "long_sol" or "short_sol"
        b_eth = signal["b_eth"]
        b_spx = signal["b_spx"]
        z = signal["z"]

        sol_price = await self.hl.get_current_price(self.sol_asset)
        eth_price = await self.hl.get_current_price(self.eth_asset)
        spx_price = await self.hl.get_current_price("xyz:SP500")
        if not all([sol_price, eth_price, spx_price]):
            log.warning("[HYBRID:STAT_ARB] Price fetch failed, skipping entry")
            return []

        sol_long = direction == "long_sol"
        sol_contracts = self.hl.round_size(self.sol_asset, self.position_size / sol_price)
        eth_contracts = self.hl.round_size(self.eth_asset, b_eth * self.position_size / sol_price)
        spx_contracts = self.hl.round_size("xyz:SP500", b_spx * self.position_size / sol_price)

        positions = [
            (self.sol_asset, sol_long, sol_contracts, sol_price),
            (self.eth_asset, not sol_long, eth_contracts, eth_price),
            ("xyz:SP500", not sol_long, spx_contracts, spx_price),
        ]

        prefix = "[DRY_RUN] " if self.dry_run else ""
        log.info(
            "[HYBRID:STAT_ARB] %sZ-score: %.3f, entering hedge: %s SOL / %s ETH / %s SPX",
            prefix, z, "LONG" if sol_long else "SHORT",
            "SHORT" if sol_long else "LONG",
            "SHORT" if sol_long else "LONG",
        )

        placed: list[dict] = []
        for asset, is_long, amount, price in positions:
            if amount <= 0:
                continue
            if not self.dry_run:
                try:
                    if is_long:
                        await self.hl.place_buy_order(asset, amount)
                    else:
                        await self.hl.place_sell_order(asset, amount)
                except Exception as exc:
                    log.error("[HYBRID:STAT_ARB] Order error for %s leg: %s", asset, exc)
                    continue
            placed.append({"asset": asset, "is_long": is_long, "amount": amount, "entry_price": price})

        self._active = {
            "direction": direction,
            "z_entry": z,
            "positions": placed,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_diary("enter", z, direction)
        return placed

    async def close_hedge_portfolio(self, reason: str = "signal") -> list[dict]:
        """Close all legs of the active hedge position."""
        if self._active is None:
            return []

        prefix = "[DRY_RUN] " if self.dry_run else ""
        log.info("[HYBRID:STAT_ARB] %sClosing hedge portfolio: %s", prefix, reason)

        closed: list[dict] = []
        for pos in self._active.get("positions", []):
            asset = pos["asset"]
            if not self.dry_run:
                try:
                    await self.hl.place_close_order(asset)
                except Exception as exc:
                    log.error("[HYBRID:STAT_ARB] Close error for %s: %s", asset, exc)
                    continue
            closed.append(asset)

        self._write_diary("exit", None, reason)
        self._active = None
        return closed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_spx_closes(self, count: int) -> list[float]:
        """Fetch recent 1-min ^GSPC closes from yfinance. Returns [] if market closed."""
        try:
            import yfinance as yf
            data = await asyncio.to_thread(
                lambda: yf.download(
                    self.spx_ticker, period="1d", interval="1m",
                    auto_adjust=True, progress=False
                )
            )
            if data is None or data.empty or len(data) < 10:
                return []
            closes = data["Close"].dropna().tolist()
            # Flatten in case yfinance returns a DataFrame with multi-level columns
            if closes and hasattr(closes[0], "__iter__"):
                closes = [float(c[0]) if hasattr(c, "__iter__") else float(c) for c in closes]
            return [float(c) for c in closes[-count:]]
        except Exception as exc:
            log.warning("[HYBRID:STAT_ARB] SPX fetch error: %s", exc)
            return []

    def _write_diary(self, action: str, z: float | None, detail: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": "hybrid:stat_arb",
            "action": action,
            "z_score": round(z, 4) if z is not None else None,
            "detail": detail,
        }
        try:
            with open(self.diary_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.error("[HYBRID:STAT_ARB] Diary write error: %s", exc)
