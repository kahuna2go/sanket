# Backtest Results Log — Momentum Scalper


---

**Asset:** SOL  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-26 10:33 UTC  |  **Entry TF:** 15m

**Strategy:** EMA9/21 crossover + RSI(14) + RVOL  [15m, TP=2×ATR, SL=1×ATR]

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| No filters | 1903 | 35.6% | 2.00 | +128.0 | +0.067 | -40.0R | NO-GO ✗ |
| + RVOL ≥ 1.5 | 725 | 34.8% | 2.00 | +31.0 | +0.043 | -33.0R | NO-GO ✗ |
| + RVOL ≥ 2.0 | 439 | 33.9% | 2.00 | +8.0 | +0.018 | -28.0R | NO-GO ✗ |
| + Session | 602 | 40.5% | 2.00 | +130.0 | +0.216 | -22.0R | GO ✓ |
| + RVOL ≥ 1.5 + Session | 241 | 41.5% | 2.00 | +59.0 | +0.245 | -13.0R | GO ✓ |
| + RVOL ≥ 2.0 + Session | 156 | 37.8% | 2.00 | +21.0 | +0.135 | -9.0R | NO-GO ✗ |
