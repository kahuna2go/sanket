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

---

**Asset:** SOL  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-28 08:01 UTC  |  **Entry TF:** 15m

**Strategy:** EMA9/21 crossover + RSI(14) + RVOL  [15m, TP=2×ATR, SL=1×ATR]

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| No filters | 1903 | 35.6% | 2.00 | +128.0 | +0.067 | -40.0R | NO-GO ✗ |
| + RVOL ≥ 1.5 | 725 | 34.8% | 2.00 | +31.0 | +0.043 | -33.0R | NO-GO ✗ |
| + RVOL ≥ 2.0 | 439 | 33.9% | 2.00 | +8.0 | +0.018 | -28.0R | NO-GO ✗ |
| + Session (London+NY) | 602 | 40.5% | 2.00 | +130.0 | +0.216 | -22.0R | GO ✓ |
| + RVOL ≥ 1.5 + Session | 241 | 41.5% | 2.00 | +59.0 | +0.245 | -13.0R | GO ✓ |
| + RVOL ≥ 2.0 + Session | 156 | 37.8% | 2.00 | +21.0 | +0.135 | -9.0R | NO-GO ✗ |
| + Session EU day (07–17) | 900 | 35.9% | 2.00 | +69.0 | +0.077 | -28.0R | NO-GO ✗ |
| + Session EU+NY ext (07–12, 14–22) | 1097 | 36.4% | 2.00 | +100.0 | +0.091 | -21.0R | NO-GO ✗ |
| + Session broad (07–22) | 1241 | 35.6% | 2.00 | +85.0 | +0.068 | -24.0R | NO-GO ✗ |
| + Session Asia (01–07) | 474 | 36.5% | 2.00 | +45.0 | +0.095 | -20.0R | NO-GO ✗ |
| + Session Asia+London+NY | 1062 | 38.8% | 2.00 | +174.0 | +0.164 | -22.0R | GO ✓ |

---

**Asset:** ETH  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-28 08:18 UTC  |  **Entry TF:** 15m

**Strategy:** EMA9/21 crossover + RSI(14) + RVOL  [15m, TP=2×ATR, SL=1×ATR]

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| No filters | 1839 | 35.0% | 2.00 | +93.0 | +0.051 | -97.0R | NO-GO ✗ |
| + RVOL ≥ 1.5 | 779 | 35.8% | 2.00 | +58.0 | +0.074 | -51.0R | NO-GO ✗ |
| + RVOL ≥ 2.0 | 515 | 33.8% | 2.00 | +7.0 | +0.014 | -37.0R | NO-GO ✗ |
| + Session (London+NY) | 545 | 35.6% | 2.00 | +37.0 | +0.068 | -38.0R | NO-GO ✗ |
| + RVOL ≥ 1.5 + Session | 245 | 34.7% | 2.00 | +10.0 | +0.041 | -29.0R | NO-GO ✗ |
| + RVOL ≥ 2.0 + Session | 155 | 30.3% | 2.00 | -14.0 | -0.090 | -23.0R | NO-GO ✗ |
| + Session EU day (07–17) | 858 | 36.9% | 2.00 | +93.0 | +0.108 | -44.0R | NO-GO ✗ |
| + Session EU+NY ext (07–12, 14–22) | 1058 | 36.4% | 2.00 | +97.0 | +0.092 | -55.0R | NO-GO ✗ |
| + Session broad (07–22) | 1183 | 35.8% | 2.00 | +89.0 | +0.075 | -56.0R | NO-GO ✗ |
| + Session Asia (01–07) | 481 | 33.1% | 2.00 | -4.0 | -0.008 | -48.0R | NO-GO ✗ |
| + Session Asia+London+NY | 1011 | 34.6% | 2.00 | +39.0 | +0.039 | -69.0R | NO-GO ✗ |
