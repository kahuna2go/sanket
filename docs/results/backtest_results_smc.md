# Backtest Results Log — SMC Scalping


---

**Asset:** SOL  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-30 20:19 UTC  |  **Entry TF:** 5m

**Strategy:** SMC Scalping: Sweep + CHoCH + FVG Fill  [1H bias, 5M entry, TP=3×risk]

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| Baseline | 226 | 28.3% | 3.00 | +30.0 | +0.133 | -14.0R | NO-GO ✗ |
| + Session (London+NY UTC) | 104 | 26.9% | 3.00 | +8.0 | +0.077 | -18.0R | NO-GO ✗ |
| + 1H Bias | 54 | 27.8% | 3.00 | +6.0 | +0.111 | -14.0R | NO-GO ✗ |
| + Session + 1H Bias | 20 | 30.0% | 3.00 | +4.0 | +0.200 | -7.0R | GO ✓ |

---

**Asset:** SOL  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-30 21:04 UTC  |  **Entry TF:** 5m

**Strategy:** SMC Scalping: Sweep + CHoCH + FVG Fill  [1H bias, 5M entry, TP=3×risk]

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| Baseline / CHoCH 12b | 226 | 28.3% | 3.00 | +30.0 | +0.133 | -14.0R | NO-GO ✗ |
| + Session / CHoCH 12b | 104 | 26.9% | 3.00 | +8.0 | +0.077 | -18.0R | NO-GO ✗ |
| + 1H Bias / CHoCH 12b | 54 | 27.8% | 3.00 | +6.0 | +0.111 | -14.0R | NO-GO ✗ |
| + Session + Bias / CHoCH 12b | 20 | 30.0% | 3.00 | +4.0 | +0.200 | -7.0R | GO ✓ |
| Baseline / CHoCH 24b | 137 | 27.0% | 3.00 | +11.0 | +0.080 | -21.0R | NO-GO ✗ |
| + Session / CHoCH 24b | 95 | 28.4% | 3.00 | +13.0 | +0.137 | -16.0R | NO-GO ✗ |
| + 1H Bias / CHoCH 24b | 82 | 25.6% | 3.00 | +2.0 | +0.024 | -18.0R | NO-GO ✗ |
| + Session + Bias / CHoCH 24b | 28 | 28.6% | 3.00 | +4.0 | +0.143 | -7.0R | NO-GO ✗ |
| Baseline / CHoCH 48b | 128 | 22.7% | 3.00 | -12.0 | -0.094 | -36.0R | NO-GO ✗ |
| + Session / CHoCH 48b | 115 | 30.4% | 3.00 | +25.0 | +0.217 | -17.0R | GO ✓ |
| + 1H Bias / CHoCH 48b | 98 | 26.5% | 3.00 | +6.0 | +0.061 | -11.0R | NO-GO ✗ |
| + Session + Bias / CHoCH 48b | 36 | 36.1% | 3.00 | +16.0 | +0.444 | -7.0R | GO ✓ |

---

**Asset:** SOL  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-31 05:07 UTC  |  **Entry TF:** 5m

**Strategy:** SMC Scalping: Sweep + CHoCH + FVG Fill  [1H bias, 5M entry, TP=3×risk]

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| Baseline / SL3 | 168 | 25.6% | 3.00 | +4.0 | +0.024 | -14.0R | NO-GO ✗ |
| + Session / SL3 | 93 | 23.7% | 3.00 | -5.0 | -0.054 | -11.0R | NO-GO ✗ |
| + 1H Bias / SL3 | 68 | 22.1% | 3.00 | -8.0 | -0.118 | -18.0R | NO-GO ✗ |
| + Session + Bias / SL3 | 48 | 31.2% | 3.00 | +12.0 | +0.250 | -13.0R | GO ✓ |
| Baseline / SL5 | 128 | 22.7% | 3.00 | -12.0 | -0.094 | -36.0R | NO-GO ✗ |
| + Session / SL5 | 115 | 30.4% | 3.00 | +25.0 | +0.217 | -17.0R | GO ✓ |
| + 1H Bias / SL5 | 98 | 26.5% | 3.00 | +6.0 | +0.061 | -11.0R | NO-GO ✗ |
| + Session + Bias / SL5 | 36 | 36.1% | 3.00 | +16.0 | +0.444 | -7.0R | GO ✓ |
| Baseline / SL7 | 85 | 23.5% | 3.00 | -5.0 | -0.059 | -22.0R | NO-GO ✗ |
| + Session / SL7 | 35 | 22.9% | 3.00 | -3.0 | -0.086 | -11.0R | NO-GO ✗ |
| + 1H Bias / SL7 | 64 | 23.4% | 3.00 | -4.0 | -0.062 | -19.0R | NO-GO ✗ |
| + Session + Bias / SL7 | 24 | 37.5% | 3.00 | +12.0 | +0.500 | -6.0R | GO ✓ |
