# Backtest Results Log — ZigZag Strategy


---

**Asset:** BTC  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-13 13:25 UTC  |  **Entry TF:** 5m

**Strategy:** 1h ZigZag MS (swing_count≥2) + VA Bounce + Confirmation bar + Partial exit

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| dev=2% no filters | 309 | 23.6% | 3.21 | -1.4 | -0.005 | -34.5% | NO-GO ✗ |
| dev=2% + RVOL≥1.2 | 270 | 21.5% | 3.23 | -24.6 | -0.091 | -47.1% | NO-GO ✗ |
| dev=3% no filters | 189 | 26.5% | 3.59 | +40.4 | +0.214 | -17.6% | NO-GO ✗ |
| dev=3% + RVOL≥1.2 | 151 | 24.5% | 3.92 | +31.1 | +0.206 | -15.6% | NO-GO ✗ |
| dev=3% + Session | 109 | 25.7% | 3.65 | +21.3 | +0.195 | -18.0% | NO-GO ✗ |
| dev=3% + RVOL≥1.2 + Session | 79 | 29.1% | 4.10 | +38.4 | +0.485 | -12.0% | GO ✓ |
| dev=5% no filters | 32 | 25.0% | 2.14 | -6.9 | -0.216 | -14.1% | NO-GO ✗ |
| dev=5% + RVOL≥1.2 | 29 | 24.1% | 2.13 | -7.1 | -0.244 | -14.0% | NO-GO ✗ |

---

**Asset:** ETH  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-13 13:26 UTC  |  **Entry TF:** 5m

**Strategy:** 1h ZigZag MS (swing_count≥2) + VA Bounce + Confirmation bar + Partial exit

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| dev=2% no filters | 437 | 23.6% | 3.18 | -6.6 | -0.015 | -84.3% | NO-GO ✗ |
| dev=2% + RVOL≥1.2 | 364 | 24.5% | 3.15 | +5.7 | +0.016 | -50.4% | NO-GO ✗ |
| dev=3% no filters | 283 | 24.4% | 2.84 | -18.0 | -0.064 | -61.1% | NO-GO ✗ |
| dev=3% + RVOL≥1.2 | 230 | 26.1% | 2.56 | -16.3 | -0.071 | -41.7% | NO-GO ✗ |
| dev=3% + Session | 152 | 23.0% | 2.47 | -30.6 | -0.201 | -41.3% | NO-GO ✗ |
| dev=3% + RVOL≥1.2 + Session | 113 | 26.5% | 2.42 | -10.4 | -0.092 | -22.9% | NO-GO ✗ |
| dev=5% no filters | 97 | 22.7% | 3.29 | -2.6 | -0.027 | -19.4% | NO-GO ✗ |
| dev=5% + RVOL≥1.2 | 86 | 26.7% | 2.56 | -4.1 | -0.048 | -18.9% | NO-GO ✗ |

---

**Asset:** SOL  |  **Period:** 2024-05-07 → 2026-05-07  |  **Run:** 2026-05-13 13:27 UTC  |  **Entry TF:** 5m

**Strategy:** 1h ZigZag MS (swing_count≥2) + VA Bounce + Confirmation bar + Partial exit

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|--------|------|---------|--------|------|-------|---------|
| dev=2% no filters | 459 | 30.1% | 2.92 | +82.1 | +0.179 | -29.0% | NO-GO ✗ |
| dev=2% + RVOL≥1.2 | 355 | 33.2% | 2.66 | +76.6 | +0.216 | -21.8% | GO ✓ |
| dev=3% no filters | 364 | 28.3% | 3.20 | +69.0 | +0.190 | -50.9% | NO-GO ✗ |
| dev=3% + RVOL≥1.2 | 298 | 31.2% | 2.70 | +46.1 | +0.155 | -38.8% | NO-GO ✗ |
| dev=3% + Session | 194 | 29.9% | 2.70 | +20.4 | +0.105 | -35.4% | NO-GO ✗ |
| dev=3% + RVOL≥1.2 + Session | 149 | 33.6% | 2.48 | +25.0 | +0.168 | -29.2% | NO-GO ✗ |
| dev=5% no filters | 191 | 23.0% | 2.48 | -37.7 | -0.197 | -44.4% | NO-GO ✗ |
| dev=5% + RVOL≥1.2 | 156 | 24.4% | 2.36 | -28.2 | -0.181 | -35.8% | NO-GO ✗ |

---

## Fib 74.5% Retracement Strategy — Filter Research (4h-scale structure on 1h bars)

**Run:** 2026-05-22  |  **Period:** 2024-05-07 → 2026-05-07  |  **Entry TF:** 5m
**Strategy:** 1h ZigZag MS (swing_count≥2) + Fib 0.745 retracement touch (no confirmation bar) + SL at 1.05× fib extension

**Research question:** The ZigZag deviations in use (3% for SOL, 2% for ETH) produce 4h-scale structure on 1h bars. Do the existing intraday filters (5m RVOL, London/NY session) suit this scale, or should we use 1h RVOL and/or drop session?

---

### SOL — dev=3%

| Config | Trades | AvgR | MaxDD | Verdict |
|--------|--------|------|-------|---------|
| no filters | — | — | — | — |
| + RVOL5m≥1.2 | — | — | — | — |
| **+ RVOL5m≥1.2 + Session [LIVE]** | **99** | **+0.443** | **-7.0%** | **GO ✓** |
| + Session only (no RVOL) | 129 | +0.315 | — | GO ✓ |
| **+ RVOL1h≥1.2 (no session) ← NEW** | **148** | **+0.598** | **-8.9%** | **GO ✓ ★ best** |
| + RVOL1h≥1.2 + Session | 77 | +0.133 | — | NO-GO ✗ |
| + swingcount≥3 + Session | — | — | — | — |

**Finding:** Session filter hurts SOL — cuts valid setups without improving edge. 1h RVOL≥1.2 alone gives the best result (+49 trades, +35% AvgR vs live). Session combined with 1h RVOL is over-filtered (NO-GO).

**Action: drop session filter for SOL; switch RVOL gate from 5m intraday to bias_1h.rvol_1h ≥ 1.2.**

---

### ETH — dev=2%

| Config | Trades | AvgR | MaxDD | Verdict |
|--------|--------|------|-------|---------|
| **+ RVOL5m≥1.2 + Session TP2only [LIVE]** | **145** | **+0.378** | — | **GO ✓** |
| + Session only TP2only ← NEW | 193 | +0.484 | — | GO ✓ ★ best |
| + RVOL1h≥1.2 + Session TP2only | 110 | +0.303 | — | GO ✓ |
| + swingcount≥3 + Session | 75 | +0.228 | — | GO ✓ |

**Finding:** 5m RVOL filter hurts ETH — drops 48 trades for −0.106 AvgR. Session filter is load-bearing and should stay. ETH already uses TP2only (no partial exit); confirmed correct.

**Action: drop RVOL filter for ETH entirely. Keep session filter. ETH prompt already reflected this; no change needed.**

---

**Implemented config after this research:**
- SOL: `dev=3%, RVOL1h≥1.2, session_filter=False`
- ETH: `dev=2%, no RVOL, session_filter=True, TP2only`
