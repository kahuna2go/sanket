# Backtest Results

## Strategy: 1h MS + VA Bounce (5m entry)

**Logic:** 1h bias via HH_HL / LH_LL (swing_count ≥ 2) + 5m Value Area Bounce entry + confirmation bar + partial exit (50% at TP1=VAH, SL→BE; 50% at TP2=Fibonacci 127.2%)

**Parameters:** STRUCT_WINDOW=200 bars, VP_WINDOW=20 bars, SL_BUFFER=0.15×VA_width, ZONE_PCT=0.30

**Period:** 2024-05-07 → 2026-05-07

---

### BTC — bounces=768, confirmed=377

| Config              | Trades | Win%  | AvgWinR | TotalR | AvgR   | MaxDD  | Verdict    |
|---------------------|--------|-------|---------|--------|--------|--------|------------|
| No filters          | 377    | 27.9% | 2.91    | +33.1  | +0.088 | -29.1% | NO-GO      |
| + RVOL ≥ 1.2        | 284    | 29.6% | 2.93    | +46.5  | +0.164 | -25.3% | NO-GO      |
| + RVOL ≥ 1.5        | 242    | 31.0% | 3.02    | +59.7  | +0.247 | -16.1% | **GO** ✓   |
| + Session           | 175    | 29.7% | 2.94    | +29.7  | +0.170 | -19.4% | NO-GO      |
| + RVOL ≥ 1.2 + Sess | 116    | 31.0% | 3.09    | +31.3  | +0.270 | -15.2% | **GO** ✓   |
| + RVOL ≥ 1.5 + Sess | 88     | 33.0% | 3.15    | +32.4  | +0.368 | -10.3% | **GO** ✓   |

### ETH — bounces=727, confirmed=349

| Config              | Trades | Win%  | AvgWinR | TotalR | AvgR    | MaxDD  | Verdict |
|---------------------|--------|-------|---------|--------|---------|--------|---------|
| No filters          | 349    | 23.8% | 2.79    | -34.4  | -0.099  | -66.3% | NO-GO   |
| + RVOL ≥ 1.2        | 286    | 22.4% | 2.68    | -50.5  | -0.176  | -68.5% | NO-GO   |
| + RVOL ≥ 1.5        | 251    | 23.5% | 2.76    | -29.2  | -0.116  | -47.2% | NO-GO   |
| + Session           | 160    | 18.1% | 2.79    | -50.1  | -0.313  | -58.5% | NO-GO   |
| + RVOL ≥ 1.2 + Sess | 112    | 17.9% | 2.54    | -41.2  | -0.368  | -42.5% | NO-GO   |
| + RVOL ≥ 1.5 + Sess | 92     | 18.5% | 2.64    | -30.1  | -0.327  | -31.5% | NO-GO   |

### SOL — bounces=884, confirmed=395

| Config              | Trades | Win%  | AvgWinR | TotalR | AvgR   | MaxDD  | Verdict    |
|---------------------|--------|-------|---------|--------|--------|--------|------------|
| No filters          | 395    | 30.1% | 2.79    | +56.1  | +0.142 | -25.6% | NO-GO      |
| + RVOL ≥ 1.2        | 301    | 33.9% | 2.56    | +62.2  | +0.207 | -24.0% | **GO** ✓   |
| + RVOL ≥ 1.5        | 255    | 33.3% | 2.61    | +51.9  | +0.204 | -19.9% | **GO** ✓   |
| + Session           | 184    | 27.7% | 2.67    | +3.1   | +0.017 | -23.1% | NO-GO      |
| + RVOL ≥ 1.2 + Sess | 124    | 33.1% | 2.73    | +29.1  | +0.235 | -10.6% | **GO** ✓   |
| + RVOL ≥ 1.5 + Sess | 93     | 31.2% | 2.89    | +19.7  | +0.212 | -9.6%  | **GO** ✓   |

---

**Live agent config:** SOL + RVOL ≥ 1.2 — best total R (+62.2R), ~150 trades/year, MaxDD -24%
