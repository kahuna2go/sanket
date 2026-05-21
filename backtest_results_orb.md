# ORB Backtest Results Log


---

**Asset:** xyz:SP500  |  **Period:** ~1y cached  |  **Run:** 2026-05-18  |  **Entry TF:** 5m

**Strategy:** Opening Range Breakout (ORB) — 15:30–15:45 CET range, breakout watch 15:45–17:30 CET, 4H 21-EMA bias filter, time stop 20:00 CET

**Exits:** TP1 at 0.5R (50%, SL→BE), TP2 at 1.0R (50%), SL = ORL/ORH ± sl_buffer × range

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict | Exit breakdown |
|--------|--------|------|---------|--------|------|-------|---------|----------------|
| SL=10% / slope≥0.02% | 10 | 100.0% | 0.71 | +7.1 | +0.713 | 0.0R | NO-GO ✗ | sl=4 time_stop=1 tp2=5 |
| SL=5%  / slope≥0.02% | 10 | 100.0% | 0.72 | +7.2 | +0.722 | 0.0R | NO-GO ✗ | sl=4 time_stop=1 tp2=5 |
| SL=10% / slope≥0.10% |  3 | 100.0% | 0.93 | +2.8 | +0.928 | 0.0R | NO-GO ✗ | tp2=3 |
| SL=5%  / slope≥0.10% |  3 | 100.0% | 0.95 | +2.8 | +0.947 | 0.0R | NO-GO ✗ | tp2=3 |
| SL=10% / no slope filter | 10 | 100.0% | 0.71 | +7.1 | +0.713 | 0.0R | NO-GO ✗ | sl=4 time_stop=1 tp2=5 |
| SL=5%  / no slope filter | 10 | 100.0% | 0.72 | +7.2 | +0.722 | 0.0R | NO-GO ✗ | sl=4 time_stop=1 tp2=5 |

**Notes:**
- All configs NO-GO due to insufficient trade count (< 20). Asset has limited history in cache.
- "win%" counts trades with r_multiple > 0; the 4 "sl" exits are post-TP1 breakeven stops (R > 0).
- Need more data (--fetch + --years 2+) to get a statistically meaningful sample.

---

**Asset:** xyz:SP500  |  **Period:** ~12 trading days cached  |  **Run:** 2026-05-21  |  **Entry TF:** 5m

**Strategy:** ORB — 4 mode combinations: entry_mode (breakout / retest) × tp_mode (range / fixed_rr)

**Breakout entry:** enter at close of first 5m candle breaking ORH/ORL  
**Retest entry:** wait for price to touch back to ORH (long) or ORL (short) after initial breakout  
**Range TP:** TP1=entry±0.5×range, TP2=entry±1.0×range  
**Fixed R:R TP:** TP1=entry±2×sl_dist, TP2=entry±3×sl_dist

| Config | Entry | TP | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict | Exits |
|--------|-------|----|--------|------|---------|--------|------|-------|---------|-------|
| SL=10% / slope≥0.02% | breakout | range | 12 | 91.7% | 0.65 | +6.2 | +0.517 | -1.0R | NO-GO ✗ | sl=7 time_stop=1 tp2=4 |
| SL=10% / slope≥0.02% | **retest** | range | **10** | **90.0%** | **0.80** | **+6.2** | **+0.623** | **-1.0R** | NO-GO ✗ | sl=4 tp2=6 |
| SL=10% / slope≥0.02% | breakout | fixed_rr | 12 | 66.7% | 0.83 | +3.5 | +0.294 | -2.0R | NO-GO ✗ | sl=3 time_stop=9 |
| SL=10% / slope≥0.02% | retest | fixed_rr | 10 | 70.0% | 1.05 | +4.4 | +0.437 | -2.0R | NO-GO ✗ | sl=3 time_stop=7 |
| SL=5%  / slope≥0.02% | breakout | range | 12 | 91.7% | 0.66 | +6.3 | +0.523 | -1.0R | NO-GO ✗ | sl=7 time_stop=1 tp2=4 |
| SL=5%  / slope≥0.02% | **retest** | range | **10** | **90.0%** | **0.82** | **+6.4** | **+0.636** | **-1.0R** | NO-GO ✗ | sl=4 tp2=6 |
| SL=5%  / slope≥0.02% | breakout | fixed_rr | 12 | 66.7% | 0.85 | +3.7 | +0.312 | -2.0R | NO-GO ✗ | sl=3 time_stop=9 |
| SL=5%  / slope≥0.02% | retest | fixed_rr | 10 | 70.0% | 1.10 | +4.7 | +0.467 | -2.0R | NO-GO ✗ | sl=3 time_stop=7 |
| SL=10% / no slope filter | breakout | range | 13 | 92.3% | 0.64 | +6.7 | +0.516 | -1.0R | NO-GO ✗ | sl=8 time_stop=1 tp2=4 |
| SL=10% / no slope filter | **retest** | range | **11** | **90.9%** | **0.73** | **+6.3** | **+0.574** | **-1.0R** | NO-GO ✗ | eod=1 sl=4 tp2=6 |
| SL=10% / no slope filter | breakout | fixed_rr | 13 | 61.5% | 0.83 | +3.2 | +0.248 | -2.3R | NO-GO ✗ | eod=1 sl=3 time_stop=9 |
| SL=10% / no slope filter | retest | fixed_rr | 11 | 72.7% | 0.93 | +4.5 | +0.405 | -2.0R | NO-GO ✗ | eod=1 sl=3 time_stop=7 |

**Key finding:** Retest + range-based TP is the standout combination:
- Fewer SL hits (4 vs 7) despite fewer trades
- Higher avgWinR (0.80 vs 0.65) — entering at ORH/ORL tightens the entry
- Zero time stops — all trades reached a clean exit (TP2 or SL)
- Same total R as breakout entry despite 2 fewer trades (missed retests)

**Caveat:** Only 10–13 trades. Statistically meaningless. Waiting for more xyz:SP500 history to accumulate.
