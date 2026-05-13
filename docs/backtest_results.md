# Backtest Results

Strategy: Momentum Breakout (crypto perps on Hyperliquid)
Go/no-go threshold: win_rate > 38% for 3:1 R:R (33% breakeven + 5pp margin for fees/slippage)

---

## ETH — 2024-05-07 → 2026-05-07 (2 years)

HTF bias: 1h | Entry: 5m | 210,240 5m bars | 17,520 1h bars

| Config                          | Trades |  Win% | Total R |  Avg R | Max DD | Verdict    |
|---------------------------------|-------:|------:|--------:|-------:|-------:|------------|
| Baseline (3:1, no filter)       |  3,646 | 29.8% |  +694.0 | +0.190 |  -42.0 | NO-GO ✗    |
| + London/NY session             |  1,089 | 32.7% |  +335.0 | +0.308 |  -24.0 | GO ✓       |
| + London/NY + Volume            |    606 | 32.0% |  +170.0 | +0.281 |  -21.0 | GO ✓       |
| + London/NY + Tight RSI (55-65) |    742 | 35.6% |  +314.0 | +0.423 |  -17.0 | GO ✓       |
| + London/NY + Volume + Tight RSI|    387 | 33.9% |  +137.0 | +0.354 |  -19.0 | GO ✓       |
| + Wide (08-22 UTC)              |  2,188 | 30.1% |  +444.0 | +0.203 |  -37.0 | GO ✓       |
| + Wide (08-22 UTC) + Volume     |  1,258 | 27.9% |  +146.0 | +0.116 |  -29.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Tight RSI  |  1,484 | 30.1% |  +300.0 | +0.202 |  -28.0 | GO ✓       |
| Reference: 2:1 R:R              |  4,216 | 37.3% |  +497.0 | +0.118 |  -34.0 | NO-GO ✗    |

**Best config: London/NY + Tight RSI** — 35.6% win rate, +0.423 avg R, -17R max drawdown.

Key takeaways:
- London/NY filter adds 2.9pp win rate and cuts max drawdown nearly in half vs. baseline.
- Tight RSI (55–65) is the single biggest quality lever: +3pp win rate, +0.115 avg R vs. session-only.
- Wide window (08-22 UTC) clears GO but is materially worse than London/NY on every metric — more trades, lower win rate, higher drawdown. Volume filter actually hurts it (drops below breakeven).
- 3:1 R:R is required: 2:1 fails to clear the threshold despite higher win rate due to lower payout.

---

## SOL — 2024-05-07 → 2026-05-07 (2 years)

HTF bias: 1h | Entry: 5m | 210,240 5m bars | 17,520 1h bars

| Config                          | Trades |  Win% | Total R |  Avg R | Max DD | Verdict    |
|---------------------------------|-------:|------:|--------:|-------:|-------:|------------|
| Baseline (3:1, no filter)       |  3,700 | 28.2% |  +480.0 | +0.130 |  -64.0 | NO-GO ✗    |
| + London/NY session             |  1,128 | 28.3% |  +148.0 | +0.131 |  -30.0 | NO-GO ✗    |
| + London/NY + Volume            |    615 | 28.3% |   +81.0 | +0.132 |  -35.0 | NO-GO ✗    |
| + London/NY + Tight RSI (55-65) |    802 | 27.4% |   +78.0 | +0.097 |  -27.0 | NO-GO ✗    |
| + London/NY + Volume + Tight RSI|    416 | 28.6% |   +60.0 | +0.144 |  -23.0 | NO-GO ✗    |
| + Wide (08-22 UTC)              |  2,170 | 28.8% |  +326.0 | +0.150 |  -42.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Volume     |  1,242 | 27.8% |  +138.0 | +0.111 |  -49.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Tight RSI  |  1,518 | 28.1% |  +186.0 | +0.123 |  -44.0 | NO-GO ✗    |
| Reference: 2:1 R:R              |  4,306 | 36.8% |  +452.0 | +0.105 |  -53.0 | NO-GO ✗    |

**Verdict: NO-GO across all configs.** Win rate consistently 27–29%, well short of the 38% threshold.

Key takeaways:
- The Momentum Breakout strategy has no edge on SOL in this period. Session and RSI filters don't rescue it — tight RSI actually makes it slightly worse (27.4%).
- SOL's higher volatility inflates ATR-based SL/TP distances without improving win rate, resulting in lower R vs. ETH across every comparable config.
- Do not trade SOL with this strategy until a separate edge is identified.

---

## ETH — 2024-05-07 → 2026-05-07 (2 years, HTF=4h)

HTF bias: 4h | Entry: 5m | 210,240 5m bars | 4,381 4h bars

| Config                          | Trades |  Win% | Total R |  Avg R | Max DD | Verdict    |
|---------------------------------|-------:|------:|--------:|-------:|-------:|------------|
| Baseline (3:1, no filter)       |  3,328 | 28.7% |  +488.0 | +0.147 |  -41.0 | NO-GO ✗    |
| + London/NY session             |  1,041 | 32.1% |  +295.0 | +0.283 |  -31.0 | GO ✓       |
| + London/NY + Volume            |    604 | 31.6% |  +160.0 | +0.265 |  -25.0 | GO ✓       |
| + London/NY + Tight RSI (55-65) |    707 | 33.2% |  +233.0 | +0.330 |  -18.0 | GO ✓       |
| + London/NY + Volume + Tight RSI|    386 | 32.4% |  +114.0 | +0.295 |  -21.0 | GO ✓       |
| + Wide (08-22 UTC)              |  2,044 | 28.8% |  +308.0 | +0.151 |  -45.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Volume     |  1,238 | 26.5% |   +74.0 | +0.060 |  -36.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Tight RSI  |  1,424 | 27.3% |  +132.0 | +0.093 |  -32.0 | NO-GO ✗    |
| Reference: 2:1 R:R              |  3,866 | 36.6% |  +382.0 | +0.099 |  -34.0 | NO-GO ✗    |

**Best config: London/NY + Tight RSI** — 33.2% win rate, +0.330 avg R, -18R max drawdown.

Key takeaways vs. 1h HTF:
- Wide session flips from GO to NO-GO with 4h bias — the wider session window needs tighter HTF confirmation to avoid noise.
- London/NY configs still all pass, but avg R drops across the board (~0.09R lower per config vs. 1h).
- 1h HTF is the better bias timeframe for ETH with this strategy.

---

## SOL — 2024-05-07 → 2026-05-07 (2 years, HTF=4h)

HTF bias: 4h | Entry: 5m | 210,240 5m bars | 4,381 4h bars

| Config                          | Trades |  Win% | Total R |  Avg R | Max DD | Verdict    |
|---------------------------------|-------:|------:|--------:|-------:|-------:|------------|
| Baseline (3:1, no filter)       |  3,511 | 28.2% |  +445.0 | +0.127 |  -41.0 | NO-GO ✗    |
| + London/NY session             |  1,090 | 29.4% |  +190.0 | +0.174 |  -24.0 | NO-GO ✗    |
| + London/NY + Volume            |    639 | 28.5% |   +89.0 | +0.139 |  -21.0 | NO-GO ✗    |
| + London/NY + Tight RSI (55-65) |    706 | 30.0% |  +142.0 | +0.201 |  -28.0 | GO ✓       |
| + London/NY + Volume + Tight RSI|    396 | 29.8% |   +76.0 | +0.192 |  -24.0 | NO-GO ✗    |
| + Wide (08-22 UTC)              |  2,089 | 28.5% |  +291.0 | +0.139 |  -29.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Volume     |  1,274 | 27.0% |  +102.0 | +0.080 |  -31.0 | NO-GO ✗    |
| + Wide (08-22 UTC) + Tight RSI  |  1,411 | 27.3% |  +129.0 | +0.091 |  -31.0 | NO-GO ✗    |
| Reference: 2:1 R:R              |  4,112 | 36.4% |  +379.0 | +0.092 |  -36.0 | NO-GO ✗    |

**One marginal GO:** London/NY + Tight RSI passes at exactly 30.0% — right at the threshold with only 706 trades.

Key takeaways vs. 1h HTF:
- 4h bias opens one crack (Tight RSI clears the bar) vs. a clean sweep NO-GO on 1h. Marginal at best.
- Wide session is NO-GO on both HTF variants for SOL.
- SOL remains not tradeable with Momentum Breakout. The single marginal GO is too thin to act on.

---

---

## MS + VA Bounce Strategy — SOL TP Split Comparison (2026-05-13)

Strategy: 1h market structure bias (swing_count ≥ 2) + 5m Value Area bounce + confirmation bar + partial exit
Period: 2024-05-07 → 2026-05-07 | Entry TF: 5m | Go/no-go: win_rate ≥ breakeven_win_rate + 5%

Two structure detection methods compared: **Old MS** (pivot-based, ±3 bar lookback) vs **ZigZag dev=2%** (2% deviation threshold).
Three TP split variants: **50/50** (baseline), **70/30**, **TP2-only** (full size runs to 127.2% fib extension, no breakeven move).

### No filters

| Approach | Split | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|----------|-------|-------:|-----:|--------:|-------:|-----:|------:|---------|
| Old MS   | 50/50   | 395 | 30.1% | 2.79 | +56.1 | +0.142 | -25.6 | NO-GO ✗ |
| Old MS   | 70/30   | 395 | 30.1% | 2.73 | +49.3 | +0.125 | -25.1 | NO-GO ✗ |
| Old MS   | TP2only | 382 | 19.4% | 5.38 | +90.4 | +0.237 | -32.1 | NO-GO ✗ |
| ZZ dev=2% | 50/50  | 459 | 30.1% | 2.92 | +82.1 | +0.179 | -29.0 | NO-GO ✗ |
| ZZ dev=2% | 70/30  | 459 | 30.1% | 2.91 | +79.9 | +0.174 | -27.6 | NO-GO ✗ |
| ZZ dev=2% | TP2only | 428 | 17.8% | 6.32 | +128.0 | +0.299 | -36.1 | NO-GO ✗ |

### RVOL ≥ 1.2

| Approach | Split | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|----------|-------|-------:|-----:|--------:|-------:|-----:|------:|---------|
| Old MS   | 50/50   | 301 | 33.9% | 2.56 | +62.2 | +0.207 | -24.0 | GO ✓ |
| Old MS   | 70/30   | 301 | 33.9% | 2.52 | +57.8 | +0.192 | -24.0 | GO ✓ |
| Old MS   | TP2only | 294 | 22.1% | 4.73 | +78.2 | +0.266 | -18.8 | NO-GO ✗ |
| ZZ dev=2% | 50/50  | 355 | 33.2% | 2.66 | +76.6 | +0.216 | -21.8 | GO ✓ |
| ZZ dev=2% | 70/30  | 355 | 33.2% | 2.66 | +76.9 | +0.217 | -19.7 | GO ✓ |
| ZZ dev=2% | TP2only | 341 | 19.9% | 5.59 | +107.0 | +0.314 | -23.7 | NO-GO ✗ |

### RVOL ≥ 1.2 + Session (London / NY)

| Approach | Split | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|----------|-------|-------:|-----:|--------:|-------:|-----:|------:|---------|
| Old MS   | 50/50   | 124 | 33.1% | 2.73 | +29.1 | +0.235 | -10.6 | GO ✓ |
| Old MS   | 70/30   | 124 | 33.1% | 2.68 | +26.9 | +0.217 |  -9.9 | GO ✓ |
| Old MS   | TP2only | 123 | 21.1% | 4.98 | +32.5 | +0.264 | -18.5 | NO-GO ✗ |
| ZZ dev=2% | 50/50  | 174 | 38.5% | 2.72 | +75.0 | +0.431 | -16.6 | GO ✓ |
| ZZ dev=2% | 70/30  | 174 | 38.5% | 2.75 | +77.0 | +0.442 | -15.5 | GO ✓ |
| ZZ dev=2% | TP2only | 171 | 22.2% | 5.66 | +82.3 | +0.481 | -15.0 | **GO ✓** |

### Key findings

- **70/30 is never better than 50/50.** Consistently equal win rate, lower or equal TotalR. No reason to use it.
- **TP2-only fails the verdict in both systems except one case:** ZZ + RVOL≥1.2 + Session, where it clears GO with the highest AvgR of any config (+0.481) and lowest MaxDD of any session config (-15.0). Trade-off: 22% win rate requires patience.
- **ZZ dev=2% is materially better than Old MS with RVOL+Session:** 38.5% vs 33.1% win rate, +0.431 vs +0.235 AvgR, 50% more trades. The ZigZag pivot definition (significance-based) appears to filter out more noise in this regime.
- **Best overall config for SOL: ZZ dev=2% + RVOL≥1.2 + Session + TP2-only** — GO verdict, +0.481 AvgR, -15.0 MaxDD.
- **ETH: NO-GO on all configs across both structure methods.** Not a SOL-specific finding — the VA Bounce strategy simply has no edge on ETH in this period.

---

## Fib 0.745 Entry vs VA Bounce — SOL SL & Entry Comparison (2026-05-13)

All tests: ZigZag market structure bias, 1h HTF, 5m entry, 2024-05-07 → 2026-05-07.
SL variants: **VA-SL** = VAL/VAH ± 15% VA-width | **TightSL** = swing_low/high ± 0.05 × swing_range (1.05 fib extension).

### Fib 0.745 retracement entry + TightSL (dev=2% and dev=3%)

Entry: 5m bar low touches swing_high − 0.745 × range (zone ±0.04), close above swing_low. No confirmation bar.

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|-------:|-----:|--------:|-------:|-----:|------:|---------|
| dev=2% zone=0.08 no filters          | 569 | 29.5% | 3.32 | +157.2 | +0.276 | -42.9 | GO ✓ |
| dev=2% zone=0.08 + RVOL≥1.2         | 409 | 29.6% | 3.14 |  +92.2 | +0.225 | -30.2 | GO ✓ |
| dev=2% zone=0.08 + RVOL≥1.2 + Sess  | 179 | 29.1% | 3.26 |  +42.5 | +0.237 | -25.0 | GO ✓ |
| dev=2% zone=0.08 + RVOL≥1.2 + Sess TP2only | 175 | 24.6% | 4.30 | +53.1 | +0.303 | -25.0 | GO ✓ |
| dev=2% zone=0.12 + RVOL≥1.2 + Sess  | 185 | 31.4% | 3.04 |  +49.5 | +0.267 | -25.0 | GO ✓ |
| dev=2% zone=0.12 + RVOL≥1.2 + Sess TP2only | 181 | 26.0% | 4.08 | +57.7 | +0.319 | -25.0 | GO ✓ |
| dev=3% zone=0.08 no filters          | 266 | 28.6% | 3.66 |  +88.1 | +0.331 | -17.7 | GO ✓ |
| dev=3% zone=0.08 + RVOL≥1.2         | 204 | 30.4% | 3.61 |  +81.7 | +0.400 | -12.4 | GO ✓ |
| dev=3% zone=0.08 + RVOL≥1.2 + Sess  |  99 | 31.3% | 3.61 |  +43.8 | +0.443 |  -7.0 | GO ✓ |
| dev=3% zone=0.08 + RVOL≥1.2 + Sess TP2only |  95 | 28.4% | 4.55 | +54.8 | +0.577 | -7.0 | **GO ✓** |

### VA Bounce entry — VA-SL vs TightSL (dev=2%)

Entry: 5m bar touches VAL/VAH zone + confirmation bar. SL either from VAL buffer or swing extreme.

| Config | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|--------|-------:|-----:|--------:|-------:|-----:|------:|---------|
| VA-SL  50/50  no filters             | 459 | 30.1% | 2.92 |  +82.1 | +0.179 | -29.0 | NO-GO ✗ |
| TightSL 50/50  no filters            | 255 | 26.7% | 3.84 |  +74.0 | +0.290 | -32.3 | GO ✓ |
| TightSL TP2only no filters           | 249 | 15.3% | 8.62 | +116.6 | +0.468 | -37.0 | NO-GO ✗ |
| VA-SL  50/50  + RVOL≥1.2             | 355 | 33.2% | 2.66 |  +76.6 | +0.216 | -21.8 | GO ✓ |
| TightSL 50/50  + RVOL≥1.2           | 164 | 29.9% | 4.02 |  +82.0 | +0.500 | -17.0 | GO ✓ |
| TightSL TP2only + RVOL≥1.2          | 161 | 18.0% | 8.77 | +122.4 | +0.760 | -19.3 | **GO ✓** |
| VA-SL  50/50  + RVOL≥1.2 + Session  | 174 | 38.5% | 2.72 |  +75.0 | +0.431 | -16.6 | GO ✓ |
| TightSL 50/50  + RVOL≥1.2 + Session |  66 | 36.4% | 2.42 |  +16.1 | +0.245 |  -9.0 | GO ✓ |
| VA-SL  TP2only + RVOL≥1.2 + Session | 171 | 22.2% | 5.66 |  +82.3 | +0.481 | -15.0 | GO ✓ |
| TightSL TP2only + RVOL≥1.2 + Session |  66 | 21.2% | 5.43 |  +24.0 | +0.364 | -12.3 | GO ✓ |

### Summary — best configs across all approaches (SOL, 2 years)

| Approach | Trades | Win% | AvgR | MaxDD | Notes |
|----------|-------:|-----:|-----:|------:|-------|
| VA Bounce + VA-SL + Session + TP2only       | 171 | 22.2% | +0.481 | -15.0 | Previous best |
| VA Bounce + TightSL + RVOL≥1.2 + TP2only   | 161 | 18.0% | +0.760 | -19.3 | Highest AvgR; low win rate, no session filter |
| Fib 0.745 + TightSL + dev=3% + RVOL≥1.2 + Session + TP2only | 95 | 28.4% | +0.577 | -7.0 | Best MaxDD; fewest trades |

### Key findings

- **TightSL hurts VA Bounce when combined with session filter** — trade count collapses (174→66) and AvgR drops. The session-filtered VA entries are mid-swing, making the swing-extreme SL too far and killing R:R on many setups.
- **TightSL helps VA Bounce without session filter** — RVOL≥1.2 + TP2only jumps from +0.216 to +0.760 AvgR. Best raw AvgR of any config tested, but 18% win rate and trades around the clock.
- **Fib 0.745 entry is the most consistent** — all 12 configs are GO, MaxDD is lowest (-7.0 best case), and win rates are in the 28–31% range (more stable than the 18–22% of TP2only VA setups).
- **Zone width (0.08 vs 0.12) has no meaningful effect** on fib entry results — the R:R filter is the active gate, not the zone boundary.
- **dev=3% produces better quality than dev=2% for fib entry** — fewer trades but higher AvgR and lower MaxDD across all configs.

---

## Selected Strategies (2026-05-13)

Period: 2024-05-07 → 2026-05-07 | Entry TF: 5m | Structure: 1h ZigZag MS (swing_count ≥ 2)
Entry: Fib 0.745 retracement of last ZZ swing | SL: swing_low/high − 0.05 × swing_range (1.05 fib extension)

| Asset | ZZ Dev | Zone | RVOL | Session | TP Split | Trades | Win% | AvgWinR | TotalR | AvgR | MaxDD | Verdict |
|-------|-------:|-----:|-----:|---------|----------|-------:|-----:|--------:|-------:|-----:|------:|---------|
| SOL | 3% | 0.08 | ≥1.2 | London/NY | 50/50 | 99 | 31.3% | 3.61 | +43.8 | +0.443 | -7.0 | GO ✓ |
| ETH | 2% | 0.08 | — | London/NY | TP2only | 193 | 27.5% | 4.40 | +93.4 | +0.484 | -23.2 | GO ✓ |

### SOL — Fib 0.745 + dev=3% + RVOL≥1.2 + Session + 50/50

- ~50 trades/year, 31.3% win rate, AvgWinR 3.61, AvgR +0.443 per trade
- SL just below last confirmed Higher Low (1.05 fib extension) — tight risk, MaxDD only -7.0R over 2 years
- 50% closes at last swing high (TP1), 50% runs to 127.2% fib extension (TP2); SL moves to breakeven after TP1
- RVOL≥1.2 filters low-volume setups; session window (London 08:30–11:30 / NY 16:00–20:00 Vienna time)
- dev=3% ZigZag produces fewer but more significant swings than dev=2% — better signal quality for SOL

### ETH — Fib 0.745 + dev=2% + Session + TP2only

- ~97 trades/year, 27.5% win rate, AvgWinR 4.40, AvgR +0.484 per trade
- Full position runs to 127.2% fib extension — no partial exit, no breakeven move; accept full -1R on losers
- Session filter (London/NY) is the only quality gate — RVOL filter consistently hurt ETH results
- dev=2% required for ETH: smaller swings need tighter ZigZag deviation to produce enough signals
- MaxDD -23.2R over 2 years is the main risk; psychologically demanding with 27.5% win rate

---

## Run log

| Date       | Asset | HTF | Period              | Command                                              |
|------------|-------|-----|---------------------|------------------------------------------------------|
| 2026-05-08 | ETH   | 1h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets ETH` |
| 2026-05-08 | SOL   | 1h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets SOL` |
| 2026-05-08 | ETH   | 4h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets ETH --htf 4h` |
| 2026-05-08 | SOL   | 4h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets SOL --htf 4h` |
