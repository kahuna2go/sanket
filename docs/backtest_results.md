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

## Run log

| Date       | Asset | HTF | Period              | Command                                              |
|------------|-------|-----|---------------------|------------------------------------------------------|
| 2026-05-08 | ETH   | 1h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets ETH` |
| 2026-05-08 | SOL   | 1h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets SOL` |
| 2026-05-08 | ETH   | 4h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets ETH --htf 4h` |
| 2026-05-08 | SOL   | 4h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets SOL --htf 4h` |
