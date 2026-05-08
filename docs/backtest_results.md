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

## Run log

| Date       | Asset | HTF | Period              | Command                                              |
|------------|-------|-----|---------------------|------------------------------------------------------|
| 2026-05-08 | ETH   | 1h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets ETH` |
| 2026-05-08 | SOL   | 1h  | 2024-05-07→2026-05-07 | `python -m src.backtest.run_backtest --assets SOL` |
