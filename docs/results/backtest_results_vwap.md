# VWAP Backtest — BTC

Run: `python -m backtest.run_backtest_vwap` (2026-10-05)
Data: Binance BTCUSDT spot, 5m (→15m) and 1h (→4h), 2023-10-06 → 2026-10-05.
Costs: 0.0432%/side taker + 0.01%/side slippage = 0.1064% round trip. Multi-day longs pay 10% APR funding.
SL-before-TP when both inside one bar. All figures in R (risk = initial SL distance).

## Summary

| Strategy | Best config | N | Net avgR | Net totR | MaxDD | Verdict |
|---|---|---|---|---|---|---|
| Pullback to session VWAP (15m) | trend_bars=4 rr=2 | 1295 | −0.40 | −522 | −525 | Reject |
| VWAP ±kσ band fade (15m) | k=2.5 min_rr=1 | 670 | −0.46 | −306 | −307 | Reject |
| Rolling VWAP trend (4h) | days=3 sl_atr=2 long-only | 145 | +0.16 | +23 | −11.5 | Weak, not robust |
| Rolling VWAP trend (4h) | days=10 sl_atr=2 long+short | 151 | +0.13 | +19 | −11.2 | Weak, not robust |

## Gross vs net (costs = 0 vs real costs)

| Config | N | Gross avgR | Net avgR |
|---|---|---|---|
| pullback 15m tb4 rr2 | 1295 | +0.041 | −0.403 |
| pullback 1h tb4 rr2 | 779 | +0.069 | −0.126 |
| bands 15m k2.5 | 761 | −0.030 | −0.424 |
| bands 1h k2.5 | 411 | −0.099 | −0.370 |

Median 15m ATR on BTC is 0.28%, so a 15m stop is ~0.3% and the 0.106% round-trip cost alone is ~0.35R per trade.

## Findings

- **Intraday VWAP on BTC has no usable edge.** The pullback signal is barely positive before costs (+0.04 to +0.07R) and costs wipe it out. The band fade loses money even before costs: BTC trends away from VWAP more often than it reverts.
- **Rolling VWAP trend (multi-day) is the only net-positive family, but it is fragile.** days=3 and days=10 are positive, while days=5 in between is negative, which is a sign of noise rather than a stable edge. The days=3 long-only profit comes from 2023–2024 (+27R); 2025–2026 is −4R.
- None of the configs passes a reasonable go-live bar.

## Full output

```
 ROLLING (4h, N-day rolling VWAP trend)
  config                                    N   /mo     WR   avgR    totR   maxDD    PF   longR  shortR    2023    2024    2025    2026
  days=3 sl_atr=2.0 long+short            256   7.2  25.0% +0.062   +15.9   -16.1  1.20   +24.8    -8.9   +14.1   +11.2    -0.4    -9.0
  days=3 sl_atr=2.0 long-only             145   4.1  25.5% +0.161   +23.3   -11.5  1.51   +23.3    +0.0   +14.7   +12.7    -0.4    -3.7
  days=3 sl_atr=3.0 long+short            256   7.2  25.0% +0.044   +11.2   -10.3  1.22   +15.8    -4.5    +9.4    +7.3    +0.3    -5.8
  days=3 sl_atr=3.0 long-only             145   4.1  25.5% +0.104   +15.0    -8.6  1.49   +15.0    +0.0    +9.8    +8.4    -0.3    -2.9
  days=5 sl_atr=2.0 long+short            250   7.1  18.0% -0.113   -28.3   -34.7  0.65   -20.8    -7.4    -2.9   -10.3   -21.5    +6.5
  days=5 sl_atr=2.0 long-only             132   3.8  18.2% -0.116   -15.3   -20.6  0.65   -15.3    +0.0    -2.0    +4.2   -15.7    -1.8
  days=5 sl_atr=3.0 long+short            250   7.1  18.0% -0.075   -18.7   -23.2  0.65   -13.1    -5.6    -1.9    -7.3   -13.9    +4.5
  days=5 sl_atr=3.0 long-only             132   3.8  18.2% -0.073    -9.7   -12.7  0.66    -9.7    +0.0    -1.4    +2.7   -10.0    -1.1
  days=10 sl_atr=2.0 long+short           151   4.3  18.5% +0.128   +19.4   -11.2  1.38    +7.9   +11.4    +5.5   +11.7    -6.9    +9.1
  days=10 sl_atr=2.0 long-only             96   2.8  16.7% +0.071    +6.9   -16.3  1.20    +6.9    +0.0    +5.5   +11.3    -1.4    -8.5
  days=10 sl_atr=3.0 long+short           151   4.3  18.5% +0.073   +11.0    -8.0  1.31    +3.4    +7.6    +3.3    +6.8    -5.1    +6.0
  days=10 sl_atr=3.0 long-only             96   2.8  16.7% +0.031    +2.9   -11.2  1.12    +2.9    +0.0    +3.3    +6.9    -1.4    -5.8
```
