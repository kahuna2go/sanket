xyz:SP500 5m: 5304 bars (cached)
xyz:SP500 4h: 584 bars (cached)

==============================================================================================================
ORB Backtest — xyz:SP500 | 5304 × 5m bars | 584 × 4h bars
==============================================================================================================

  BREAKOUT FUNNEL DIAGNOSTIC
  ────────────────────────────────────────────────────────────
  Trading days total          : 68
  Days with valid OR (≥2 bars): 68  (100.0% of days)
  Bias — bull=39  bear=22  neutral=7  (57.4% / 32.4% / 10.3%)
  Raw breakouts (5m close)    : 84  (123.5% of OR days)  — long=49  short=35
  Bias-aligned (taken)        : 44  (52.4% of raw breakouts)  — long=31  short=13
  Filtered out by bias        : 40  — long filtered=18 (bias≠bull)  short filtered=22 (bias≠bear)


  RETEST entry    |  SL=OR extreme+buf  |  TP=tp2_partial (70%@TP2, 30% swing trail sw=1)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_partial/retest/or_extreme/sw1] |   40 trades | win= 72.5% | avgWinR=0.83 | totalR=+13.4 | avgR=+0.336 | maxDD=-2.2R | GO ✓ | sl=10 time_stop=8 trail=22
  SL=10% / slope≥0.02% [tp2_partial/retest/or_extreme/sw2] |   40 trades | win= 72.5% | avgWinR=0.82 | totalR=+13.3 | avgR=+0.333 | maxDD=-2.2R | GO ✓ | sl=10 time_stop=8 trail=22
  SL=10% / slope≥0.02% [tp2_partial/retest/or_extreme/sw3] |   40 trades | win= 72.5% | avgWinR=0.83 | totalR=+13.4 | avgR=+0.336 | maxDD=-2.2R | GO ✓ | sl=10 time_stop=9 trail=21
  SL=5%  / slope≥0.02% [tp2_partial/retest/or_extreme/sw1] |   40 trades | win= 70.0% | avgWinR=0.87 | totalR=+12.7 | avgR=+0.318 | maxDD=-2.2R | GO ✓ | sl=11 time_stop=8 trail=21
  SL=5%  / slope≥0.02% [tp2_partial/retest/or_extreme/sw2] |   40 trades | win= 70.0% | avgWinR=0.86 | totalR=+12.6 | avgR=+0.315 | maxDD=-2.2R | GO ✓ | sl=11 time_stop=8 trail=21
  SL=5%  / slope≥0.02% [tp2_partial/retest/or_extreme/sw3] |   40 trades | win= 70.0% | avgWinR=0.87 | totalR=+12.7 | avgR=+0.319 | maxDD=-2.2R | GO ✓ | sl=11 time_stop=9 trail=20
  SL=10% / slope≥0.10% [tp2_partial/retest/or_extreme/sw1] |   28 trades | win= 75.0% | avgWinR=0.84 | totalR=+10.6 | avgR=+0.378 | maxDD=-2.0R | GO ✓ | sl=7 time_stop=5 trail=16
  SL=10% / slope≥0.10% [tp2_partial/retest/or_extreme/sw2] |   28 trades | win= 75.0% | avgWinR=0.85 | totalR=+10.8 | avgR=+0.387 | maxDD=-2.0R | GO ✓ | sl=7 time_stop=5 trail=16
  SL=10% / slope≥0.10% [tp2_partial/retest/or_extreme/sw3] |   28 trades | win= 75.0% | avgWinR=0.86 | totalR=+11.0 | avgR=+0.394 | maxDD=-2.0R | GO ✓ | sl=7 time_stop=6 trail=15
  SL=5%  / slope≥0.10% [tp2_partial/retest/or_extreme/sw1] |   28 trades | win= 71.4% | avgWinR=0.88 | totalR=+9.6 | avgR=+0.342 | maxDD=-2.2R | GO ✓ | sl=8 time_stop=5 trail=15
  SL=5%  / slope≥0.10% [tp2_partial/retest/or_extreme/sw2] |   28 trades | win= 71.4% | avgWinR=0.89 | totalR=+9.8 | avgR=+0.351 | maxDD=-2.2R | GO ✓ | sl=8 time_stop=5 trail=15
  SL=5%  / slope≥0.10% [tp2_partial/retest/or_extreme/sw3] |   28 trades | win= 71.4% | avgWinR=0.90 | totalR=+10.1 | avgR=+0.360 | maxDD=-2.2R | GO ✓ | sl=8 time_stop=6 trail=14
  SL=10% / no slope filter [tp2_partial/retest/or_extreme/sw1] |   44 trades | win= 70.5% | avgWinR=0.82 | totalR=+12.9 | avgR=+0.292 | maxDD=-4.0R | GO ✓ | sl=12 time_stop=9 trail=23
  SL=10% / no slope filter [tp2_partial/retest/or_extreme/sw2] |   44 trades | win= 70.5% | avgWinR=0.82 | totalR=+12.7 | avgR=+0.289 | maxDD=-4.0R | GO ✓ | sl=12 time_stop=9 trail=23
  SL=10% / no slope filter [tp2_partial/retest/or_extreme/sw3] |   44 trades | win= 70.5% | avgWinR=0.82 | totalR=+12.8 | avgR=+0.292 | maxDD=-4.0R | GO ✓ | sl=12 time_stop=10 trail=22
  SL=5%  / no slope filter [tp2_partial/retest/or_extreme/sw1] |   44 trades | win= 68.2% | avgWinR=0.86 | totalR=+12.2 | avgR=+0.277 | maxDD=-4.0R | GO ✓ | sl=13 time_stop=9 trail=22
  SL=5%  / no slope filter [tp2_partial/retest/or_extreme/sw2] |   44 trades | win= 68.2% | avgWinR=0.85 | totalR=+12.1 | avgR=+0.274 | maxDD=-4.0R | GO ✓ | sl=13 time_stop=9 trail=22
  SL=5%  / no slope filter [tp2_partial/retest/or_extreme/sw3] |   44 trades | win= 68.2% | avgWinR=0.86 | totalR=+12.2 | avgR=+0.278 | maxDD=-4.0R | GO ✓ | sl=13 time_stop=10 trail=21
==============================================================================================================

