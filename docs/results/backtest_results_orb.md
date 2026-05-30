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


  RETEST entry    |  SL=OR extreme+buf  |  TP=range (0.5R/1.0R)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [range/retest/or_extreme] |   40 trades | win= 77.5% | avgWinR=0.79 | totalR=+15.9 | avgR=+0.397 | maxDD=-2.0R | GO ✓ | sl=17 time_stop=4 tp2=19
  SL=5%  / slope≥0.02% [range/retest/or_extreme] |   40 trades | win= 75.0% | avgWinR=0.80 | totalR=+14.3 | avgR=+0.358 | maxDD=-2.0R | GO ✓ | sl=18 time_stop=4 tp2=18
  SL=10% / slope≥0.10% [range/retest/or_extreme] |   28 trades | win= 82.1% | avgWinR=0.75 | totalR=+12.2 | avgR=+0.436 | maxDD=-2.0R | GO ✓ | sl=14 time_stop=1 tp2=13
  SL=5%  / slope≥0.10% [range/retest/or_extreme] |   28 trades | win= 78.6% | avgWinR=0.75 | totalR=+10.5 | avgR=+0.376 | maxDD=-2.0R | GO ✓ | sl=15 time_stop=1 tp2=12
  SL=10% / no slope filter [range/retest/or_extreme] |   44 trades | win= 77.3% | avgWinR=0.79 | totalR=+17.2 | avgR=+0.390 | maxDD=-2.0R | GO ✓ | sl=19 time_stop=5 tp2=20
  SL=5%  / no slope filter [range/retest/or_extreme] |   44 trades | win= 75.0% | avgWinR=0.79 | totalR=+15.6 | avgR=+0.355 | maxDD=-2.0R | GO ✓ | sl=20 time_stop=5 tp2=19

  RETEST entry    |  SL=OR extreme+buf  |  TP=trail (0.5×range trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [trail/retest/or_extreme] |   40 trades | win= 77.5% | avgWinR=0.83 | totalR=+17.3 | avgR=+0.432 | maxDD=-2.0R | GO ✓ | sl=8 time_stop=4 trail=28
  SL=5%  / slope≥0.02% [trail/retest/or_extreme] |   40 trades | win= 75.0% | avgWinR=0.85 | totalR=+15.9 | avgR=+0.398 | maxDD=-2.1R | GO ✓ | sl=9 time_stop=4 trail=27
  SL=10% / slope≥0.10% [trail/retest/or_extreme] |   28 trades | win= 82.1% | avgWinR=0.81 | totalR=+13.7 | avgR=+0.488 | maxDD=-2.0R | GO ✓ | sl=5 time_stop=2 trail=21
  SL=5%  / slope≥0.10% [trail/retest/or_extreme] |   28 trades | win= 78.6% | avgWinR=0.83 | totalR=+12.2 | avgR=+0.434 | maxDD=-2.1R | GO ✓ | sl=6 time_stop=2 trail=20
  SL=10% / no slope filter [trail/retest/or_extreme] |   44 trades | win= 77.3% | avgWinR=0.82 | totalR=+18.5 | avgR=+0.420 | maxDD=-2.0R | GO ✓ | sl=9 time_stop=5 trail=30
  SL=5%  / no slope filter [trail/retest/or_extreme] |   44 trades | win= 75.0% | avgWinR=0.84 | totalR=+17.1 | avgR=+0.390 | maxDD=-2.1R | GO ✓ | sl=10 time_stop=5 trail=29

  RETEST entry    |  SL=OR extreme+buf  |  TP=swing_trail (50%@TP1, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [swing_trail/retest/or_extreme] |   40 trades | win= 77.5% | avgWinR=0.80 | totalR=+16.3 | avgR=+0.408 | maxDD=-2.0R | GO ✓ | sl=8 time_stop=2 trail=30
  SL=5%  / slope≥0.02% [swing_trail/retest/or_extreme] |   40 trades | win= 75.0% | avgWinR=0.82 | totalR=+14.9 | avgR=+0.374 | maxDD=-2.4R | GO ✓ | sl=9 time_stop=2 trail=29
  SL=10% / slope≥0.10% [swing_trail/retest/or_extreme] |   28 trades | win= 82.1% | avgWinR=0.78 | totalR=+13.0 | avgR=+0.463 | maxDD=-2.0R | GO ✓ | sl=5 time_stop=1 trail=22
  SL=5%  / slope≥0.10% [swing_trail/retest/or_extreme] |   28 trades | win= 78.6% | avgWinR=0.79 | totalR=+11.5 | avgR=+0.410 | maxDD=-2.4R | GO ✓ | sl=6 time_stop=1 trail=21
  SL=10% / no slope filter [swing_trail/retest/or_extreme] |   44 trades | win= 77.3% | avgWinR=0.79 | totalR=+17.3 | avgR=+0.392 | maxDD=-2.0R | GO ✓ | sl=9 time_stop=2 trail=33
  SL=5%  / no slope filter [swing_trail/retest/or_extreme] |   44 trades | win= 75.0% | avgWinR=0.80 | totalR=+15.9 | avgR=+0.362 | maxDD=-2.4R | GO ✓ | sl=10 time_stop=2 trail=32

  RETEST entry    |  SL=OR extreme+buf  |  TP=tp2_swing (full@TP2, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_swing/retest/or_extreme] |   40 trades | win= 72.5% | avgWinR=0.85 | totalR=+14.2 | avgR=+0.355 | maxDD=-2.5R | GO ✓ | sl=10 time_stop=8 trail=22
  SL=5%  / slope≥0.02% [tp2_swing/retest/or_extreme] |   40 trades | win= 70.0% | avgWinR=0.90 | totalR=+13.7 | avgR=+0.343 | maxDD=-2.5R | GO ✓ | sl=11 time_stop=8 trail=21
  SL=10% / slope≥0.10% [tp2_swing/retest/or_extreme] |   28 trades | win= 75.0% | avgWinR=0.88 | totalR=+11.5 | avgR=+0.409 | maxDD=-2.0R | GO ✓ | sl=7 time_stop=5 trail=16
  SL=5%  / slope≥0.10% [tp2_swing/retest/or_extreme] |   28 trades | win= 71.4% | avgWinR=0.94 | totalR=+10.7 | avgR=+0.383 | maxDD=-2.5R | GO ✓ | sl=8 time_stop=5 trail=15
  SL=10% / no slope filter [tp2_swing/retest/or_extreme] |   44 trades | win= 70.5% | avgWinR=0.83 | totalR=+13.3 | avgR=+0.302 | maxDD=-4.0R | GO ✓ | sl=12 time_stop=9 trail=23
  SL=5%  / no slope filter [tp2_swing/retest/or_extreme] |   44 trades | win= 68.2% | avgWinR=0.88 | totalR=+12.9 | avgR=+0.292 | maxDD=-4.0R | GO ✓ | sl=13 time_stop=9 trail=22

  RETEST entry    |  SL=OR extreme+buf  |  TP=tp2_partial (70%@TP2+BE, 30% swing trail sw=1)
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
  SL=10% / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw1] |   40 trades | win= 55.0% | avgWinR=0.87 | totalR=+10.7 | avgR=+0.266 | maxDD=-2.0R | NO-GO ✗ | be=9 sl=8 time_stop=4 trail=19
  SL=10% / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw2] |   40 trades | win= 55.0% | avgWinR=0.87 | totalR=+10.6 | avgR=+0.264 | maxDD=-2.0R | NO-GO ✗ | be=9 sl=8 time_stop=4 trail=19
  SL=10% / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw3] |   40 trades | win= 55.0% | avgWinR=0.87 | totalR=+10.7 | avgR=+0.267 | maxDD=-2.0R | NO-GO ✗ | be=9 sl=8 time_stop=5 trail=18
  SL=5%  / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw1] |   40 trades | win= 52.5% | avgWinR=0.92 | totalR=+9.7 | avgR=+0.242 | maxDD=-2.2R | NO-GO ✗ | be=9 sl=9 time_stop=4 trail=18
  SL=5%  / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw2] |   40 trades | win= 52.5% | avgWinR=0.91 | totalR=+9.6 | avgR=+0.240 | maxDD=-2.2R | NO-GO ✗ | be=9 sl=9 time_stop=4 trail=18
  SL=5%  / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw3] |   40 trades | win= 52.5% | avgWinR=0.92 | totalR=+9.8 | avgR=+0.244 | maxDD=-2.2R | NO-GO ✗ | be=9 sl=9 time_stop=5 trail=17
  SL=10% / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw1] |   28 trades | win= 50.0% | avgWinR=0.91 | totalR=+7.8 | avgR=+0.279 | maxDD=-2.0R | NO-GO ✗ | be=9 sl=5 time_stop=1 trail=13
  SL=10% / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw2] |   28 trades | win= 50.0% | avgWinR=0.93 | totalR=+8.1 | avgR=+0.289 | maxDD=-2.0R | NO-GO ✗ | be=9 sl=5 time_stop=1 trail=13
  SL=10% / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw3] |   28 trades | win= 50.0% | avgWinR=0.95 | totalR=+8.3 | avgR=+0.296 | maxDD=-2.0R | NO-GO ✗ | be=9 sl=5 time_stop=2 trail=12
  SL=5%  / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw1] |   28 trades | win= 46.4% | avgWinR=0.97 | totalR=+6.6 | avgR=+0.234 | maxDD=-2.2R | NO-GO ✗ | be=9 sl=6 time_stop=1 trail=12
  SL=5%  / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw2] |   28 trades | win= 46.4% | avgWinR=0.99 | totalR=+6.9 | avgR=+0.245 | maxDD=-2.2R | NO-GO ✗ | be=9 sl=6 time_stop=1 trail=12
  SL=5%  / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw3] |   28 trades | win= 46.4% | avgWinR=1.01 | totalR=+7.1 | avgR=+0.254 | maxDD=-2.2R | NO-GO ✗ | be=9 sl=6 time_stop=2 trail=11
  SL=10% / no slope filter [tp2_partial+BE/retest/or_extreme/sw1] |   44 trades | win= 54.5% | avgWinR=0.86 | totalR=+11.1 | avgR=+0.252 | maxDD=-2.0R | NO-GO ✗ | be=10 sl=9 time_stop=5 trail=20
  SL=10% / no slope filter [tp2_partial+BE/retest/or_extreme/sw2] |   44 trades | win= 54.5% | avgWinR=0.86 | totalR=+11.0 | avgR=+0.250 | maxDD=-2.0R | NO-GO ✗ | be=10 sl=9 time_stop=5 trail=20
  SL=10% / no slope filter [tp2_partial+BE/retest/or_extreme/sw3] |   44 trades | win= 54.5% | avgWinR=0.86 | totalR=+11.1 | avgR=+0.252 | maxDD=-2.0R | NO-GO ✗ | be=10 sl=9 time_stop=6 trail=19
  SL=5%  / no slope filter [tp2_partial+BE/retest/or_extreme/sw1] |   44 trades | win= 52.3% | avgWinR=0.90 | totalR=+10.2 | avgR=+0.231 | maxDD=-2.2R | NO-GO ✗ | be=10 sl=10 time_stop=5 trail=19
  SL=5%  / no slope filter [tp2_partial+BE/retest/or_extreme/sw2] |   44 trades | win= 52.3% | avgWinR=0.90 | totalR=+10.1 | avgR=+0.229 | maxDD=-2.2R | NO-GO ✗ | be=10 sl=10 time_stop=5 trail=19
  SL=5%  / no slope filter [tp2_partial+BE/retest/or_extreme/sw3] |   44 trades | win= 52.3% | avgWinR=0.91 | totalR=+10.3 | avgR=+0.233 | maxDD=-2.2R | NO-GO ✗ | be=10 sl=10 time_stop=6 trail=18

  RETEST entry    |  SL=OR extreme+buf  |  TP=fixed R:R (2×/3×SL)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [fixed_rr/retest/or_extreme] |   40 trades | win= 62.5% | avgWinR=0.99 | totalR=+10.2 | avgR=+0.255 | maxDD=-3.2R | GO ✓ | sl=14 time_stop=24 tp2=2
  SL=5%  / slope≥0.02% [fixed_rr/retest/or_extreme] |   40 trades | win= 62.5% | avgWinR=1.01 | totalR=+10.7 | avgR=+0.268 | maxDD=-3.1R | GO ✓ | sl=14 time_stop=24 tp2=2
  SL=10% / slope≥0.10% [fixed_rr/retest/or_extreme] |   28 trades | win= 64.3% | avgWinR=1.10 | totalR=+9.8 | avgR=+0.351 | maxDD=-2.7R | GO ✓ | sl=10 time_stop=16 tp2=2
  SL=5%  / slope≥0.10% [fixed_rr/retest/or_extreme] |   28 trades | win= 64.3% | avgWinR=1.12 | totalR=+10.2 | avgR=+0.365 | maxDD=-2.6R | GO ✓ | sl=10 time_stop=16 tp2=2
  SL=10% / no slope filter [fixed_rr/retest/or_extreme] |   44 trades | win= 61.4% | avgWinR=0.98 | totalR=+9.8 | avgR=+0.224 | maxDD=-4.0R | GO ✓ | sl=16 time_stop=26 tp2=2
  SL=5%  / no slope filter [fixed_rr/retest/or_extreme] |   44 trades | win= 61.4% | avgWinR=1.00 | totalR=+10.5 | avgR=+0.238 | maxDD=-4.0R | GO ✓ | sl=16 time_stop=26 tp2=2
==============================================================================================================

