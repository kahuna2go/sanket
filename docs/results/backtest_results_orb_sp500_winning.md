xyz:SP500 5m: 41327 bars (cached)
xyz:SP500 4h: 1410 bars (cached)

==============================================================================================================
ORB Backtest — xyz:SP500 | 41327 × 5m bars | 1410 × 4h bars
==============================================================================================================

  BREAKOUT FUNNEL DIAGNOSTIC
  ────────────────────────────────────────────────────────────
  Trading days total          : 498
  Days with valid OR (≥2 bars): 498  (100.0% of days)
  Bias — bull=294  bear=143  neutral=61  (59.0% / 28.7% / 12.2%)
  Raw breakouts (5m close)    : 631  (126.7% of OR days)  — long=332  short=299
  Bias-aligned (taken)        : 312  (49.4% of raw breakouts)  — long=217  short=95
  Filtered out by bias        : 319  — long filtered=115 (bias≠bull)  short filtered=204 (bias≠bear)


  RETEST entry    |  SL=retest low+buf  |  TP=trail (0.5×range trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [trail/retest/retest_low] |  238 trades | win= 47.1% | avgWinR=2.83 | totalR=+191.6 | avgR=+0.805 | maxDD=-9.0R | GO ✓ | sl=125 time_stop=6 trail=107
  SL=5%  / slope≥0.02% [trail/retest/retest_low] |  238 trades | win= 43.7% | avgWinR=4.29 | totalR=+312.6 | avgR=+1.313 | maxDD=-10.1R | GO ✓ | sl=134 time_stop=4 trail=100
  SL=10% / slope≥0.10% [trail/retest/retest_low] |  102 trades | win= 52.0% | avgWinR=2.77 | totalR=+97.8 | avgR=+0.959 | maxDD=-7.3R | GO ✓ | sl=49 time_stop=2 trail=51
  SL=5%  / slope≥0.10% [trail/retest/retest_low] |  102 trades | win= 49.0% | avgWinR=4.07 | totalR=+151.7 | avgR=+1.487 | maxDD=-6.4R | GO ✓ | sl=52 time_stop=2 trail=48
  SL=10% / no slope filter [trail/retest/retest_low] |  263 trades | win= 46.8% | avgWinR=2.76 | totalR=+199.3 | avgR=+0.758 | maxDD=-10.0R | GO ✓ | sl=139 time_stop=6 trail=118
  SL=5%  / no slope filter [trail/retest/retest_low] |  263 trades | win= 43.3% | avgWinR=4.15 | totalR=+324.0 | avgR=+1.232 | maxDD=-10.0R | GO ✓ | sl=149 time_stop=4 trail=110
==============================================================================================================


==========================================================================================
  Detail: SL=5% / no slope filter [trail/retest/retest_low]
==========================================================================================

  YEAR-BY-YEAR
  ─────────────────────────────────────────────────────────────────────────────────────
  2024  |    74 trades | win= 41.9% | totalR=  +76.9 | avgR=+1.039 | maxDD=-10.0R
  2025  |   128 trades | win= 46.9% | totalR= +186.6 | avgR=+1.458 | maxDD=-7.1R
  2026  |    61 trades | win= 37.7% | totalR=  +60.5 | avgR=+0.992 | maxDD=-9.0R
  ─────────────────────────────────────────────────────────────────────────────────────
  TOTAL  |   263 trades | win= 43.3% | totalR= +324.0 | avgR=+1.232 | maxDD=-10.0R

  LONG vs SHORT
  ─────────────────────────────────────────────────────────────────────────────────────
  LONG   |   182 trades | win= 39.0% | totalR= +173.9 | avgR=+0.955 | maxDD=-10.2R
  SHORT  |    81 trades | win= 53.1% | totalR= +150.1 | avgR=+1.853 | maxDD=-6.0R
==========================================================================================

