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


  BREAKOUT entry  |  SL=OR extreme+buf  |  TP=range (0.5R/1.0R)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [range/breakout/or_extreme] |  312 trades | win= 80.4% | avgWinR=0.43 | totalR=+51.0 | avgR=+0.163 | maxDD=-3.5R | GO ✓ | sl=139 time_stop=27 tp2=146
  SL=5%  / slope≥0.02% [range/breakout/or_extreme] |  312 trades | win= 79.8% | avgWinR=0.45 | totalR=+52.4 | avgR=+0.168 | maxDD=-3.4R | GO ✓ | sl=141 time_stop=25 tp2=146
  SL=10% / slope≥0.10% [range/breakout/or_extreme] |  137 trades | win= 79.6% | avgWinR=0.46 | totalR=+23.8 | avgR=+0.174 | maxDD=-3.0R | GO ✓ | sl=54 time_stop=9 tp2=74
  SL=5%  / slope≥0.10% [range/breakout/or_extreme] |  137 trades | win= 79.6% | avgWinR=0.48 | totalR=+25.9 | avgR=+0.189 | maxDD=-3.0R | GO ✓ | sl=54 time_stop=9 tp2=74
  SL=10% / no slope filter [range/breakout/or_extreme] |  343 trades | win= 79.0% | avgWinR=0.42 | totalR=+47.2 | avgR=+0.138 | maxDD=-5.1R | GO ✓ | sl=161 time_stop=28 tp2=154
  SL=5%  / no slope filter [range/breakout/or_extreme] |  343 trades | win= 78.4% | avgWinR=0.44 | totalR=+49.0 | avgR=+0.143 | maxDD=-5.1R | GO ✓ | sl=163 time_stop=26 tp2=154

  RETEST entry    |  SL=OR extreme+buf  |  TP=range (0.5R/1.0R)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [range/retest/or_extreme] |  238 trades | win= 74.4% | avgWinR=0.51 | totalR=+29.5 | avgR=+0.124 | maxDD=-5.5R | GO ✓ | sl=118 time_stop=18 tp2=102
  SL=5%  / slope≥0.02% [range/retest/or_extreme] |  238 trades | win= 73.9% | avgWinR=0.53 | totalR=+31.9 | avgR=+0.134 | maxDD=-5.4R | GO ✓ | sl=120 time_stop=16 tp2=102
  SL=10% / slope≥0.10% [range/retest/or_extreme] |  102 trades | win= 72.5% | avgWinR=0.49 | totalR=+8.3 | avgR=+0.081 | maxDD=-4.6R | GO ✓ | sl=55 time_stop=6 tp2=41
  SL=5%  / slope≥0.10% [range/retest/or_extreme] |  102 trades | win= 72.5% | avgWinR=0.51 | totalR=+10.0 | avgR=+0.098 | maxDD=-4.6R | GO ✓ | sl=55 time_stop=6 tp2=41
  SL=10% / no slope filter [range/retest/or_extreme] |  263 trades | win= 73.8% | avgWinR=0.50 | totalR=+28.4 | avgR=+0.108 | maxDD=-7.1R | GO ✓ | sl=136 time_stop=19 tp2=108
  SL=5%  / no slope filter [range/retest/or_extreme] |  263 trades | win= 73.4% | avgWinR=0.52 | totalR=+31.2 | avgR=+0.119 | maxDD=-7.2R | GO ✓ | sl=138 time_stop=17 tp2=108

  RETEST entry    |  SL=retest low+buf  |  TP=range (0.5R/1.0R)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [range/retest/retest_low] |  238 trades | win= 47.1% | avgWinR=2.59 | totalR=+163.9 | avgR=+0.689 | maxDD=-9.0R | GO ✓ | sl=165 time_stop=4 tp2=69
  SL=5%  / slope≥0.02% [range/retest/retest_low] |  238 trades | win= 43.7% | avgWinR=3.90 | totalR=+271.1 | avgR=+1.139 | maxDD=-9.7R | GO ✓ | sl=171 time_stop=2 tp2=65
  SL=10% / slope≥0.10% [range/retest/retest_low] |  102 trades | win= 52.0% | avgWinR=2.35 | totalR=+75.6 | avgR=+0.741 | maxDD=-9.0R | GO ✓ | sl=67 time_stop=1 tp2=34
  SL=5%  / slope≥0.10% [range/retest/retest_low] |  102 trades | win= 49.0% | avgWinR=3.42 | totalR=+119.0 | avgR=+1.167 | maxDD=-7.9R | GO ✓ | sl=69 time_stop=1 tp2=32
  SL=10% / no slope filter [range/retest/retest_low] |  263 trades | win= 46.8% | avgWinR=2.55 | totalR=+174.1 | avgR=+0.662 | maxDD=-10.0R | GO ✓ | sl=184 time_stop=4 tp2=75
  SL=5%  / no slope filter [range/retest/retest_low] |  263 trades | win= 43.3% | avgWinR=3.80 | totalR=+284.3 | avgR=+1.081 | maxDD=-10.0R | GO ✓ | sl=191 time_stop=2 tp2=70

  BREAKOUT entry  |  SL=OR extreme+buf  |  TP=trail (0.5×range trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [trail/breakout/or_extreme] |  312 trades | win= 80.4% | avgWinR=0.45 | totalR=+56.2 | avgR=+0.180 | maxDD=-3.4R | GO ✓ | sl=53 time_stop=31 trail=228
  SL=5%  / slope≥0.02% [trail/breakout/or_extreme] |  312 trades | win= 79.8% | avgWinR=0.47 | totalR=+57.9 | avgR=+0.185 | maxDD=-3.4R | GO ✓ | sl=56 time_stop=29 trail=227
  SL=10% / slope≥0.10% [trail/breakout/or_extreme] |  137 trades | win= 79.6% | avgWinR=0.50 | totalR=+28.3 | avgR=+0.207 | maxDD=-3.0R | GO ✓ | sl=26 time_stop=11 trail=100
  SL=5%  / slope≥0.10% [trail/breakout/or_extreme] |  137 trades | win= 79.6% | avgWinR=0.52 | totalR=+30.6 | avgR=+0.223 | maxDD=-3.0R | GO ✓ | sl=26 time_stop=11 trail=100
  SL=10% / no slope filter [trail/breakout/or_extreme] |  343 trades | win= 79.0% | avgWinR=0.45 | totalR=+53.7 | avgR=+0.156 | maxDD=-5.4R | GO ✓ | sl=64 time_stop=32 trail=247
  SL=5%  / no slope filter [trail/breakout/or_extreme] |  343 trades | win= 78.4% | avgWinR=0.47 | totalR=+55.7 | avgR=+0.162 | maxDD=-5.4R | GO ✓ | sl=67 time_stop=30 trail=246

  RETEST entry    |  SL=OR extreme+buf  |  TP=trail (0.5×range trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [trail/retest/or_extreme] |  238 trades | win= 74.4% | avgWinR=0.54 | totalR=+36.1 | avgR=+0.152 | maxDD=-5.6R | GO ✓ | sl=58 time_stop=21 trail=159
  SL=5%  / slope≥0.02% [trail/retest/or_extreme] |  238 trades | win= 73.9% | avgWinR=0.57 | totalR=+38.9 | avgR=+0.163 | maxDD=-5.5R | GO ✓ | sl=60 time_stop=19 trail=159
  SL=10% / slope≥0.10% [trail/retest/or_extreme] |  102 trades | win= 72.5% | avgWinR=0.57 | totalR=+13.9 | avgR=+0.137 | maxDD=-4.4R | GO ✓ | sl=28 time_stop=6 trail=68
  SL=5%  / slope≥0.10% [trail/retest/or_extreme] |  102 trades | win= 72.5% | avgWinR=0.59 | totalR=+15.9 | avgR=+0.156 | maxDD=-4.4R | GO ✓ | sl=28 time_stop=6 trail=68
  SL=10% / no slope filter [trail/retest/or_extreme] |  263 trades | win= 73.8% | avgWinR=0.53 | totalR=+35.0 | avgR=+0.133 | maxDD=-6.8R | GO ✓ | sl=66 time_stop=22 trail=175
  SL=5%  / no slope filter [trail/retest/or_extreme] |  263 trades | win= 73.4% | avgWinR=0.56 | totalR=+38.0 | avgR=+0.145 | maxDD=-6.7R | GO ✓ | sl=68 time_stop=20 trail=175

  RETEST entry    |  SL=retest low+buf  |  TP=trail (0.5×range trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [trail/retest/retest_low] |  238 trades | win= 47.1% | avgWinR=2.83 | totalR=+191.6 | avgR=+0.805 | maxDD=-9.0R | GO ✓ | sl=125 time_stop=6 trail=107
  SL=5%  / slope≥0.02% [trail/retest/retest_low] |  238 trades | win= 43.7% | avgWinR=4.29 | totalR=+312.6 | avgR=+1.313 | maxDD=-10.1R | GO ✓ | sl=134 time_stop=4 trail=100
  SL=10% / slope≥0.10% [trail/retest/retest_low] |  102 trades | win= 52.0% | avgWinR=2.77 | totalR=+97.8 | avgR=+0.959 | maxDD=-7.3R | GO ✓ | sl=49 time_stop=2 trail=51
  SL=5%  / slope≥0.10% [trail/retest/retest_low] |  102 trades | win= 49.0% | avgWinR=4.07 | totalR=+151.7 | avgR=+1.487 | maxDD=-6.4R | GO ✓ | sl=52 time_stop=2 trail=48
  SL=10% / no slope filter [trail/retest/retest_low] |  263 trades | win= 46.8% | avgWinR=2.76 | totalR=+199.3 | avgR=+0.758 | maxDD=-10.0R | GO ✓ | sl=139 time_stop=6 trail=118
  SL=5%  / no slope filter [trail/retest/retest_low] |  263 trades | win= 43.3% | avgWinR=4.15 | totalR=+324.0 | avgR=+1.232 | maxDD=-10.0R | GO ✓ | sl=149 time_stop=4 trail=110

  BREAKOUT entry  |  SL=OR extreme+buf  |  TP=swing_trail (50%@TP1, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [swing_trail/breakout/or_extreme] |  312 trades | win= 80.4% | avgWinR=0.46 | totalR=+57.7 | avgR=+0.185 | maxDD=-3.2R | GO ✓ | sl=53 time_stop=26 trail=233
  SL=5%  / slope≥0.02% [swing_trail/breakout/or_extreme] |  312 trades | win= 79.8% | avgWinR=0.48 | totalR=+59.4 | avgR=+0.190 | maxDD=-3.2R | GO ✓ | sl=56 time_stop=24 trail=232
  SL=10% / slope≥0.10% [swing_trail/breakout/or_extreme] |  137 trades | win= 79.6% | avgWinR=0.52 | totalR=+30.1 | avgR=+0.220 | maxDD=-3.0R | GO ✓ | sl=26 time_stop=10 trail=101
  SL=5%  / slope≥0.10% [swing_trail/breakout/or_extreme] |  137 trades | win= 79.6% | avgWinR=0.54 | totalR=+32.4 | avgR=+0.237 | maxDD=-3.0R | GO ✓ | sl=26 time_stop=10 trail=101
  SL=10% / no slope filter [swing_trail/breakout/or_extreme] |  343 trades | win= 79.0% | avgWinR=0.46 | totalR=+55.9 | avgR=+0.163 | maxDD=-5.2R | GO ✓ | sl=64 time_stop=26 trail=253
  SL=5%  / no slope filter [swing_trail/breakout/or_extreme] |  343 trades | win= 78.4% | avgWinR=0.48 | totalR=+57.9 | avgR=+0.169 | maxDD=-5.2R | GO ✓ | sl=67 time_stop=24 trail=252

  RETEST entry    |  SL=OR extreme+buf  |  TP=swing_trail (50%@TP1, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [swing_trail/retest/or_extreme] |  238 trades | win= 74.4% | avgWinR=0.53 | totalR=+33.8 | avgR=+0.142 | maxDD=-6.6R | GO ✓ | eod=1 sl=58 time_stop=13 trail=166
  SL=5%  / slope≥0.02% [swing_trail/retest/or_extreme] |  238 trades | win= 73.9% | avgWinR=0.56 | totalR=+36.4 | avgR=+0.153 | maxDD=-6.5R | GO ✓ | eod=1 sl=60 time_stop=11 trail=166
  SL=10% / slope≥0.10% [swing_trail/retest/or_extreme] |  102 trades | win= 72.5% | avgWinR=0.55 | totalR=+12.6 | avgR=+0.124 | maxDD=-5.1R | GO ✓ | sl=28 time_stop=4 trail=70
  SL=5%  / slope≥0.10% [swing_trail/retest/or_extreme] |  102 trades | win= 72.5% | avgWinR=0.57 | totalR=+14.5 | avgR=+0.143 | maxDD=-5.0R | GO ✓ | sl=28 time_stop=4 trail=70
  SL=10% / no slope filter [swing_trail/retest/or_extreme] |  263 trades | win= 73.8% | avgWinR=0.52 | totalR=+33.7 | avgR=+0.128 | maxDD=-5.9R | GO ✓ | eod=1 sl=66 time_stop=13 trail=183
  SL=5%  / no slope filter [swing_trail/retest/or_extreme] |  263 trades | win= 73.4% | avgWinR=0.55 | totalR=+36.7 | avgR=+0.139 | maxDD=-5.7R | GO ✓ | eod=1 sl=68 time_stop=11 trail=183

  RETEST entry    |  SL=retest low+buf  |  TP=swing_trail (50%@TP1, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [swing_trail/retest/retest_low] |  238 trades | win= 47.1% | avgWinR=2.52 | totalR=+156.7 | avgR=+0.659 | maxDD=-9.2R | GO ✓ | eod=1 sl=125 time_stop=1 trail=111
  SL=5%  / slope≥0.02% [swing_trail/retest/retest_low] |  238 trades | win= 43.7% | avgWinR=3.73 | totalR=+254.3 | avgR=+1.068 | maxDD=-10.0R | GO ✓ | eod=1 sl=134 trail=103
  SL=10% / slope≥0.10% [swing_trail/retest/retest_low] |  102 trades | win= 52.0% | avgWinR=2.42 | totalR=+79.2 | avgR=+0.776 | maxDD=-7.6R | GO ✓ | sl=49 trail=53
  SL=5%  / slope≥0.10% [swing_trail/retest/retest_low] |  102 trades | win= 49.0% | avgWinR=3.46 | totalR=+120.8 | avgR=+1.184 | maxDD=-6.9R | GO ✓ | sl=52 trail=50
  SL=10% / no slope filter [swing_trail/retest/retest_low] |  263 trades | win= 46.8% | avgWinR=2.52 | totalR=+169.9 | avgR=+0.646 | maxDD=-10.2R | GO ✓ | eod=1 sl=139 time_stop=1 trail=122
  SL=5%  / no slope filter [swing_trail/retest/retest_low] |  263 trades | win= 43.3% | avgWinR=3.67 | totalR=+268.9 | avgR=+1.022 | maxDD=-10.0R | GO ✓ | eod=1 sl=149 trail=113

  BREAKOUT entry  |  SL=OR extreme+buf  |  TP=tp2_swing (full@TP2, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_swing/breakout/or_extreme] |  312 trades | win= 70.8% | avgWinR=0.75 | totalR=+88.0 | avgR=+0.282 | maxDD=-4.5R | GO ✓ | eod=1 sl=73 time_stop=56 trail=182
  SL=5%  / slope≥0.02% [tp2_swing/breakout/or_extreme] |  312 trades | win= 70.5% | avgWinR=0.79 | totalR=+92.5 | avgR=+0.296 | maxDD=-4.5R | GO ✓ | eod=1 sl=76 time_stop=53 trail=182
  SL=10% / slope≥0.10% [tp2_swing/breakout/or_extreme] |  137 trades | win= 73.0% | avgWinR=0.82 | totalR=+48.2 | avgR=+0.352 | maxDD=-3.0R | GO ✓ | eod=1 sl=33 time_stop=18 trail=85
  SL=5%  / slope≥0.10% [tp2_swing/breakout/or_extreme] |  137 trades | win= 73.0% | avgWinR=0.85 | totalR=+51.5 | avgR=+0.376 | maxDD=-3.0R | GO ✓ | eod=1 sl=33 time_stop=18 trail=85
  SL=10% / no slope filter [tp2_swing/breakout/or_extreme] |  343 trades | win= 68.5% | avgWinR=0.76 | totalR=+82.9 | avgR=+0.242 | maxDD=-5.5R | GO ✓ | eod=1 sl=90 time_stop=59 trail=193
  SL=5%  / no slope filter [tp2_swing/breakout/or_extreme] |  343 trades | win= 68.2% | avgWinR=0.79 | totalR=+87.9 | avgR=+0.256 | maxDD=-5.5R | GO ✓ | eod=1 sl=93 time_stop=56 trail=193

  RETEST entry    |  SL=OR extreme+buf  |  TP=tp2_swing (full@TP2, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_swing/retest/or_extreme] |  238 trades | win= 64.3% | avgWinR=0.89 | totalR=+58.3 | avgR=+0.245 | maxDD=-6.1R | GO ✓ | eod=1 sl=75 time_stop=38 trail=124
  SL=5%  / slope≥0.02% [tp2_swing/retest/or_extreme] |  238 trades | win= 63.9% | avgWinR=0.94 | totalR=+62.5 | avgR=+0.263 | maxDD=-6.0R | GO ✓ | eod=1 sl=78 time_stop=35 trail=124
  SL=10% / slope≥0.10% [tp2_swing/retest/or_extreme] |  102 trades | win= 65.7% | avgWinR=0.99 | totalR=+32.2 | avgR=+0.316 | maxDD=-3.0R | GO ✓ | eod=1 sl=34 time_stop=13 trail=54
  SL=5%  / slope≥0.10% [tp2_swing/retest/or_extreme] |  102 trades | win= 65.7% | avgWinR=1.04 | totalR=+35.4 | avgR=+0.347 | maxDD=-3.0R | GO ✓ | eod=1 sl=34 time_stop=13 trail=54
  SL=10% / no slope filter [tp2_swing/retest/or_extreme] |  263 trades | win= 62.4% | avgWinR=0.90 | totalR=+55.1 | avgR=+0.210 | maxDD=-8.0R | GO ✓ | eod=1 sl=89 time_stop=40 trail=133
  SL=5%  / no slope filter [tp2_swing/retest/or_extreme] |  263 trades | win= 62.0% | avgWinR=0.94 | totalR=+59.9 | avgR=+0.228 | maxDD=-8.0R | GO ✓ | eod=1 sl=92 time_stop=37 trail=133

  RETEST entry    |  SL=retest low+buf  |  TP=tp2_swing (full@TP2, swing trail)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_swing/retest/retest_low] |  238 trades | win= 33.6% | avgWinR=4.69 | totalR=+218.4 | avgR=+0.918 | maxDD=-17.6R | GO ✓ | sl=156 time_stop=5 trail=77
  SL=5%  / slope≥0.02% [tp2_swing/retest/retest_low] |  238 trades | win= 30.3% | avgWinR=6.60 | totalR=+309.8 | avgR=+1.302 | maxDD=-16.6R | GO ✓ | sl=165 time_stop=3 trail=70
  SL=10% / slope≥0.10% [tp2_swing/retest/retest_low] |  102 trades | win= 41.2% | avgWinR=4.54 | totalR=+130.5 | avgR=+1.279 | maxDD=-12.6R | GO ✓ | sl=60 time_stop=1 trail=41
  SL=5%  / slope≥0.10% [tp2_swing/retest/retest_low] |  102 trades | win= 36.3% | avgWinR=5.99 | totalR=+156.4 | avgR=+1.534 | maxDD=-12.0R | GO ✓ | sl=65 time_stop=1 trail=36
  SL=10% / no slope filter [tp2_swing/retest/retest_low] |  263 trades | win= 33.1% | avgWinR=4.61 | totalR=+226.1 | avgR=+0.860 | maxDD=-18.6R | GO ✓ | sl=174 time_stop=5 trail=84
  SL=5%  / no slope filter [tp2_swing/retest/retest_low] |  263 trades | win= 29.7% | avgWinR=6.44 | totalR=+318.6 | avgR=+1.211 | maxDD=-17.6R | GO ✓ | sl=184 time_stop=3 trail=76

  RETEST entry    |  SL=OR extreme+buf  |  TP=tp2_partial (70%@TP2+BE, 30% swing trail sw=1)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_partial/retest/or_extreme/sw1] |  238 trades | win= 64.3% | avgWinR=0.85 | totalR=+51.9 | avgR=+0.218 | maxDD=-6.0R | GO ✓ | eod=1 sl=75 time_stop=38 trail=124
  SL=10% / slope≥0.02% [tp2_partial/retest/or_extreme/sw2] |  238 trades | win= 64.3% | avgWinR=0.84 | totalR=+50.6 | avgR=+0.213 | maxDD=-6.0R | GO ✓ | eod=2 sl=75 time_stop=45 trail=116
  SL=10% / slope≥0.02% [tp2_partial/retest/or_extreme/sw3] |  238 trades | win= 64.3% | avgWinR=0.84 | totalR=+50.4 | avgR=+0.212 | maxDD=-6.0R | GO ✓ | eod=3 sl=75 time_stop=52 trail=108
  SL=5%  / slope≥0.02% [tp2_partial/retest/or_extreme/sw1] |  238 trades | win= 63.9% | avgWinR=0.89 | totalR=+55.9 | avgR=+0.235 | maxDD=-6.0R | GO ✓ | eod=1 sl=78 time_stop=35 trail=124
  SL=5%  / slope≥0.02% [tp2_partial/retest/or_extreme/sw2] |  238 trades | win= 63.9% | avgWinR=0.89 | totalR=+54.5 | avgR=+0.229 | maxDD=-6.0R | GO ✓ | eod=2 sl=78 time_stop=42 trail=116
  SL=5%  / slope≥0.02% [tp2_partial/retest/or_extreme/sw3] |  238 trades | win= 63.9% | avgWinR=0.88 | totalR=+54.3 | avgR=+0.228 | maxDD=-6.0R | GO ✓ | eod=3 sl=78 time_stop=49 trail=108
  SL=10% / slope≥0.10% [tp2_partial/retest/or_extreme/sw1] |  102 trades | win= 65.7% | avgWinR=0.88 | totalR=+24.8 | avgR=+0.243 | maxDD=-3.0R | GO ✓ | eod=1 sl=34 time_stop=13 trail=54
  SL=10% / slope≥0.10% [tp2_partial/retest/or_extreme/sw2] |  102 trades | win= 65.7% | avgWinR=0.87 | totalR=+24.0 | avgR=+0.235 | maxDD=-3.0R | GO ✓ | eod=1 sl=34 time_stop=15 trail=52
  SL=10% / slope≥0.10% [tp2_partial/retest/or_extreme/sw3] |  102 trades | win= 65.7% | avgWinR=0.88 | totalR=+24.4 | avgR=+0.239 | maxDD=-3.0R | GO ✓ | eod=2 sl=34 time_stop=18 trail=48
  SL=5%  / slope≥0.10% [tp2_partial/retest/or_extreme/sw1] |  102 trades | win= 65.7% | avgWinR=0.92 | totalR=+27.6 | avgR=+0.271 | maxDD=-3.0R | GO ✓ | eod=1 sl=34 time_stop=13 trail=54
  SL=5%  / slope≥0.10% [tp2_partial/retest/or_extreme/sw2] |  102 trades | win= 65.7% | avgWinR=0.91 | totalR=+26.8 | avgR=+0.263 | maxDD=-3.0R | GO ✓ | eod=1 sl=34 time_stop=15 trail=52
  SL=5%  / slope≥0.10% [tp2_partial/retest/or_extreme/sw3] |  102 trades | win= 65.7% | avgWinR=0.92 | totalR=+27.1 | avgR=+0.266 | maxDD=-3.0R | GO ✓ | eod=2 sl=34 time_stop=18 trail=48
  SL=10% / no slope filter [tp2_partial/retest/or_extreme/sw1] |  263 trades | win= 62.4% | avgWinR=0.85 | totalR=+47.4 | avgR=+0.180 | maxDD=-8.0R | GO ✓ | eod=1 sl=89 time_stop=40 trail=133
  SL=10% / no slope filter [tp2_partial/retest/or_extreme/sw2] |  263 trades | win= 62.4% | avgWinR=0.84 | totalR=+45.7 | avgR=+0.174 | maxDD=-8.0R | GO ✓ | eod=2 sl=89 time_stop=47 trail=125
  SL=10% / no slope filter [tp2_partial/retest/or_extreme/sw3] |  263 trades | win= 62.4% | avgWinR=0.84 | totalR=+45.5 | avgR=+0.173 | maxDD=-8.0R | GO ✓ | eod=3 sl=89 time_stop=54 trail=117
  SL=5%  / no slope filter [tp2_partial/retest/or_extreme/sw1] |  263 trades | win= 62.0% | avgWinR=0.90 | totalR=+51.8 | avgR=+0.197 | maxDD=-8.0R | GO ✓ | eod=1 sl=92 time_stop=37 trail=133
  SL=5%  / no slope filter [tp2_partial/retest/or_extreme/sw2] |  263 trades | win= 62.0% | avgWinR=0.88 | totalR=+50.1 | avgR=+0.190 | maxDD=-8.0R | GO ✓ | eod=2 sl=92 time_stop=44 trail=125
  SL=5%  / no slope filter [tp2_partial/retest/or_extreme/sw3] |  263 trades | win= 62.0% | avgWinR=0.88 | totalR=+49.8 | avgR=+0.189 | maxDD=-8.0R | GO ✓ | eod=3 sl=92 time_stop=51 trail=117
  SL=10% / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw1] |  238 trades | win= 49.2% | avgWinR=0.88 | totalR=+42.4 | avgR=+0.178 | maxDD=-6.5R | NO-GO ✗ | be=60 sl=58 time_stop=21 trail=99
  SL=10% / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw2] |  238 trades | win= 49.2% | avgWinR=0.87 | totalR=+41.7 | avgR=+0.175 | maxDD=-6.5R | NO-GO ✗ | be=60 eod=1 sl=58 time_stop=27 trail=92
  SL=10% / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw3] |  238 trades | win= 49.2% | avgWinR=0.87 | totalR=+41.1 | avgR=+0.173 | maxDD=-6.5R | NO-GO ✗ | be=60 eod=2 sl=58 time_stop=32 trail=86
  SL=5%  / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw1] |  238 trades | win= 48.7% | avgWinR=0.92 | totalR=+45.4 | avgR=+0.191 | maxDD=-6.4R | NO-GO ✗ | be=60 sl=60 time_stop=19 trail=99
  SL=5%  / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw2] |  238 trades | win= 48.7% | avgWinR=0.91 | totalR=+44.6 | avgR=+0.188 | maxDD=-6.4R | NO-GO ✗ | be=60 eod=1 sl=60 time_stop=25 trail=92
  SL=5%  / slope≥0.02% [tp2_partial+BE/retest/or_extreme/sw3] |  238 trades | win= 48.7% | avgWinR=0.91 | totalR=+44.1 | avgR=+0.185 | maxDD=-6.4R | NO-GO ✗ | be=60 eod=2 sl=60 time_stop=30 trail=86
  SL=10% / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw1] |  102 trades | win= 46.1% | avgWinR=0.91 | totalR=+14.6 | avgR=+0.143 | maxDD=-5.2R | NO-GO ✗ | be=27 sl=28 time_stop=8 trail=39
  SL=10% / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw2] |  102 trades | win= 46.1% | avgWinR=0.90 | totalR=+14.1 | avgR=+0.139 | maxDD=-5.2R | NO-GO ✗ | be=27 sl=28 time_stop=9 trail=38
  SL=10% / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw3] |  102 trades | win= 46.1% | avgWinR=0.90 | totalR=+14.1 | avgR=+0.138 | maxDD=-5.2R | NO-GO ✗ | be=27 eod=1 sl=28 time_stop=10 trail=36
  SL=5%  / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw1] |  102 trades | win= 46.1% | avgWinR=0.95 | totalR=+16.6 | avgR=+0.163 | maxDD=-5.2R | NO-GO ✗ | be=27 sl=28 time_stop=8 trail=39
  SL=5%  / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw2] |  102 trades | win= 46.1% | avgWinR=0.94 | totalR=+16.1 | avgR=+0.158 | maxDD=-5.2R | NO-GO ✗ | be=27 sl=28 time_stop=9 trail=38
  SL=5%  / slope≥0.10% [tp2_partial+BE/retest/or_extreme/sw3] |  102 trades | win= 46.1% | avgWinR=0.94 | totalR=+16.1 | avgR=+0.158 | maxDD=-5.2R | NO-GO ✗ | be=27 eod=1 sl=28 time_stop=10 trail=36
  SL=10% / no slope filter [tp2_partial+BE/retest/or_extreme/sw1] |  263 trades | win= 47.1% | avgWinR=0.88 | totalR=+41.1 | avgR=+0.156 | maxDD=-6.1R | NO-GO ✗ | be=70 sl=66 time_stop=22 trail=105
  SL=10% / no slope filter [tp2_partial+BE/retest/or_extreme/sw2] |  263 trades | win= 47.1% | avgWinR=0.87 | totalR=+40.0 | avgR=+0.152 | maxDD=-6.1R | NO-GO ✗ | be=70 eod=1 sl=66 time_stop=28 trail=98
  SL=10% / no slope filter [tp2_partial+BE/retest/or_extreme/sw3] |  263 trades | win= 47.1% | avgWinR=0.87 | totalR=+39.4 | avgR=+0.150 | maxDD=-6.1R | NO-GO ✗ | be=70 eod=2 sl=66 time_stop=33 trail=92
  SL=5%  / no slope filter [tp2_partial+BE/retest/or_extreme/sw1] |  263 trades | win= 46.8% | avgWinR=0.93 | totalR=+44.4 | avgR=+0.169 | maxDD=-6.0R | NO-GO ✗ | be=70 sl=68 time_stop=20 trail=105
  SL=5%  / no slope filter [tp2_partial+BE/retest/or_extreme/sw2] |  263 trades | win= 46.8% | avgWinR=0.92 | totalR=+43.2 | avgR=+0.164 | maxDD=-6.0R | NO-GO ✗ | be=70 eod=1 sl=68 time_stop=26 trail=98
  SL=5%  / no slope filter [tp2_partial+BE/retest/or_extreme/sw3] |  263 trades | win= 46.8% | avgWinR=0.91 | totalR=+42.7 | avgR=+0.162 | maxDD=-6.0R | NO-GO ✗ | be=70 eod=2 sl=68 time_stop=31 trail=92

  BREAKOUT entry  |  SL=OR extreme+buf  |  TP=tp2_partial (70%@TP2+BE, 30% swing trail sw=1)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [tp2_partial/breakout/or_extreme/sw1] |  312 trades | win= 70.8% | avgWinR=0.71 | totalR=+78.6 | avgR=+0.252 | maxDD=-4.5R | GO ✓ | eod=1 sl=73 time_stop=56 trail=182
  SL=10% / slope≥0.02% [tp2_partial/breakout/or_extreme/sw2] |  312 trades | win= 70.8% | avgWinR=0.72 | totalR=+80.3 | avgR=+0.257 | maxDD=-4.5R | GO ✓ | eod=2 sl=73 time_stop=73 trail=164
  SL=10% / slope≥0.02% [tp2_partial/breakout/or_extreme/sw3] |  312 trades | win= 70.8% | avgWinR=0.72 | totalR=+80.0 | avgR=+0.256 | maxDD=-4.5R | GO ✓ | eod=3 sl=73 time_stop=82 trail=154
  SL=5%  / slope≥0.02% [tp2_partial/breakout/or_extreme/sw1] |  312 trades | win= 70.5% | avgWinR=0.74 | totalR=+82.8 | avgR=+0.265 | maxDD=-4.5R | GO ✓ | eod=1 sl=76 time_stop=53 trail=182
  SL=5%  / slope≥0.02% [tp2_partial/breakout/or_extreme/sw2] |  312 trades | win= 70.5% | avgWinR=0.75 | totalR=+84.5 | avgR=+0.271 | maxDD=-4.5R | GO ✓ | eod=2 sl=76 time_stop=70 trail=164
  SL=5%  / slope≥0.02% [tp2_partial/breakout/or_extreme/sw3] |  312 trades | win= 70.5% | avgWinR=0.75 | totalR=+84.2 | avgR=+0.270 | maxDD=-4.5R | GO ✓ | eod=3 sl=76 time_stop=79 trail=154
  SL=10% / slope≥0.10% [tp2_partial/breakout/or_extreme/sw1] |  137 trades | win= 73.0% | avgWinR=0.75 | totalR=+40.8 | avgR=+0.298 | maxDD=-3.0R | GO ✓ | eod=1 sl=33 time_stop=18 trail=85
  SL=10% / slope≥0.10% [tp2_partial/breakout/or_extreme/sw2] |  137 trades | win= 73.0% | avgWinR=0.75 | totalR=+41.2 | avgR=+0.301 | maxDD=-3.0R | GO ✓ | eod=1 sl=33 time_stop=24 trail=79
  SL=10% / slope≥0.10% [tp2_partial/breakout/or_extreme/sw3] |  137 trades | win= 73.0% | avgWinR=0.75 | totalR=+41.4 | avgR=+0.302 | maxDD=-3.0R | GO ✓ | eod=2 sl=33 time_stop=30 trail=72
  SL=5%  / slope≥0.10% [tp2_partial/breakout/or_extreme/sw1] |  137 trades | win= 73.0% | avgWinR=0.78 | totalR=+43.8 | avgR=+0.320 | maxDD=-3.0R | GO ✓ | eod=1 sl=33 time_stop=18 trail=85
  SL=5%  / slope≥0.10% [tp2_partial/breakout/or_extreme/sw2] |  137 trades | win= 73.0% | avgWinR=0.78 | totalR=+44.2 | avgR=+0.323 | maxDD=-3.0R | GO ✓ | eod=1 sl=33 time_stop=24 trail=79
  SL=5%  / slope≥0.10% [tp2_partial/breakout/or_extreme/sw3] |  137 trades | win= 73.0% | avgWinR=0.78 | totalR=+44.4 | avgR=+0.324 | maxDD=-3.0R | GO ✓ | eod=2 sl=33 time_stop=30 trail=72
  SL=10% / no slope filter [tp2_partial/breakout/or_extreme/sw1] |  343 trades | win= 68.5% | avgWinR=0.71 | totalR=+72.0 | avgR=+0.210 | maxDD=-5.6R | GO ✓ | eod=1 sl=90 time_stop=59 trail=193
  SL=10% / no slope filter [tp2_partial/breakout/or_extreme/sw2] |  343 trades | win= 68.5% | avgWinR=0.72 | totalR=+73.4 | avgR=+0.214 | maxDD=-5.5R | GO ✓ | eod=2 sl=90 time_stop=76 trail=175
  SL=10% / no slope filter [tp2_partial/breakout/or_extreme/sw3] |  343 trades | win= 68.5% | avgWinR=0.72 | totalR=+72.9 | avgR=+0.213 | maxDD=-5.5R | GO ✓ | eod=3 sl=90 time_stop=85 trail=165
  SL=5%  / no slope filter [tp2_partial/breakout/or_extreme/sw1] |  343 trades | win= 68.2% | avgWinR=0.74 | totalR=+76.6 | avgR=+0.223 | maxDD=-5.5R | GO ✓ | eod=1 sl=93 time_stop=56 trail=193
  SL=5%  / no slope filter [tp2_partial/breakout/or_extreme/sw2] |  343 trades | win= 68.2% | avgWinR=0.75 | totalR=+78.0 | avgR=+0.227 | maxDD=-5.5R | GO ✓ | eod=2 sl=93 time_stop=73 trail=175
  SL=5%  / no slope filter [tp2_partial/breakout/or_extreme/sw3] |  343 trades | win= 68.2% | avgWinR=0.75 | totalR=+77.5 | avgR=+0.226 | maxDD=-5.5R | GO ✓ | eod=3 sl=93 time_stop=82 trail=165
  SL=10% / slope≥0.02% [tp2_partial+BE/breakout/or_extreme/sw1] |  312 trades | win= 52.9% | avgWinR=0.75 | totalR=+66.8 | avgR=+0.214 | maxDD=-3.5R | NO-GO ✗ | be=86 sl=53 time_stop=32 trail=141
  SL=10% / slope≥0.02% [tp2_partial+BE/breakout/or_extreme/sw2] |  312 trades | win= 52.9% | avgWinR=0.75 | totalR=+67.4 | avgR=+0.216 | maxDD=-3.5R | NO-GO ✗ | be=86 eod=1 sl=53 time_stop=42 trail=130
  SL=10% / slope≥0.02% [tp2_partial+BE/breakout/or_extreme/sw3] |  312 trades | win= 52.9% | avgWinR=0.75 | totalR=+66.6 | avgR=+0.214 | maxDD=-3.5R | NO-GO ✗ | be=86 eod=2 sl=53 time_stop=49 trail=122
  SL=5%  / slope≥0.02% [tp2_partial+BE/breakout/or_extreme/sw1] |  312 trades | win= 52.6% | avgWinR=0.78 | totalR=+69.1 | avgR=+0.221 | maxDD=-3.4R | NO-GO ✗ | be=86 sl=55 time_stop=30 trail=141
  SL=5%  / slope≥0.02% [tp2_partial+BE/breakout/or_extreme/sw2] |  312 trades | win= 52.6% | avgWinR=0.78 | totalR=+69.6 | avgR=+0.223 | maxDD=-3.4R | NO-GO ✗ | be=86 eod=1 sl=55 time_stop=40 trail=130
  SL=5%  / slope≥0.02% [tp2_partial+BE/breakout/or_extreme/sw3] |  312 trades | win= 52.6% | avgWinR=0.78 | totalR=+68.9 | avgR=+0.221 | maxDD=-3.4R | NO-GO ✗ | be=86 eod=2 sl=55 time_stop=47 trail=122
  SL=10% / slope≥0.10% [tp2_partial+BE/breakout/or_extreme/sw1] |  137 trades | win= 59.1% | avgWinR=0.77 | totalR=+36.2 | avgR=+0.264 | maxDD=-3.0R | NO-GO ✗ | be=28 sl=26 time_stop=12 trail=71
  SL=10% / slope≥0.10% [tp2_partial+BE/breakout/or_extreme/sw2] |  137 trades | win= 59.1% | avgWinR=0.78 | totalR=+36.7 | avgR=+0.268 | maxDD=-3.0R | NO-GO ✗ | be=28 sl=26 time_stop=17 trail=66
  SL=10% / slope≥0.10% [tp2_partial+BE/breakout/or_extreme/sw3] |  137 trades | win= 59.1% | avgWinR=0.79 | totalR=+37.1 | avgR=+0.271 | maxDD=-3.0R | NO-GO ✗ | be=28 eod=1 sl=26 time_stop=22 trail=60
  SL=5%  / slope≥0.10% [tp2_partial+BE/breakout/or_extreme/sw1] |  137 trades | win= 59.1% | avgWinR=0.81 | totalR=+38.7 | avgR=+0.283 | maxDD=-3.0R | NO-GO ✗ | be=28 sl=26 time_stop=12 trail=71
  SL=5%  / slope≥0.10% [tp2_partial+BE/breakout/or_extreme/sw2] |  137 trades | win= 59.1% | avgWinR=0.81 | totalR=+39.2 | avgR=+0.286 | maxDD=-3.0R | NO-GO ✗ | be=28 sl=26 time_stop=17 trail=66
  SL=5%  / slope≥0.10% [tp2_partial+BE/breakout/or_extreme/sw3] |  137 trades | win= 59.1% | avgWinR=0.82 | totalR=+39.7 | avgR=+0.290 | maxDD=-3.0R | NO-GO ✗ | be=28 eod=1 sl=26 time_stop=22 trail=60
  SL=10% / no slope filter [tp2_partial+BE/breakout/or_extreme/sw1] |  343 trades | win= 50.7% | avgWinR=0.75 | totalR=+63.3 | avgR=+0.185 | maxDD=-5.5R | NO-GO ✗ | be=97 sl=64 time_stop=33 trail=149
  SL=10% / no slope filter [tp2_partial+BE/breakout/or_extreme/sw2] |  343 trades | win= 50.7% | avgWinR=0.76 | totalR=+63.8 | avgR=+0.186 | maxDD=-5.5R | NO-GO ✗ | be=97 eod=1 sl=64 time_stop=43 trail=138
  SL=10% / no slope filter [tp2_partial+BE/breakout/or_extreme/sw3] |  343 trades | win= 50.7% | avgWinR=0.75 | totalR=+62.9 | avgR=+0.183 | maxDD=-5.5R | NO-GO ✗ | be=97 eod=2 sl=64 time_stop=50 trail=130
  SL=5%  / no slope filter [tp2_partial+BE/breakout/or_extreme/sw1] |  343 trades | win= 50.4% | avgWinR=0.78 | totalR=+65.9 | avgR=+0.192 | maxDD=-5.4R | NO-GO ✗ | be=97 sl=66 time_stop=31 trail=149
  SL=5%  / no slope filter [tp2_partial+BE/breakout/or_extreme/sw2] |  343 trades | win= 50.4% | avgWinR=0.79 | totalR=+66.4 | avgR=+0.194 | maxDD=-5.4R | NO-GO ✗ | be=97 eod=1 sl=66 time_stop=41 trail=138
  SL=5%  / no slope filter [tp2_partial+BE/breakout/or_extreme/sw3] |  343 trades | win= 50.4% | avgWinR=0.78 | totalR=+65.4 | avgR=+0.191 | maxDD=-5.4R | NO-GO ✗ | be=97 eod=2 sl=66 time_stop=48 trail=130

  BREAKOUT entry  |  SL=OR extreme+buf  |  TP=fixed R:R (2×/3×SL)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [fixed_rr/breakout/or_extreme] |  312 trades | win= 58.7% | avgWinR=1.14 | totalR=+98.7 | avgR=+0.316 | maxDD=-5.7R | GO ✓ | eod=6 sl=104 time_stop=186 tp2=16
  SL=5%  / slope≥0.02% [fixed_rr/breakout/or_extreme] |  312 trades | win= 58.3% | avgWinR=1.18 | totalR=+103.8 | avgR=+0.333 | maxDD=-5.7R | GO ✓ | eod=6 sl=107 time_stop=178 tp2=21
  SL=10% / slope≥0.10% [fixed_rr/breakout/or_extreme] |  137 trades | win= 60.6% | avgWinR=1.33 | totalR=+62.2 | avgR=+0.454 | maxDD=-5.4R | GO ✓ | eod=3 sl=49 time_stop=76 tp2=9
  SL=5%  / slope≥0.10% [fixed_rr/breakout/or_extreme] |  137 trades | win= 60.6% | avgWinR=1.38 | totalR=+66.6 | avgR=+0.486 | maxDD=-5.4R | GO ✓ | eod=3 sl=49 time_stop=71 tp2=14
  SL=10% / no slope filter [fixed_rr/breakout/or_extreme] |  343 trades | win= 56.9% | avgWinR=1.15 | totalR=+96.0 | avgR=+0.280 | maxDD=-6.7R | GO ✓ | eod=6 sl=123 time_stop=196 tp2=18
  SL=5%  / no slope filter [fixed_rr/breakout/or_extreme] |  343 trades | win= 56.6% | avgWinR=1.19 | totalR=+101.6 | avgR=+0.296 | maxDD=-6.7R | GO ✓ | eod=6 sl=126 time_stop=188 tp2=23

  RETEST entry    |  SL=OR extreme+buf  |  TP=fixed R:R (2×/3×SL)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [fixed_rr/retest/or_extreme] |  238 trades | win= 54.6% | avgWinR=1.26 | totalR=+64.6 | avgR=+0.271 | maxDD=-6.2R | GO ✓ | eod=4 sl=98 time_stop=118 tp2=18
  SL=5%  / slope≥0.02% [fixed_rr/retest/or_extreme] |  238 trades | win= 54.6% | avgWinR=1.32 | totalR=+71.5 | avgR=+0.301 | maxDD=-6.2R | GO ✓ | eod=4 sl=103 time_stop=110 tp2=21
  SL=10% / slope≥0.10% [fixed_rr/retest/or_extreme] |  102 trades | win= 55.9% | avgWinR=1.40 | totalR=+36.1 | avgR=+0.354 | maxDD=-5.0R | GO ✓ | eod=2 sl=44 time_stop=46 tp2=10
  SL=5%  / slope≥0.10% [fixed_rr/retest/or_extreme] |  102 trades | win= 56.9% | avgWinR=1.43 | totalR=+40.7 | avgR=+0.399 | maxDD=-4.9R | GO ✓ | eod=2 sl=45 time_stop=43 tp2=12
  SL=10% / no slope filter [fixed_rr/retest/or_extreme] |  263 trades | win= 52.9% | avgWinR=1.30 | totalR=+64.8 | avgR=+0.246 | maxDD=-8.0R | GO ✓ | eod=4 sl=114 time_stop=123 tp2=22
  SL=5%  / no slope filter [fixed_rr/retest/or_extreme] |  263 trades | win= 53.2% | avgWinR=1.35 | totalR=+74.0 | avgR=+0.281 | maxDD=-8.0R | GO ✓ | eod=4 sl=119 time_stop=115 tp2=25

  RETEST entry    |  SL=retest low+buf  |  TP=fixed R:R (2×/3×SL)
  ---------------------------------------------------------------------------------------------------------
  SL=10% / slope≥0.02% [fixed_rr/retest/retest_low] |  238 trades | win= 45.0% | avgWinR=2.17 | totalR=+102.1 | avgR=+0.429 | maxDD=-10.2R | GO ✓ | sl=150 time_stop=6 tp2=82
  SL=5%  / slope≥0.02% [fixed_rr/retest/retest_low] |  238 trades | win= 44.5% | avgWinR=2.24 | totalR=+105.8 | avgR=+0.444 | maxDD=-10.5R | GO ✓ | sl=150 time_stop=1 tp2=87
  SL=10% / slope≥0.10% [fixed_rr/retest/retest_low] |  102 trades | win= 50.0% | avgWinR=2.21 | totalR=+61.5 | avgR=+0.603 | maxDD=-6.0R | GO ✓ | sl=60 time_stop=2 tp2=40
  SL=5%  / slope≥0.10% [fixed_rr/retest/retest_low] |  102 trades | win= 49.0% | avgWinR=2.14 | totalR=+55.0 | avgR=+0.539 | maxDD=-8.5R | GO ✓ | sl=64 tp2=38
  SL=10% / no slope filter [fixed_rr/retest/retest_low] |  263 trades | win= 45.2% | avgWinR=2.19 | totalR=+117.6 | avgR=+0.447 | maxDD=-9.5R | GO ✓ | sl=164 time_stop=6 tp2=93
  SL=5%  / no slope filter [fixed_rr/retest/retest_low] |  263 trades | win= 44.5% | avgWinR=2.27 | totalR=+119.3 | avgR=+0.454 | maxDD=-10.0R | GO ✓ | sl=164 time_stop=1 tp2=98
==============================================================================================================

