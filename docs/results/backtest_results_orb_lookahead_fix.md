# ORB backtest — retest_low lookahead fix (2026-09-30)

Data: xyz:SP500 cache, 5m 2025-07-11 → 2026-07-10 (21,616 bars), 4h for bias.
Command: `python -m backtest.run_backtest_orb --asset xyz:SP500 --entry-mode retest`

## What was wrong

`sl_mode="retest_low"` anchored the SL to the *completed* retest bar's low/high while
entering intrabar at ORH/ORL. The retest bar could therefore never stop the trade out,
and the forward sim skipped the rest of the entry bar entirely. Live `orb.py` reads the
still-forming candle at the touch, so its SL sits ~5% of the range from entry and gets
hit within seconds (live Jul 11–Sep 29: 29 trades, 5 wins).

Fix: `retest_low` SL = ORH − buf (long) / ORL + buf (short), i.e. what is knowable at the
touch; retest entries also check the remainder of the entry bar and take −1R if it reaches
the SL (bar order unknown → conservative). `or_extreme` results are unchanged.

Note: R excludes fees. At ~$5.30 round trip on ~4 units, fees ≈ 1.3 pts/trade — ~1R on
retest_low's ~1pt stops, ~0.1R on or_extreme's ~range-sized stops.

## Before (lookahead)

```
  SL=10% / slope≥0.02% [range/retest/retest_low] |  124 trades | win= 41.1% | avgWinR=2.53 | totalR=+56.2 | avgR=+0.453 | maxDD=-8.5R | GO ✓ | sl=95 time_stop=2 tp2=27
  SL=5%  / slope≥0.02% [range/retest/retest_low] |  124 trades | win= 37.9% | avgWinR=3.84 | totalR=+103.5 | avgR=+0.834 | maxDD=-9.0R | GO ✓ | sl=98 time_stop=1 tp2=25
  SL=10% / slope≥0.10% [range/retest/retest_low] |   51 trades | win= 39.2% | avgWinR=2.48 | totalR=+18.6 | avgR=+0.364 | maxDD=-9.0R | GO ✓ | sl=39 time_stop=1 tp2=11
  SL=5%  / slope≥0.10% [range/retest/retest_low] |   51 trades | win= 39.2% | avgWinR=3.69 | totalR=+42.7 | avgR=+0.838 | maxDD=-7.9R | GO ✓ | sl=39 time_stop=1 tp2=11
  SL=10% / no slope filter [range/retest/retest_low] |  134 trades | win= 40.3% | avgWinR=2.60 | totalR=+60.2 | avgR=+0.449 | maxDD=-9.5R | GO ✓ | sl=103 time_stop=2 tp2=29
  SL=5%  / no slope filter [range/retest/retest_low] |  134 trades | win= 36.6% | avgWinR=3.92 | totalR=+106.9 | avgR=+0.798 | maxDD=-9.0R | GO ✓ | sl=107 time_stop=1 tp2=26
  SL=10% / slope≥0.02% [trail/retest/retest_low] |  124 trades | win= 41.1% | avgWinR=2.80 | totalR=+70.1 | avgR=+0.565 | maxDD=-7.0R | GO ✓ | sl=73 time_stop=5 trail=46
  SL=5%  / slope≥0.02% [trail/retest/retest_low] |  124 trades | win= 37.9% | avgWinR=4.31 | totalR=+125.8 | avgR=+1.015 | maxDD=-9.0R | GO ✓ | sl=77 time_stop=4 trail=43
  SL=10% / slope≥0.10% [trail/retest/retest_low] |   51 trades | win= 39.2% | avgWinR=3.00 | totalR=+29.0 | avgR=+0.569 | maxDD=-7.3R | GO ✓ | sl=31 time_stop=2 trail=18
  SL=5%  / slope≥0.10% [trail/retest/retest_low] |   51 trades | win= 39.2% | avgWinR=4.42 | totalR=+57.3 | avgR=+1.124 | maxDD=-6.5R | GO ✓ | sl=31 time_stop=2 trail=18
  SL=10% / no slope filter [trail/retest/retest_low] |  134 trades | win= 40.3% | avgWinR=2.81 | totalR=+71.9 | avgR=+0.536 | maxDD=-7.0R | GO ✓ | sl=80 time_stop=5 trail=49
  SL=5%  / no slope filter [trail/retest/retest_low] |  134 trades | win= 36.6% | avgWinR=4.34 | totalR=+127.6 | avgR=+0.953 | maxDD=-9.0R | GO ✓ | sl=85 time_stop=4 trail=45
  SL=10% / slope≥0.02% [swing_trail/retest/retest_low] |  124 trades | win= 41.1% | avgWinR=2.62 | totalR=+60.8 | avgR=+0.490 | maxDD=-8.4R | GO ✓ | sl=73 trail=51
  SL=5%  / slope≥0.02% [swing_trail/retest/retest_low] |  124 trades | win= 37.9% | avgWinR=3.95 | totalR=+108.8 | avgR=+0.878 | maxDD=-9.0R | GO ✓ | sl=77 trail=47
  SL=10% / slope≥0.10% [swing_trail/retest/retest_low] |   51 trades | win= 39.2% | avgWinR=2.37 | totalR=+16.4 | avgR=+0.321 | maxDD=-7.6R | GO ✓ | sl=31 trail=20
  SL=5%  / slope≥0.10% [swing_trail/retest/retest_low] |   51 trades | win= 39.2% | avgWinR=3.42 | totalR=+37.5 | avgR=+0.734 | maxDD=-6.9R | GO ✓ | sl=31 trail=20
  SL=10% / no slope filter [swing_trail/retest/retest_low] |  134 trades | win= 40.3% | avgWinR=2.67 | totalR=+63.9 | avgR=+0.477 | maxDD=-9.4R | GO ✓ | sl=80 trail=54
  SL=5%  / no slope filter [swing_trail/retest/retest_low] |  134 trades | win= 36.6% | avgWinR=3.96 | totalR=+109.2 | avgR=+0.815 | maxDD=-9.0R | GO ✓ | sl=85 trail=49
  SL=10% / slope≥0.02% [tp2_swing/retest/retest_low] |  124 trades | win= 25.8% | avgWinR=5.56 | totalR=+86.9 | avgR=+0.701 | maxDD=-17.6R | GO ✓ | sl=91 time_stop=3 trail=30
  SL=5%  / slope≥0.02% [tp2_swing/retest/retest_low] |  124 trades | win= 22.6% | avgWinR=8.06 | totalR=+130.5 | avgR=+1.052 | maxDD=-16.6R | GO ✓ | sl=95 time_stop=2 trail=27
  SL=10% / slope≥0.10% [tp2_swing/retest/retest_low] |   51 trades | win= 27.5% | avgWinR=4.77 | totalR=+29.8 | avgR=+0.584 | maxDD=-12.7R | GO ✓ | sl=37 time_stop=1 trail=13
  SL=5%  / slope≥0.10% [tp2_swing/retest/retest_low] |   51 trades | win= 25.5% | avgWinR=6.32 | totalR=+44.2 | avgR=+0.866 | maxDD=-12.0R | GO ✓ | sl=38 time_stop=1 trail=12
  SL=10% / no slope filter [tp2_swing/retest/retest_low] |  134 trades | win= 25.4% | avgWinR=5.40 | totalR=+84.6 | avgR=+0.631 | maxDD=-18.6R | GO ✓ | sl=99 time_stop=3 trail=32
  SL=5%  / no slope filter [tp2_swing/retest/retest_low] |  134 trades | win= 21.6% | avgWinR=7.93 | totalR=+125.8 | avgR=+0.939 | maxDD=-17.6R | GO ✓ | sl=104 time_stop=2 trail=28
  SL=10% / slope≥0.02% [fixed_rr/retest/retest_low] |  124 trades | win= 39.5% | avgWinR=2.09 | totalR=+28.2 | avgR=+0.228 | maxDD=-12.5R | GO ✓ | sl=86 time_stop=2 tp2=36
  SL=5%  / slope≥0.02% [fixed_rr/retest/retest_low] |  124 trades | win= 39.5% | avgWinR=2.26 | totalR=+35.5 | avgR=+0.286 | maxDD=-9.5R | GO ✓ | sl=83 tp2=41
  SL=10% / slope≥0.10% [fixed_rr/retest/retest_low] |   51 trades | win= 41.2% | avgWinR=2.14 | totalR=+15.0 | avgR=+0.294 | maxDD=-8.5R | GO ✓ | sl=35 tp2=16
  SL=5%  / slope≥0.10% [fixed_rr/retest/retest_low] |   51 trades | win= 41.2% | avgWinR=2.21 | totalR=+16.5 | avgR=+0.324 | maxDD=-8.5R | GO ✓ | sl=34 tp2=17
  SL=10% / no slope filter [fixed_rr/retest/retest_low] |  134 trades | win= 38.8% | avgWinR=2.11 | totalR=+28.7 | avgR=+0.214 | maxDD=-11.5R | GO ✓ | sl=93 time_stop=2 tp2=39
  SL=5%  / no slope filter [fixed_rr/retest/retest_low] |  134 trades | win= 38.1% | avgWinR=2.26 | totalR=+32.5 | avgR=+0.243 | maxDD=-9.5R | GO ✓ | sl=91 tp2=43
```

## After (fixed)

```
  SL=10% / slope≥0.02% [range/retest/retest_low] |  124 trades | win= 14.5% | avgWinR=5.50 | totalR=-7.0 | avgR=-0.056 | maxDD=-23.0R | NO-GO ✗ | sl=113 time_stop=1 tp2=10
  SL=5%  / slope≥0.02% [range/retest/retest_low] |  124 trades | win=  9.7% | avgWinR=10.83 | totalR=+18.0 | avgR=+0.145 | maxDD=-31.0R | NO-GO ✗ | sl=117 tp2=7
  SL=10% / slope≥0.10% [range/retest/retest_low] |   51 trades | win= 13.7% | avgWinR=5.36 | totalR=-6.5 | avgR=-0.127 | maxDD=-23.0R | NO-GO ✗ | sl=47 tp2=4
  SL=5%  / slope≥0.10% [range/retest/retest_low] |   51 trades | win=  7.8% | avgWinR=12.50 | totalR=+3.0 | avgR=+0.059 | maxDD=-26.0R | NO-GO ✗ | sl=48 tp2=3
  SL=10% / no slope filter [range/retest/retest_low] |  134 trades | win= 14.9% | avgWinR=5.45 | totalR=-5.0 | avgR=-0.037 | maxDD=-23.0R | NO-GO ✗ | sl=122 time_stop=1 tp2=11
  SL=5%  / no slope filter [range/retest/retest_low] |  134 trades | win=  9.7% | avgWinR=10.38 | totalR=+14.0 | avgR=+0.104 | maxDD=-33.0R | NO-GO ✗ | sl=127 tp2=7
  SL=10% / slope≥0.02% [trail/retest/retest_low] |  124 trades | win= 14.5% | avgWinR=5.71 | totalR=-3.3 | avgR=-0.026 | maxDD=-14.6R | NO-GO ✗ | sl=106 time_stop=3 trail=15
  SL=5%  / slope≥0.02% [trail/retest/retest_low] |  124 trades | win=  9.7% | avgWinR=11.22 | totalR=+22.6 | avgR=+0.182 | maxDD=-22.3R | NO-GO ✗ | sl=112 time_stop=1 trail=11
  SL=10% / slope≥0.10% [trail/retest/retest_low] |   51 trades | win= 13.7% | avgWinR=5.86 | totalR=-3.0 | avgR=-0.058 | maxDD=-20.5R | NO-GO ✗ | sl=44 time_stop=1 trail=6
  SL=5%  / slope≥0.10% [trail/retest/retest_low] |   51 trades | win=  7.8% | avgWinR=13.53 | totalR=+7.1 | avgR=+0.140 | maxDD=-26.0R | NO-GO ✗ | sl=47 time_stop=1 trail=3
  SL=10% / no slope filter [trail/retest/retest_low] |  134 trades | win= 14.9% | avgWinR=5.54 | totalR=-3.2 | avgR=-0.024 | maxDD=-17.0R | NO-GO ✗ | sl=114 time_stop=3 trail=17
  SL=5%  / no slope filter [trail/retest/retest_low] |  134 trades | win=  9.7% | avgWinR=10.74 | totalR=+18.7 | avgR=+0.139 | maxDD=-22.6R | NO-GO ✗ | sl=121 time_stop=1 trail=12
  SL=10% / slope≥0.02% [swing_trail/retest/retest_low] |  124 trades | win= 14.5% | avgWinR=4.84 | totalR=-18.9 | avgR=-0.153 | maxDD=-28.9R | NO-GO ✗ | sl=106 trail=18
  SL=5%  / slope≥0.02% [swing_trail/retest/retest_low] |  124 trades | win=  9.7% | avgWinR=9.67 | totalR=+4.0 | avgR=+0.032 | maxDD=-33.0R | NO-GO ✗ | sl=112 trail=12
  SL=10% / slope≥0.10% [swing_trail/retest/retest_low] |   51 trades | win= 13.7% | avgWinR=4.18 | totalR=-14.7 | avgR=-0.289 | maxDD=-22.8R | NO-GO ✗ | sl=44 trail=7
  SL=5%  / slope≥0.10% [swing_trail/retest/retest_low] |   51 trades | win=  7.8% | avgWinR=9.65 | totalR=-8.4 | avgR=-0.165 | maxDD=-26.7R | NO-GO ✗ | sl=47 trail=4
  SL=10% / no slope filter [swing_trail/retest/retest_low] |  134 trades | win= 14.9% | avgWinR=4.67 | totalR=-20.5 | avgR=-0.153 | maxDD=-29.8R | NO-GO ✗ | sl=114 trail=20
  SL=5%  / no slope filter [swing_trail/retest/retest_low] |  134 trades | win=  9.7% | avgWinR=9.31 | totalR=+0.0 | avgR=+0.000 | maxDD=-36.0R | NO-GO ✗ | sl=121 trail=13
  SL=10% / slope≥0.02% [tp2_swing/retest/retest_low] |  124 trades | win=  8.9% | avgWinR=9.40 | totalR=-9.6 | avgR=-0.077 | maxDD=-35.0R | NO-GO ✗ | sl=113 time_stop=1 trail=10
  SL=5%  / slope≥0.02% [tp2_swing/retest/retest_low] |  124 trades | win=  5.6% | avgWinR=19.39 | totalR=+18.8 | avgR=+0.151 | maxDD=-50.0R | NO-GO ✗ | sl=117 trail=7
  SL=10% / slope≥0.10% [tp2_swing/retest/retest_low] |   51 trades | win=  7.8% | avgWinR=8.33 | totalR=-13.7 | avgR=-0.268 | maxDD=-31.7R | NO-GO ✗ | sl=47 trail=4
  SL=5%  / slope≥0.10% [tp2_swing/retest/retest_low] |   51 trades | win=  5.9% | avgWinR=18.87 | totalR=+8.6 | avgR=+0.169 | maxDD=-30.4R | NO-GO ✗ | sl=48 trail=3
  SL=10% / no slope filter [tp2_swing/retest/retest_low] |  134 trades | win=  9.0% | avgWinR=9.04 | totalR=-13.6 | avgR=-0.101 | maxDD=-36.0R | NO-GO ✗ | sl=122 time_stop=1 trail=11
  SL=5%  / no slope filter [tp2_swing/retest/retest_low] |  134 trades | win=  5.2% | avgWinR=19.39 | totalR=+8.8 | avgR=+0.065 | maxDD=-52.0R | NO-GO ✗ | sl=127 trail=7
  SL=10% / slope≥0.02% [fixed_rr/retest/retest_low] |  124 trades | win= 18.5% | avgWinR=2.24 | totalR=-49.5 | avgR=-0.399 | maxDD=-53.5R | NO-GO ✗ | sl=105 tp2=19
  SL=5%  / slope≥0.02% [fixed_rr/retest/retest_low] |  124 trades | win= 15.3% | avgWinR=2.42 | totalR=-59.0 | avgR=-0.476 | maxDD=-64.5R | NO-GO ✗ | sl=106 tp2=18
  SL=10% / slope≥0.10% [fixed_rr/retest/retest_low] |   51 trades | win= 19.6% | avgWinR=2.35 | totalR=-17.5 | avgR=-0.343 | maxDD=-22.5R | NO-GO ✗ | sl=42 tp2=9
  SL=5%  / slope≥0.10% [fixed_rr/retest/retest_low] |   51 trades | win= 13.7% | avgWinR=2.50 | totalR=-26.5 | avgR=-0.520 | maxDD=-28.5R | NO-GO ✗ | sl=44 tp2=7
  SL=10% / no slope filter [fixed_rr/retest/retest_low] |  134 trades | win= 18.7% | avgWinR=2.26 | totalR=-52.5 | avgR=-0.392 | maxDD=-56.5R | NO-GO ✗ | sl=113 tp2=21
  SL=5%  / no slope filter [fixed_rr/retest/retest_low] |  134 trades | win= 14.9% | avgWinR=2.42 | totalR=-65.5 | avgR=-0.489 | maxDD=-71.0R | NO-GO ✗ | sl=115 tp2=19
```

## Unaffected reference: or_extreme (no lookahead in SL)

```
  SL=10% / slope≥0.02% [range/retest/or_extreme] |  124 trades | win= 77.4% | avgWinR=0.48 | totalR=+18.7 | avgR=+0.150 | maxDD=-4.2R | GO ✓ | sl=61 time_stop=13 tp2=50
  SL=5%  / slope≥0.02% [range/retest/or_extreme] |  124 trades | win= 76.6% | avgWinR=0.50 | totalR=+19.2 | avgR=+0.155 | maxDD=-3.9R | GO ✓ | sl=62 time_stop=12 tp2=50
  SL=10% / slope≥0.10% [range/retest/or_extreme] |   51 trades | win= 76.5% | avgWinR=0.45 | totalR=+5.5 | avgR=+0.108 | maxDD=-2.3R | GO ✓ | sl=28 time_stop=5 tp2=18
  SL=5%  / slope≥0.10% [range/retest/or_extreme] |   51 trades | win= 76.5% | avgWinR=0.47 | totalR=+6.4 | avgR=+0.125 | maxDD=-2.3R | GO ✓ | sl=28 time_stop=5 tp2=18
  SL=10% / no slope filter [range/retest/or_extreme] |  134 trades | win= 77.6% | avgWinR=0.47 | totalR=+19.7 | avgR=+0.147 | maxDD=-3.9R | GO ✓ | sl=68 time_stop=14 tp2=52
  SL=5%  / no slope filter [range/retest/or_extreme] |  134 trades | win= 76.9% | avgWinR=0.49 | totalR=+20.5 | avgR=+0.153 | maxDD=-3.7R | GO ✓ | sl=69 time_stop=13 tp2=52
  SL=10% / slope≥0.02% [trail/retest/or_extreme] |  124 trades | win= 77.4% | avgWinR=0.52 | totalR=+22.4 | avgR=+0.181 | maxDD=-4.2R | GO ✓ | sl=26 time_stop=19 trail=79
  SL=5%  / slope≥0.02% [trail/retest/or_extreme] |  124 trades | win= 76.6% | avgWinR=0.54 | totalR=+23.1 | avgR=+0.187 | maxDD=-3.9R | GO ✓ | sl=27 time_stop=18 trail=79
  SL=10% / slope≥0.10% [trail/retest/or_extreme] |   51 trades | win= 76.5% | avgWinR=0.53 | totalR=+8.8 | avgR=+0.173 | maxDD=-2.4R | GO ✓ | sl=12 time_stop=5 trail=34
  SL=5%  / slope≥0.10% [trail/retest/or_extreme] |   51 trades | win= 76.5% | avgWinR=0.56 | totalR=+9.8 | avgR=+0.192 | maxDD=-2.4R | GO ✓ | sl=12 time_stop=5 trail=34
  SL=10% / no slope filter [trail/retest/or_extreme] |  134 trades | win= 77.6% | avgWinR=0.51 | totalR=+23.4 | avgR=+0.175 | maxDD=-3.8R | GO ✓ | sl=28 time_stop=20 trail=86
  SL=5%  / no slope filter [trail/retest/or_extreme] |  134 trades | win= 76.9% | avgWinR=0.53 | totalR=+24.3 | avgR=+0.181 | maxDD=-3.7R | GO ✓ | sl=29 time_stop=19 trail=86
  SL=10% / slope≥0.02% [swing_trail/retest/or_extreme] |  124 trades | win= 77.4% | avgWinR=0.52 | totalR=+22.7 | avgR=+0.183 | maxDD=-4.2R | GO ✓ | sl=26 time_stop=9 trail=89
  SL=5%  / slope≥0.02% [swing_trail/retest/or_extreme] |  124 trades | win= 76.6% | avgWinR=0.55 | totalR=+23.5 | avgR=+0.189 | maxDD=-4.2R | GO ✓ | sl=27 time_stop=8 trail=89
  SL=10% / slope≥0.10% [swing_trail/retest/or_extreme] |   51 trades | win= 76.5% | avgWinR=0.46 | totalR=+6.1 | avgR=+0.120 | maxDD=-2.7R | GO ✓ | sl=12 time_stop=3 trail=36
  SL=5%  / slope≥0.10% [swing_trail/retest/or_extreme] |   51 trades | win= 76.5% | avgWinR=0.49 | totalR=+7.0 | avgR=+0.136 | maxDD=-2.7R | GO ✓ | sl=12 time_stop=3 trail=36
  SL=10% / no slope filter [swing_trail/retest/or_extreme] |  134 trades | win= 77.6% | avgWinR=0.51 | totalR=+23.6 | avgR=+0.176 | maxDD=-3.9R | GO ✓ | sl=28 time_stop=9 trail=97
  SL=5%  / no slope filter [swing_trail/retest/or_extreme] |  134 trades | win= 76.9% | avgWinR=0.53 | totalR=+24.5 | avgR=+0.183 | maxDD=-3.8R | GO ✓ | sl=29 time_stop=8 trail=97
  SL=10% / slope≥0.02% [tp2_swing/retest/or_extreme] |  124 trades | win= 66.1% | avgWinR=0.85 | totalR=+31.6 | avgR=+0.255 | maxDD=-5.0R | GO ✓ | eod=1 sl=36 time_stop=27 trail=60
  SL=5%  / slope≥0.02% [tp2_swing/retest/or_extreme] |  124 trades | win= 65.3% | avgWinR=0.89 | totalR=+33.3 | avgR=+0.268 | maxDD=-5.0R | GO ✓ | eod=1 sl=37 time_stop=26 trail=60
  SL=10% / slope≥0.10% [tp2_swing/retest/or_extreme] |   51 trades | win= 68.6% | avgWinR=0.78 | totalR=+12.2 | avgR=+0.239 | maxDD=-2.8R | GO ✓ | eod=1 sl=15 time_stop=12 trail=23
  SL=5%  / slope≥0.10% [tp2_swing/retest/or_extreme] |   51 trades | win= 68.6% | avgWinR=0.82 | totalR=+13.5 | avgR=+0.264 | maxDD=-2.8R | GO ✓ | eod=1 sl=15 time_stop=12 trail=23
  SL=10% / no slope filter [tp2_swing/retest/or_extreme] |  134 trades | win= 64.9% | avgWinR=0.84 | totalR=+30.2 | avgR=+0.225 | maxDD=-3.6R | GO ✓ | eod=1 sl=41 time_stop=29 trail=63
  SL=5%  / no slope filter [tp2_swing/retest/or_extreme] |  134 trades | win= 64.2% | avgWinR=0.88 | totalR=+32.0 | avgR=+0.239 | maxDD=-5.0R | GO ✓ | eod=1 sl=42 time_stop=28 trail=63
  SL=10% / slope≥0.02% [fixed_rr/retest/or_extreme] |  124 trades | win= 60.5% | avgWinR=1.01 | totalR=+30.8 | avgR=+0.248 | maxDD=-4.6R | GO ✓ | eod=2 sl=43 time_stop=74 tp2=5
  SL=5%  / slope≥0.02% [fixed_rr/retest/or_extreme] |  124 trades | win= 59.7% | avgWinR=1.07 | totalR=+32.9 | avgR=+0.265 | maxDD=-6.0R | GO ✓ | eod=2 sl=44 time_stop=72 tp2=6
  SL=10% / slope≥0.10% [fixed_rr/retest/or_extreme] |   51 trades | win= 60.8% | avgWinR=0.97 | totalR=+10.6 | avgR=+0.209 | maxDD=-5.0R | GO ✓ | eod=2 sl=19 time_stop=29 tp2=1
  SL=5%  / slope≥0.10% [fixed_rr/retest/or_extreme] |   51 trades | win= 60.8% | avgWinR=1.00 | totalR=+11.7 | avgR=+0.229 | maxDD=-4.9R | GO ✓ | eod=2 sl=19 time_stop=29 tp2=1
  SL=10% / no slope filter [fixed_rr/retest/or_extreme] |  134 trades | win= 59.0% | avgWinR=1.02 | totalR=+29.8 | avgR=+0.222 | maxDD=-4.4R | GO ✓ | eod=2 sl=49 time_stop=77 tp2=6
  SL=5%  / no slope filter [fixed_rr/retest/or_extreme] |  134 trades | win= 58.2% | avgWinR=1.08 | totalR=+31.9 | avgR=+0.238 | maxDD=-6.0R | GO ✓ | eod=2 sl=50 time_stop=75 tp2=7
```
