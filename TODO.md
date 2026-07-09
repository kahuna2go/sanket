# TODO

## ORB strategy (src/strategies/orb.py)

- [ ] **Trail distance has no floor for tiny opening ranges.** Trail step is
  `0.5 * or_range` (`_manage_trade`, trail branch, ~line 569). On a day like
  2026-07-09 the OR was only 2.10 points, so the trail/breakeven stop sat
  right on top of the entry — the position got stopped ~60s after TP1 on
  ordinary noise instead of running. Consider a floor (e.g.
  `max(0.5 * or_range, min_ticks_or_dollars)`) or a minimum OR-range filter
  that skips the trade entirely when the range is too thin to give the trail
  room to work. Check this against the ORB backtest results before changing
  live behavior.

- [ ] **`_warm_up`'s trail-state inference is fragile if a past SL replacement
  ever failed.** It infers `trail_active` from `sl_price >= entry_px` (long)
  / `<= entry_px` (short) — i.e. "if the resting SL is at/through breakeven,
  TP1 must have already happened." If the SL→BE replacement had silently
  failed (the exact bug fixed 2026-07-09 in `_place_protective_sl`) and the
  original SL was left resting, a restart would misjudge `trail_active=False`
  post-TP1 and could re-trigger a second partial close on the already-halved
  position. Now that #1 (unverified SL replacement) is fixed this should be
  much rarer, but the warm-up logic itself doesn't independently verify —
  worth a follow-up once there's warm-up test coverage to change it safely.
