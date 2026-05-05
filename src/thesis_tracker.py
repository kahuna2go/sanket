"""Stateful thesis-strength tracker with deterministic auto-close enforcement.

Persists per-asset score history to thesis_history.json.
Must be called after every Claude decision cycle, before trade execution.
"""

import json
import os

HISTORY_PATH = "thesis_history.json"
MAX_HISTORY = 5


def _load() -> dict:
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(data: dict):
    with open(HISTORY_PATH, "w") as f:
        json.dump(data, f)


def update_and_check(decisions: dict, active_trades: list) -> dict:
    """Apply deterministic close rules based on thesis_strength history.

    Args:
        decisions: Output from decide_trade — {"reasoning": ..., "trade_decisions": [...]}.
        active_trades: List of active trade dicts from main loop (each has "asset", "is_long").

    Returns:
        decisions with auto-close overrides applied where rules trigger.
    """
    history = _load()

    open_positions = {tr["asset"]: tr.get("is_long", True) for tr in active_trades}

    trade_decisions = decisions.get("trade_decisions", [])

    for item in trade_decisions:
        asset = item.get("asset")
        if not asset:
            continue

        ts = item.get("thesis_strength", 3)

        # Record score before applying rules
        if asset not in history:
            history[asset] = []
        history[asset].append(ts)
        history[asset] = history[asset][-MAX_HISTORY:]

        if asset not in open_positions:
            continue

        action = item.get("action", "hold")
        if action != "hold":
            # Already an active close/flip — rules don't need to override
            continue

        is_long = open_positions[asset]
        close_action = "sell" if is_long else "buy"
        scores = history[asset]

        # Rule 1: thesis_strength == 1 → immediate close
        if ts == 1:
            item["action"] = close_action
            item["allocation_usd"] = 0.0
            item["close_fraction"] = 1.0
            item["rationale"] = f"[AUTO-CLOSE thesis_strength=1] " + item.get("rationale", "")
            continue

        # Rule 2: last 3 scores all <= 2 → exit regardless of P&L
        if len(scores) >= 3 and all(s <= 2 for s in scores[-3:]):
            item["action"] = close_action
            item["allocation_usd"] = 0.0
            item["close_fraction"] = 1.0
            item["rationale"] = f"[AUTO-CLOSE 3x weak thesis {scores[-3:]}] " + item.get("rationale", "")

    # Reset history for assets whose position is being closed this cycle
    closed_assets = {
        item["asset"]
        for item in trade_decisions
        if item.get("asset") in open_positions
        and item.get("action") in ("buy", "sell")
    }
    for asset in closed_assets:
        history.pop(asset, None)

    _save(history)
    decisions["trade_decisions"] = trade_decisions
    return decisions
