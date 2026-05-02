"""Entry-point script that wires together the trading agent, data feeds, and API."""

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.agent.decision_maker import TradingAgent
from src.indicators.local_indicators import compute_all, last_n, latest
from src.risk_manager import RiskManager
from src.trading.hyperliquid_api import HyperliquidAPI
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone, timedelta
import math  # For Sharpe
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_interval_seconds(interval_str):
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    from src.config_loader import CONFIG
    assets_env = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")
    if (not args.assets or len(args.assets) == 0) and assets_env:
        # Support space or comma separated
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    hyperliquid = HyperliquidAPI()
    agent = TradingAgent(hyperliquid=hyperliquid)
    risk_mgr = RiskManager()


    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = []  # For Sharpe: list of returns
    active_trades = []  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    recent_events = deque(maxlen=200)
    diary_path = "diary.jsonl"
    initial_account_value = None
    prev_positions_count = None
    prev_account_value = None
    prev_asset_prices = {}
    last_state = {}
    model_usage = {"sonnet": 0, "skipped": 0}
    last_sonnet_time: datetime | None = None
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
        logging.info(msg)

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        nonlocal invocation_count, initial_account_value, prev_positions_count, prev_account_value, prev_asset_prices, last_state, model_usage, last_sonnet_time

        # Pre-load meta cache for correct order sizing
        await hyperliquid.get_meta_and_ctxs()
        # Pre-load HIP-3 dex meta for any dex:asset in the asset list
        hip3_dexes = set()
        for a in args.assets:
            if ":" in a:
                hip3_dexes.add(a.split(":")[0])
        for dex in hip3_dexes:
            await hyperliquid.get_meta_and_ctxs(dex=dex)
            add_event(f"Loaded HIP-3 meta for dex: {dex}")
        if hip3_dexes:
            hyperliquid.register_perp_dexs(list(hip3_dexes))

        # Reverse map: short coin name → full "dex:coin" form, e.g. "CL" → "xyz:CL"
        # Hyperliquid user_state returns HIP-3 positions with just the short name.
        hip3_coin_map = {a.split(":", 1)[1]: a for a in args.assets if ":" in a}

        def _resolve_coin(raw_coin: str) -> str:
            """Return the canonical asset name: maps 'CL' → 'xyz:CL' if needed."""
            return hip3_coin_map.get(raw_coin, raw_coin)

        # Reconstruct active_trades from diary so TP/SL order IDs survive restarts.
        # Replay: buy/sell = open, reconcile_close/risk_force_close = closed.
        # Then cross-reference against live positions to drop anything that closed
        # while the agent was down (TP/SL hit, manual close, etc.).
        try:
            diary_trades = {}  # asset -> latest open trade diary entry
            if os.path.exists(diary_path):
                with open(diary_path, "r") as _df:
                    for _line in _df:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _entry = json.loads(_line)
                        except json.JSONDecodeError:
                            continue
                        _asset = _entry.get("asset")
                        _action = _entry.get("action")
                        if not _asset:
                            continue
                        if _action in ("buy", "sell"):
                            diary_trades[_asset] = _entry
                        elif _action in ("reconcile_close", "risk_force_close"):
                            diary_trades.pop(_asset, None)
                        elif _action == "tpsl_update" and _asset in diary_trades:
                            for _k in ("tp_price", "tp_oid", "sl_price", "sl_oid"):
                                if _entry.get(_k) is not None:
                                    diary_trades[_asset][_k] = _entry[_k]
            if diary_trades:
                _live = await hyperliquid.get_user_state()
                _live_coins = {
                    _resolve_coin(pos.get("coin") or "")
                    for pos in _live.get("positions", [])
                    if abs(float(pos.get("szi", 0) or 0)) > 0
                }
                for _asset, _entry in diary_trades.items():
                    if _asset not in set(args.assets):
                        add_event(f"Skipping diary restore for {_asset} — not in ASSETS (manual management)")
                        continue
                    if _asset in _live_coins:
                        active_trades.append({
                            "asset": _asset,
                            "is_long": _entry.get("action") == "buy",
                            "amount": _entry.get("amount"),
                            "entry_price": _entry.get("entry_price"),
                            "tp_oid": _entry.get("tp_oid"),
                            "sl_oid": _entry.get("sl_oid"),
                            "tp_price": _entry.get("tp_price"),
                            "sl_price": _entry.get("sl_price"),
                            "exit_plan": _entry.get("exit_plan", ""),
                            "opened_at": _entry.get("opened_at"),
                        })
                        add_event(
                            f"Restored active_trade from diary: {_asset} "
                            f"tp_oid={_entry.get('tp_oid')} sl_oid={_entry.get('sl_oid')}"
                        )
        except Exception as _restore_err:
            add_event(f"active_trades restore failed (non-fatal): {_restore_err}")

        while True:
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state
            state = await hyperliquid.get_user_state()
            total_value = state.get('total_value') or state['balance'] + sum(p.get('pnl', 0) for p in state['positions'])
            sharpe = calculate_sharpe(trade_log)

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            positions = []
            for pos_wrap in state['positions']:
                pos = pos_wrap
                coin = _resolve_coin(pos.get('coin') or '')
                current_px = await hyperliquid.get_current_price(coin) if coin else None
                positions.append({
                    "symbol": coin,
                    "quantity": round_or_none(pos.get('szi'), 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })

            # --- RISK: Force-close positions that exceed max loss ---
            try:
                managed_positions = [p for p in state['positions'] if _resolve_coin(p.get('coin') or '') in set(args.assets)]
                positions_to_close = risk_mgr.check_losing_positions(managed_positions)
                for ptc in positions_to_close:
                    coin = ptc["coin"]
                    size = ptc["size"]
                    is_long = ptc["is_long"]
                    add_event(f"RISK FORCE-CLOSE: {coin} at {ptc['loss_pct']}% loss (PnL: ${ptc['pnl']})")
                    try:
                        if is_long:
                            await hyperliquid.place_sell_order(coin, size)
                        else:
                            await hyperliquid.place_buy_order(coin, size)
                        await hyperliquid.cancel_all_orders(coin)
                        # Remove from active trades
                        for tr in active_trades[:]:
                            if tr.get('asset') == coin:
                                active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": coin,
                                "action": "risk_force_close",
                                "loss_pct": ptc["loss_pct"],
                                "pnl": ptc["pnl"],
                            }) + "\n")
                    except Exception as fc_err:
                        add_event(f"Force-close error for {coin}: {fc_err}")
            except Exception as risk_err:
                add_event(f"Risk check error: {risk_err}")

            recent_diary = []
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        entry = json.loads(line)
                        recent_diary.append(entry)
            except Exception:
                pass

            open_orders_struct = []
            try:
                open_orders = await hyperliquid.get_open_orders()
                for o in open_orders[:50]:
                    raw_coin = o.get('coin') or ''
                    coin = _resolve_coin(raw_coin)
                    ts = o.get('timestamp')
                    placed_at = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else None
                    open_orders_struct.append({
                        "coin": coin,
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "is_trigger": o.get('isTrigger', False),
                        "order_type": o.get('orderType'),
                        "placed_at": placed_at
                    })
            except Exception:
                open_orders = []

            recent_fills_struct = []
            fills = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=50)
                for f_entry in fills[-20:]:
                    try:
                        t_raw = f_entry.get('time') or f_entry.get('timestamp')
                        timestamp = None
                        if t_raw is not None:
                            try:
                                t_int = int(t_raw)
                                if t_int > 1e12:
                                    timestamp = datetime.fromtimestamp(t_int / 1000, tz=timezone.utc).isoformat()
                                else:
                                    timestamp = datetime.fromtimestamp(t_int, tz=timezone.utc).isoformat()
                            except Exception:
                                timestamp = str(t_raw)
                        is_buy = f_entry.get('isBuy')
                        if is_buy is None:
                            is_buy = f_entry.get('side') == 'B'
                        raw_fc = f_entry.get('coin') or f_entry.get('asset') or ''
                        recent_fills_struct.append({
                            "timestamp": timestamp,
                            "coin": _resolve_coin(raw_fc),
                            "is_buy": is_buy,
                            "size": round_or_none(f_entry.get('sz') or f_entry.get('size'), 6),
                            "price": round_or_none(f_entry.get('px') or f_entry.get('price'), 2),
                            "fee": round_or_none(f_entry.get('fee'), 6),
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            # Reconcile active trades
            try:
                assets_with_positions = set()
                for pos in state['positions']:
                    try:
                        if abs(float(pos.get('szi') or 0)) > 0:
                            assets_with_positions.add(_resolve_coin(pos.get('coin') or ''))
                    except Exception:
                        continue
                assets_with_orders = {_resolve_coin(o.get('coin') or '') for o in (open_orders or []) if o.get('coin')}
                for tr in active_trades[:]:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions and asset not in assets_with_orders:
                        add_event(f"Reconciling stale active trade for {asset} (no position, no orders)")
                        active_trades.remove(tr)
                        # Find the most recent fill for this asset to capture exit price
                        short_asset = asset.split(":", 1)[1] if ":" in asset else asset
                        exit_fill = next(
                            (f for f in reversed(fills or [])
                             if (f.get('coin') or f.get('asset') or '') in (asset, short_asset)),
                            None
                        )
                        exit_price = round_or_none(exit_fill.get('px') or exit_fill.get('price'), 2) if exit_fill else None
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "reconcile_close",
                                "reason": "no_position_no_orders",
                                "opened_at": tr.get('opened_at'),
                                "exit_price": exit_price,
                                "exit_is_buy": exit_fill.get('isBuy') if exit_fill else None,
                            }) + "\n")
            except Exception:
                pass

            # For each active trade with a live position: cancel orphaned entry limits
            # and ensure TP/SL orders are always on the book.
            try:
                def _is_trigger(o):
                    return (
                        o.get('isTrigger')
                        or (isinstance(o.get('orderType'), dict) and 'trigger' in o.get('orderType', {}))
                        or o.get('triggerPx') is not None
                    )

                def _trigger_price_matches(o, price, tol=0.001):
                    try:
                        return abs(float(o.get('triggerPx', 0)) - price) / price <= tol
                    except (TypeError, ZeroDivisionError):
                        return False

                trigger_oids = {o.get('oid') for o in (open_orders or []) if _is_trigger(o)}
                for tr in active_trades:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions:
                        continue
                    # Cancel resting entry limits (non-trigger) for assets with an open position
                    orphaned = [
                        o for o in (open_orders or [])
                        if hyperliquid._coin_matches(o.get('coin', ''), asset)
                        and not _is_trigger(o)
                    ]
                    if orphaned:
                        await hyperliquid.cancel_limit_orders(asset)
                        add_event(f"Cancelled {len(orphaned)} orphaned entry limit(s) for {asset}")
                    # Collect all trigger orders for this asset
                    asset_triggers = [
                        o for o in (open_orders or [])
                        if hyperliquid._coin_matches(o.get('coin', ''), asset) and _is_trigger(o)
                    ]
                    # Learn tp_price/sl_price from existing orders when not stored in tr.
                    # Uses orderType string ("Take Profit Market" vs "Stop Market") to classify.
                    # This handles restarts where the diary was written without these fields.
                    _tpsl_learnt = False
                    for _o in asset_triggers:
                        _ot = ((_o.get('orderType') or '')).lower()
                        _px = _o.get('triggerPx')
                        _oid = _o.get('oid')
                        if _px is None:
                            continue
                        if not tr.get('tp_price') and 'take profit' in _ot:
                            tr['tp_price'] = float(_px)
                            tr['tp_oid'] = _oid
                            add_event(f"Learnt TP for {asset} from existing order: {tr['tp_price']}")
                            _tpsl_learnt = True
                        elif not tr.get('sl_price') and 'stop' in _ot:
                            tr['sl_price'] = float(_px)
                            tr['sl_oid'] = _oid
                            add_event(f"Learnt SL for {asset} from existing order: {tr['sl_price']}")
                            _tpsl_learnt = True
                    if _tpsl_learnt:
                        with open(diary_path, "a") as _dlf:
                            _dlf.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "tpsl_update",
                                "tp_price": tr.get('tp_price'),
                                "tp_oid": tr.get('tp_oid'),
                                "sl_price": tr.get('sl_price'),
                                "sl_oid": tr.get('sl_oid'),
                            }) + "\n")
                    # Deduplicate: group trigger orders by triggerPx, cancel extras
                    seen_prices: dict = {}
                    for o in asset_triggers:
                        px = o.get('triggerPx')
                        oid = o.get('oid')
                        if px is not None and oid is not None:
                            if px in seen_prices:
                                try:
                                    await hyperliquid.cancel_order(asset, oid)
                                    add_event(f"Cancelled duplicate trigger order for {asset} at {px}")
                                except Exception as _de:
                                    add_event(f"Failed to cancel duplicate trigger for {asset}: {_de}")
                            else:
                                seen_prices[px] = oid
                    # Re-place TP if missing from book
                    tp_on_book = (
                        (tr.get('tp_oid') and tr['tp_oid'] in trigger_oids)
                        or (tr.get('tp_price') and any(
                            o for o in (open_orders or [])
                            if hyperliquid._coin_matches(o.get('coin', ''), asset)
                            and _is_trigger(o)
                            and _trigger_price_matches(o, tr['tp_price'])
                        ))
                    )
                    if not tp_on_book:
                        tp_price = tr.get('tp_price')
                        if not tp_price:
                            entry = tr.get('entry_price') or asset_prices.get(asset)
                            if entry:
                                pct = risk_mgr.mandatory_tp_pct / 100.0
                                tp_price = round(
                                    entry * (1 + pct) if tr['is_long'] else entry * (1 - pct), 4
                                )
                                add_event(f"No TP stored for {asset} — mandatory fallback at {tp_price}")
                        if tp_price:
                            try:
                                tp_order = await hyperliquid.place_take_profit(
                                    asset, tr['is_long'], tr['amount'], tp_price
                                )
                                tp_oids = hyperliquid.extract_oids(tp_order)
                                tr['tp_oid'] = tp_oids[0] if tp_oids else None
                                tr['tp_price'] = tp_price
                                add_event(f"Re-placed missing TP for {asset} at {tp_price}")
                                with open(diary_path, "a") as _dtf:
                                    _dtf.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "tpsl_update",
                                        "tp_price": tr['tp_price'],
                                        "tp_oid": tr['tp_oid'],
                                    }) + "\n")
                            except Exception as e:
                                add_event(f"Failed to re-place TP for {asset}: {e}")
                    # Re-place SL if missing — mandatory fallback if no sl_price stored
                    sl_on_book = (
                        (tr.get('sl_oid') and tr['sl_oid'] in trigger_oids)
                        or (tr.get('sl_price') and any(
                            o for o in (open_orders or [])
                            if hyperliquid._coin_matches(o.get('coin', ''), asset)
                            and _is_trigger(o)
                            and _trigger_price_matches(o, tr['sl_price'])
                        ))
                    )
                    if not sl_on_book:
                        sl_price = tr.get('sl_price')
                        if not sl_price:
                            entry = tr.get('entry_price') or asset_prices.get(asset)
                            if entry:
                                pct = risk_mgr.mandatory_sl_pct / 100.0
                                sl_price = round(entry * (1 - pct) if tr['is_long'] else entry * (1 + pct), 4)
                                add_event(f"No SL price stored for {asset} — using mandatory fallback at {sl_price}")
                        if sl_price:
                            try:
                                sl_order = await hyperliquid.place_stop_loss(asset, tr['is_long'], tr['amount'], sl_price)
                                sl_oids = hyperliquid.extract_oids(sl_order)
                                tr['sl_oid'] = sl_oids[0] if sl_oids else None
                                tr['sl_price'] = sl_price
                                add_event(f"Re-placed missing SL for {asset} at {sl_price}")
                                with open(diary_path, "a") as _dsf:
                                    _dsf.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "tpsl_update",
                                        "sl_price": tr['sl_price'],
                                        "sl_oid": tr['sl_oid'],
                                    }) + "\n")
                            except Exception as e:
                                add_event(f"Failed to re-place SL for {asset}: {e}")
                # Adopt positions not tracked in active_trades.
                # Covers positions opened in a prior session or before the coin-name
                # bug was fixed. For each: cancel any resting entry limits, place a
                # mandatory SL using live position data, add to active_trades for this
                # session, and write a diary entry so it survives the next restart.
                tracked_assets = {tr.get('asset') for tr in active_trades}
                for asset in (assets_with_positions - tracked_assets) & set(args.assets):
                    untracked_limits = [
                        o for o in (open_orders or [])
                        if hyperliquid._coin_matches(o.get('coin', ''), asset)
                        and not o.get('isTrigger')
                    ]
                    if untracked_limits:
                        await hyperliquid.cancel_limit_orders(asset)
                        add_event(f"Cancelled {len(untracked_limits)} entry limit(s) for untracked position {asset}")

                    pos_data = next(
                        (p for p in state['positions']
                         if _resolve_coin(p.get('coin') or '') == asset),
                        None
                    )
                    if not pos_data:
                        continue
                    size = float(pos_data.get('szi') or 0)
                    entry_px = float(pos_data.get('entryPx') or 0)
                    if not size or not entry_px:
                        continue
                    is_long = size > 0
                    amount = abs(size)

                    existing_triggers = [
                        o for o in (open_orders or [])
                        if hyperliquid._coin_matches(o.get('coin', ''), asset)
                        and isinstance(o.get('orderType'), dict)
                        and 'trigger' in o.get('orderType', {})
                    ]
                    existing_tpsl = {
                        o.get('orderType', {}).get('trigger', {}).get('tpsl')
                        for o in existing_triggers
                    }

                    adopted_sl_oid = None
                    adopted_sl_price = None
                    if 'sl' not in existing_tpsl:
                        pct = risk_mgr.mandatory_sl_pct / 100.0
                        adopted_sl_price = round(
                            entry_px * (1 - pct) if is_long else entry_px * (1 + pct), 4
                        )
                        try:
                            sl_order = await hyperliquid.place_stop_loss(asset, is_long, amount, adopted_sl_price)
                            sl_oids = hyperliquid.extract_oids(sl_order)
                            adopted_sl_oid = sl_oids[0] if sl_oids else None
                            add_event(f"Placed mandatory SL for untracked {asset} at {adopted_sl_price}")
                        except Exception as _sl_err:
                            add_event(f"Failed to place SL for untracked {asset}: {_sl_err}")

                    adopted_tp_oid = None
                    adopted_tp_price = None
                    if 'tp' not in existing_tpsl:
                        pct = risk_mgr.mandatory_tp_pct / 100.0
                        adopted_tp_price = round(
                            entry_px * (1 + pct) if is_long else entry_px * (1 - pct), 4
                        )
                        try:
                            tp_order = await hyperliquid.place_take_profit(asset, is_long, amount, adopted_tp_price)
                            tp_oids = hyperliquid.extract_oids(tp_order)
                            adopted_tp_oid = tp_oids[0] if tp_oids else None
                            add_event(f"Placed mandatory TP for untracked {asset} at {adopted_tp_price}")
                        except Exception as _tp_err:
                            add_event(f"Failed to place TP for untracked {asset}: {_tp_err}")

                    new_tr = {
                        "asset": asset,
                        "is_long": is_long,
                        "amount": amount,
                        "entry_price": entry_px,
                        "tp_oid": adopted_tp_oid,
                        "sl_oid": adopted_sl_oid,
                        "tp_price": adopted_tp_price,
                        "sl_price": adopted_sl_price,
                        "exit_plan": "adopted from untracked live position",
                        "opened_at": datetime.now(timezone.utc).isoformat(),
                    }
                    active_trades.append(new_tr)
                    add_event(f"Adopted untracked {asset} into active_trades (long={is_long}, amount={amount:.4f}, entry={entry_px})")
                    try:
                        with open(diary_path, "a") as _df:
                            _df.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "buy" if is_long else "sell",
                                "order_type": "adopted",
                                "amount": amount,
                                "entry_price": entry_px,
                                "sl_price": adopted_sl_price,
                                "sl_oid": adopted_sl_oid,
                                "tp_price": adopted_tp_price,
                                "tp_oid": adopted_tp_oid,
                                "exit_plan": "Adopted from untracked live position",
                                "opened_at": new_tr["opened_at"],
                                "filled": True,
                            }) + "\n")
                    except Exception as _dw_err:
                        add_event(f"Failed to write adopted trade diary for {asset}: {_dw_err}")
            except Exception as _rec_err:
                add_event(f"TP/SL reconcile error (non-fatal): {_rec_err}")

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
                "sharpe_ratio": round_or_none(sharpe, 3),
                "positions": positions,
                "active_trades": [
                    {
                        "asset": tr.get('asset'),
                        "is_long": tr.get('is_long'),
                        "amount": round_or_none(tr.get('amount'), 6),
                        "entry_price": round_or_none(tr.get('entry_price'), 2),
                        "tp_oid": tr.get('tp_oid'),
                        "sl_oid": tr.get('sl_oid'),
                        "exit_plan": tr.get('exit_plan'),
                        "opened_at": tr.get('opened_at')
                    }
                    for tr in active_trades
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
            }

            # Gather data for ALL assets first (using Hyperliquid candles + local indicators)
            market_sections = []
            asset_prices = {}
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    # Fetch candles from Hyperliquid and compute indicators locally
                    candles_5m = await hyperliquid.get_candles(asset, "5m", 100)
                    candles_4h = await hyperliquid.get_candles(asset, "4h", 100)

                    intra = compute_all(candles_5m)
                    lt = compute_all(candles_4h)

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-10:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        "intraday": {
                            "ema20": round_or_none(latest(intra.get("ema20", [])), 2),
                            "macd": round_or_none(latest(intra.get("macd", [])), 2),
                            "rsi7": round_or_none(latest(intra.get("rsi7", [])), 2),
                            "rsi14": round_or_none(latest(intra.get("rsi14", [])), 2),
                            "series": {
                                "ema20": round_series(last_n(intra.get("ema20", []), 10), 2),
                                "macd": round_series(last_n(intra.get("macd", []), 10), 2),
                                "rsi7": round_series(last_n(intra.get("rsi7", []), 10), 2),
                                "rsi14": round_series(last_n(intra.get("rsi14", []), 10), 2),
                            }
                        },
                        "long_term": {
                            "ema20": round_or_none(latest(lt.get("ema20", [])), 2),
                            "ema50": round_or_none(latest(lt.get("ema50", [])), 2),
                            "atr3": round_or_none(latest(lt.get("atr3", [])), 2),
                            "atr14": round_or_none(latest(lt.get("atr14", [])), 2),
                            "macd_series": round_series(last_n(lt.get("macd", []), 10), 2),
                            "rsi_series": round_series(last_n(lt.get("rsi14", []), 10), 2),
                        },
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids,
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # Single LLM call with all assets
            context_payload = OrderedDict([
                ("invocation", {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count
                }),
                ("account", dashboard),
                ("risk_limits", risk_mgr.get_risk_summary()),
                ("market_data", market_sections),
                ("instructions", {
                    "assets": args.assets,
                    "requirement": "Decide actions for all assets and return a strict JSON object matching the schema."
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            with open("prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

            # Escalation logic: call Sonnet only when something actionable is happening.
            # Otherwise skip the LLM entirely and auto-hold (zero cost).
            #
            # Sonnet fires when ANY of:
            #   1. First run (no prior state)
            #   2. Any asset price moved > SONNET_PRICE_MOVE_PCT
            #   3. Any open position is within SONNET_TPSL_PROXIMITY_PCT of its TP or SL
            #   4. Open positions exist and Sonnet hasn't run in SONNET_HEALTH_CHECK_MINUTES
            price_move_threshold = float(CONFIG.get("sonnet_price_move_pct") or 0.5) / 100.0
            tpsl_proximity = float(CONFIG.get("sonnet_tpsl_proximity_pct") or 1.25) / 100.0
            health_check_minutes = int(float(CONFIG.get("sonnet_health_check_minutes") or 60))

            first_run = prev_positions_count is None

            price_moved = any(
                prev_asset_prices.get(a) is not None
                and prev_asset_prices[a] != 0
                and asset_prices.get(a) is not None
                and abs(asset_prices[a] - prev_asset_prices[a]) / prev_asset_prices[a] > price_move_threshold
                for a in args.assets
            )

            tpsl_near = False
            for tr in active_trades:
                asset = tr.get('asset')
                price = asset_prices.get(asset)
                if not price:
                    continue
                for level in (tr.get('tp_price'), tr.get('sl_price')):
                    if level and abs(price - level) / price <= tpsl_proximity:
                        tpsl_near = True
                        break
                if tpsl_near:
                    break

            health_check_due = (
                bool(active_trades)
                and (
                    last_sonnet_time is None
                    or (datetime.now(timezone.utc) - last_sonnet_time).total_seconds() / 60 >= health_check_minutes
                )
            )

            use_sonnet = first_run or price_moved or tpsl_near or health_check_due

            if not use_sonnet:
                model_usage["skipped"] += 1
                add_event(
                    f"Skipping LLM (no trigger): price_moved={price_moved}, "
                    f"tpsl_near={tpsl_near}, health_check_due={health_check_due} — auto-hold"
                )
                prev_positions_count = len(active_trades)
                prev_account_value = account_value
                prev_asset_prices = dict(asset_prices)
                continue

            model_usage["sonnet"] += 1
            last_sonnet_time = datetime.now(timezone.utc)
            add_event(
                f"Sonnet triggered: first_run={first_run}, price_moved={price_moved}, "
                f"tpsl_near={tpsl_near}, health_check_due={health_check_due}"
            )

            def _is_failed_outputs(outs):
                """Return True when outputs are missing or clearly invalid."""
                if not isinstance(outs, dict):
                    return True
                decisions = outs.get("trade_decisions")
                if not isinstance(decisions, list) or not decisions:
                    return True
                try:
                    return all(
                        isinstance(o, dict)
                        and (o.get('action') == 'hold')
                        and ('parse error' in (o.get('rationale', '').lower()))
                        for o in decisions
                    )
                except Exception:
                    return True

            try:
                outputs = agent.decide_trade(args.assets, context, model=agent.model)
                if not isinstance(outputs, dict):
                    add_event(f"Invalid output format (expected dict): {outputs}")
                    outputs = {}
            except Exception as e:
                import traceback
                add_event(f"Agent error: {e}")
                add_event(f"Traceback: {traceback.format_exc()}")
                outputs = {}

            # Retry once on failure/parse error; always use Sonnet for the retry
            if _is_failed_outputs(outputs):
                add_event("Retrying LLM once due to invalid/parse-error output")
                context_retry_payload = OrderedDict([
                    ("retry_instruction", "Return ONLY a JSON object with exactly one key: \"trade_decisions\" (array of per-asset objects). No prose, no markdown, no code fences, no reasoning field."),
                    ("original_context", context_payload)
                ])
                context_retry = json.dumps(context_retry_payload, default=json_default)
                try:
                    outputs = agent.decide_trade(args.assets, context_retry, model=agent.model)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

            reasoning_text = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
            if reasoning_text:
                add_event(f"LLM reasoning summary: {reasoning_text}")

            # Log full cycle decisions for the dashboard
            cycle_decisions = []
            for d in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                cycle_decisions.append({
                    "asset": d.get("asset"),
                    "action": d.get("action", "hold"),
                    "allocation_usd": d.get("allocation_usd", 0),
                    "rationale": d.get("rationale", ""),
                })
            cycle_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": invocation_count,
                "reasoning": reasoning_text[:2000] if reasoning_text else "",
                "decisions": cycle_decisions,
                "account_value": round_or_none(account_value, 2),
                "balance": round_or_none(state['balance'], 2),
                "positions_count": len([p for p in state['positions'] if abs(float(p.get('szi') or 0)) > 0]),
            }
            try:
                with open("decisions.jsonl", "a") as f:
                    f.write(json.dumps(cycle_log) + "\n")
            except Exception:
                pass

            # Execute trades for each asset
            for output in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue
                    action = output.get("action")
                    current_price = asset_prices.get(asset, 0)
                    action = output["action"]
                    rationale = output.get("rationale", "")
                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")
                    if action == "cancel_limits":
                        try:
                            result = await hyperliquid.cancel_limit_orders(asset)
                            add_event(f"CANCEL_LIMITS {asset}: {result.get('cancelled_count', 0)} order(s) cancelled — {rationale}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "cancel_limits",
                                    "cancelled_count": result.get("cancelled_count", 0),
                                    "rationale": rationale,
                                }) + "\n")
                        except Exception as cl_err:
                            add_event(f"Cancel limits error {asset}: {cl_err}")
                        continue
                    elif action in ("buy", "sell"):
                        is_buy = action == "buy"

                        # Detect close: LLM is selling an existing long or buying an existing short.
                        # In that case use the tracked position size — allocation_usd is irrelevant
                        # and risk checks for new exposure do not apply.
                        existing_tr = next(
                            (tr for tr in active_trades
                             if tr.get('asset') == asset
                             and tr.get('is_long') != is_buy),  # opposite direction = close
                            None
                        )
                        if existing_tr:
                            amount = existing_tr['amount']
                            alloc_usd = amount * current_price
                            add_event(f"CLOSE {asset}: using tracked size {amount:.4f} (LLM allocation ignored)")
                        else:
                            alloc_usd = float(output.get("allocation_usd", 0.0))
                            if alloc_usd <= 0:
                                add_event(f"Holding {asset}: zero/negative allocation")
                                continue

                            # --- RISK: Validate trade before execution ---
                            output["current_price"] = current_price
                            allowed, reason, output = risk_mgr.validate_trade(
                                output, state, initial_account_value or 0, open_orders_struct
                            )
                            if not allowed:
                                add_event(f"RISK BLOCKED {asset}: {reason}")
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "risk_blocked",
                                        "reason": reason,
                                        "original_alloc_usd": alloc_usd,
                                    }) + "\n")
                                continue
                            alloc_usd = float(output.get("allocation_usd", alloc_usd))
                            amount = alloc_usd / current_price

                        # Place market or limit order
                        order_type = output.get("order_type", "market")
                        limit_price = output.get("limit_price")

                        # Cancel resting entry limit orders before opening a new one.
                        # If cancel fails, skip the order — don't stack on uncancelled limits.
                        try:
                            cancelled = await hyperliquid.cancel_limit_orders(asset)
                            if cancelled.get("status") == "error":
                                add_event(f"Pre-trade cancel failed for {asset}: {cancelled.get('message')} — skipping order")
                                continue
                            if cancelled.get("cancelled_count", 0) > 0:
                                add_event(f"Cancelled {cancelled['cancelled_count']} entry limit(s) for {asset} before new {action} order")
                        except Exception as _ce:
                            add_event(f"Pre-trade cancel error for {asset}: {_ce} — skipping order")
                            continue

                        # Enforce exchange-side leverage before every order.
                        # check_leverage() only validates allocation/balance ratio; the
                        # actual position size is determined by the exchange leverage
                        # setting, which must be capped here explicitly.
                        try:
                            max_lev = int(risk_mgr.max_leverage)
                            await hyperliquid.set_leverage(asset, max_lev)
                        except Exception as _lev_err:
                            add_event(f"WARNING: Failed to set leverage for {asset}: {_lev_err}")

                        tp_oid = None
                        sl_oid = None
                        if order_type == "limit" and limit_price:
                            limit_price = float(limit_price)
                            tp_price_val = output.get("tp_price")
                            sl_price_val = output.get("sl_price")
                            if tp_price_val and sl_price_val:
                                # Bracket order: TP/SL only activate after entry fills
                                order = await hyperliquid.place_limit_with_tpsl(
                                    asset, is_buy, amount, limit_price,
                                    float(tp_price_val), float(sl_price_val)
                                )
                                oids = hyperliquid.extract_oids(order)
                                # statuses: [0]=entry, [1]=tp, [2]=sl
                                tp_oid = oids[1] if len(oids) > 1 else None
                                sl_oid = oids[2] if len(oids) > 2 else None
                                add_event(f"LIMIT {action.upper()} {asset} {amount:.4f} @ ${limit_price} with bracket TP={tp_price_val} SL={sl_price_val}")
                            else:
                                # No TP/SL prices — plain limit
                                if is_buy:
                                    order = await hyperliquid.place_limit_buy(asset, amount, limit_price)
                                else:
                                    order = await hyperliquid.place_limit_sell(asset, amount, limit_price)
                                add_event(f"LIMIT {action.upper()} {asset} amount {amount:.4f} at limit ${limit_price}")
                        else:
                            order = await hyperliquid.place_buy_order(asset, amount) if is_buy else await hyperliquid.place_sell_order(asset, amount)

                        # Confirm by checking recent fills for this asset shortly after placing
                        await asyncio.sleep(1)
                        fills_check = await hyperliquid.get_recent_fills(limit=10)
                        filled = False
                        for fc in reversed(fills_check):
                            try:
                                if (fc.get('coin') == asset or fc.get('asset') == asset):
                                    filled = True
                                    break
                            except Exception:
                                continue
                        trade_log.append({"type": action, "price": current_price, "amount": amount, "exit_plan": output["exit_plan"], "filled": filled})
                        # For market orders, place TP/SL immediately (position is open now)
                        if order_type != "limit":
                            if output.get("tp_price"):
                                tp_order = await hyperliquid.place_take_profit(asset, is_buy, amount, output["tp_price"])
                                tp_oids = hyperliquid.extract_oids(tp_order)
                                tp_oid = tp_oids[0] if tp_oids else None
                                add_event(f"TP placed {asset} at {output['tp_price']}")
                            if output.get("sl_price"):
                                sl_order = await hyperliquid.place_stop_loss(asset, is_buy, amount, output["sl_price"])
                                sl_oids = hyperliquid.extract_oids(sl_order)
                                sl_oid = sl_oids[0] if sl_oids else None
                                add_event(f"SL placed {asset} at {output['sl_price']}")
                        # Reconcile: if opposite-side position exists or TP/SL just filled, clear stale active_trades for this asset
                        for existing in active_trades[:]:
                            if existing.get('asset') == asset:
                                try:
                                    active_trades.remove(existing)
                                except ValueError:
                                    pass
                        active_trades.append({
                            "asset": asset,
                            "is_long": is_buy,
                            "amount": amount,
                            "entry_price": current_price,
                            "tp_oid": tp_oid,
                            "sl_oid": sl_oid,
                            "tp_price": output.get("tp_price"),
                            "sl_price": output.get("sl_price"),
                            "exit_plan": output["exit_plan"],
                            "opened_at": datetime.now().isoformat()
                        })
                        add_event(f"{action.upper()} {asset} amount {amount:.4f} at ~{current_price}")
                        if rationale:
                            add_event(f"Post-trade rationale for {asset}: {rationale}")
                        # Write to diary after confirming fills status
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": action,
                                "order_type": order_type,
                                "limit_price": limit_price,
                                "allocation_usd": alloc_usd,
                                "amount": amount,
                                "entry_price": current_price,
                                "tp_price": output.get("tp_price"),
                                "tp_oid": tp_oid,
                                "sl_price": output.get("sl_price"),
                                "sl_oid": sl_oid,
                                "exit_plan": output.get("exit_plan", ""),
                                "rationale": output.get("rationale", ""),
                                "order_result": str(order),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "filled": filled
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                    else:  # hold
                        add_event(f"Hold {asset}: {output.get('rationale', '')}")
                        # Write hold to diary
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "asset": asset,
                                "action": "hold",
                                "rationale": output.get("rationale", "")
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                except Exception as e:
                    import traceback
                    add_event(f"Execution error {asset}: {e}")
                    add_event(f"Traceback: {traceback.format_exc()}")

            prev_positions_count = len(active_trades)
            prev_account_value = account_value
            prev_asset_prices = dict(asset_prices)
            last_state = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "session_start": start_time.isoformat(),
                "initial_account_value": initial_account_value,
                "account": dashboard,
            }

            await asyncio.sleep(get_interval_seconds(args.interval))

    async def handle_diary(request):
        """Return diary entries as JSON or newline-delimited text."""
        try:
            raw = request.query.get('raw')
            download = request.query.get('download')
            if raw or download:
                if not os.path.exists(diary_path):
                    return web.Response(text="", content_type="text/plain")
                with open(diary_path, "r") as f:
                    data = f.read()
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename=diary.jsonl"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(request.query.get('limit', '200'))
            with open(diary_path, "r") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            entries = [json.loads(l) for l in lines[start:]]
            return web.json_response({"entries": entries})
        except FileNotFoundError:
            return web.json_response({"entries": []})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path = request.query.get('path', 'llm_requests.log')
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_param) if limit_param else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_state(request):
        """Return full dashboard state as JSON for the dashboard UI."""
        try:
            recent_decisions = []
            try:
                with open("decisions.jsonl", "r") as f:
                    lines = f.readlines()
                for line in lines[-20:]:
                    try:
                        recent_decisions.append(json.loads(line.strip()))
                    except Exception:
                        pass
            except FileNotFoundError:
                pass
            return web.json_response({
                "status": "running",
                "uptime_minutes": round((datetime.now(timezone.utc) - start_time).total_seconds() / 60, 1),
                "invocation_count": invocation_count,
                "model_usage": model_usage,
                "recent_decisions": recent_decisions,
                **last_state,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # Cache for get_all_fills so dashboard polling doesn't hammer the API
    _fills_cache: dict = {"data": None, "expires": 0.0}

    def _parse_ts(ts_str):
        """Parse an ISO or epoch-ms timestamp to epoch seconds, or None."""
        if ts_str is None:
            return None
        try:
            t_int = int(ts_str)
            return t_int / 1000 if t_int > 1e10 else float(t_int)
        except (TypeError, ValueError):
            pass
        try:
            import time as _time
            s = str(ts_str).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    async def handle_history(request):
        """Return all-time realized P&L directly from Hyperliquid fill history."""
        import time as _time
        from collections import defaultdict, deque
        try:
            # Fetch full fill history, cached for 60 s
            now = _time.time()
            if _fills_cache["expires"] < now:
                all_fills = await hyperliquid.get_all_fills()
                _fills_cache["data"] = all_fills
                _fills_cache["expires"] = now + 60.0
            else:
                all_fills = _fills_cache["data"] or []

            SANKET_START_TS = _parse_ts("2026-04-24T00:00:00+00:00")

            # Sort all fills chronologically once
            def _fl_ts(fl):
                return _parse_ts(fl.get("time") or fl.get("timestamp")) or 0.0
            sorted_fills = sorted(all_fills, key=_fl_ts)

            # Build FIFO open-fill stacks per asset (for entry price recovery)
            open_stacks: dict = defaultdict(deque)
            for fl in sorted_fills:
                ts = _fl_ts(fl)
                if ts < SANKET_START_TS:
                    continue
                if "Open" in (fl.get("dir") or ""):
                    coin = fl.get("coin") or fl.get("asset") or ""
                    open_stacks[coin].append(fl)

            # Time windows for daily / weekly buckets (UTC)
            now_dt = datetime.now(timezone.utc)
            day_start_ts = now_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            week_start_ts = (now_dt - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

            # Single pass: accumulate fees from all fills; record completed trades from closing fills
            total_closed_pnl = 0.0
            total_fees = 0.0
            day_closed_pnl = 0.0
            day_fees = 0.0
            day_trades = 0
            week_closed_pnl = 0.0
            week_fees = 0.0
            week_trades = 0
            closing_trades = []

            for fl in sorted_fills:
                ts = _fl_ts(fl)
                if ts < SANKET_START_TS:
                    continue
                fee = float(fl.get("fee") or 0)
                total_fees += fee
                if ts >= week_start_ts:
                    week_fees += fee
                if ts >= day_start_ts:
                    day_fees += fee
                closed_pnl = float(fl.get("closedPnl") or 0)
                if closed_pnl == 0:
                    continue  # opening fill — counted fee already, no P&L to record
                total_closed_pnl += closed_pnl
                if ts >= week_start_ts:
                    week_closed_pnl += closed_pnl
                    week_trades += 1
                if ts >= day_start_ts:
                    day_closed_pnl += closed_pnl
                    day_trades += 1
                coin = fl.get("coin") or fl.get("asset") or ""
                # Pop the earliest unmatched open fill to get entry price
                entry_fill = open_stacks[coin].popleft() if open_stacks[coin] else None
                entry_price = round_or_none(float(entry_fill.get("px") or 0), 4) if entry_fill else None
                closing_trades.append({
                    "closed_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "asset": coin,
                    "dir": fl.get("dir") or "",
                    "entry_price": entry_price,
                    "exit_price": round_or_none(float(fl.get("px") or 0), 4),
                    "size": round_or_none(float(fl.get("sz") or 0), 6),
                    "pnl": round(closed_pnl, 4),
                    "fee": round(fee, 6),
                })

            return web.json_response({
                "total_realized_pnl": round(total_closed_pnl - total_fees, 4),
                "gross_pnl": round(total_closed_pnl, 4),
                "total_fees": round(total_fees, 4),
                "completed_trades": len(closing_trades),
                "daily": {
                    "net_pnl": round(day_closed_pnl - day_fees, 4),
                    "gross_pnl": round(day_closed_pnl, 4),
                    "fees": round(day_fees, 4),
                    "trades": day_trades,
                },
                "weekly": {
                    "net_pnl": round(week_closed_pnl - week_fees, 4),
                    "gross_pnl": round(week_closed_pnl, 4),
                    "fees": round(week_fees, 4),
                    "trades": week_trades,
                },
                "trades": closing_trades,
            })
        except Exception as e:
            import traceback
            logging.error("handle_history error: %s\n%s", e, traceback.format_exc())
            return web.json_response({"error": str(e)}, status=500)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/logs', handle_logs)
        app.router.add_get('/state', handle_state)
        app.router.add_get('/history', handle_history)

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        @web.middleware
        async def cors(request, handler):
            resp = await handler(request)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        app = web.Application(middlewares=[cors])
        await start_api(app)
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        await run_loop()

    def calculate_total_return(state, trade_log):
        """Compute percent return relative to an assumed initial balance."""
        initial = 10000
        current = state['balance'] + sum(p.get('pnl', 0) for p in state.get('positions', []))
        return ((current - initial) / initial) * 100 if initial else 0

    def calculate_sharpe(returns):
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    async def check_exit_condition(trade, hyperliquid_api):
        """Evaluate whether a given trade's exit plan triggers a close."""
        plan = (trade.get("exit_plan") or "").lower()
        if not plan:
            return False
        try:
            candles_4h = await hyperliquid_api.get_candles(trade["asset"], "4h", 60)
            indicators = compute_all(candles_4h)
            if "macd" in plan and "below" in plan:
                macd_val = latest(indicators.get("macd", []))
                threshold = float(plan.split("below")[-1].strip())
                return macd_val is not None and macd_val < threshold
            if "close above ema50" in plan:
                ema50_val = latest(indicators.get("ema50", []))
                current = await hyperliquid_api.get_current_price(trade["asset"])
                return ema50_val is not None and current > ema50_val
        except Exception:
            return False
        return False

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
