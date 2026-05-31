"""Entry-point script that wires together the trading agent, data feeds, and API."""

# websocket-client ignores SSL_CERT_FILE and uses load_default_certs() which
# requires macOS system keychain — unavailable in launcher subprocesses.
# Patch run_forever() to always pass certifi's CA bundle.
import certifi as _certifi
from hyperliquid import websocket_manager as _wm
_orig_ws_run = _wm.WebsocketManager.run
def _ws_run(self):
    self.ping_sender.start()
    self.ws.run_forever(sslopt={"ca_certs": _certifi.where()})
_wm.WebsocketManager.run = _ws_run

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
_ROOT = pathlib.Path(__file__).parent.parent
from src.agent.decision_maker import TradingAgent
from src.thesis_tracker import update_and_check
from src.macro_filter import get_macro_context
from src.indicators.local_indicators import compute_all, last_n, latest, swing_structure, zz_structure, volume_profile, rvol, ema
from src.risk_manager import RiskManager
from src.trading.hyperliquid_api import HyperliquidAPI
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import math  # For Sharpe
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series
from src.utils.display import (
    setup_logging, print_loop_header, print_position_table,
    print_decision, print_claude_stats, print_orb_status,
)

load_dotenv()

setup_logging()


_VIENNA_TZ = ZoneInfo("Europe/Vienna")

# Per-asset strategy config
# Fib assets use ZigZag + Fib 0.745 retracement entry; others use VA Bounce
_FIB_ASSETS = {"SOL", "ETH"}
_ZZ_DEV     = {"SOL": 3.0, "ETH": 2.0}   # ZigZag deviation_pct per asset
_ZZ_DEV_DEFAULT = 2.0

# Active trading windows (Vienna local time)
_LONDON_START = 8 + 30 / 60  # 08:30
_LONDON_END   = 11.5          # 11:30
_NY_START     = 16.0          # 16:00
_NY_END       = 20.0          # 20:00

# S&P 500 ORB session window (Vienna local time)
_SP500_PRESESSION = 15.0      # 15:00 — pre-session bias eval starts
_SP500_END        = 20.0      # 20:00 — time stop / session end
_SP500_ASSETS     = frozenset({"xyz:SP500"})


def _get_session(utc_now: datetime) -> dict:
    hf = utc_now.astimezone(_VIENNA_TZ)
    hf = hf.hour + hf.minute / 60
    if _LONDON_START <= hf < _LONDON_END:
        return {"name": "london", "active": True,  "interval_secs": 300, "move_pct": 0.003}
    elif _SP500_PRESESSION <= hf < _NY_START:
        return {"name": "pre-ny", "active": True,  "interval_secs": 300, "move_pct": 0.003}
    elif _NY_START <= hf < _NY_END:
        return {"name": "ny",     "active": True,  "interval_secs": 300, "move_pct": 0.003}
    else:
        return {"name": "off",    "active": False, "interval_secs": 900, "move_pct": 0.005}


def _vhour(ts_ms: int) -> float:
    """Return candle open time as decimal hours in Vienna local time."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(_VIENNA_TZ)
    return dt.hour + dt.minute / 60 + dt.second / 3600


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

def _sig_round(x, sig=4):
    if x == 0:
        return 0
    mag = math.floor(math.log10(abs(x)))
    return round(x, max(0, sig - 1 - mag))


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

    strategy = (CONFIG.get("strategy") or "llm").lower()
    if strategy == "llm" and (not args.assets or not args.interval):
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    hyperliquid = HyperliquidAPI()
    agent = TradingAgent(hyperliquid=hyperliquid)
    risk_mgr = RiskManager()


    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = []  # For Sharpe: list of returns
    active_trades = []  # {'asset','is_long','amount','entry_price','tp_oid','sl_oid','exit_plan'}
    orb_state: dict = {}  # Per-asset ORB day state for SP500 session
    orb_phase_tracker: dict = {}  # asset → last phase seen (lightweight, runs every loop)
    recent_events = deque(maxlen=200)
    diary_path = str(_ROOT / "data" / "diary.jsonl")
    initial_account_value = None
    prev_positions_count = None
    prev_account_value = None
    prev_asset_prices = {}
    last_state = {}
    model_usage = {"sonnet": 0, "skipped": 0}
    last_sonnet_time: datetime | None = None
    last_sonnet_prices: dict = {}
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
        logging.info(msg)

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        nonlocal invocation_count, initial_account_value, prev_positions_count, prev_account_value, prev_asset_prices, last_state, model_usage, last_sonnet_time, last_sonnet_prices

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
                                if _k in _entry:
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
                            "orb_tp1": _entry.get("orb_tp1"),
                            "orb_tp2": _entry.get("orb_tp2"),
                            "orb_or_range": _entry.get("orb_or_range"),
                            "orb_trail_active": _entry.get("orb_trail_active", False),
                            "orb_trail_max": _entry.get("orb_trail_max"),
                        })
                        add_event(
                            f"Restored active_trade from diary: {_asset} "
                            f"tp_oid={_entry.get('tp_oid')} sl_oid={_entry.get('sl_oid')}"
                        )
        except Exception as _restore_err:
            add_event(f"active_trades restore failed (non-fatal): {_restore_err}")

        while True:
            invocation_count += 1
            _loop_now = datetime.now(timezone.utc)
            _session = _get_session(_loop_now)
            print_loop_header(_session["name"], _session["interval_secs"], args.assets)
            minutes_since_start = (_loop_now - start_time).total_seconds() / 60

            macro_ctx = await get_macro_context()
            if not _session["active"] and not macro_ctx.get("block_new_opens"):
                macro_ctx["block_new_opens"] = True
            macro_ctx["session"] = _session["name"]
            with open(_ROOT / "logs" / "llm_requests.log", "a", encoding="utf-8") as _mf:
                _mf.write(f"\n=== Macro context {datetime.now(timezone.utc).isoformat()} ===\n{json.dumps(macro_ctx)}\n")
            if macro_ctx["block_new_opens"]:
                logging.info("Session filter: block_new_opens=True (session=%s)", _session["name"])

            # Global account state
            try:
                state = await hyperliquid.get_user_state()
            except Exception as _state_err:
                logging.error("[LLM] get_user_state failed, retrying next cycle: %s", _state_err)
                await asyncio.sleep(60)
                continue
            total_value = state.get('total_value') or state['balance'] + sum(p.get('pnl', 0) for p in state['positions'])
            sharpe = calculate_sharpe(trade_log)

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            # Fetch current prices once for all managed assets; reused throughout the cycle
            asset_prices = {}
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                except Exception as e:
                    add_event(f"Price fetch error {asset}: {e}")

            positions = []
            for pos_wrap in state['positions']:
                pos = pos_wrap
                coin = _resolve_coin(pos.get('coin') or '')
                current_px = asset_prices.get(coin) if coin else None
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
                    for line in lines[-5:]:
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
                    open_orders_struct.append({
                        "coin": coin,
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('limitPx') or o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "is_trigger": o.get('isTrigger', False),
                        "order_type": o.get('orderType'),
                    })
            except Exception:
                open_orders = []

            recent_fills_struct = []
            fills = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=10)
                for f_entry in fills[-5:]:
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

            # Sync entry_price from live exchange position — corrects limit orders once filled.
            try:
                live_entry = {p["symbol"]: p["entry_price"] for p in positions if p.get("entry_price")}
                for tr in active_trades:
                    ep = live_entry.get(tr.get("asset"))
                    if ep and ep != tr.get("entry_price"):
                        tr["entry_price"] = ep
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

                orders_by_coin: dict = {}
                for _o in (open_orders or []):
                    orders_by_coin.setdefault(_o.get('coin', ''), []).append(_o)

                def _orders_for(a: str) -> list:
                    return orders_by_coin.get(a.split(":", 1)[1] if ":" in a else a, [])

                trigger_oids = {o.get('oid') for o in (open_orders or []) if _is_trigger(o)}
                for tr in active_trades:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions:
                        continue
                    # Cancel resting entry limits (non-trigger) for assets with an open position
                    orphaned = [o for o in _orders_for(asset) if not _is_trigger(o)]
                    if orphaned:
                        await hyperliquid.cancel_limit_orders(asset, cached_orders=open_orders)
                        add_event(f"Cancelled {len(orphaned)} orphaned entry limit(s) for {asset}")
                    # Collect all trigger orders for this asset
                    asset_triggers = [o for o in _orders_for(asset) if _is_trigger(o)]
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
                            o for o in _orders_for(asset)
                            if _is_trigger(o) and _trigger_price_matches(o, tr['tp_price'])
                        ))
                    )
                    if not tp_on_book:
                        tp_price = tr.get('tp_price')
                        cur_px = next((p.get('current_price') for p in positions if p.get('symbol') == asset), None)
                        tp_pct = risk_mgr.mandatory_tp_pct / 100.0
                        # Validate direction: long TP must be above current price; short TP below
                        if tp_price and cur_px:
                            wrong_side = (tr['is_long'] and tp_price <= cur_px) or (not tr['is_long'] and tp_price >= cur_px)
                            if wrong_side:
                                add_event(f"TP {tp_price} for {asset} wrong side of current {cur_px} — recomputing")
                                tp_price = None
                                tr['tp_price'] = None
                        if not tp_price:
                            ref = cur_px or tr.get('entry_price')
                            if ref:
                                tp_price = _sig_round(ref * (1 + tp_pct) if tr['is_long'] else ref * (1 - tp_pct))
                                add_event(f"No valid TP for {asset} — fallback at {tp_price}")
                        if tp_price:
                            try:
                                tp_order = await hyperliquid.place_take_profit(
                                    asset, tr['is_long'], tr['amount'], tp_price
                                )
                                tp_oids = hyperliquid.extract_oids(tp_order)
                                if tp_oids:
                                    tr['tp_oid'] = tp_oids[0]
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
                                else:
                                    add_event(f"TP order rejected for {asset} at {tp_price}: {tp_order}")
                                    tr['tp_price'] = None
                                    tr['tp_oid'] = None
                                    with open(diary_path, "a") as _dtf:
                                        _dtf.write(json.dumps({
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "asset": asset, "action": "tpsl_update",
                                            "tp_price": None, "tp_oid": None,
                                        }) + "\n")
                            except Exception as e:
                                add_event(f"Failed to re-place TP for {asset}: {e}")
                    # Re-place SL if missing — mandatory fallback if no sl_price stored
                    sl_on_book = (
                        (tr.get('sl_oid') and tr['sl_oid'] in trigger_oids)
                        or (tr.get('sl_price') and any(
                            o for o in _orders_for(asset)
                            if _is_trigger(o) and _trigger_price_matches(o, tr['sl_price'])
                        ))
                    )
                    if not sl_on_book:
                        sl_price = tr.get('sl_price')
                        cur_px = next((p.get('current_price') for p in positions if p.get('symbol') == asset), None)
                        sl_pct = risk_mgr.mandatory_sl_pct / 100.0
                        # Validate direction: short SL must be above current price; long SL below
                        if sl_price and cur_px:
                            wrong_side = (tr['is_long'] and sl_price >= cur_px) or (not tr['is_long'] and sl_price <= cur_px)
                            if wrong_side:
                                add_event(f"SL {sl_price} for {asset} wrong side of current {cur_px} — recomputing")
                                sl_price = None
                                tr['sl_price'] = None
                        if not sl_price:
                            ref = cur_px or tr.get('entry_price')
                            if ref:
                                sl_price = _sig_round(ref * (1 - sl_pct) if tr['is_long'] else ref * (1 + sl_pct))
                                add_event(f"No valid SL for {asset} — fallback at {sl_price}")
                        if sl_price:
                            try:
                                sl_order = await hyperliquid.place_stop_loss(asset, tr['is_long'], tr['amount'], sl_price)
                                sl_oids = hyperliquid.extract_oids(sl_order)
                                if sl_oids:
                                    tr['sl_oid'] = sl_oids[0]
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
                                else:
                                    add_event(f"SL order rejected for {asset} at {sl_price}: {sl_order}")
                                    tr['sl_price'] = None
                                    tr['sl_oid'] = None
                                    with open(diary_path, "a") as _dsf:
                                        _dsf.write(json.dumps({
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "asset": asset, "action": "tpsl_update",
                                            "sl_price": None, "sl_oid": None,
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
                    untracked_limits = [o for o in _orders_for(asset) if not o.get('isTrigger')]
                    if untracked_limits:
                        await hyperliquid.cancel_limit_orders(asset, cached_orders=open_orders)
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
                        o for o in _orders_for(asset)
                        if isinstance(o.get('orderType'), dict) and 'trigger' in o.get('orderType', {})
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

            _now_orb = datetime.now(timezone.utc).astimezone(_VIENNA_TZ)
            _hf_orb = _now_orb.hour + _now_orb.minute / 60 + _now_orb.second / 3600
            _today_orb = _now_orb.date()

            # ORB time stop: force-close any open SP500 position at 20:00 CET (14:00 ET).
            # Mechanical — does not rely on LLM.
            if _hf_orb >= _SP500_END:
                for tr in active_trades[:]:
                    if tr.get('asset') not in _SP500_ASSETS:
                        continue
                    try:
                        await hyperliquid.place_close_order(tr['asset'])
                        add_event(f"ORB time stop: closed {tr['asset']} @ {asset_prices.get(tr['asset'])}")
                        with open(diary_path, 'a') as _df:
                            _df.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": tr['asset'], "action": "reconcile_close",
                                "reason": "orb_time_stop",
                            }) + "\n")
                        active_trades.remove(tr)
                    except Exception as _tse:
                        add_event(f"ORB time stop failed {tr.get('asset')}: {_tse}")

            # ORB tp2_swing: autonomous tick-level management, no LLM involvement.
            # Strategy: hold full position until TP2 (+1×range) is reached.
            # At TP2: cancel TP order, move SL to TP1 level, activate swing trail.
            # Swing trail: each tick fetch recent 5m candles, trail SL to the most
            # recent confirmed 5m swing low (long) or swing high (short).
            try:
                for tr in active_trades:
                    _tr_asset = tr.get('asset')
                    if _tr_asset not in _SP500_ASSETS:
                        continue
                    if not tr.get('orb_tp2') or not tr.get('orb_or_range'):
                        continue
                    _tr_px = asset_prices.get(_tr_asset)
                    if not _tr_px:
                        continue
                    _tr_long = tr['is_long']
                    _tr_tp2  = tr['orb_tp2']
                    _tr_tp1  = tr.get('orb_tp1') or _tr_tp2  # SL floor when trail activates
                    _tr_range = tr['orb_or_range']

                    if not tr.get('orb_trail_active'):
                        # Wait for TP2 hit before activating trail
                        _tp2_hit = (_tr_long and _tr_px >= _tr_tp2) or (not _tr_long and _tr_px <= _tr_tp2)
                        if not _tp2_hit:
                            continue
                        # Move SL to TP1 level (locks in +0.5×range profit floor)
                        _floor_sl = _sig_round(_tr_tp1)
                        try:
                            if tr.get('sl_oid'):
                                await hyperliquid.cancel_order(_tr_asset, tr['sl_oid'])
                            _sl_order = await hyperliquid.place_stop_loss(_tr_asset, _tr_long, tr['amount'], _floor_sl)
                            _sl_oids = hyperliquid.extract_oids(_sl_order)
                            tr['sl_oid'] = _sl_oids[0] if _sl_oids else None
                            tr['sl_price'] = _floor_sl
                            add_event(f"ORB trail: TP2 hit {_tr_asset} @ {_tr_px} — SL → {_floor_sl} (TP1 floor), swing trail active")
                        except Exception as _fe:
                            add_event(f"ORB trail: SL to TP1 floor failed {_tr_asset}: {_fe}")
                        # Cancel fixed TP2 order — swing trail takes over
                        if tr.get('tp_oid'):
                            try:
                                await hyperliquid.cancel_order(_tr_asset, tr['tp_oid'])
                            except Exception:
                                pass
                        tr['tp_oid'] = None
                        tr['tp_price'] = None
                        tr['orb_trail_active'] = True
                        tr['orb_trail_max'] = _tr_px
                        with open(diary_path, 'a') as _df:
                            _df.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": _tr_asset,
                                "action": "tpsl_update",
                                "sl_price": _floor_sl,
                                "sl_oid": tr.get('sl_oid'),
                                "tp_price": None,
                                "tp_oid": None,
                                "orb_trail_active": True,
                                "orb_trail_max": _tr_px,
                            }) + "\n")
                    else:
                        # Swing trail: find the highest confirmed swing low (long) in recent 5m bars
                        try:
                            _candles = await hyperliquid.get_candles(_tr_asset, "5m", 20)
                        except Exception:
                            _candles = []
                        _new_sl = tr.get('sl_price') or (_tr_tp1 if _tr_long else _tr_tp2)
                        if _candles and len(_candles) >= 4:
                            # Exclude last bar (potentially still forming)
                            _closed = _candles[:-1]
                            for _i in range(1, len(_closed) - 1):
                                _b0, _b1, _b2 = _closed[_i - 1], _closed[_i], _closed[_i + 1]
                                if _tr_long:
                                    if _b1["low"] < _b0["low"] and _b1["low"] < _b2["low"]:
                                        if _b1["low"] > _new_sl:
                                            _new_sl = _b1["low"]
                                else:
                                    if _b1["high"] > _b0["high"] and _b1["high"] > _b2["high"]:
                                        if _b1["high"] < _new_sl:
                                            _new_sl = _b1["high"]
                        _new_sl = _sig_round(_new_sl)
                        _cur_sl = tr.get('sl_price') or 0
                        _moved = (_tr_long and _new_sl > _cur_sl) or (not _tr_long and _new_sl < _cur_sl)
                        if not _moved:
                            continue
                        try:
                            if tr.get('sl_oid'):
                                await hyperliquid.cancel_order(_tr_asset, tr['sl_oid'])
                            _sl_order = await hyperliquid.place_stop_loss(_tr_asset, _tr_long, tr['amount'], _new_sl)
                            _sl_oids = hyperliquid.extract_oids(_sl_order)
                            tr['sl_oid'] = _sl_oids[0] if _sl_oids else None
                            tr['sl_price'] = _new_sl
                            add_event(f"ORB trail: swing SL → {_new_sl} {_tr_asset}")
                            with open(diary_path, 'a') as _df:
                                _df.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": _tr_asset,
                                    "action": "tpsl_update",
                                    "sl_price": _new_sl,
                                    "sl_oid": tr.get('sl_oid'),
                                }) + "\n")
                        except Exception as _te:
                            add_event(f"ORB trail: SL update failed {_tr_asset}: {_te}")
            except Exception as _trail_err:
                add_event(f"ORB trail error (non-fatal): {_trail_err}")

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "available_margin": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
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
                    if tr.get('asset') in assets_with_positions
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
            }

            # Compact position summary table
            _pos_rows = []
            for _tr in active_trades:
                _a = _tr.get("asset")
                _cur = asset_prices.get(_a)
                _entry = _tr.get("entry_price")
                _is_long = _tr.get("is_long")
                _amt = _tr.get("amount")
                _pnl = None
                if _cur and _entry and _amt:
                    _pnl = (_cur - _entry) * _amt * (1 if _is_long else -1)
                _sl = _tr.get("sl_price")
                _tp = _tr.get("tp_price")
                _pos_rows.append({
                    "asset": _a,
                    "side": "long" if _is_long else "short",
                    "amount": _amt,
                    "entry": _entry,
                    "current": _cur,
                    "pnl_usd": _pnl,
                    "sl_dist_pts": abs(_cur - _sl) if _cur and _sl else None,
                    "tp_dist_pts": abs(_cur - _tp) if _cur and _tp else None,
                })
            if _pos_rows:
                print_position_table(_pos_rows)

            # Escalation logic: check triggers before building expensive prompt.
            # Sonnet fires when ANY of:
            #   1. First run (no prior state)
            #   2. Any asset price moved > SONNET_PRICE_MOVE_PCT
            #   3. Any open position is within SONNET_TPSL_PROXIMITY_PCT of its TP or SL
            #   4. Open positions exist and Sonnet hasn't run in SONNET_HEALTH_CHECK_MINUTES
            price_move_threshold = float(CONFIG.get("sonnet_price_move_pct") or 0.5) / 100.0
            if _session["active"]:
                price_move_threshold = min(price_move_threshold, _session["move_pct"])
            tpsl_proximity = float(CONFIG.get("sonnet_tpsl_proximity_pct") or 1.25) / 100.0
            health_check_minutes = int(float(CONFIG.get("sonnet_health_check_minutes") or 60))

            first_run = prev_positions_count is None

            price_moved = any(
                last_sonnet_prices.get(a) is not None
                and last_sonnet_prices[a] != 0
                and asset_prices.get(a) is not None
                and abs(asset_prices[a] - last_sonnet_prices[a]) / last_sonnet_prices[a] > price_move_threshold
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
                last_sonnet_time is None
                or (datetime.now(timezone.utc) - last_sonnet_time).total_seconds() / 60 >= health_check_minutes
            )

            # ORB triggers — all time-based, run every loop, no LLM needed.

            # Compute current ORB phase from clock alone (mirrors the in-LLM logic).
            def _orb_phase_now(hf):
                if hf < _SP500_PRESESSION:      return "pre_session"
                elif hf < 15.5:                 return "pre_open"
                elif hf < 15.75:                return "or_formation"
                elif hf < 17.5:                 return "breakout_watch"
                elif hf < _SP500_END:           return "in_session"
                else:                           return "time_stop"

            # Phase-transition trigger: fire LLM when clock enters or_formation (15:30)
            # or breakout_watch (15:45). This guarantees ORH/ORL are computed from real
            # OR candles before any breakout check can succeed.
            orb_phase_trigger = False
            for _sp_asset in _SP500_ASSETS:
                if _sp_asset not in args.assets:
                    continue
                _orb_cached = orb_state.get(_sp_asset)
                if _orb_cached and _orb_cached.get("trade_taken"):
                    continue
                _phase_now = _orb_phase_now(_hf_orb)
                _phase_key = f"{_sp_asset}:{_today_orb}"
                _phase_prev = orb_phase_tracker.get(_phase_key)
                if _phase_now != _phase_prev:
                    orb_phase_tracker[_phase_key] = _phase_now
                    if _phase_now in ("or_formation", "breakout_watch"):
                        orb_phase_trigger = True
                        add_event(f"ORB phase → {_phase_now} for {_sp_asset} — firing LLM")

            # Retest-entry ORB: don't trigger LLM on initial breakout — wait for price to
            # touch back to ORH (long) or ORL (short), then enter at that level.
            # Phase 1 — breakout detected: record direction in breakout_pending, no LLM.
            # Phase 2 — retest: price touches back to ORH/ORL → trigger LLM.
            # Failure: price fully retraces through OR → clear breakout_pending.
            orb_breakout = False
            for _sp_asset in _SP500_ASSETS:
                if _sp_asset not in args.assets:
                    continue
                if _today_orb.weekday() >= 5:  # Saturday=5, Sunday=6 — no ORB on weekends
                    continue
                _orb_cached = orb_state.get(_sp_asset)
                if not _orb_cached or _orb_cached.get("trade_taken"):
                    continue
                _orh_c = _orb_cached.get("orh")
                _orl_c = _orb_cached.get("orl")
                _bias_c = _orb_cached.get("bias")
                _fund_long_c = _orb_cached.get("funding_ok_long", True)
                _fund_short_c = _orb_cached.get("funding_ok_short", True)
                if _orh_c is None or _orl_c is None:
                    continue
                if not (15.75 <= _hf_orb < 17.5):
                    continue
                _sp_price = asset_prices.get(_sp_asset)
                if _sp_price is None:
                    continue

                _bp = _orb_cached.get("breakout_pending")

                if _bp is None:
                    # Phase 1: watch for initial breakout.
                    # Use the last COMPLETED candle close, not live mid-price, to avoid
                    # setting breakout_pending on wick spikes that never close above ORH.
                    _last_close = _sp_price  # fallback to live price if fetch fails
                    try:
                        _bo_candles = await hyperliquid.get_candles(_sp_asset, "5m", 3)
                        if len(_bo_candles) >= 2:
                            _last_close = _bo_candles[-2]["close"]  # -1 is still forming
                    except Exception:
                        pass
                    if _last_close > _orh_c and _bias_c == "bull" and _fund_long_c:
                        _orb_cached["breakout_pending"] = "long"
                        add_event(f"ORB {_sp_asset}: long breakout (close {_last_close:.2f} > ORH {_orh_c:.2f}) — waiting for retest")
                    elif _last_close < _orl_c and _bias_c == "bear" and _fund_short_c:
                        _orb_cached["breakout_pending"] = "short"
                        add_event(f"ORB {_sp_asset}: short breakout (close {_last_close:.2f} < ORL {_orl_c:.2f}) — waiting for retest")
                elif _bp == "long":
                    if _sp_price < _orl_c:
                        _orb_cached["breakout_pending"] = None
                        add_event(f"ORB {_sp_asset}: long breakout failed (price {_sp_price:.2f} < ORL {_orl_c:.2f}) — cleared")
                    elif _sp_price <= _orh_c:
                        # Price touched back to ORH — retest confirmed, trigger LLM
                        orb_breakout = True
                        add_event(f"ORB {_sp_asset}: long retest at {_sp_price:.2f} (ORH {_orh_c:.2f}) — triggering LLM")
                        break
                    # else: still above ORH, waiting
                else:  # _bp == "short"
                    if _sp_price > _orh_c:
                        _orb_cached["breakout_pending"] = None
                        add_event(f"ORB {_sp_asset}: short breakout failed (price {_sp_price:.2f} > ORH {_orh_c:.2f}) — cleared")
                    elif _sp_price >= _orl_c:
                        # Price touched back to ORL — retest confirmed, trigger LLM
                        orb_breakout = True
                        add_event(f"ORB {_sp_asset}: short retest at {_sp_price:.2f} (ORL {_orl_c:.2f}) — triggering LLM")
                        break
                    # else: still below ORL, waiting

            # Mechanical ORB entry — fires when retest is confirmed and all prerequisites pass.
            # Bypasses LLM entirely: every prerequisite is already computed by the state machine.
            if orb_breakout and not macro_ctx.get("block_new_opens"):
                for _sp_asset in _SP500_ASSETS:
                    if _sp_asset not in args.assets:
                        continue
                    _orb_e = orb_state.get(_sp_asset, {})
                    _bias_e   = _orb_e.get("bias", "neutral")
                    _long_e   = _orb_e.get("breakout_pending") == "long" or _orb_e.get("breakout_long")
                    _short_e  = _orb_e.get("breakout_pending") == "short" or _orb_e.get("breakout_short")
                    _taken_e  = _orb_e.get("trade_taken", False)
                    _fund_ok  = _orb_e.get("funding_ok_long") if _long_e else _orb_e.get("funding_ok_short")
                    _phase_e  = _orb_e.get("phase", "")
                    _ready = (
                        _phase_e == "breakout_watch"
                        and not _taken_e
                        and _bias_e != "neutral"
                        and ((_long_e and _bias_e == "bull") or (_short_e and _bias_e == "bear"))
                        and _fund_ok
                        and not any(t.get("asset") == _sp_asset for t in active_trades)
                    )
                    if not _ready:
                        continue
                    _is_buy_e = _long_e
                    _entry_px  = asset_prices.get(_sp_asset, 0)
                    _tp_price  = _orb_e.get("tp2_long" if _is_buy_e else "tp2_short")
                    _sl_price  = _orb_e.get("sl_long"  if _is_buy_e else "sl_short")
                    if not _tp_price or not _sl_price or not _entry_px:
                        continue
                    # Risk-size the position using the same path as LLM trades
                    _syn_trade = {
                        "asset": _sp_asset, "action": "buy" if _is_buy_e else "sell",
                        "allocation_usd": 100.0,  # placeholder; validate_trade resizes
                        "tp_price": _tp_price, "sl_price": _sl_price, "current_price": _entry_px,
                    }
                    _managed_state = {**state, "positions": [p for p in state["positions"] if p.get("coin") in set(args.assets)]}
                    _allowed, _reason, _syn_trade = risk_mgr.validate_trade(_syn_trade, _managed_state, initial_account_value or 0, open_orders_struct)
                    if not _allowed:
                        add_event(f"ORB mechanical entry blocked by risk: {_reason}")
                        continue
                    _amount = float(_syn_trade.get("allocation_usd", 100.0)) / _entry_px
                    _amount = hyperliquid.round_size(_sp_asset, _amount)
                    if _amount <= 0:
                        continue
                    try:
                        _order = await (hyperliquid.place_buy_order(_sp_asset, _amount)
                                        if _is_buy_e else hyperliquid.place_sell_order(_sp_asset, _amount))
                        _tp_oid = _sl_oid_e = None
                        await asyncio.sleep(0.5)
                        _tp_order = await hyperliquid.place_take_profit(_sp_asset, _is_buy_e, _amount, _tp_price)
                        _tp_oids  = hyperliquid.extract_oids(_tp_order)
                        _tp_oid   = _tp_oids[0] if _tp_oids else None
                        _sl_order_e = await hyperliquid.place_stop_loss(_sp_asset, _is_buy_e, _amount, _sl_price)
                        _sl_oids_e  = hyperliquid.extract_oids(_sl_order_e)
                        _sl_oid_e   = _sl_oids_e[0] if _sl_oids_e else None
                        _orb_tp1_e = _orb_e.get("tp1_long" if _is_buy_e else "tp1_short")
                        _orb_tp2_e = _tp_price
                        active_trades.append({
                            "asset": _sp_asset, "is_long": _is_buy_e, "amount": _amount,
                            "entry_price": _entry_px, "tp_oid": _tp_oid, "sl_oid": _sl_oid_e,
                            "tp_price": _tp_price, "sl_price": _sl_price,
                            "exit_plan": "System ORB tp2_swing", "opened_at": datetime.now().isoformat(),
                            "orb_tp1": _orb_tp1_e, "orb_tp2": _orb_tp2_e,
                            "orb_or_range": _orb_e.get("or_range"),
                            "orb_trail_active": False, "orb_trail_max": None,
                        })
                        orb_state[_sp_asset]["trade_taken"] = True
                        orb_breakout = False  # handled — don't pass to LLM
                        side_str = "LONG" if _is_buy_e else "SHORT"
                        add_event(f"ORB mechanical entry: {side_str} {_sp_asset} {_amount:.4f} @ {_entry_px}  TP {_tp_price}  SL {_sl_price}")
                        print_decision(_sp_asset, "buy" if _is_buy_e else "sell",
                                       f"Mechanical ORB entry: bias={_bias_e} phase={_phase_e}", 5,
                                       extra=f"{_amount:.4f} @ {_entry_px}  TP {_tp_price}  SL {_sl_price}")
                        with open(diary_path, "a") as _f:
                            _f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": _sp_asset,
                                "action": "buy" if _is_buy_e else "sell",
                                "order_type": "market",
                                "amount": _amount,
                                "entry_price": _entry_px,
                                "tp_price": _tp_price, "tp_oid": _tp_oid,
                                "sl_price": _sl_price, "sl_oid": _sl_oid_e,
                                "orb_tp1": _orb_tp1_e, "orb_tp2": _orb_tp2_e,
                                "orb_or_range": _orb_e.get("or_range"),
                                "rationale": "mechanical ORB entry",
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                            }) + "\n")
                    except Exception as _me:
                        add_event(f"ORB mechanical entry failed {_sp_asset}: {_me}")

            # During or_formation + breakout_watch, poll every 60s to catch phase transitions promptly.
            _orb_watching = any(
                _sp_asset in args.assets
                and 15.5 <= _hf_orb < 17.5
                and not orb_state.get(_sp_asset, {}).get("trade_taken")
                for _sp_asset in _SP500_ASSETS
            )
            _effective_interval_secs = 60 if _orb_watching else _session["interval_secs"]

            use_sonnet = first_run or price_moved or tpsl_near or health_check_due or orb_breakout or orb_phase_trigger

            if use_sonnet and macro_ctx.get("block_new_opens") and not active_trades and not health_check_due and not price_moved and not orb_breakout and not orb_phase_trigger:
                use_sonnet = False
                add_event("Skipping LLM: block_new_opens=True and no open positions to manage")

            if not use_sonnet:
                model_usage["skipped"] += 1
                add_event(
                    f"Skipping LLM (no trigger): price_moved={price_moved}, "
                    f"tpsl_near={tpsl_near}, health_check_due={health_check_due} — auto-hold"
                )
                prev_positions_count = len(active_trades)
                prev_account_value = account_value
                prev_asset_prices = dict(asset_prices)
                await asyncio.sleep(_effective_interval_secs)
                continue

            model_usage["sonnet"] += 1
            last_sonnet_time = datetime.now(timezone.utc)
            _prev_sonnet_prices = dict(last_sonnet_prices)  # snapshot before overwrite
            last_sonnet_prices = dict(asset_prices)

            # Asset triage: on health-check or first-run, evaluate everything.
            # Otherwise only evaluate assets that moved or have an open position.
            assets_with_position = {tr['asset'] for tr in active_trades}
            _sp500_window_active = _SP500_PRESESSION <= _hf_orb < _SP500_END

            def _sp500_needed(asset):
                """SP500 only needs an LLM call inside its 15:00–20:00 window or when a position is open."""
                if asset not in _SP500_ASSETS:
                    return True
                return _sp500_window_active or asset in assets_with_position

            if first_run or health_check_due:
                assets_to_evaluate = [a for a in args.assets if _sp500_needed(a)]
                assets_auto_hold = [a for a in args.assets if a not in assets_to_evaluate]
            else:
                assets_to_evaluate = []
                assets_auto_hold = []
                for _a in args.assets:
                    if not _sp500_needed(_a):
                        assets_auto_hold.append(_a)
                        continue
                    _cur = asset_prices.get(_a)
                    _last = _prev_sonnet_prices.get(_a)
                    _moved = (
                        _cur is not None and _last is not None and _last != 0
                        and abs(_cur - _last) / _last > price_move_threshold
                    )
                    _orb_relevant = _a in _SP500_ASSETS and (orb_breakout or orb_phase_trigger)
                    if _a in assets_with_position or _moved or _orb_relevant:
                        assets_to_evaluate.append(_a)
                    else:
                        assets_auto_hold.append(_a)

            add_event(
                f"Sonnet triggered: first_run={first_run}, price_moved={price_moved}, "
                f"tpsl_near={tpsl_near}, health_check_due={health_check_due}, "
                f"orb_phase={orb_phase_trigger}, orb_breakout={orb_breakout} — "
                f"evaluating {len(assets_to_evaluate)}/{len(args.assets)} assets"
                + (f" | auto-hold (quiet): {assets_auto_hold}" if assets_auto_hold else "")
            )

            # Heavy: gather full market data (OI, funding, candles, indicators) only when LLM fires
            market_sections = []
            for asset in assets_to_evaluate:
                try:
                    current_price = asset_prices.get(asset)
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    # Fetch candles from Hyperliquid and compute indicators locally
                    candles_5m = await hyperliquid.get_candles(asset, "5m", 100)
                    intra = compute_all(candles_5m, current_price=current_price)

                    if asset in _SP500_ASSETS:
                        # --- ORB state machine for S&P 500 ---
                        _now_v = datetime.now(timezone.utc).astimezone(_VIENNA_TZ)
                        _today_v = _now_v.date()
                        _hf_v = _now_v.hour + _now_v.minute / 60 + _now_v.second / 3600

                        _orb = orb_state.get(asset)
                        if _orb is None or _orb.get("date") != _today_v:
                            _orb = {
                                "date": _today_v,
                                "bias": None,
                                "funding_ok_long": True,
                                "funding_ok_short": True,
                                "orh": None,
                                "orl": None,
                                "trade_taken": False,
                                "bias_evaluated": False,
                                "breakout_pending": None,  # "long"/"short" — waiting for retest
                            }
                            orb_state[asset] = _orb

                        # Phase
                        if _hf_v < _SP500_PRESESSION:
                            _orb_phase = "pre_session"
                        elif _hf_v < 15.5:
                            _orb_phase = "pre_open"
                        elif _hf_v < 15.75:
                            _orb_phase = "or_formation"
                        elif _hf_v < 17.5:
                            _orb_phase = "breakout_watch"
                        elif _hf_v < _SP500_END:
                            _orb_phase = "in_session"
                        else:
                            _orb_phase = "time_stop"

                        # One-time bias + funding evaluation at/after 15:00 CET
                        if not _orb["bias_evaluated"] and _hf_v >= _SP500_PRESESSION:
                            _candles_4h = await hyperliquid.get_candles(asset, "4h", 50)
                            if len(_candles_4h) >= 21:
                                _closes_4h = [c["close"] for c in _candles_4h]
                                _ema21 = ema(_closes_4h, 21)
                                _last_c4 = _closes_4h[-1]
                                _last_e = next((v for v in reversed(_ema21) if v is not None), None)
                                _prev_e = next((v for v in reversed(_ema21[:-1]) if v is not None), None)
                                if _last_e and _prev_e:
                                    _slope = (_last_e - _prev_e) / _prev_e
                                    _SLOPE_THRESH = 0.0002  # matches backtest default
                                    if _last_c4 > _last_e and _slope > _SLOPE_THRESH:
                                        _orb["bias"] = "bull"
                                    elif _last_c4 < _last_e and _slope < -_SLOPE_THRESH:
                                        _orb["bias"] = "bear"
                                    else:
                                        _orb["bias"] = "neutral"
                            _fund_rate = funding
                            if _fund_rate is not None:
                                _FUND_THRESH = 0.0003  # 0.03% per 8h
                                _orb["funding_ok_long"] = _fund_rate <= _FUND_THRESH
                                _orb["funding_ok_short"] = _fund_rate >= -_FUND_THRESH
                            _orb["bias_evaluated"] = True

                        # OR formation: derive ORH/ORL from 5m candles in [15:30, 15:45) CET
                        if _orb["orh"] is None and _hf_v >= 15.5:
                            _or_candles = [
                                c for c in candles_5m
                                if c.get("t") and 15.5 <= _vhour(c["t"]) < 15.75
                            ]
                            if _or_candles:
                                _orb["orh"] = max(c["high"] for c in _or_candles)
                                _orb["orl"] = min(c["low"]  for c in _or_candles)

                        # Retest detection: set breakout_long/short when price is at ORH/ORL
                        # after an initial breakout (breakout_pending set by the pre-check above).
                        _breakout_long = False
                        _breakout_short = False
                        if _orb_phase == "breakout_watch" and _orb["orh"] is not None and not _orb["trade_taken"]:
                            _bp_dir = _orb.get("breakout_pending")
                            _cp = current_price or 0
                            if _bp_dir == "long" and _cp <= _orh_v:
                                _breakout_long = True
                            elif _bp_dir == "short" and _cp >= _orl_v:
                                _breakout_short = True

                        # Pre-compute TP/SL levels (based on ORH/ORL as proxy for entry)
                        _orh_v = _orb["orh"]
                        _orl_v = _orb["orl"]
                        _or_range = (_orh_v - _orl_v) if (_orh_v is not None and _orl_v is not None) else None
                        _SL_BUF = 0.1
                        if _or_range:
                            _tp1_l = round(_orh_v + 0.5 * _or_range, 2)
                            _tp2_l = round(_orh_v + 1.0 * _or_range, 2)
                            _sl_l  = round(_orl_v - _SL_BUF * _or_range, 2)
                            _tp1_s = round(_orl_v - 0.5 * _or_range, 2)
                            _tp2_s = round(_orl_v - 1.0 * _or_range, 2)
                            _sl_s  = round(_orh_v + _SL_BUF * _or_range, 2)
                        else:
                            _tp1_l = _tp2_l = _sl_l = _tp1_s = _tp2_s = _sl_s = None

                        bias_section = {
                            "phase":             _orb_phase,
                            "bias":              _orb["bias"],
                            "funding_ok_long":   _orb["funding_ok_long"],
                            "funding_ok_short":  _orb["funding_ok_short"],
                            "orh":               round_or_none(_orh_v, 2),
                            "orl":               round_or_none(_orl_v, 2),
                            "or_range":          round_or_none(_or_range, 2),
                            "trade_taken_today": _orb["trade_taken"],
                            "breakout_long":     _breakout_long,
                            "breakout_short":    _breakout_short,
                            "tp1_long":          _tp1_l,
                            "tp2_long":          _tp2_l,
                            "sl_long":           _sl_l,
                            "tp1_short":         _tp1_s,
                            "tp2_short":         _tp2_s,
                            "sl_short":          _sl_s,
                        }
                        # Compute R:R for display — entry is at ORH/ORL (retest entry)
                        _rr_display = None
                        if _breakout_long and _tp2_l and _sl_l and _orh_v:
                            _rr_display = round((_tp2_l - _orh_v) / max(_orh_v - _sl_l, 0.01), 2)
                        elif _breakout_short and _tp2_s and _sl_s and _orl_v:
                            _rr_display = round((_orl_v - _tp2_s) / max(_sl_s - _orl_v, 0.01), 2)
                        print_orb_status(
                            phase=_orb_phase,
                            bias=_orb["bias"],
                            orh=_orh_v,
                            orl=_orl_v,
                            breakout_long=_breakout_long,
                            breakout_short=_breakout_short,
                            trade_taken=_orb["trade_taken"],
                            rr=_rr_display,
                            breakout_pending=_orb.get("breakout_pending"),
                        )

                    else:
                        candles_1h = await hyperliquid.get_candles(asset, "1h", 220)

                        # 1h bias: zz_structure per asset (dev varies); VP only for non-fib assets
                        dev_pct = _ZZ_DEV.get(asset, _ZZ_DEV_DEFAULT)
                        struct_1h = zz_structure(candles_1h[-200:], deviation_pct=dev_pct, current_price=current_price) if len(candles_1h) >= 5 else None
                        rvol_1h_val = latest(rvol(candles_1h, period=20))

                        swing_count_1h = struct_1h.get("swing_count", 0) if struct_1h else 0
                        trend_1h = struct_1h.get("trend") if struct_1h else None
                        if trend_1h == "HH_HL" and swing_count_1h >= 2:
                            bias_dir = "bull"
                        elif trend_1h == "LH_LL" and swing_count_1h >= 2:
                            bias_dir = "bear"
                        else:
                            bias_dir = None

                        _session_active = _get_session(datetime.now(timezone.utc))["active"]

                        if asset in _FIB_ASSETS:
                            sw_high = struct_1h.get("last_swing_high") if struct_1h else None
                            sw_low  = struct_1h.get("last_swing_low")  if struct_1h else None
                            sw_rng  = struct_1h.get("swing_range")      if struct_1h else None
                            if sw_high is not None and sw_low is not None and sw_rng and sw_rng > 0:
                                fib_entry_long  = round(sw_high - 0.745 * sw_rng, 4)
                                fib_entry_short = round(sw_low  + 0.745 * sw_rng, 4)
                                sl_long         = round(sw_low  - 0.05  * sw_rng, 4)
                                sl_short        = round(sw_high + 0.05  * sw_rng, 4)
                            else:
                                fib_entry_long = fib_entry_short = sl_long = sl_short = None
                            bias_section = {
                                "bias":            bias_dir,
                                "trend":           trend_1h,
                                "swing_count":     swing_count_1h,
                                "rvol_1h":         round_or_none(rvol_1h_val, 3),
                                "last_swing_high": round_or_none(sw_high, 4),
                                "last_swing_low":  round_or_none(sw_low,  4),
                                "swing_range":     round_or_none(sw_rng,  4),
                                "fib_entry_long":  fib_entry_long,
                                "fib_entry_short": fib_entry_short,
                                "sl_long":         sl_long,
                                "sl_short":        sl_short,
                                "tp1_long":        round_or_none(sw_high, 4),
                                "tp1_short":       round_or_none(sw_low,  4),
                                "tp2_long":        round_or_none(struct_1h.get("tp_speculative_long")  if struct_1h else None, 4),
                                "tp2_short":       round_or_none(struct_1h.get("tp_speculative_short") if struct_1h else None, 4),
                                "session_active":  _session_active,
                            }
                        else:
                            vp_1h = volume_profile(candles_1h[-20:]) if len(candles_1h) >= 20 else None
                            vah_1h = vp_1h.get("vah") if vp_1h else None
                            val_1h = vp_1h.get("val") if vp_1h else None
                            va_width_1h = (vah_1h - val_1h) if (vah_1h is not None and val_1h is not None) else None
                            bias_section = {
                                "bias":                   bias_dir,
                                "trend":                  trend_1h,
                                "swing_count":            swing_count_1h,
                                "val":                    round_or_none(val_1h, 4),
                                "vah":                    round_or_none(vah_1h, 4),
                                "va_width":               round_or_none(va_width_1h, 4),
                                "rvol_1h":                round_or_none(rvol_1h_val, 3),
                                "tp_speculative_long":    round_or_none(struct_1h.get("tp_speculative_long")  if struct_1h else None, 4),
                                "tp_speculative_short":   round_or_none(struct_1h.get("tp_speculative_short") if struct_1h else None, 4),
                            }

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-3:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    # OBV direction: True when OBV rose over the last 3 bars
                    obv_vals = last_n(intra.get("obv", []), 5)
                    obv_rising = len(obv_vals) >= 3 and obv_vals[-1] > obv_vals[-3]

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        "intraday": {
                            "rsi14": round_or_none(latest(intra.get("rsi14", [])), 2),
                            "rvol": round_or_none(latest(intra.get("rvol", [])), 3),
                            "obv_rising": obv_rising,
                        },
                        "bias_1h": bias_section,
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "recent_mid_prices": recent_mids,
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # Single LLM call with all assets
            _now = datetime.now(timezone.utc)
            _is_weekend = _now.weekday() >= 5
            _invocation = {
                "current_time": _now.isoformat(),
                "day_of_week": _now.strftime("%A"),
            }
            if _is_weekend:
                _invocation["weekend_liquidity_note"] = (
                    "CEX-linked markets (commodities, indices, equities) are closed today. "
                    "Hyperliquid candles for these assets may reflect near-zero real volume — "
                    "treat RSI/MACD/EMA signals with lower confidence and require stronger "
                    "confluence before opening new positions."
                )
            context_payload = OrderedDict([
                ("invocation", _invocation),
                ("account", dashboard),
                ("risk_limits", risk_mgr.get_risk_summary()),
                ("market_data", market_sections),
                ("instructions", {
                    "assets": assets_to_evaluate,
                    "requirement": "Decide actions for all assets and return a strict JSON object matching the schema."
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(assets_to_evaluate)} assets")
            with open(_ROOT / "logs" / "prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

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
                outputs = await agent.decide_trade(assets_to_evaluate, context, model=agent.model, macro_context=macro_ctx)
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
                    outputs = await agent.decide_trade(assets_to_evaluate, context_retry, model=agent.model, macro_context=macro_ctx)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

            if isinstance(outputs, dict) and outputs.get("trade_decisions"):
                outputs = update_and_check(outputs, active_trades)

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
                    "thesis_strength": d.get("thesis_strength"),
                })
            cycle_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": invocation_count,
                "reasoning": reasoning_text[:2000] if reasoning_text else "",
                "decisions": cycle_decisions,
                "account_value": round_or_none(account_value, 2),
                "available_margin": round_or_none(state['balance'], 2),
                "positions_count": len([p for p in state['positions'] if abs(float(p.get('szi') or 0)) > 0]),
            }
            try:
                with open(_ROOT / "data" / "decisions.jsonl", "a") as f:
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
                    thesis_strength = output.get("thesis_strength")
                    if action == "cancel_limits":
                        # For Fib-entry assets (ETH, SOL): protect a pending limit entry order.
                        # If there are resting limits on the book but no open position, the order
                        # is a valid unfilled entry — hold it regardless of thesis_strength.
                        # cancel_limits should only execute when there IS a position (orphan cleanup)
                        # or there are genuinely no orders to protect.
                        if asset in _FIB_ASSETS and asset not in assets_with_positions:
                            _pending = [o for o in (open_orders or [])
                                        if hyperliquid._coin_matches(o.get('coin', ''), asset)
                                        and not hyperliquid._is_trigger_order(o)]
                            if _pending:
                                add_event(
                                    f"CANCEL_LIMITS {asset} blocked — {len(_pending)} pending entry "
                                    f"limit(s) preserved (set-and-forget; no position open)"
                                )
                                print_decision(asset, "hold", rationale, thesis_strength,
                                               extra="set-and-forget: limit preserved")
                                continue
                        try:
                            result = await hyperliquid.cancel_limit_orders(asset, cached_orders=open_orders)
                            n = result.get("cancelled_count", 0)
                            print_decision(asset, "cancel_limits", rationale, thesis_strength, extra=f"{n} order(s) cancelled")
                            add_event(f"CANCEL_LIMITS {asset}: {n} order(s) cancelled")
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
                    elif action == "update_tpsl":
                        tr = next((t for t in active_trades if t.get("asset") == asset), None)
                        new_tp = output.get("tp_price")
                        new_sl = output.get("sl_price")
                        # close_fraction applies to TP size (e.g. 0.5 for TP1); SL always full position
                        cf = float(output.get("close_fraction") or 1.0)
                        cf = max(0.01, min(1.0, cf))
                        try:
                            if not tr:
                                # No tracked trade (e.g. after restart) — resolve from exchange.
                                # Execute the LLM's intent fully: cancel all existing trigger
                                # orders for this asset, then place the requested TP/SL.
                                state = await hyperliquid.get_user_state()
                                live_pos = next(
                                    (p for p in state.get("positions", [])
                                     if hyperliquid._coin_matches(p.get("coin", ""), asset)),
                                    None
                                )
                                if not live_pos or float(live_pos.get("szi", 0)) == 0:
                                    add_event(f"UPDATE_TPSL {asset}: no active position on exchange — skipping")
                                    continue
                                szi = float(live_pos["szi"])
                                is_long = szi > 0
                                amount = abs(szi)
                                open_orders = await hyperliquid.get_open_orders()
                                for o in open_orders:
                                    if not hyperliquid._coin_matches(o.get("coin", ""), asset):
                                        continue
                                    if not hyperliquid._is_trigger_order(o):
                                        continue
                                    oid = o.get("oid")
                                    if not oid:
                                        continue
                                    try:
                                        await hyperliquid.cancel_order(asset, oid)
                                    except Exception as cancel_err:
                                        add_event(f"UPDATE_TPSL {asset}: cancel oid {oid} failed — {cancel_err}")
                                tp_price_placed = None
                                sl_price_placed = None
                                tp_oid_placed = None
                                sl_oid_placed = None
                                if new_tp is not None:
                                    tp_amount = round(amount * cf, 8)
                                    tp_order = await hyperliquid.place_take_profit(asset, is_long, tp_amount, float(new_tp))
                                    tp_oids = hyperliquid.extract_oids(tp_order)
                                    tp_oid_placed = tp_oids[0] if tp_oids else None
                                    tp_price_placed = float(new_tp)
                                    add_event(f"UPDATE_TPSL {asset}: TP → {new_tp} (size {tp_amount}, no tracked trade)")
                                if new_sl is not None:
                                    sl_order = await hyperliquid.place_stop_loss(asset, is_long, amount, float(new_sl))
                                    sl_oids = hyperliquid.extract_oids(sl_order)
                                    sl_oid_placed = sl_oids[0] if sl_oids else None
                                    sl_price_placed = float(new_sl)
                                    add_event(f"UPDATE_TPSL {asset}: SL → {new_sl} (size {amount}, no tracked trade)")
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "tpsl_update",
                                        "tp_price": tp_price_placed,
                                        "tp_oid": tp_oid_placed,
                                        "sl_price": sl_price_placed,
                                        "sl_oid": sl_oid_placed,
                                        "rationale": rationale,
                                    }) + "\n")
                            else:
                                is_long = tr.get("is_long")
                                amount = tr.get("amount")
                                if new_tp is not None:
                                    if tr.get("tp_oid"):
                                        try:
                                            await hyperliquid.cancel_order(asset, tr["tp_oid"])
                                        except Exception as cancel_err:
                                            add_event(f"UPDATE_TPSL {asset}: cancel tp_oid {tr['tp_oid']} failed — {cancel_err}")
                                    tp_amount = round(amount * cf, 8)
                                    tp_order = await hyperliquid.place_take_profit(asset, is_long, tp_amount, float(new_tp))
                                    tp_oids = hyperliquid.extract_oids(tp_order)
                                    tr["tp_oid"] = tp_oids[0] if tp_oids else None
                                    tr["tp_price"] = float(new_tp)
                                    add_event(f"UPDATE_TPSL {asset}: TP → {new_tp} (size {tp_amount})")
                                if new_sl is not None:
                                    if tr.get("sl_oid"):
                                        try:
                                            await hyperliquid.cancel_order(asset, tr["sl_oid"])
                                        except Exception as cancel_err:
                                            add_event(f"UPDATE_TPSL {asset}: cancel sl_oid {tr['sl_oid']} failed — {cancel_err}")
                                    sl_order = await hyperliquid.place_stop_loss(asset, is_long, amount, float(new_sl))
                                    sl_oids = hyperliquid.extract_oids(sl_order)
                                    tr["sl_oid"] = sl_oids[0] if sl_oids else None
                                    tr["sl_price"] = float(new_sl)
                                    add_event(f"UPDATE_TPSL {asset}: SL → {new_sl} (size {amount})")
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "tpsl_update",
                                        "tp_price": tr.get("tp_price"),
                                        "tp_oid": tr.get("tp_oid"),
                                        "sl_price": tr.get("sl_price"),
                                        "sl_oid": tr.get("sl_oid"),
                                        "rationale": rationale,
                                    }) + "\n")
                        except Exception as tpsl_err:
                            add_event(f"UPDATE_TPSL error {asset}: {tpsl_err}")
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
                            close_fraction = float(output.get("close_fraction") or 1.0)
                            close_fraction = max(0.01, min(1.0, close_fraction))
                            amount = existing_tr['amount'] * close_fraction
                            alloc_usd = amount * current_price
                            if close_fraction < 1.0:
                                add_event(f"PARTIAL CLOSE {asset}: {close_fraction:.0%} of {existing_tr['amount']:.4f} = {amount:.4f}")
                            else:
                                add_event(f"CLOSE {asset}: using tracked size {amount:.4f} (LLM allocation ignored)")
                        else:
                            alloc_usd = float(output.get("allocation_usd", 0.0))
                            if alloc_usd <= 0:
                                add_event(f"Holding {asset}: zero/negative allocation")
                                continue

                            # --- RISK: Validate trade before execution ---
                            output["current_price"] = current_price
                            _managed_assets = set(args.assets)
                            _managed_state = {**state, "positions": [p for p in state["positions"] if p.get("coin") in _managed_assets]}
                            allowed, reason, output = risk_mgr.validate_trade(
                                output, _managed_state, initial_account_value or 0, open_orders_struct
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

                        # Cancel resting entry limit orders before opening a new one —
                        # EXCEPT for Fib assets placing a new limit: multiple concurrent
                        # entry limits are allowed there (each risk-validated individually).
                        # Market opens still cancel first (reconcile loop would orphan them anyway).
                        _skip_precancel = (asset in _FIB_ASSETS and order_type == "limit" and not existing_tr)
                        if not _skip_precancel:
                            try:
                                cancelled = await hyperliquid.cancel_limit_orders(asset, cached_orders=open_orders)
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
                                entry_oid = oids[0] if oids else None
                                tp_oid = oids[1] if len(oids) > 1 else None
                                sl_oid = oids[2] if len(oids) > 2 else None
                                if not entry_oid:
                                    add_event(f"ORDER REJECTED by exchange for {asset} — not tracking. Raw response: {order}")
                                    continue
                                add_event(f"LIMIT {action.upper()} {asset} {amount:.4f} @ ${limit_price} with bracket TP={tp_price_val} SL={sl_price_val}")
                            else:
                                # No TP/SL prices — plain limit
                                if is_buy:
                                    order = await hyperliquid.place_limit_buy(asset, amount, limit_price)
                                else:
                                    order = await hyperliquid.place_limit_sell(asset, amount, limit_price)
                                plain_oids = hyperliquid.extract_oids(order)
                                entry_oid = plain_oids[0] if plain_oids else None
                                if not entry_oid:
                                    add_event(f"ORDER REJECTED by exchange for {asset} — not tracking. Raw response: {order}")
                                    continue
                                add_event(f"LIMIT {action.upper()} {asset} amount {amount:.4f} at limit ${limit_price}")
                        else:
                            if existing_tr and close_fraction >= 1.0:
                                # Full close: reduceOnly market order, can never flip
                                order = await hyperliquid.place_close_order(asset)
                            elif is_buy:
                                order = await hyperliquid.place_buy_order(asset, amount)
                            else:
                                order = await hyperliquid.place_sell_order(asset, amount)

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
                        if existing_tr:
                            if close_fraction < 1.0:
                                # Partial close: reduce tracked amount, leave trade active
                                existing_tr['amount'] = round(existing_tr['amount'] - amount, 6)
                                add_event(f"Partial close recorded: {asset} remaining {existing_tr['amount']:.4f}")
                            else:
                                # Full close: remove from active_trades
                                for existing in active_trades[:]:
                                    if existing.get('asset') == asset:
                                        try:
                                            active_trades.remove(existing)
                                        except ValueError:
                                            pass
                        else:
                            # New open or flip: remove old entry, add new
                            for existing in active_trades[:]:
                                if existing.get('asset') == asset:
                                    try:
                                        active_trades.remove(existing)
                                    except ValueError:
                                        pass
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
                            _orb_tp1 = None
                            _orb_tp2 = None
                            _orb_or_range = None
                            if asset in _SP500_ASSETS and asset in orb_state:
                                _orb_s = orb_state[asset]
                                _orb_tp1 = _orb_s.get("tp1_long" if is_buy else "tp1_short")
                                _orb_tp2 = _orb_s.get("tp2_long" if is_buy else "tp2_short")
                                _orb_or_range = _orb_s.get("or_range")
                            active_trades.append({
                                "asset": asset,
                                "is_long": is_buy,
                                "amount": amount,
                                "entry_price": limit_price if order_type == "limit" and limit_price else current_price,
                                "tp_oid": tp_oid,
                                "sl_oid": sl_oid,
                                "tp_price": output.get("tp_price"),
                                "sl_price": output.get("sl_price"),
                                "exit_plan": output["exit_plan"],
                                "opened_at": datetime.now().isoformat(),
                                "orb_tp1": _orb_tp1,
                                "orb_tp2": _orb_tp2,
                                "orb_or_range": _orb_or_range,
                                "orb_trail_active": False,
                                "orb_trail_max": None,
                            })
                            if asset in _SP500_ASSETS and asset in orb_state:
                                orb_state[asset]["trade_taken"] = True
                        tp_str = f"TP {output.get('tp_price')}  " if output.get("tp_price") else ""
                        sl_str = f"SL {output.get('sl_price')}" if output.get("sl_price") else ""
                        _display_entry = limit_price if order_type == "limit" and limit_price else current_price
                        print_decision(asset, action, rationale, thesis_strength,
                                       extra=f"{amount:.4f} @ {_display_entry}  {tp_str}{sl_str}")
                        # Write to diary after confirming fills status
                        with open(diary_path, "a") as f:
                            if existing_tr:
                                diary_entry = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "reconcile_close",
                                    "reason": "llm_close",
                                    "order_result": str(order),
                                    "filled": filled,
                                }
                            else:
                                diary_entry = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": action,
                                    "order_type": order_type,
                                    "limit_price": limit_price,
                                    "allocation_usd": alloc_usd,
                                    "amount": amount,
                                    "entry_price": limit_price if order_type == "limit" and limit_price else current_price,
                                    "tp_price": output.get("tp_price"),
                                    "tp_oid": tp_oid,
                                    "sl_price": output.get("sl_price"),
                                    "sl_oid": sl_oid,
                                    "exit_plan": output.get("exit_plan", ""),
                                    "rationale": output.get("rationale", ""),
                                    "order_result": str(order),
                                    "opened_at": datetime.now(timezone.utc).isoformat(),
                                    "filled": filled,
                                    "orb_tp1": _orb_tp1,
                                    "orb_tp2": _orb_tp2,
                                    "orb_or_range": _orb_or_range,
                                }
                            f.write(json.dumps(diary_entry) + "\n")
                    else:  # hold
                        print_decision(asset, "hold", rationale, thesis_strength)
                        # ORB: after LLM holds at a retest, clear breakout_pending so the
                        # state machine requires a fresh breakout+retest before firing again.
                        if asset in _SP500_ASSETS and asset in orb_state:
                            _orb_s = orb_state[asset]
                            if _orb_s.get("breakout_pending") and not _orb_s.get("trade_taken"):
                                _orb_s["breakout_pending"] = None
                                add_event(f"ORB {asset}: LLM held at retest — breakout_pending cleared")
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

            await asyncio.sleep(_effective_interval_secs)

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
            path = request.query.get('path', str(_ROOT / "logs" / "llm_requests.log"))
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
                with open(_ROOT / "data" / "decisions.jsonl", "r") as f:
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
                if (fl.get("coin") or fl.get("asset") or "") == "BTC":
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
                if (fl.get("coin") or fl.get("asset") or "") == "BTC":
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

    async def handle_dashboard(request):
        dashboard = pathlib.Path(__file__).parent.parent / "dashboard.html"
        return web.FileResponse(dashboard)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/', handle_dashboard)
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/logs', handle_logs)
        app.router.add_get('/state', handle_state)
        app.router.add_get('/history', handle_history)

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        import base64

        _dash_user = os.getenv("DASHBOARD_USER")
        _dash_pass = os.getenv("DASHBOARD_PASSWORD")
        # Skip auth when running as a launcher subprocess (port 3001/3002 = internal only)
        _api_port = int(os.getenv("API_PORT") or os.getenv("APP_PORT") or "3000")
        _apply_auth = bool(_dash_user and _dash_pass and _api_port == 3000)

        @web.middleware
        async def auth(request, handler):
            if _apply_auth:
                header = request.headers.get("Authorization", "")
                if header.startswith("Basic "):
                    try:
                        decoded = base64.b64decode(header[6:]).decode()
                        u, p = decoded.split(":", 1)
                        if u == _dash_user and p == _dash_pass:
                            return await handler(request)
                    except Exception:
                        pass
                return web.Response(
                    status=401,
                    headers={"WWW-Authenticate": 'Basic realm="Sanket"'},
                    text="Unauthorized",
                )
            return await handler(request)

        @web.middleware
        async def cors(request, handler):
            resp = await handler(request)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        app = web.Application(middlewares=[auth, cors])
        await start_api(app)
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        strategy = (CFG.get("strategy") or "llm").lower()
        if strategy == "hybrid":
            from src.strategies.hybrid.hybrid_manager import HybridManager
            mgr = HybridManager(hyperliquid, risk_mgr)
            await mgr.run()
        elif strategy == "sol_momentum":
            from src.strategies.sol_momentum import SolMomentum
            from src.config_loader import CONFIG as _CFG
            risk_pct = float(_CFG.get("sol_momentum_risk_pct") or 0.015)
            dry_run  = _CFG.get("dry_run", False)
            strat = SolMomentum(hyperliquid, risk_pct=risk_pct, dry_run=dry_run)
            await strat.run()
        else:
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
