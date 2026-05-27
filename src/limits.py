"""Check live positions against configured risk limits."""

import asyncio, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.config_loader import CONFIG
from src.trading.hyperliquid_api import HyperliquidAPI


async def main():
    api = HyperliquidAPI()
    assets_raw = (CONFIG.get("assets") or "").split()
    dexs = list({a.split(":")[0] for a in assets_raw if ":" in a})
    if dexs:
        api.register_perp_dexs(dexs)

    state = await api.get_user_state()
    total_value = float(state.get("total_value", 0))
    balance     = float(state.get("balance", 0))
    positions   = state.get("positions", [])

    lim_pos_pct = float(CONFIG["max_position_pct"])
    lim_lev     = float(CONFIG["max_leverage"])
    lim_exp_pct = float(CONFIG["max_total_exposure_pct"])
    lim_max_pos = int(CONFIG["max_concurrent_positions"])

    print(f"Account total value: ${total_value:,.2f}  |  Balance: ${balance:,.2f}")
    print(f"Limits: pos<{lim_pos_pct}%  lev<{lim_lev}x  total_exp<{lim_exp_pct}%  max_positions={lim_max_pos}\n")

    open_pos = []
    for pos in positions:
        szi = float(pos.get("szi", 0))
        if szi == 0:
            continue
        coin     = pos.get("coin", "")
        notional = abs(float(pos.get("notional_entry") or 0))
        lev_raw  = pos.get("leverage")
        leverage = float(lev_raw.get("value", 0)) if isinstance(lev_raw, dict) else float(lev_raw or 0)
        pnl      = float(pos.get("pnl") or 0)
        pct      = (notional / total_value * 100) if total_value else 0

        flags = []
        if pct > lim_pos_pct:
            flags.append(f"SIZE {pct:.1f}% > {lim_pos_pct}%")
        if leverage > lim_lev:
            flags.append(f"LEV {leverage:.1f}x > {lim_lev}x")

        open_pos.append({"notional": notional, "flags": flags})
        tag = "BREACH" if flags else "ok"
        print(f"{coin:18s} | size={szi:+.4f} | notional=${notional:,.0f} ({pct:.1f}%) | lev={leverage:.1f}x | pnl={pnl:+.2f} | {tag} {', '.join(flags)}")

    total_exp_pct = sum(p["notional"] for p in open_pos) / total_value * 100 if total_value else 0
    print(f"\nTotal exposure: {total_exp_pct:.1f}% (limit {lim_exp_pct}%)")
    print(f"Open positions: {len(open_pos)} (limit {lim_max_pos})")

    breaches = [f for p in open_pos for f in p["flags"]]
    if total_exp_pct > lim_exp_pct:
        breaches.append(f"total exposure {total_exp_pct:.1f}% > {lim_exp_pct}%")
    if len(open_pos) > lim_max_pos:
        breaches.append(f"{len(open_pos)} open positions > {lim_max_pos} max")

    print("\n--- " + ("BREACHES:" if breaches else "All within limits") + " ---")
    for b in breaches:
        print(f"  !  {b}")


asyncio.run(main())
