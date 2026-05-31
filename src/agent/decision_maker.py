"""Decision-making agent that orchestrates LLM prompts and indicator lookups.

Uses the Anthropic Claude API directly for trade decisions.
"""

import asyncio
import json
import logging
import pathlib
from datetime import datetime

_LOG_PATH = pathlib.Path(__file__).parent.parent.parent / "logs" / "llm_requests.log"

import anthropic

from src.config_loader import CONFIG
from src.utils.display import print_claude_stats
from src.indicators.local_indicators import (
    compute_all, last_n, latest,
    ema as _ema, sma as _sma, rsi as _rsi, atr as _atr, rvol as _rvol,
    swing_structure as _swing_structure,
    volume_profile as _volume_profile,
)


class TradingAgent:
    """High-level trading agent that delegates reasoning to Claude."""

    def __init__(self, hyperliquid=None):
        self.model = CONFIG["llm_model"]
        self.client = anthropic.AsyncAnthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        self.sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-4-5-20251001"
        self.haiku_model = CONFIG.get("haiku_model") or "claude-haiku-4-5-20251001"
        self.max_tokens = int(CONFIG.get("max_tokens") or 4096)

    async def decide_trade(self, assets, context, model=None, macro_context=None):
        """Decide for multiple assets in one call."""
        return await self._decide(context, assets=assets, model=model, macro_context=macro_context)

    async def _decide(self, context, assets, model=None, macro_context=None):
        """Dispatch decision request to Claude and enforce output contract."""
        enable_tools = CONFIG.get("enable_tool_calling", False)

        tool_instruction = (
            "Tools\n"
            "- Use fetch_indicator (indicator: ema/sma/rsi/macd/bbands/atr/adx/obv/vwap/stoch_rsi/rvol/swing_structure/volume_profile/all, asset, interval: 5m/4h, optional period) when an extra datapoint sharpens your thesis. Summarize findings in rationale — never paste raw output into JSON.\n\n"
        ) if enable_tools else ""

        system_prompt = (
            "You are a senior quantitative trader managing perpetual futures on Hyperliquid, optimizing risk-adjusted returns under real execution, margin, and funding constraints.\n"
            "You receive market + account context including per-asset intraday (5m) and 1h bias data, active trades with exit plans, recent history, and hard-enforced risk limits.\n\n"
            "Always use the 'current time' to evaluate cooldown expirations and timed exit plans. "
            "If 'is_weekend' is true or day_of_week is Saturday/Sunday: CEX-linked markets (commodities, indices, equities) are closed — candles may reflect near-zero volume, making indicators unreliable. Require significantly stronger confluence before opening new positions in such assets.\n\n"
            "Goal: decisive, first-principles decisions per asset — minimize churn, capture edge, control downside.\n\n"
            "Core policy\n"
            "1) Respect prior plans: Do NOT close or flip early unless the explicit invalidation in exit_plan has occurred (or a stronger one has).\n"
            "2) Hysteresis: To flip direction, require BOTH (a) 1h market structure shifted to opposite trend (HH_HL → LH_LL or vice versa) AND (b) intraday BoS in new direction with RVOL ≥ 1.5. Otherwise hold or update_tpsl.\n"
            "3) Cooldown: After any direction change, impose at least 3 bars before another. Encode in exit_plan (e.g. \"cooldown_bars:3 until 2026-06-01T10:00Z\") and honor it on future cycles.\n"
            "4) Funding is a tilt, not a trigger: Do not flip solely due to funding unless it meaningfully exceeds expected edge (>~0.25×ATR over your holding horizon).\n"
            "5) Prefer adjustments over flips: If thesis weakens but is not invalidated — tighten stop (update_tpsl), trail TP, or take partial profits (buy/sell with close_fraction < 1.0). RSI extremes alone are not reversals. Flip only on hard invalidation + fresh confluence.\n\n"
            "Thesis strength (required every cycle, every asset)\n"
            "Rate the conviction behind the current thesis as an integer 1–5:\n"
            "  5 = Strong trend + full confluence (4h + 5m aligned, all signals green)\n"
            "  4 = Good setup, minor mixed signals (e.g. RSI neutral)\n"
            "  3 = Neutral / no clear edge — do not open new positions at this level or below\n"
            "  2 = Thesis weakening: structure eroding, signals diverging\n"
            "  1 = Thesis broken: invalidation triggered, structure reversed\n"
            "Rules enforced in code: thesis_strength == 1 + open position → immediate close (hold forbidden). "
            "thesis_strength <= 2 for 3+ consecutive cycles + open position → exit regardless of P&L.\n\n"
            "Core Entry Logic — SOL\n"
            "1h ZigZag (dev=3%) structure + Fib 0.745 retracement entry:\n"
            "  Prerequisites (all required):\n"
            "    - bias_1h.bias = 'bull' (long) or 'bear' (short)\n"
            "    - bias_1h.swing_count ≥ 2\n"
            "    - bias_1h.rvol_1h ≥ 1.2 (1h relative volume — no session filter for SOL)\n"
            "    If any is missing: no new opens, hold only.\n"
            "  5m entry — touch of Fib 0.745 retracement:\n"
            "    Long:  5m bar.low  ≤ bias_1h.fib_entry_long  (74.5% retracement from swing high)\n"
            "    Short: 5m bar.high ≥ bias_1h.fib_entry_short (74.5% retracement from swing low)\n"
            "    Touch alone is sufficient — no confirmation bar required.\n"
            "  SL: bias_1h.sl_long (long) or bias_1h.sl_short (short) — 1.05× fib extension\n"
            "  TP1 (50% of position): bias_1h.tp1_long or bias_1h.tp1_short (swing extreme)\n"
            "    When TP1 hit: close half, move SL to entry (breakeven)\n"
            "  TP2 (remaining 50%): bias_1h.tp2_long or bias_1h.tp2_short (127.2% extension)\n"
            "    Set tp_price = TP2. Encode TP1 and partial-exit plan in exit_plan.\n"
            "  R:R check: (TP1 − entry) ÷ (entry − SL) ≥ 1.5. If below, skip.\n"
            "  Exit: bias_1h.bias flips → close position.\n"
            "  Minimum thesis_strength to open: 4.\n\n"
            "Core Entry Logic — ETH\n"
            "1h ZigZag (dev=2%) structure + Fib 0.745 retracement entry:\n"
            "  Prerequisites (all required):\n"
            "    - bias_1h.bias = 'bull' (long) or 'bear' (short)\n"
            "    - bias_1h.swing_count ≥ 2\n"
            "    - bias_1h.session_active = true (London 08:30–11:30 or NY 16:00–20:00 Vienna)\n"
            "    Note: no RVOL filter for ETH.\n"
            "    If any prerequisite missing: no new opens, hold only.\n"
            "  5m entry — touch of Fib 0.745 retracement:\n"
            "    Long:  5m bar.low  ≤ bias_1h.fib_entry_long\n"
            "    Short: 5m bar.high ≥ bias_1h.fib_entry_short\n"
            "    Touch alone is sufficient — no confirmation bar required.\n"
            "  SL: bias_1h.sl_long (long) or bias_1h.sl_short (short)\n"
            "  TP (full position): bias_1h.tp2_long or bias_1h.tp2_short — 127.2% extension, no partial exit.\n"
            "  R:R check: (TP2 − entry) ÷ (entry − SL) ≥ 1.5. If below, skip.\n"
            "  Exit: bias_1h.bias flips → close position.\n"
            "  Limit order management (SET & FORGET): once an ETH entry limit is on the book,\n"
            "    NEVER issue cancel_limits unless bias_1h.bias has flipped direction since the order\n"
            "    was placed. Low thesis_strength, quiet sessions, or RSI extremes are NOT reasons\n"
            "    to cancel. Use hold until price touches the Fib level or bias flips.\n"
            "  Minimum thesis_strength to open: 4.\n\n"
            "Core Entry Logic — BTC and other crypto perps\n"
            "1h ZigZag market structure + 5m Value Area Bounce:\n"
            "  1h bias (all required before considering any entry):\n"
            "    - bias_1h.bias = 'bull' (long) or 'bear' (short)\n"
            "    - bias_1h.swing_count ≥ 2\n"
            "    - bias_1h.va_width > 0 (valid volume profile)\n"
            "    If bias_1h.bias is null or swing_count < 2: no new opens, hold only.\n"
            "  5m entry — Value Area Bounce:\n"
            "    Long:  5m bar.low ≤ bias_1h.val + 0.30×va_width  AND  bar.close > bias_1h.val\n"
            "    Short: 5m bar.high ≥ bias_1h.vah − 0.30×va_width  AND  bar.close < bias_1h.vah\n"
            "    Volume filter: intraday.rvol ≥ 1.2 on the signal bar. If below, skip.\n"
            "    Confirmation: next 5m bar must close above bias_1h.val (long) or below bias_1h.vah (short).\n"
            "  SL: bias_1h.val − 0.15×va_width (long) or bias_1h.vah + 0.15×va_width (short)\n"
            "  TP1 (50%): bias_1h.vah (long) or bias_1h.val (short) — close half, move SL to breakeven\n"
            "  TP2 (50%): bias_1h.tp_speculative_long or bias_1h.tp_speculative_short (127.2% extension)\n"
            "    Set tp_price = TP2. Encode TP1 and partial-exit plan in exit_plan.\n"
            "  R:R check: (TP1 − entry) ÷ (entry − SL) ≥ 1.5. If below, do not open.\n"
            "  Exit: bias_1h.bias flips → close position.\n"
            "  Minimum thesis_strength to open: 4.\n\n"
            "Core Entry Logic — Gold (xyz:GOLD)\n"
            "Use Range Breakout only — do NOT apply the crypto Momentum Breakout rules to Gold:\n"
            "  4h setup (both required): BBands squeeze (band width at its narrowest in the last 8 bars) AND ADX rising (ADX now > ADX 5 bars ago)\n"
            "  5m entry: close breaks above 20-bar range high + 0.3×ATR14 (long) or below 20-bar range low − 0.3×ATR14 (short)\n"
            "  TP: 2.0×ATR14 from entry. SL: 0.5×ATR14 from entry (R:R = 4:1)\n"
            "  DXY rising (dxy_rising=true in macro context) → reduce long Gold exposure by ~30%, favor short setups\n"
            "  Weekend: Gold is a CEX-linked market — no new opens Saturday/Sunday\n"
            "  Minimum thesis_strength to open: 4\n\n"
            "Core Entry Logic — S&P 500 Perp (xyz:SP500)\n"
            "Opening Range Breakout (ORB) — mechanical, time-boxed US session strategy.\n"
            "The bias_1h field contains ORB state computed by the system. Use it directly — do not override with other indicators.\n\n"
            "  Phase meanings (bias_1h.phase):\n"
            "    pre_session / pre_open — before 15:30 CET. No action; hold only.\n"
            "    or_formation (15:30–15:45 CET) — Opening Range forming. No action.\n"
            "    breakout_watch (15:45–17:30 CET) — Watch for first breakout.\n"
            "    in_session (17:30–20:00 CET) — Window closed; manage open trade only.\n"
            "    time_stop (≥20:00 CET) — Close any open SP500 position immediately.\n\n"
            "  Entry rules (breakout_watch phase ONLY):\n"
            "    Prerequisites (ALL required to open):\n"
            "      - bias_1h.phase = 'breakout_watch'\n"
            "      - bias_1h.trade_taken_today = false\n"
            "      - bias_1h.breakout_long = true  (long) OR bias_1h.breakout_short = true  (short)\n"
            "        (set when price retests ORH after long breakout, or ORL after short breakout)\n"
            "      - bias_1h.bias ≠ 'neutral'\n"
            "      - bias_1h.funding_ok_long = true (for longs) OR bias_1h.funding_ok_short = true (for shorts)\n"
            "      - block_new_opens = false in macro context (FOMC/high-impact events suppress entry)\n"
            "    If all met → action=buy (long) or action=sell (short), order_type='market'.\n"
            "    Entry price is bias_1h.orh (long) or bias_1h.orl (short) — the retest level.\n"
            "    Only ONE trade per day — system sets trade_taken_today after entry.\n\n"
            "  TP/SL: use pre-computed levels from bias_1h (based on Opening Range size).\n"
            "    Long:  tp_price = bias_1h.tp2_long,  sl_price = bias_1h.sl_long\n"
            "    Short: tp_price = bias_1h.tp2_short, sl_price = bias_1h.sl_short\n"
            "    Exit management: fully system-managed after entry — do NOT issue partial closes, update_tpsl, or SL moves for SP500 positions. The system trails the SL automatically after TP2 is reached.\n"
            "  R:R: ORB R:R is ~0.9 by construction (TP2=+1×range, SL=−1.1×range from entry). Do NOT apply the 1.5 R:R gate here — enter whenever all prerequisites are met.\n\n"
            "  Time stop (time_stop phase): if open SP500 position exists → close immediately (action=sell or buy, close_fraction=1.0).\n"
            "  Do NOT apply any crypto-style (ZZ/Fib/VA) rules to xyz:SP500.\n"
            "  Weekend: SP500 is a CEX-linked market — no new opens Saturday/Sunday.\n"
            "  Minimum thesis_strength to open: 4.\n\n"
            "CRITICAL — what 'hold' does\n"
            "- action=hold places ZERO new orders. TP/SL levels in rationale have no effect on the exchange.\n"
            "- To move TP/SL: use update_tpsl — the only way to change protective orders on the exchange.\n"
            "- To protect an unprotected position: use update_tpsl, not close + re-open.\n\n"
            "Open order review (mandatory every cycle)\n"
            "- For every asset with open orders, decide explicitly in your rationale.\n"
            "- Entry limits (is_trigger=false): thesis holds → hold (no duplicate); want better price → buy/sell with order_type=limit (existing orders auto-cancelled first).\n"
            "  cancel_limits requires explicit invalidation: name the specific level or condition that negated the setup (e.g. 'price closed below key support', 'structure broken on 1h'). Low thesis_strength alone is NOT sufficient — use hold if the setup is merely quiet or uncertain.\n"
            "- TP/SL (is_trigger=true): if levels appear misplaced, use update_tpsl to correct them.\n\n"
            "Decision discipline\n"
            "- Choose one per asset: buy / sell / hold / cancel_limits / update_tpsl.\n"
            "- allocation_usd: the system overrides this with fixed-risk sizing (target ~$50 risk per trade based on your SL distance). Set it to any non-zero value as a signal to trade — the actual notional is computed as target_risk_usd / sl_distance_pct. Your sl_price placement is what drives position size.\n"
            "- order_type: \"market\" (default) or \"limit\". Limit requires limit_price; market sets it null.\n"
            "- TP/SL sanity: BUY → tp_price > current_price, sl_price < current_price. SELL → tp_price < current_price, sl_price > current_price. Use null if levels can't be set. Mandatory SL auto-applied on buy/sell opens if not set.\n"
            "- exit_plan: at least one explicit invalidation trigger + any cooldown guidance.\n"
            "- Leverage: system enforces a hard cap. Treat allocation_usd as notional exposure consistent with available margin.\n\n"
            + tool_instruction
            + "Reasoning: assess Structure (1h trend, swing_count, VA levels), Momentum (5m RVOL, RSI slope), Volatility (ATR), Positioning (funding, OI). Favor 1h+5m alignment.\n\n"
            "Output contract\n"
            "- Return ONLY a strict JSON object with one key: \"trade_decisions\" (array ordered to match assets list).\n"
            "- Each item: asset, action, allocation_usd, order_type, limit_price, tp_price, sl_price, exit_plan, rationale, close_fraction, thesis_strength.\n"
            "  • thesis_strength: integer 1–5, required for every item every cycle.\n"
            "  • close_fraction: 0.01–1.0. For buy/sell close actions: fraction of position to exit. For update_tpsl: fraction of position the TP order covers (e.g. 0.5 for TP1 at 50%); SL always covers full position. Defaults to 1.0.\n"
            "  • cancel_limits: allocation_usd=0, order_type=\"market\", limit_price=null, tp_price=null, sl_price=null.\n"
            "  • update_tpsl: allocation_usd=0, order_type=\"market\", limit_price=null. null tp/sl = keep existing. close_fraction sets TP size.\n"
            "- No Markdown, no code fences, no extra properties.\n"
        )

        tools = [{
            "name": "fetch_indicator",
            "description": (
                "Fetch technical indicators computed locally from Hyperliquid candle data. "
                "Works for ALL Hyperliquid perp markets including crypto (BTC, ETH, SOL), "
                "commodities (OIL, GOLD, SILVER), indices (SPX), and more. "
                "Available indicators: ema, sma, rsi, macd, bbands, atr, adx, obv, vwap, stoch_rsi, all. "
                "Returns the latest values and recent series."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "enum": ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "obv", "vwap", "stoch_rsi", "rvol", "swing_structure", "volume_profile", "all"],
                    },
                    "asset": {
                        "type": "string",
                        "description": "Hyperliquid asset symbol, e.g. BTC, ETH, OIL, GOLD, SPX",
                    },
                    "interval": {
                        "type": "string",
                        "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                    },
                    "period": {
                        "type": "integer",
                        "description": "Indicator period (default varies by indicator)",
                    },
                },
                "required": ["indicator", "asset", "interval"],
            },
        }]

        user_content = context
        if macro_context:
            fg = macro_context.get("fear_greed", 50)
            fg_label = (
                "extreme fear"  if fg < 25 else
                "fear"          if fg < 45 else
                "neutral"       if fg < 56 else
                "greed"         if fg < 76 else
                "extreme greed"
            )
            _session_name = macro_context.get("session", "unknown").upper()
            macro_section = (
                f"Macro context (current cycle):\n"
                f"- Trading session: {_session_name}\n"
                f"- Fear & Greed Index: {fg} ({fg_label})\n"
                f"- DXY trend: {'rising — reduce long allocation on crypto by ~30%' if macro_context.get('dxy_rising') else 'neutral/falling'}\n"
                f"- High-impact macro event within 60 min: {macro_context.get('high_impact_event_imminent', False)}\n"
                f"- Minimum thesis_strength to open new positions this cycle: {macro_context.get('min_thesis_strength_to_open', 3)}\n"
                f"- New opens blocked this cycle: {macro_context.get('block_new_opens', False)}\n\n"
                "Apply these constraints strictly. If block_new_opens is true, action must be hold "
                "or update_tpsl for all assets — no buy or sell opens regardless of signal strength.\n\n"
            )
            user_content = macro_section + context

        messages = [{"role": "user", "content": user_content}]

        effective_model = model or self.model
        thinking_enabled = CONFIG.get("thinking_enabled")
        thinking_budget = int(CONFIG.get("thinking_budget_tokens") or 10000)

        async def _call_claude(msgs, use_tools=True):
            kwargs = {
                "model": effective_model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": msgs,
            }
            if use_tools and enable_tools:
                kwargs["tools"] = tools
            if thinking_enabled:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                kwargs["max_tokens"] = max(self.max_tokens, 16000)

            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {effective_model}\n")
                f.write(f"Messages count: {len(msgs)}\n")
                last = msgs[-1]
                f.write(f"Last message role: {last.get('role')}\n")
                f.write(f"Last message content (truncated): {str(last.get('content', ''))[:500]}\n")

                response = await self.client.messages.create(**kwargs)
                u = response.usage
                cache_hit = getattr(u, "cache_read_input_tokens", 0) or 0
                cache_create = getattr(u, "cache_creation_input_tokens", 0) or 0
                print_claude_stats(
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cache_create=cache_create,
                    cache_read=cache_hit,
                    stop_reason=response.stop_reason,
                )
                if response.stop_reason == "max_tokens":
                    logging.warning(
                        "Response truncated at max_tokens=%d (used %d output tokens) — increase MAX_TOKENS if JSON is cut off",
                        kwargs["max_tokens"], u.output_tokens,
                    )
                f.write(f"Response stop_reason: {response.stop_reason}\n")
                f.write(f"Usage: input={u.input_tokens}, output={u.output_tokens}, cache_create={cache_create}, cache_read={cache_hit}\n")

            return response

        async def _handle_tool_call(tool_name, tool_input):
            if tool_name != "fetch_indicator":
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            try:
                asset     = tool_input["asset"]
                interval  = tool_input["interval"]
                indicator = tool_input["indicator"]
                period    = tool_input.get("period")

                candles = await self.hyperliquid.get_candles(asset, interval, 100)

                if indicator == "all":
                    all_indicators = compute_all(candles)
                    result = {}
                    for k, v in all_indicators.items():
                        if not isinstance(v, list):
                            result[k] = v
                        else:
                            result[k] = {"latest": latest(v), "series": last_n(v, 10)}
                elif indicator == "macd":
                    ai = compute_all(candles)
                    result = {
                        "macd":      {"latest": latest(ai.get("macd", [])),           "series": last_n(ai.get("macd", []), 10)},
                        "signal":    {"latest": latest(ai.get("macd_signal", [])),    "series": last_n(ai.get("macd_signal", []), 10)},
                        "histogram": {"latest": latest(ai.get("macd_histogram", [])), "series": last_n(ai.get("macd_histogram", []), 10)},
                    }
                elif indicator == "bbands":
                    ai = compute_all(candles)
                    result = {
                        "upper":  {"latest": latest(ai.get("bbands_upper", [])),  "series": last_n(ai.get("bbands_upper", []), 10)},
                        "middle": {"latest": latest(ai.get("bbands_middle", [])), "series": last_n(ai.get("bbands_middle", []), 10)},
                        "lower":  {"latest": latest(ai.get("bbands_lower", [])),  "series": last_n(ai.get("bbands_lower", []), 10)},
                    }
                elif indicator == "ema":
                    p = period or 20
                    series = _ema([c["close"] for c in candles], p)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": p}
                elif indicator == "sma":
                    p = period or 20
                    series = _sma([c["close"] for c in candles], p)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": p}
                elif indicator == "rsi":
                    p = period or 14
                    series = _rsi(candles, p)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": p}
                elif indicator == "atr":
                    p = period or 14
                    series = _atr(candles, p)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": p}
                elif indicator == "rvol":
                    p = period or 20
                    series = _rvol(candles, p)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": p}
                elif indicator == "swing_structure":
                    current_px = candles[-1]["close"] if candles else None
                    result = _swing_structure(candles, current_price=current_px) or {"error": "insufficient data"}
                elif indicator == "volume_profile":
                    result = _volume_profile(candles) or {"error": "insufficient data"}
                else:
                    # adx, obv, vwap, stoch_rsi
                    ai = compute_all(candles)
                    key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap", "stoch_rsi": "stoch_rsi"}
                    mapped = key_map.get(indicator, indicator)
                    series = ai.get(mapped, [])
                    result = {
                        "latest": latest(series) if isinstance(series, list) else series,
                        "series": last_n(series, 10) if isinstance(series, list) else series,
                    }

                return json.dumps(result, default=str)
            except Exception as ex:
                logging.error("Tool call error: %s", ex)
                return json.dumps({"error": str(ex)})

        async def _sanitize_output(raw_content: str, assets_list):
            try:
                response = await self.client.messages.create(
                    model=self.sanitize_model,
                    max_tokens=max(self.max_tokens, 4096),
                    system=(
                        "You are a strict JSON normalizer. Return ONLY a JSON object with one key: "
                        "\"trade_decisions\" (array). "
                        "Each trade_decisions item must have: asset, action (buy/sell/hold), "
                        "allocation_usd (number), order_type (\"market\" or \"limit\"), "
                        "limit_price (number or null), tp_price (number or null), sl_price (number or null), "
                        "exit_plan (string), rationale (string), thesis_strength (integer 1-5), "
                        "close_fraction (number 0.01-1.0). "
                        f"Valid assets: {json.dumps(list(assets_list))}. "
                        "If input is wrapped in markdown or has prose, extract just the JSON. Do not add fields."
                    ),
                    messages=[{"role": "user", "content": raw_content}],
                )
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                if not content.strip().startswith("{"):
                    brace_pos = content.find("{")
                    if brace_pos >= 0:
                        content = content[brace_pos:]
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "trade_decisions" in parsed:
                    return parsed
                return {"reasoning": "", "trade_decisions": []}
            except Exception as se:
                logging.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        def _fallback_hold(reason: str):
            return {
                "reasoning": reason,
                "trade_decisions": [{
                    "asset": a,
                    "action": "hold",
                    "allocation_usd": 0.0,
                    "order_type": "market",
                    "limit_price": None,
                    "tp_price": None,
                    "sl_price": None,
                    "exit_plan": "",
                    "rationale": reason,
                    "close_fraction": 1.0,
                    "thesis_strength": 3,
                } for a in assets]
            }

        # Main loop: up to 6 iterations to handle tool calls
        for iteration in range(6):
            try:
                response = await _call_claude(messages)
            except anthropic.APIError as e:
                logging.error("Claude API error: %s", e)
                with open(_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"API Error: {e}\n")
                break

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks     = [b for b in response.content if b.type == "text"]

            if tool_use_blocks and response.stop_reason == "tool_use":
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif block.type == "thinking":
                        assistant_content.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.signature,
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute all tool calls in parallel
                results = await asyncio.gather(
                    *[_handle_tool_call(b.name, b.input) for b in tool_use_blocks]
                )
                tool_results = [
                    {"type": "tool_result", "tool_use_id": b.id, "content": result_str}
                    for b, result_str in zip(tool_use_blocks, results)
                ]
                messages.append({"role": "user", "content": tool_results})
                continue

            # No tool calls — parse the text response as JSON
            raw_text = "".join(b.text for b in text_blocks)

            if not raw_text.strip():
                logging.error("Empty response from Claude")
                break

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
            if not cleaned.startswith("{"):
                brace_pos = cleaned.find("{")
                if brace_pos >= 0:
                    cleaned = cleaned[brace_pos:]

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    logging.error("Expected dict, got: %s; attempting sanitize", type(parsed))
                    return await _sanitize_output(raw_text, assets)

                reasoning_text = parsed.get("reasoning", "") or ""
                decisions = parsed.get("trade_decisions")

                if isinstance(decisions, list):
                    normalized = []
                    for item in decisions:
                        if isinstance(item, dict):
                            item.setdefault("allocation_usd", 0.0)
                            item.setdefault("order_type", "market")
                            item.setdefault("limit_price", None)
                            item.setdefault("tp_price", None)
                            item.setdefault("sl_price", None)
                            item.setdefault("close_fraction", 1.0)
                            item.setdefault("exit_plan", "")
                            item.setdefault("rationale", "")
                            ts = item.get("thesis_strength")
                            if not isinstance(ts, int) or not (1 <= ts <= 5):
                                item["thesis_strength"] = 3
                            normalized.append(item)
                    return {"reasoning": reasoning_text, "trade_decisions": normalized}

                logging.error("trade_decisions missing or invalid; attempting sanitize")
                sanitized = await _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, raw_text[:200])
                sanitized = await _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return _fallback_hold("Parse error")

        return _fallback_hold("tool loop cap")
