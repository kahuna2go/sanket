"""Decision-making agent that orchestrates LLM prompts and indicator lookups.

Uses the Anthropic Claude API directly for trade decisions.
"""

import asyncio
import anthropic
from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
import json
import logging
from datetime import datetime


class TradingAgent:
    """High-level trading agent that delegates reasoning to Claude."""

    def __init__(self, hyperliquid=None):
        self.model = CONFIG["llm_model"]
        self.client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        self.hyperliquid = hyperliquid
        self.sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-4-5-20251001"
        self.haiku_model = CONFIG.get("haiku_model") or "claude-haiku-4-5-20251001"
        self.max_tokens = int(CONFIG.get("max_tokens") or 4096)

    def decide_trade(self, assets, context, model=None, macro_context=None):
        """Decide for multiple assets in one call."""
        return self._decide(context, assets=assets, model=model, macro_context=macro_context)

    def _decide(self, context, assets, model=None, macro_context=None):
        """Dispatch decision request to Claude and enforce output contract."""
        system_prompt = (
            "You are a senior quantitative trader managing perpetual futures on Hyperliquid, optimizing risk-adjusted returns under real execution, margin, and funding constraints.\n"
            "You receive market + account context for: "
            f"assets = {json.dumps(list(assets))}, "
            "per-asset intraday (5m) and 4h metrics, active trades with exit plans, recent history, and hard-enforced risk limits.\n\n"
            "Always use the 'current time' to evaluate cooldown expirations and timed exit plans. "
            "If 'is_weekend' is true or day_of_week is Saturday/Sunday: CEX-linked markets (commodities, indices, equities) are closed — candles may reflect near-zero volume, making indicators unreliable. Require significantly stronger confluence before opening new positions in such assets.\n\n"
            "Goal: decisive, first-principles decisions per asset — minimize churn, capture edge, control downside.\n\n"
            "Core policy\n"
            "1) Respect prior plans: Do NOT close or flip early unless the explicit invalidation in exit_plan has occurred (or a stronger one has).\n"
            "2) Hysteresis: To flip direction, require BOTH (a) 4h structure supporting the new direction (EMA cross, MACD regime) AND (b) intraday confirmation (break >~0.5×ATR + momentum alignment). Otherwise hold or update_tpsl.\n"
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
            "Core Entry Logic — crypto assets (BTC, ETH, SOL, and other pure-crypto perps)\n"
            "Use Momentum Breakout only:\n"
            "  4h bias (all required): EMA20 > EMA50, MACD histogram positive, ADX > 25\n"
            "  5m entry (all required): close breaks above previous bar high, OBV rising, RSI 50–70\n"
            "  TP: 1.5× ATR14 above entry. SL: 0.75× ATR14 below entry (R:R = 2:1)\n"
            "  No new opens when ADX < 25 on any asset.\n\n"
            "CRITICAL — what 'hold' does\n"
            "- action=hold places ZERO new orders. TP/SL levels in rationale have no effect on the exchange.\n"
            "- To move TP/SL: use update_tpsl — the only way to change protective orders on the exchange.\n"
            "- To protect an unprotected position: use update_tpsl, not close + re-open.\n\n"
            "Open order review (mandatory every cycle)\n"
            "- For every asset with open orders, decide explicitly in your rationale.\n"
            "- Entry limits (is_trigger=false): thesis holds → hold (no duplicate); invalidated → cancel_limits; want better price → buy/sell with order_type=limit (existing orders auto-cancelled first).\n"
            "- TP/SL (is_trigger=true): if levels appear misplaced, use update_tpsl to correct them.\n\n"
            "Decision discipline\n"
            "- Choose one per asset: buy / sell / hold / cancel_limits / update_tpsl.\n"
            "- allocation_usd controls position size (system caps it per risk limits).\n"
            "- order_type: \"market\" (default) or \"limit\". Limit requires limit_price; market sets it null.\n"
            "- TP/SL sanity: BUY → tp_price > current_price, sl_price < current_price. SELL → tp_price < current_price, sl_price > current_price. Use null if levels can't be set. Mandatory SL auto-applied on buy/sell opens if not set.\n"
            "- exit_plan: at least one explicit invalidation trigger + any cooldown guidance.\n"
            "- Leverage: system enforces a hard cap. Treat allocation_usd as notional exposure consistent with available margin.\n\n"
            "Tools\n"
            "- Use fetch_indicator (indicator: ema/sma/rsi/macd/bbands/atr/adx/obv/vwap/stoch_rsi/all, asset, interval: 5m/4h, optional period) when an extra datapoint sharpens your thesis. Summarize findings in rationale — never paste raw output into JSON.\n\n"
            "Reasoning: assess Structure (trend, EMA slopes/cross), Momentum (MACD, RSI slope), Volatility (ATR), Positioning (funding, OI). Favor 4h+5m alignment.\n\n"
            "Output contract\n"
            "- Return ONLY a strict JSON object with one key: \"trade_decisions\" (array ordered to match assets list).\n"
            "- Each item: asset, action, allocation_usd, order_type, limit_price, tp_price, sl_price, exit_plan, rationale, close_fraction, thesis_strength.\n"
            "  • thesis_strength: integer 1–5, required for every item every cycle.\n"
            "  • close_fraction: 0.01–1.0 for closing an existing position (1.0 = full, 0.5 = half). Ignored when opening.\n"
            "  • cancel_limits: allocation_usd=0, order_type=\"market\", limit_price=null, tp_price=null, sl_price=null.\n"
            "  • update_tpsl: allocation_usd=0, order_type=\"market\", limit_price=null. null tp/sl = keep existing.\n"
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
                        "enum": ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "obv", "vwap", "stoch_rsi", "all"],
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
            "cache_control": {"type": "ephemeral"},
        }]

        user_content = context
        if macro_context:
            fg = macro_context.get("fear_greed", 50)
            fg_label = (
                "extreme fear" if fg < 20 else
                "extreme greed" if fg > 80 else
                "neutral"
            )
            macro_section = (
                f"Macro context (current cycle):\n"
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

        def _log_request(model, messages_to_log):
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {model}\n")
                f.write(f"Messages count: {len(messages_to_log)}\n")
                # Log last message content (truncated)
                last = messages_to_log[-1]
                content_str = str(last.get("content", ""))[:500]
                f.write(f"Last message role: {last.get('role')}\n")
                f.write(f"Last message content (truncated): {content_str}\n")

        effective_model = model or self.model
        enable_tools = CONFIG.get("enable_tool_calling", False)

        def _call_claude(msgs, use_tools=True):
            """Make a Claude API call with optional tool use."""
            _log_request(effective_model, msgs)
            kwargs = {
                "model": effective_model,
                "max_tokens": self.max_tokens,
                "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                "messages": msgs,
            }
            if use_tools and enable_tools:
                kwargs["tools"] = tools
            if CONFIG.get("thinking_enabled"):
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(CONFIG.get("thinking_budget_tokens") or 10000),
                }
                # When thinking is enabled, max_tokens must be larger
                kwargs["max_tokens"] = max(self.max_tokens, 16000)

            response = self.client.messages.create(**kwargs)
            logging.info("Claude response: stop_reason=%s, usage=%s",
                        response.stop_reason, response.usage)
            if response.stop_reason == "max_tokens":
                logging.warning(
                    "Response truncated at max_tokens=%d (used %d output tokens) — increase MAX_TOKENS if JSON is cut off",
                    kwargs["max_tokens"], response.usage.output_tokens,
                )
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"Response stop_reason: {response.stop_reason}\n")
                f.write(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}\n")
            return response

        def _handle_tool_call(tool_name, tool_input):
            """Execute a tool call and return the result string."""
            if tool_name != "fetch_indicator":
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            try:
                asset = tool_input["asset"]
                interval = tool_input["interval"]
                indicator = tool_input["indicator"]

                # Fetch candles from Hyperliquid
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        candles = pool.submit(
                            asyncio.run,
                            self.hyperliquid.get_candles(asset, interval, 100)
                        ).result(timeout=30)
                else:
                    candles = asyncio.run(self.hyperliquid.get_candles(asset, interval, 100))

                all_indicators = compute_all(candles)

                if indicator == "all":
                    result = {k: {"latest": latest(v) if isinstance(v, list) else v,
                                  "series": last_n(v, 10) if isinstance(v, list) else v}
                              for k, v in all_indicators.items()}
                elif indicator == "macd":
                    result = {
                        "macd": {"latest": latest(all_indicators.get("macd", [])), "series": last_n(all_indicators.get("macd", []), 10)},
                        "signal": {"latest": latest(all_indicators.get("macd_signal", [])), "series": last_n(all_indicators.get("macd_signal", []), 10)},
                        "histogram": {"latest": latest(all_indicators.get("macd_histogram", [])), "series": last_n(all_indicators.get("macd_histogram", []), 10)},
                    }
                elif indicator == "bbands":
                    result = {
                        "upper": {"latest": latest(all_indicators.get("bbands_upper", [])), "series": last_n(all_indicators.get("bbands_upper", []), 10)},
                        "middle": {"latest": latest(all_indicators.get("bbands_middle", [])), "series": last_n(all_indicators.get("bbands_middle", []), 10)},
                        "lower": {"latest": latest(all_indicators.get("bbands_lower", [])), "series": last_n(all_indicators.get("bbands_lower", []), 10)},
                    }
                elif indicator in ("ema", "sma"):
                    period = tool_input.get("period", 20)
                    from src.indicators.local_indicators import ema as _ema, sma as _sma
                    closes = [c["close"] for c in candles]
                    series = _ema(closes, period) if indicator == "ema" else _sma(closes, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "rsi":
                    period = tool_input.get("period", 14)
                    from src.indicators.local_indicators import rsi as _rsi
                    series = _rsi(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "atr":
                    period = tool_input.get("period", 14)
                    from src.indicators.local_indicators import atr as _atr
                    series = _atr(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                else:
                    key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap", "stoch_rsi": "stoch_rsi"}
                    mapped = key_map.get(indicator, indicator)
                    series = all_indicators.get(mapped, [])
                    result = {"latest": latest(series) if isinstance(series, list) else series,
                              "series": last_n(series, 10) if isinstance(series, list) else series}

                return json.dumps(result, default=str)
            except Exception as ex:
                logging.error("Tool call error: %s", ex)
                return json.dumps({"error": str(ex)})

        def _sanitize_output(raw_content: str, assets_list):
            """Use a cheap Claude model to normalize malformed output."""
            try:
                response = self.client.messages.create(
                    model=self.sanitize_model,
                    max_tokens=max(self.max_tokens, 4096),
                    system=(
                        "You are a strict JSON normalizer. Return ONLY a JSON object with one key: "
                        "\"trade_decisions\" (array). "
                        "Each trade_decisions item must have: asset, action (buy/sell/hold), "
                        "allocation_usd (number), order_type (\"market\" or \"limit\"), "
                        "limit_price (number or null), tp_price (number or null), sl_price (number or null), "
                        "exit_plan (string), rationale (string), thesis_strength (integer 1-5). "
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

        # Main loop: up to 6 iterations to handle tool calls
        for iteration in range(6):
            try:
                response = _call_claude(messages)
            except anthropic.APIError as e:
                logging.error("Claude API error: %s", e)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"API Error: {e}\n")
                break

            # Check if the response contains tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_use_blocks and response.stop_reason == "tool_use":
                # Build assistant message with all content blocks
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
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                # Process each tool call
                tool_results = []
                for block in tool_use_blocks:
                    result_str = _handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # No tool calls — parse the text response as JSON
            raw_text = ""
            for block in text_blocks:
                raw_text += block.text

            if not raw_text.strip():
                logging.error("Empty response from Claude")
                break

            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

            # If Claude prefaced the JSON with prose, skip to the first '{'
            if not cleaned.startswith("{"):
                brace_pos = cleaned.find("{")
                if brace_pos >= 0:
                    cleaned = cleaned[brace_pos:]

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    logging.error("Expected dict, got: %s; attempting sanitize", type(parsed))
                    return _sanitize_output(raw_text, assets)

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
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {"reasoning": reasoning_text, "trade_decisions": []}

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logging.error("JSON parse error: %s, content: %s", e, raw_text[:200])
                sanitized = _sanitize_output(raw_text, assets)
                if sanitized.get("trade_decisions"):
                    return sanitized
                return {
                    "reasoning": "Parse error",
                    "trade_decisions": [{
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error"
                    } for a in assets]
                }

        # Exhausted tool loop
        return {
            "reasoning": "tool loop cap",
            "trade_decisions": [{
                "asset": a,
                "action": "hold",
                "allocation_usd": 0.0,
                "tp_price": None,
                "sl_price": None,
                "exit_plan": "",
                "rationale": "tool loop cap"
            } for a in assets]
        }
