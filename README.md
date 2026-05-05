# Hyperliquid AI Trading Agent

An AI-powered trading agent that uses Claude to analyze markets and execute perpetual futures trades on Hyperliquid. Supports crypto, stocks, commodities, indices, and forex via HIP-3 markets.

## How It Works

Each loop iteration:

1. **Macro filter** — fetches Fear & Greed Index, DXY trend, and Finnhub economic calendar concurrently. Blocks new opens within 60 min of high-impact events; raises entry threshold during extreme sentiment; signals Claude to reduce long allocation when DXY is rising.
2. **Account state** — fetches balance, positions, open orders, recent fills from Hyperliquid.
3. **Risk checks** — force-closes any position at or beyond the max-loss threshold before Claude is called.
4. **Market data** — fetches 5m and 4h candles, computes indicators locally (EMA, RSI, MACD, ATR, ADX, OBV, VWAP, BBands).
5. **Claude decision** — sends full context (account state, market data, macro context, risk limits) to Claude. Claude returns a structured JSON decision for every asset with `thesis_strength`, action, allocation, TP/SL, and rationale.
6. **Thesis tracker** — two deterministic close rules enforced in code before execution: `thesis_strength == 1` on an open position → immediate close; three consecutive cycles at `thesis_strength ≤ 2` → exit regardless of P&L.
7. **Risk validation** — risk manager caps allocation, enforces leverage limits, checks circuit breakers.
8. **Execution** — places market or limit orders with TP/SL. Reconciles TP/SL orders every cycle.

## Entry Strategy — Crypto (Momentum Breakout)

For pure-crypto perps (BTC, ETH, SOL, etc.), Claude uses a single defined setup:

**4h bias** (all required): EMA20 > EMA50, MACD histogram positive, ADX > 25
**5m entry** (all required): close breaks above previous bar high, OBV rising, RSI 50–70
**TP**: 1.5× ATR14 above entry. **SL**: 0.75× ATR14 below entry (R:R = 2:1)
**Minimum thesis_strength to open**: 4. No new opens at 3 or below.
**ADX gate**: no new opens on any asset when ADX < 25.

## Thesis Strength

Claude scores every asset every cycle on a 1–5 scale:

| Score | Meaning |
|-------|---------|
| 5 | Strong trend, full 4h + 5m confluence |
| 4 | Good setup, minor mixed signals — minimum to open |
| 3 | Neutral, no clear edge — hold only |
| 2 | Thesis weakening, signals diverging |
| 1 | Thesis broken — immediate close enforced in code |

Two auto-close rules are enforced in code (not just prompt):
- `thesis_strength == 1` + open position → override to close
- Three consecutive scores ≤ 2 + open position → override to close

Score history persists to `thesis_history.json` across restarts.

## Macro Filter

Fetched at the top of every cycle via three concurrent sources:

| Signal | Source | Effect |
|--------|--------|--------|
| Fear & Greed Index | alternative.me (free) | Score < 20 or > 80 → raises min thesis to open to 4 |
| Economic calendar | Finnhub (free tier) | High-impact event within 60 min → blocks all new opens |
| DXY trend | yfinance, 5-day EMA | Rising → Claude instructed to reduce long crypto allocation ~30% |

Requires `FINNHUB_API_KEY` in `.env`. All sources fail gracefully — the trading loop is never blocked by a fetch error.

## Safety Guards

All enforced in code, not just LLM prompts. Configurable via `.env`:

| Guard | Default | Description |
|-------|---------|-------------|
| Max Position Size | 20% | Single position capped at 20% of portfolio |
| Force Close | -20% | Auto-close positions at 20% loss |
| Max Leverage | 10x | Hard leverage cap |
| Total Exposure | 80% | All positions combined capped at 80% |
| Daily Circuit Breaker | -25% | Stops new trades at 25% daily drawdown |
| Mandatory Stop-Loss | 5% | Auto-set SL if Claude doesn't provide one |
| Max Positions | 10 | Concurrent position limit |
| Balance Reserve | 10% | Don't trade below 10% of initial balance |

## Tradeable Markets

All 229+ Hyperliquid perp markets plus HIP-3 tradfi assets:

- **Crypto**: BTC, ETH, SOL, HYPE, AVAX, SUI, ARB, LINK, and 200+ more
- **Stocks**: xyz:TSLA, xyz:NVDA, xyz:AAPL, xyz:GOOGL, xyz:AMZN, xyz:META, xyz:MSFT, xyz:COIN, xyz:PLTR...
- **Commodities**: xyz:GOLD, xyz:SILVER, xyz:BRENTOIL, xyz:CL, xyz:COPPER, xyz:NATGAS, xyz:PLATINUM
- **Indices**: xyz:SP500, xyz:XYZ100
- **Forex**: xyz:EUR, xyz:JPY

## Setup

### Prerequisites
- Python 3.12+
- Anthropic API key
- Hyperliquid wallet (agent wallet as signer + main wallet with funds)
- Finnhub API key (free tier — for economic calendar)

### Install

```bash
git clone <repo>
cd sanket
~/Library/Python/3.9/bin/poetry install
cp .env.example .env
# Fill in .env
```

### Configuration

Required:
- `ANTHROPIC_API_KEY` — Claude API key
- `HYPERLIQUID_PRIVATE_KEY` — Agent/API wallet private key (signer only)
- `HYPERLIQUID_VAULT_ADDRESS` — Main wallet address (holds funds)
- `ASSETS` — Space-separated list of assets, e.g. `"BTC ETH SOL"`
- `INTERVAL` — Trading loop interval, e.g. `5m`

Recommended:
- `FINNHUB_API_KEY` — Free at finnhub.io. Enables economic calendar gate.

### Agent Wallet Setup

1. Go to app.hyperliquid.xyz → Settings → API Wallets
2. Add your agent wallet address as an authorized signer
3. Set `HYPERLIQUID_VAULT_ADDRESS` to your main wallet address in `.env`

The agent wallet signs trades on behalf of your main wallet. It cannot withdraw funds.

### Run

```bash
./run.sh
# or
~/Library/Python/3.9/bin/poetry run python -m src.main --assets BTC ETH SOL --interval 5m
```

## Recommended Pre-Live Checklist

1. **Backtest** — run on BTC, ETH, SOL with 2 years of data. Threshold: win rate > 38%, ≥ 200 trades.
2. **Paper trade** — run on Hyperliquid testnet for at least 2 weeks.
3. **Compliance check** — verify Claude follows entry rules using `decisions.jsonl` from paper trading.
4. **Go live** — minimal capital only after steps 1–3 pass.

```bash
# Fetch 2 years of candle history (Binance for 5m, Hyperliquid for 4h)
~/Library/Python/3.9/bin/poetry run python -m src.backtest.fetch_history --assets BTC ETH SOL --intervals 5m 4h --years 2

# Run backtest
~/Library/Python/3.9/bin/poetry run python -m src.backtest.run_backtest --assets BTC ETH SOL

# After paper trading — check Claude followed the rules
~/Library/Python/3.9/bin/poetry run python -m src.backtest.compliance_check
```

## API Endpoints

When running, serves a local API on port 3000:

| Endpoint | Description |
|----------|-------------|
| `GET /diary` | Trade diary entries (opens, closes, holds, risk blocks) |
| `GET /logs` | LLM request logs with macro context |
| `GET /state` | Live dashboard state (positions, account value, recent decisions) |
| `GET /history` | All-time realized P&L from Hyperliquid fill history |

## Project Structure

```
src/
  main.py                  # Trading loop, escalation logic, API server
  config_loader.py         # Environment config
  risk_manager.py          # Position limits, loss protection, circuit breakers
  macro_filter.py          # Fear & Greed, DXY, economic calendar
  thesis_tracker.py        # Per-asset thesis_strength history, auto-close rules
  agent/
    decision_maker.py      # Claude API integration, prompt, tool calling
  indicators/
    local_indicators.py    # EMA, RSI, MACD, ATR, BBands, ADX, OBV, VWAP
  trading/
    hyperliquid_api.py     # Order execution, candles, state queries
  backtest/
    fetch_history.py       # Fetch + cache candle history (Binance fallback for 5m)
    run_backtest.py        # Momentum Breakout simulation, filter comparison table
    compliance_check.py    # Audit decisions.jsonl for rule violations post-live
  utils/
    formatting.py
    prompt_utils.py
data/
  candles/                 # Cached candle data (gitignored)
```

## License

Use at your own risk. No guarantee of returns. This code has not been audited.
