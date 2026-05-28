"""Rich-based display layer for Sanket.

Replaces the default logging.basicConfig formatter with a compact, coloured
terminal output.  Everything that was previously a logging.info() call is still
logged the same way internally — we just swap the handler so it renders nicely.

Public API used by main.py / decision_maker.py:
    setup_logging()            — call once at startup instead of basicConfig()
    print_loop_header()        — separator printed at the start of each loop tick
    print_position_table()     — compact single-row-per-asset position summary
    print_decision()           — coloured HOLD / BUY / SELL line with short reason
    print_claude_stats()       — dim token-usage line
    print_orb_status()         — compact ORB state line
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── colour palette ────────────────────────────────────────────────────────────
_THEME = Theme({
    "hold":      "dim white",
    "buy":       "bold green",
    "sell":      "bold red",
    "pnl.pos":   "green",
    "pnl.neg":   "red",
    "pnl.zero":  "dim white",
    "asset":     "bold cyan",
    "price":     "white",
    "dim":       "dim white",
    "warn":      "yellow",
    "orb.hit":   "bold magenta",
    "orb.miss":  "dim white",
    "label":     "dim cyan",
})

console = Console(theme=_THEME, highlight=False)

# ── logging setup ─────────────────────────────────────────────────────────────

# Patterns for log lines we suppress entirely (HTTP plumbing, not trading info)
_SUPPRESS_PATTERNS = [
    re.compile(r"HTTP Request:"),
    re.compile(r"httpx"),
]

# Patterns we dim heavily (internal Claude response stats — shown via print_claude_stats instead)
_DIM_PATTERNS = [
    re.compile(r"Claude response:"),
    re.compile(r"Decision rationale for"),
]


class _SanketFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for p in _SUPPRESS_PATTERNS:
            if p.search(msg):
                return False
        return True


class _SanketFormatter(logging.Formatter):
    """Compact single-line formatter used by RichHandler."""

    # We shorten very long messages (rationale walls) to one readable line.
    _RATIONALE_RE = re.compile(r"^Decision rationale for ([^:]+): (.+)$", re.DOTALL)

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        m = self._RATIONALE_RE.match(msg)
        if m:
            # Already rendered by print_decision(); skip duplicate here.
            return ""

        return msg


def setup_logging() -> None:
    """Replace default basicConfig logging with Rich-formatted output."""
    root = logging.getLogger()
    root.handlers.clear()

    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.addFilter(_SanketFilter())

    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "anthropic", "aiohttp"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── loop header ───────────────────────────────────────────────────────────────

def print_loop_header(session_name: str, interval_secs: int, assets: list[str]) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    assets_str = "  ".join(f"[asset]{a}[/asset]" for a in assets)
    console.rule(
        f"[dim]{now}[/dim]  [label]{session_name}[/label]  [dim]{interval_secs}s[/dim]  {assets_str}",
        style="dim",
    )


# ── position table ────────────────────────────────────────────────────────────

def print_position_table(rows: list[dict]) -> None:
    """Print a compact table with one row per asset.

    Each row dict:
        asset, side (long/short/—), amount, entry, current, pnl_usd,
        sl_dist_pts (float|None), tp_dist_pts (float|None)
    """
    if not rows:
        return

    t = Table(box=box.SIMPLE, show_header=True, header_style="label", padding=(0, 1))
    t.add_column("Asset",   style="asset",  no_wrap=True)
    t.add_column("Side",    no_wrap=True)
    t.add_column("Size",    justify="right", style="dim")
    t.add_column("Entry",   justify="right", style="dim")
    t.add_column("Now",     justify="right", style="price")
    t.add_column("PnL",     justify="right")
    t.add_column("SL dist", justify="right", style="dim")
    t.add_column("TP dist", justify="right", style="dim")

    for r in rows:
        side = r.get("side", "—")
        if side == "long":
            side_txt = Text("LONG",  style="buy")
        elif side == "short":
            side_txt = Text("SHORT", style="sell")
        else:
            side_txt = Text("—",     style="dim")

        pnl = r.get("pnl_usd")
        if pnl is None:
            pnl_txt = Text("—", style="dim")
        elif pnl > 0:
            pnl_txt = Text(f"+${pnl:.2f}", style="pnl.pos")
        elif pnl < 0:
            pnl_txt = Text(f"-${abs(pnl):.2f}", style="pnl.neg")
        else:
            pnl_txt = Text("$0.00", style="pnl.zero")

        def _fmt_dist(v):
            if v is None:
                return Text("—", style="dim")
            return Text(f"{v:.1f}pt", style="dim")

        t.add_row(
            r.get("asset", "?"),
            side_txt,
            f"{r['amount']:.4f}" if r.get("amount") else "—",
            f"{r['entry']:.2f}"  if r.get("entry")  else "—",
            f"{r['current']:.2f}" if r.get("current") else "—",
            pnl_txt,
            _fmt_dist(r.get("sl_dist_pts")),
            _fmt_dist(r.get("tp_dist_pts")),
        )

    console.print(t)


# ── decision line ─────────────────────────────────────────────────────────────

_RATIONALE_LIMIT = 160   # chars shown for hold; buy/sell show a bit more


def print_decision(
    asset: str,
    action: str,           # "hold", "buy", "sell", "update_tpsl", "cancel_limits"
    rationale: str,
    thesis_strength: Optional[int] = None,
    extra: str = "",       # e.g. "TP → 2055.5  SL → 2160.1"
) -> None:
    action_up = action.upper()
    if action == "hold":
        action_style = "hold"
    elif action == "buy":
        action_style = "buy"
    elif action == "sell":
        action_style = "sell"
    else:
        action_style = "warn"

    # Truncate rationale to the first sentence / N chars
    short = _first_sentence(rationale, _RATIONALE_LIMIT)
    ts_str = f"  [label]ts={thesis_strength}[/label]" if thesis_strength else ""
    extra_str = f"  [dim]{extra}[/dim]" if extra else ""

    console.print(
        f"  [{action_style}]{action_up:14s}[/{action_style}] "
        f"[asset]{asset}[/asset]{ts_str}{extra_str}\n"
        f"    [dim]{short}[/dim]"
    )


def _first_sentence(text: str, limit: int) -> str:
    """Return the first sentence of text, capped at limit chars."""
    text = text.strip()
    # Try to break at first full stop followed by space or end
    m = re.search(r"\.(?:\s|$)", text)
    if m and m.end() <= limit:
        return text[: m.end()].strip()
    # Fallback: hard truncate at limit
    if len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


# ── claude token stats ────────────────────────────────────────────────────────

def print_claude_stats(
    input_tokens: int,
    output_tokens: int,
    cache_create: int,
    cache_read: int,
    stop_reason: str,
) -> None:
    flag = ""
    if stop_reason == "max_tokens":
        flag = "  [warn]⚠ max_tokens[/warn]"
    console.print(
        f"  [dim]Claude  in={input_tokens:,}  out={output_tokens:,}  "
        f"cache_create={cache_create:,}  cache_read={cache_read:,}{flag}[/dim]"
    )


# ── ORB status ────────────────────────────────────────────────────────────────

def print_orb_status(
    phase: str,
    bias: Optional[str],
    orh: Optional[float],
    orl: Optional[float],
    breakout_long: bool,
    breakout_short: bool,
    trade_taken: bool,
    rr: Optional[float] = None,
    breakout_pending: Optional[str] = None,
) -> None:
    hit = breakout_long or breakout_short
    if hit:
        direction = "▲ LONG retest" if breakout_long else "▼ SHORT retest"
        style = "orb.hit"
    elif breakout_pending == "long":
        direction = "▲ broke out — retest pending"
        style = "orb.miss"
    elif breakout_pending == "short":
        direction = "▼ broke out — retest pending"
        style = "orb.miss"
    else:
        direction = "—"
        style = "orb.miss"
    rr_str = f"  R:R {rr:.2f}" if rr is not None else ""
    taken_str = "  [warn]trade taken[/warn]" if trade_taken else ""

    range_str = ""
    if orh is not None and orl is not None:
        range_str = f"  ORH {orh:.1f}  ORL {orl:.1f}  range {orh - orl:.1f}pt"

    bias_str = f"  [dim]{bias}[/dim]" if bias else ""

    console.print(
        f"  [label]ORB[/label]  [dim]{phase}[/dim]{range_str}{bias_str}  "
        f"[{style}]{direction}[/{style}]{rr_str}{taken_str}"
    )
