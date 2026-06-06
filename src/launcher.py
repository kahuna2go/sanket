"""Strategy launcher — manages LLM and SOL Momentum as subprocesses.

Runs an aiohttp dashboard server on port 3000.
Each strategy gets its own port:
  LLM strategy:   port 3001
  SOL Momentum:   port 3002

Usage:
  python -m src.launcher
  Then open http://localhost:3000
"""

import asyncio
import json
import logging
import logging.handlers
import os
import pathlib
import signal
import sys
from collections import deque

from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [launcher] %(message)s",
    datefmt="%H:%M:%S",
)

DASHBOARD_HTML = pathlib.Path(__file__).parent.parent / "dashboard.html"
PYTHON = sys.executable
_ROOT = pathlib.Path(__file__).parent.parent
_PAUSE_FILE = _ROOT / "data" / "llm_pause.json"


def _get_llm_paused() -> bool:
    try:
        return json.loads(_PAUSE_FILE.read_text()).get("paused", False)
    except Exception:
        return False


def _set_llm_paused(v: bool):
    _PAUSE_FILE.parent.mkdir(exist_ok=True)
    _PAUSE_FILE.write_text(json.dumps({"paused": v}))

# Per-strategy state
_strategies: dict[str, dict] = {
    "llm": {
        "port": 3001,
        "proc": None,
        "logs": deque(maxlen=300),
        "subscribers": set(),
    },
    "smoby": {
        "port": 3002,
        "proc": None,
        "logs": deque(maxlen=300),
        "subscribers": set(),
    },
    "smc": {
        "port": 3004,
        "proc": None,
        "logs": deque(maxlen=300),
        "subscribers": set(),
    },
    "orb": {
        "port": 3003,
        "proc": None,
        "logs": deque(maxlen=300),
        "subscribers": set(),
    },
}


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _is_running(name: str) -> bool:
    proc = _strategies[name]["proc"]
    return proc is not None and proc.returncode is None


def _get_file_logger(name: str) -> logging.Logger:
    log_path = _ROOT / "logs" / f"{name}.log"
    logger = logging.getLogger(f"strategy.{name}")
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


async def _read_logs(name: str, proc: asyncio.subprocess.Process):
    """Read subprocess stdout+stderr and push lines to buffer + SSE subscribers."""
    buf = _strategies[name]["logs"]
    subs = _strategies[name]["subscribers"]
    file_log = _get_file_logger(name)
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            buf.append(line)
            file_log.info(line)
            dead = set()
            for q in subs:
                try:
                    q.put_nowait(line)
                except asyncio.QueueFull:
                    dead.add(q)
            subs -= dead
    except Exception:
        pass
    logging.info("%s process exited (rc=%s)", name, proc.returncode)


async def _kill_port(port: int):
    """Kill any process occupying the given port so the strategy can bind to it."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "fuser", "-k", f"{port}/tcp",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        await asyncio.sleep(0.5)
    except FileNotFoundError:
        pass  # fuser not available; port assumed free


async def _start(name: str, extra_env: dict | None = None, args: list[str] | None = None):
    if _is_running(name):
        return {"ok": False, "error": f"{name} already running"}

    s = _strategies[name]
    await _kill_port(s["port"])
    import certifi
    ca = certifi.where()
    env = {**os.environ, "API_PORT": str(s["port"]), "SSL_CERT_FILE": ca, "WEBSOCKET_CLIENT_CA_BUNDLE": ca}
    if extra_env:
        env.update(extra_env)

    cmd = [PYTHON, "-m", "src.main"] + (args or [])
    logging.info("Starting %s: %s", name, " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd=str(pathlib.Path(__file__).parent.parent),
    )
    s["proc"] = proc
    asyncio.ensure_future(_read_logs(name, proc))
    return {"ok": True, "pid": proc.pid}


async def _stop(name: str):
    s = _strategies[name]
    proc = s["proc"]
    if proc is None or proc.returncode is not None:
        return {"ok": False, "error": f"{name} not running"}
    proc.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=8)
    except asyncio.TimeoutError:
        proc.kill()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Account state (read-only, proxied from running strategy API or direct HL call)
# ---------------------------------------------------------------------------

async def _get_account():
    """Try to fetch account state from a running strategy's API, fall back to empty."""
    import aiohttp as _aio
    for name, s in _strategies.items():
        if _is_running(name):
            try:
                url = f"http://127.0.0.1:{s['port']}/state"
                async with _aio.ClientSession() as sess:
                    async with sess.get(url, timeout=_aio.ClientTimeout(total=3)) as r:
                        if r.status == 200:
                            data = await r.json()
                            return {
                                "balance": data.get("balance", 0),
                                "total_value": data.get("total_value", 0),
                                "positions": data.get("positions", []),
                            }
            except Exception:
                continue
    return {"balance": None, "total_value": None, "positions": []}


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_index(request):
    if not DASHBOARD_HTML.exists():
        return web.Response(text="dashboard.html not found", status=404)
    return web.FileResponse(DASHBOARD_HTML)


async def handle_status(request):
    account = await _get_account()
    return web.json_response({
        "strategies": {
            name: {
                "running": _is_running(name),
                "pid": s["proc"].pid if _is_running(name) else None,
                "port": s["port"],
            }
            for name, s in _strategies.items()
        },
        "account": account,
    })


async def handle_start(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)

    name = body.get("strategy")
    if name not in _strategies:
        return web.json_response({"ok": False, "error": "unknown strategy"}, status=400)

    _strategy_env = {"smoby": "sol_momentum"}  # name → STRATEGY env var when different

    if name == "llm":
        assets = body.get("assets", [])
        interval = body.get("interval", "5m")
        if not assets:
            return web.json_response({"ok": False, "error": "assets required for LLM strategy"}, status=400)
        args = ["--assets"] + assets + ["--interval", interval]
        result = await _start(name, args=args)
    else:
        result = await _start(name, extra_env={"STRATEGY": _strategy_env.get(name, name)})

    return web.json_response(result)


async def handle_stop(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)

    name = body.get("strategy")
    if name not in _strategies:
        return web.json_response({"ok": False, "error": "unknown strategy"}, status=400)

    result = await _stop(name)
    return web.json_response(result)


_STUB_STATE = {
    "status": "stopped", "balance": 0, "total_value": 0,
    "positions": [], "uptime_minutes": 0, "invocation_count": 0,
    "model_usage": {}, "recent_decisions": [],
}

async def handle_llm_pause(request):
    """Toggle or query the LLM pause flag.

    Works before the LLM strategy starts (state persisted in a file).
    When the strategy is running, also syncs the flag to the subprocess.
    """
    import aiohttp as _aio
    if request.method == "POST":
        _set_llm_paused(not _get_llm_paused())
        # Best-effort sync to running process
        if _is_running("llm"):
            try:
                port = _strategies["llm"]["port"]
                async with _aio.ClientSession() as sess:
                    await sess.post(f"http://127.0.0.1:{port}/llm-pause",
                                    timeout=_aio.ClientTimeout(total=2))
            except Exception:
                pass
    return web.json_response({"llm_paused": _get_llm_paused()})


async def handle_proxy(request):
    """Proxy LLM strategy API endpoints to port 3001 so dashboard works unchanged.
    When LLM is not running, return stub data so the dashboard renders (overlay clears)
    and the launcher control bar is visible.
    """
    import aiohttp as _aio
    path = request.path
    port = _strategies["llm"]["port"]
    url  = f"http://127.0.0.1:{port}{path}"
    try:
        async with _aio.ClientSession() as sess:
            async with sess.request(request.method, url, timeout=_aio.ClientTimeout(total=5)) as r:
                body  = await r.read()
                ctype = r.headers.get("Content-Type", "application/json")
                return web.Response(body=body, content_type=ctype.split(";")[0].strip(), status=r.status)
    except Exception:
        # LLM strategy not running — return stub so dashboard renders
        if path == "/state":
            return web.json_response(_STUB_STATE)
        return web.json_response([] if path in ("/history", "/diary") else {})


async def handle_logs_sse(request):
    name = request.match_info["strategy"]
    if name not in _strategies:
        return web.Response(status=404)

    s = _strategies[name]
    response = web.StreamResponse()
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Access-Control-Allow-Origin"] = "*"
    await response.prepare(request)

    # Flush existing buffer to new subscriber
    for line in list(s["logs"]):
        await response.write(f"data: {json.dumps(line)}\n\n".encode())

    # Stream live lines via queue
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    s["subscribers"].add(queue)
    try:
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=20)
                await response.write(f"data: {json.dumps(line)}\n\n".encode())
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")  # prevent proxy timeout
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        s["subscribers"].discard(queue)

    return response


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

def build_app() -> web.Application:
    import base64

    _dash_user = os.getenv("DASHBOARD_USER")
    _dash_pass = os.getenv("DASHBOARD_PASSWORD")

    _NO_AUTH_PATHS = {'/llm-pause', '/favicon.ico', '/apple-touch-icon.png', '/apple-touch-icon-precomposed.png'}

    @web.middleware
    async def auth(request, handler):
        if _dash_user and _dash_pass and request.path not in _NO_AUTH_PATHS:
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
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    app = web.Application(middlewares=[auth, cors])
    app.router.add_get("/",                     handle_index)
    app.router.add_get("/api/status",           handle_status)
    app.router.add_post("/api/start",           handle_start)
    app.router.add_post("/api/stop",            handle_stop)
    app.router.add_get("/api/logs/{strategy}",  handle_logs_sse)
    # Proxy LLM strategy endpoints so dashboard.html works unchanged on port 3000
    for path in ("/state", "/history", "/diary", "/logs"):
        app.router.add_get(path, handle_proxy)
    app.router.add_get("/llm-pause",  handle_llm_pause)
    app.router.add_post("/llm-pause", handle_llm_pause)
    return app


async def main():
    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 3000)
    await site.start()
    logging.info("Dashboard running at http://localhost:3000")
    try:
        await asyncio.Event().wait()  # run forever
    finally:
        # Stop any running strategies on shutdown
        for name in _strategies:
            if _is_running(name):
                await _stop(name)
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
