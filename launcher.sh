#!/bin/bash
# Start the launcher dashboard server, then open http://localhost:3000 in the browser
cd "$(dirname "$0")"
PYTHON="/Users/peermagnus/Library/Caches/pypoetry/virtualenvs/trading-agent-iHVPNt-P-py3.12/bin/python3.12"
open http://localhost:3000
exec "$PYTHON" -m src.launcher
