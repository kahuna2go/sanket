#!/bin/bash
# Launch the Sanket trading agent with correct SSL certificates for macOS + Python 3.12
cd "$(dirname "$0")"
CERT=$(~/Library/Python/3.9/bin/poetry run python3 -c "import certifi; print(certifi.where())" 2>/dev/null | tail -1)
export SSL_CERT_FILE="$CERT"
export REQUESTS_CA_BUNDLE="$CERT"
exec ~/Library/Python/3.9/bin/poetry run python -m src.main
