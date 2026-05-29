FROM python:3.12-slim

WORKDIR /app

# Install build tools needed by some deps (web3, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "hyperliquid-python-sdk>=0.20.0,<0.21.0" \
    "python-dotenv>=1.1.1,<2.0.0" \
    "web3>=7.14.0,<8.0.0" \
    "aiohttp>=3.13.1,<4.0.0" \
    "anthropic>=0.52.0,<1.0.0" \
    "requests>=2.32.5,<3.0.0" \
    "rich>=14.2.0,<15.0.0" \
    "yfinance>=0.2.0" \
    "pyyaml>=6.0" \
    "numpy>=1.24" \
    "certifi"

# Copy source and config
COPY src ./src
COPY config ./config

# Runtime data dirs (mounted as volumes in production)
RUN mkdir -p data logs

ENTRYPOINT ["python", "-m", "src.main"]
