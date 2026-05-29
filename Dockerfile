FROM python:3.12-slim

WORKDIR /app

# Install build tools needed by some deps (web3, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry, then use it to install exact locked versions
RUN pip install --no-cache-dir poetry==1.8.5

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main --no-interaction --no-ansi

# Copy source and config
COPY src ./src
COPY config ./config

# Runtime data dirs (mounted as volumes in production)
RUN mkdir -p data logs

ENTRYPOINT ["python", "-m", "src.main"]
