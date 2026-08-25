# The build context is the *workspace* directory, not this repo, because
# llm-client-kit is a sibling path dependency. `docker compose build` is run
# from here and sets the context accordingly (see docker-compose.yml).
FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependency manifests first so a source-only change does not invalidate the
# dependency layer.
COPY inference-gateway/pyproject.toml ./inference-gateway/
COPY llm-client-kit/pyproject.toml ./llm-client-kit/
COPY llm-client-kit/src ./llm-client-kit/src

WORKDIR /app/inference-gateway
RUN uv venv && uv pip install --no-cache fastapi uvicorn httpx -e ../llm-client-kit

COPY inference-gateway/src ./src
COPY inference-gateway/scripts ./scripts

ENV PATH="/app/inference-gateway/.venv/bin:$PATH" \
    PYTHONPATH=/app/inference-gateway/src \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker on purpose: the cache, ledger and idempotency store are
# in-process, so a second worker would serve inconsistent results. Scaling out
# means moving that state into the Redis and Postgres services this compose
# file already declares. Stated in the README's Limitations.
CMD ["uvicorn", "inference_gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
