FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates stockfish \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    CHECKPOINTS_DIR=/app/weights \
    STOCKFISH_PATH=/usr/games/stockfish

WORKDIR /app
COPY . .

RUN uv sync --frozen --no-dev --package chess-backend

ARG WEIGHTS_TAG=v0.1.0
RUN mkdir -p weights/bucket_1000_1 weights/bucket_2000_1 \
    && curl -fL https://github.com/Melsso/human-move/releases/download/${WEIGHTS_TAG}/bucket_1000.pt -o weights/bucket_1000_1/best.pt \
    && curl -fL https://github.com/Melsso/human-move/releases/download/${WEIGHTS_TAG}/bucket_2000.pt -o weights/bucket_2000_1/best.pt

CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn chess_backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]