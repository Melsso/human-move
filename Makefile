.PHONY: test lint format typecheck check sync reset play serve prepare train count-games

EPOCHS ?= 10

sync:
	uv sync --all-packages

reset:
	rm -rf .venv
	uv cache clean
	uv sync --all-packages

test: sync
	uv run --all-packages pytest -v

lint: sync
	uv run --all-packages ruff check --no-cache .

format: sync
	uv run --all-packages ruff format --no-cache .

typecheck: sync
	uv run --all-packages mypy

check: sync
	uv run --all-packages ruff format --no-cache --check .
	uv run --all-packages ruff check --no-cache .
	uv run --all-packages mypy
	uv run --all-packages pytest -v

count-games: sync
	uv run --package chess-data python -m chess_data.count_games $(SOURCE)

prepare: sync
	uv run --package chess-data python -m chess_data.prepare \
		$(SOURCE) \
		data/processed \
		--chunk-size $(CHUNK_SIZE) \
		$(if $(BUCKETS),--buckets $(BUCKETS),) \
		$(if $(MAX_CHUNKS),--max-chunks $(MAX_CHUNKS),)

train: sync
	bash scripts/train_all_chunks.sh "$(BUCKETS)" "$(EPOCHS)"

serve: sync
	uv run --package chess-backend uvicorn chess_backend.main:app --reload --port 8000