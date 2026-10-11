.PHONY: test lint format typecheck check sync reset serve prepare train count-games eval-stockfish eval-round-robin fetch-weights vendor

EPOCHS ?= 10
WEIGHTS_TAG ?= v0.1.0
WEIGHTS_BASE ?= https://github.com/Melsso/human-move/releases/download/$(WEIGHTS_TAG)

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

fetch-weights:
	mkdir -p weights/bucket_1000_1 weights/bucket_2000_1
	curl -fL $(WEIGHTS_BASE)/bucket_1000.pt -o weights/bucket_1000_1/best.pt
	curl -fL $(WEIGHTS_BASE)/bucket_2000.pt -o weights/bucket_2000_1/best.pt
	@echo "done. run: CHECKPOINTS_DIR=weights make serve"

vendor:
	bash scripts/vendor_static.sh

eval-stockfish: sync
	uv run --package chess-eval python -m chess_eval.stockfish $(TIER) \
		$(if $(STOCKFISH_ELOS),--stockfish-elos $(STOCKFISH_ELOS),) \
		$(if $(GAMES),--games $(GAMES),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		--out eval/results/$(TIER)_stockfish.json

eval-round-robin: sync
	uv run --package chess-eval python -m chess_eval.round_robin $(TIERS) \
		$(if $(GAMES),--games $(GAMES),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		--out eval/results/round_robin.json