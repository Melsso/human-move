.PHONY: test lint format typecheck check sync reset prepare-1 prepare-2 prepare-3 train-1000 train-1500 train-2000 play serve

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

prepare-1: sync
	uv run --package chess-data python -m chess_data.prepare \
		data/downloads/sample.pgn.zst \
		data/processed/sample_bucket_1000.npz \
		--min-elo 900 --max-elo 1100 \
		--max-games 300000

prepare-2: sync
	uv run --package chess-data python -m chess_data.prepare \
		data/downloads/sample.pgn.zst \
		data/processed/sample_bucket_1500.npz \
		--min-elo 1400 --max-elo 1600 \
		--max-games 300000

prepare-3: sync
	uv run --package chess-data python -m chess_data.prepare \
		data/downloads/sample.pgn.zst \
		data/processed/sample_bucket_2000.npz \
		--min-elo 1900 --max-elo 2100 \
		--max-games 300000

train-1000: sync
	uv run --package chess-training python -m chess_training.train \
		data/processed/bucket_1000.npz \
		--out-dir training/checkpoints/bucket_1000 \
		--epochs 10 --batch-size 256 --lr 1e-3

train-1500: sync
	uv run --package chess-training python -m chess_training.train \
		data/processed/bucket_1500.npz \
		--out-dir training/checkpoints/bucket_1500 \
		--epochs 10 --batch-size 256 --lr 1e-3

train-2000: sync
	uv run --package chess-training python -m chess_training.train \
		data/processed/bucket_2000.npz \
		--out-dir training/checkpoints/bucket_2000 \
		--epochs 10 --batch-size 256 --lr 1e-3

# usage: make play CHECKPOINT=training/checkpoints/bucket_1000/best.pt
play: sync
	uv run --package chess-training python -m chess_training.play $(CHECKPOINT)

serve: sync
	uv run --package chess-backend uvicorn chess_backend.main:app --reload --port 8000