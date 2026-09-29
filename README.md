# chess-ai

A from-scratch chess engine with three human-like rating tiers (~1000, ~1500,
~2000). Policy networks are trained on Lichess games bucketed by rating (the
Maia approach). At play time each tier adds a temperature and, for the two
stronger tiers, a small policy-guided alpha-beta search, tuned for playing
strength and move accuracy. Fully local.

## Layout

```
chess-ai/
├── shared/     # encodings, masked inference, search, tier config, move selection
├── data/       # Lichess PGN -> rating-bucketed training tensors
├── training/   # policy network + training loop
├── backend/    # FastAPI inference API + static board UI (backend/static/)
├── eval/       # Stockfish calibration + inter-tier round robin
└── scripts/    # train_all_chunks.sh
```

`shared` is the single source of truth for board and move encoding and for how
a tier picks a move. Training, serving and evaluation all use it, so they
can't drift apart. It has no torch dependency.

## Tiers

Defined in `shared/chess_shared/tiers.py`:

| Tier | Checkpoint | Temperature | Search |
|---|---|---|---|
| 1000 | bucket 1000 | 0.3 | none |
| 1500 | bucket 1500 | 0 | depth 3, weight 1 |
| 2000 | bucket 1500 | 0 | depth 6, weight 4 |

There is no separate 2000 model. Tier 2000 is the 1500 model searched deeper
and trusted less. The search re-scores the network's top 6 moves as
`log(policy_prob) + weight * eval_in_pawns` (details in `shared/README.md`).
Data buckets are defined in `data/chess_data/brackets.py`.

## Setup

A `uv` workspace with one lockfile.

```bash
uv sync --all-packages
```

Use `make <target>` instead of raw `uv` commands. Every target syncs with
`--all-packages` first, because a bare `uv sync` uninstalls the workspace
members. If you hit `ModuleNotFoundError: No module named 'chess_X'`, run
`make reset`. After editing any `pyproject.toml`, run `uv lock`.

## Workflow

```bash
make count-games SOURCE=data/downloads/lichess_2024-06.pgn.zst
make prepare SOURCE=data/downloads/lichess_2024-06.pgn.zst CHUNK_SIZE=700000
make train
make serve
make eval-round-robin TIERS=1000,1500,2000 GAMES=100
make eval-stockfish TIER=1500
make check
```

- `make prepare` reads the source once and builds every bucket in chunks
  (`data/processed/bucket_<name>_<chunk>.npz`).
- `make train` trains each bucket through its chunks in order, resuming each
  from the previous one. It writes `training/checkpoints/bucket_<name>_<chunk>/`.
  Chunks that already have a `best.pt` are skipped.
- `make serve` uses the highest chunk per bucket. Open http://127.0.0.1:8000.

### Makefile variables

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `SOURCE` | `count-games`, `prepare` | required | `.pgn` / `.pgn.zst` dump |
| `CHUNK_SIZE` | `prepare` | required | Games scanned per chunk |
| `BUCKETS` | `prepare`, `train` | all | e.g. `1000,1500` |
| `MAX_CHUNKS` | `prepare` | all | Stop after N chunks |
| `EPOCHS` | `train` | `10` | Epochs per chunk |
| `TIER` | `eval-stockfish` | required | Tier to benchmark |
| `TIERS` | `eval-round-robin` | required | Comma-separated tiers |
| `STOCKFISH_ELOS` | `eval-stockfish` | package default | `UCI_Elo` levels |
| `GAMES` | eval targets | package default | Games per level or pairing |
| `WORKERS` | eval targets | CPU count | Parallel worker processes |

`make check` runs the format check, ruff, strict mypy and all tests. `make
test`, `make lint`, `make format` and `make typecheck` run them individually.

## Package docs

[`shared`](shared/README.md), [`data`](data/README.md),
[`training`](training/README.md), [`backend`](backend/README.md),
[`eval`](eval/README.md).