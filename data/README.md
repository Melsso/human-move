# chess-data

Filters Lichess games into rating-bucketed `(board, move)` training tensors.
It depends on `chess-shared` for the encodings.

## 1. Get a PGN dump

Lichess publishes monthly dumps at https://database.lichess.org/ (30-40GB compressed). The pipeline reads `.zst` directly, with no decompression step.

```bash
cd data/downloads
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst
```

For quick iteration, download a truncated prefix (`curl -r 0-200000000 -o sample.pgn.zst <url>`). The reader processes everything up to the cut-off.

## 2. Run the pipeline

One pass over the file builds every bucket at once.

```bash
make count-games SOURCE=data/downloads/lichess_db_standard_rated_2024-06.pgn.zst
make prepare SOURCE=data/downloads/lichess_db_standard_rated_2024-06.pgn.zst CHUNK_SIZE=700000
```

Output is `data/processed/bucket_<name>_<chunk>.npz`, one file per (bucket, chunk). A bucket with no matching games in a chunk gets no file for it.
`--chunk-size` is the number of games scanned per chunk, not the number kept.
Each `.npz` contains:

- `boards`: `(N, 20, 8, 8)` float32, from the side to move's perspective (`chess_shared.board_encoding`)
- `moves`: `(N,)` int64, an index into the 20480-way mover-perspective move space (`chess_shared.move_encoding`)
- `game_ids`: `(N,)` int64, the source game's global index, used by training to split train and validation by game

## Design decisions

- Buckets are defined once in `chess_data/brackets.py`:

  | Bucket | Average Elo range |
  |---|---|
  | `1000` | 900-1100 |
  | `2000` | 1900-2100 |

  Players must be within max_elo_gap (200) of each other. There is no 1500 bucket: tiers 1500 and 2000 both use the 2000 model.
- Only **rated Rapid**, standard-variant, human games are kept. Bots and abandoned or rules-infraction games are dropped.
- The first `--skip-first-n-plies` (default 10) plies of each game are skipped, since opening moves are mostly memorised theory.
- At most `--max-positions-per-game` (default 40) positions are sampled per game, evenly spaced.
- Each bucket's arrays are staged in growable memory-mapped temp files under `out_dir/.prepare-tmp-*`, so RAM use stays flat. They are removed when the chunk finishes.
- Only rated Rapid (including Rapid tournament games), standard-variant, human games are kept. Blitz, bullet and classical games are dropped because their quality was too poor, along with bots and abandoned or rules-infraction games.

## Modules

- `prepare.py` — chunked pipeline and CLI.
- `filters.py` — game filtering and the multi-bucket PGN visitor.
- `brackets.py` — bucket definitions.
- `pgn_io.py` — opens `.pgn` and `.pgn.zst` streams.
- `array_store.py` — growable memory-mapped array used for staging.
- `count_games.py` — counts games in a dump.

## Commands

```bash
uv run --package chess-data python -m chess_data.count_games <pgn_path>
uv run --package chess-data python -m chess_data.prepare <pgn_path> <out_dir> --chunk-size N [options]
uv run --package chess-data pytest data/tests/ -v
```

### `chess_data.prepare` arguments

| Argument | Default | Meaning |
|---|---|---|
| `pgn_path`, `out_dir` | — | Source dump and output directory |
| `--chunk-size` | required | Games scanned per chunk, across all buckets |
| `--buckets` | all | Comma-separated bucket names |
| `--max-chunks` | until exhausted | Stop early for a sample run |
| `--max-positions-per-game` | `40` | Position cap per game |
| `--skip-first-n-plies` | `10` | Opening plies skipped |

Check the per-chunk position counts as it runs. If a bucket ends with zero positions, its Elo range in `brackets.py` never matched anything.