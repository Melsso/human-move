# chess-data

Downloads and filters Lichess games into rating-bucketed `(board, move)`
training tensors. Depends on `chess-shared` for board/move encoding — see
the root README for install order.

## 1. Get a PGN dump

Lichess publishes every rated game ever played, one file per month, at
https://database.lichess.org/. Files are named like
`lichess_db_standard_rated_2024-06.pgn.zst` and are large — a recent month
is commonly 30-40GB compressed. You do **not** need to decompress it; the
pipeline reads directly from the `.zst` stream.

Pick ONE recent month to start with (more than enough games). Download it
into `data/downloads/`:

```bash
cd data/downloads
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst
```

If you don't want a 30GB download while you're still developing the
pipeline, Lichess also hosts small sample files for testing — search
"lichess database" for the current sample links, or just cap
`--max-chunks` low (see below) and let the script stop early; the
streaming reader means it only pulls as much of the file as it actually
reads, but a plain `curl` download still fetches the whole thing, so for
quick iteration truncate a downloaded file locally instead:

```bash
# grab roughly the first 200MB to iterate on the pipeline quickly
curl -r 0-200000000 -o sample.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst
```

(A truncated `.zst` will error at the point it cuts off — that's fine, the
script processes everything up to that point and stops.)

## 2. Run the pipeline

One command processes the WHOLE file in one pass, building every rating
bucket (1000/1500/2000) at once — see `chess_data/prepare.py`'s docstring
for why one pass for every bucket beats one pass per bucket. Commands
below assume you're in `data/` — `uv run --package` finds the workspace
root automatically):

```bash
# optional: see how many games are in the file first, to help pick a chunk size
uv run --package chess-data python -m chess_data.count_games \
    downloads/lichess_db_standard_rated_2024-06.pgn.zst

uv run --package chess-data python -m chess_data.prepare \
    downloads/lichess_db_standard_rated_2024-06.pgn.zst \
    processed \
    --chunk-size 700000
```

That builds `processed/bucket_1000_1.npz`, `bucket_1000_2.npz`, ...,
`bucket_1500_1.npz`, `bucket_2000_1.npz`, etc — one file per (bucket,
chunk) pair, chunked so no single file gets too large to comfortably
train on (see the root README's "Processing a full Lichess dump"
section for why). Use `--buckets 1000,2000` to only build specific
buckets, and `--max-chunks N` to stop early for a quick sample run
instead of processing the whole file.

`--chunk-size` caps how many games get scanned *per chunk*, not how many
get kept — a chunk with very few matches for a bucket is normal, not an
error; the run just keeps going and produces more chunks until the source
is exhausted.

Each output `.npz` has two arrays:
- `boards`: `(N, 18, 8, 8)` float32 — see `chess_shared.board_encoding` for
  the plane layout
- `moves`: `(N,)` int64 — indices into the 4096-way move space, see
  `chess_shared.move_encoding`

## 3. Sanity-check the output before training on it

Before handing this to the training package, worth eyeballing:
- The per-chunk position counts `chess_data.prepare` prints as it runs,
  and the final summary across all chunks — if a bucket's total is much
  smaller than expected, its elo range (`chess_data/brackets.py`) or
  `max_elo_gap` filter is probably too strict for the source file, and
  the summary prints an explicit warning if a bucket ended up with zero
  positions across the entire run.
- Spot-check a handful of `(board, move)` pairs by decoding a board tensor
  back into a readable position and confirming the labeled move is
  actually legal there (a good exercise, and a good gut check that the
  data pipeline and `chess_shared` encoding agree with each other).

## Design decisions worth knowing about

- The rating buckets themselves (names and elo ranges) are defined once in
  `chess_data/brackets.py`, not passed as CLI flags — edit `DEFAULT_BRACKETS`
  there if you want different ranges or an extra tier; every other file
  that needs to know "what are our buckets" (this pipeline, the Makefile,
  the training/backend naming convention) reads from that one place, so it
  can't drift out of sync.
- Games are filtered to **rated, standard-variant, human-vs-human** only
  (see `chess_data/filters.py`) — bot games are excluded so the model
  learns from humans, not engines.
- Players must be within `max_elo_gap` (default 200, set per-bracket in
  `chess_data/brackets.py`) of each other — otherwise you're mostly
  training on "how a 1000 plays when they're about to get crushed by a
  2200," which isn't representative.
- The first `skip_first_n_plies` (default 6) moves of every game are
  dropped — early opening moves are largely memorized book theory and look
  similar across every rating band, so they carry little signal about
  skill level and would just dilute the dataset.
- Each game contributes at most `max_positions_per_game` (default 40)
  positions, so no single very long game dominates the dataset.

## Full command reference

```bash
uv run --package chess-data python -m chess_data.count_games <pgn_path>

uv run --package chess-data python -m chess_data.prepare <pgn_path> <out_dir> --chunk-size N [options]

uv run --package chess-data pytest data/tests/ -v
```

Or, from the repo root, `make count-games SOURCE=...` and `make prepare
SOURCE=... CHUNK_SIZE=...` wrap the same commands (see the root README).

### `chess_data.count_games`

| Argument | Meaning |
|---|---|
| `pgn_path` (positional) | Path to `.pgn` or `.pgn.zst` |

### `chess_data.prepare`

| Argument | Default | Meaning |
|---|---|---|
| `pgn_path` (positional) | — | Path to `.pgn` or `.pgn.zst` source dump |
| `out_dir` (positional) | — | Directory to write `bucket_<name>_<chunk>.npz` files into |
| `--chunk-size` | required | Games scanned per chunk, across ALL buckets together |
| `--buckets` | all (`1000,1500,2000`) | Comma-separated bucket names to build — see `chess_data/brackets.py` |
| `--max-chunks` | none (process until exhausted) | Stop after this many chunks — useful for a quick sample run |
| `--max-positions-per-game` | `40` | Cap on positions extracted per game |
| `--skip-first-n-plies` | `6` | Opening plies skipped per game |

## Running tests

```bash
uv run --package chess-data pytest data/tests/ -v
# or, from the repo root:
make test
```