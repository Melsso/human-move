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
"lichess database" for the current sample links, or just cap `--max-games`
low (see below) and let the script stop early; the streaming reader means
it only pulls as much of the file as it actually reads, but a plain `curl`
download still fetches the whole thing, so for quick iteration truncate a
downloaded file locally instead:

```bash
# grab roughly the first 200MB to iterate on the pipeline quickly
curl -r 0-200000000 -o sample.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst
```

(A truncated `.zst` will error at the point it cuts off — that's fine, the
script processes everything up to that point and stops.)

## 2. Run the pipeline

One bucket per rating tier, run separately (commands below assume you're
in `data/` — `uv run --package` finds the workspace root automatically):

```bash
uv run --package chess-data python -m chess_data.prepare \
    downloads/lichess_db_standard_rated_2024-06.pgn.zst \
    processed/bucket_1000.npz \
    --min-elo 900 --max-elo 1100 \
    --max-games 300000

uv run --package chess-data python -m chess_data.prepare \
    downloads/lichess_db_standard_rated_2024-06.pgn.zst \
    processed/bucket_1500.npz \
    --min-elo 1400 --max-elo 1600 \
    --max-games 300000

uv run --package chess-data python -m chess_data.prepare \
    downloads/lichess_db_standard_rated_2024-06.pgn.zst \
    processed/bucket_2000.npz \
    --min-elo 1900 --max-elo 2100 \
    --max-games 300000
```

`--max-games` caps how many games get *scanned* (not kept) — use it to
keep early iterations fast. Drop it for a full run once you trust the
pipeline.

Each run produces one `.npz` with two arrays:
- `boards`: `(N, 18, 8, 8)` float32 — see `chess_shared.board_encoding` for
  the plane layout
- `moves`: `(N,)` int64 — indices into the 4096-way move space, see
  `chess_shared.move_encoding`

## 3. Sanity-check the output before training on it

Before handing this to the training package, worth eyeballing:
- `boards.shape[0]` — how many positions did you actually get? If it's
  much smaller than expected, your elo range or `max_elo_gap` filter is
  probably too strict for the month/bucket you picked.
- Spot-check a handful of `(board, move)` pairs by decoding a board tensor
  back into a readable position and confirming the labeled move is
  actually legal there (a good exercise, and a good gut check that the
  data pipeline and `chess_shared` encoding agree with each other).

## Design decisions worth knowing about

- Games are filtered to **rated, standard-variant, human-vs-human** only
  (see `chess_data/filters.py`) — bot games are excluded so the model
  learns from humans, not engines.
- Players must be within `max_elo_gap` (default 200) of each other —
  otherwise you're mostly training on "how a 1000 plays when they're about
  to get crushed by a 2200," which isn't representative.
- The first `skip_first_n_plies` (default 6) moves of every game are
  dropped — early opening moves are largely memorized book theory and look
  similar across every rating band, so they carry little signal about
  skill level and would just dilute the dataset.
- Each game contributes at most `max_positions_per_game` (default 40)
  positions, so no single very long game dominates the dataset.