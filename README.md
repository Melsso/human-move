# chess-ai

A from-scratch chess engine trained on real human games, bucketed by rating,
so it plays *like* a ~1000-rated player or a ~2000-rated player rather than
just being weakened-Stockfish. Approach follows the same idea as
[Maia Chess](https://www.maiachess.com/): supervised learning on human games
filtered by rating, no search, no self-play RL.

Fully local, nothing deployed anywhere (yet).

## Layout

```
chess-ai/
├── shared/     # board/move encoding + inference logic — the ONE source of truth
├── data/       # downloads Lichess PGNs, filters by rating, converts to training tensors
├── training/   # model definition, training loop, checkpoints, terminal play script
├── backend/    # FastAPI inference API + serves the static frontend
└── (backend/static/ is the frontend — a single-page board UI, no separate build step)
```

`shared` exists because board/move encoding has to be byte-for-byte
identical between training and inference — if it drifts, the model gets
fed different input at serving time than it was trained on, and fails
silently (still returns *a* move, just usually a bad one, with nothing that
looks like an error).

## Setup

This is a [`uv`](https://docs.astral.sh/uv/) workspace — one `uv.lock` at
the root pins exact versions for every package, and `shared` is wired up as
a normal dependency of `data`/`training`/`backend` via `[tool.uv.sources]`
rather than a fragile relative path.

```bash
# install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync --all-packages
```

That creates `.venv/` at the root with everything installed, `chess-shared`
included in editable mode — so edits to `shared/chess_shared/*.py` are
picked up immediately by every other package, no reinstall needed.

Run a package's code or tests via `uv run --package <name> ...`:

```bash
uv run --package chess-shared pytest shared/tests/ -v
uv run --package chess-data pytest data/tests/ -v
uv run --package chess-training pytest training/tests/ -v
uv run --package chess-backend pytest backend/tests/ -v
uv run --package chess-data python -m chess_data.prepare ...
```

Adding a dependency to one package: `uv add --package chess-data <pkg>` —
this updates that package's `pyproject.toml` *and* re-resolves the shared
`uv.lock`, so the lockfile never drifts out of sync with what's declared.

**Always use `make <target>` rather than raw `uv run`/`uv sync` commands.**
A bare `uv sync` (no `--all-packages`) only syncs the workspace *root*
project and silently uninstalls `chess-data`/`chess-shared` and their
dependencies — that's expected `uv` behavior for a multi-package workspace,
not a bug, but it's easy to trigger by habit. Every Makefile target
depends on `sync` (which always passes `--all-packages`), so `make test`,
`make prepare`, etc. self-heal regardless of what ran before them. If
you ever do need a raw command, always add `--all-packages`.

## Status

- [x] `shared` — board encoding (18x8x8 planes) + move encoding (4096-way
      from/to classification), fully tested (`shared/tests/`)
- [x] `data` — PGN filtering + streaming `.zst` preprocessing into rating
      buckets, fully tested (`data/tests/`). See `data/README.md` for how
      to actually pull a Lichess dump and run it.
- [x] `training` — 6-block/64-filter residual CNN policy network (matches
      the original Maia Chess paper's architecture), training loop with
      MPS/CUDA/CPU auto-detection, checkpointing, fully tested
      (`training/tests/`). `chess_training.play` lets you play a game
      against a checkpoint right in the terminal. Elo calibration against
      Stockfish is not built yet.
- [x] `backend` — FastAPI service (`chess_backend`) that discovers
      checkpoints under `training/checkpoints/`, grouping chunked-training
      directories (`bucket_<name>_<chunk>`) by name and exposing only the
      most-trained (highest) chunk per bucket as a clean tier, loads/caches
      models lazily, and exposes `/api/tiers` + `/api/move`. Also serves the
      static frontend at `/` from the same process (no CORS setup needed
      for a local project like this). Fully tested (`backend/tests/`) with
      synthetic checkpoints, and smoke-tested end to end (boot the server,
      hit every route with curl) before being handed off.
- [x] `frontend` — single-page board UI at `backend/static/index.html`
      (chessboard.js + chess.js off cdnjs, no build step). Drag pieces to
      move, pick a tier from the dropdown, see the model's top-5
      candidate moves and probabilities after each of its turns.

## Playing against your trained models

```bash
make serve
# then open http://127.0.0.1:8000 in a browser
```

The tier dropdown is populated from whatever's actually sitting under
`training/checkpoints/` — `bucket_<name>_<chunk>` directories (what
chunked training via `make train` produces) are grouped by `<name>`, so
you see one clean "1000"/"1500"/"2000" entry per bucket pointing at its
most-trained chunk, not a confusing list of every individual chunk. A
directory that doesn't match that pattern still shows up as its own
tier, unchanged.

## Day-to-day commands

Everything goes through the Makefile — every target depends on `sync`
(always `uv sync --all-packages`), so it's never possible to run a command
against a stale or partially-installed environment:

```bash
make check          # format check + lint + strict mypy + full test suite
make test            # just the tests
make lint / format / typecheck   # individually
make reset            # rm -rf .venv + uv cache clean + fresh sync -- use this if
                       # you ever hit "ModuleNotFoundError: No module named 'chess_X'"
                       # after a sync; it's a known venv-corruption pattern, not a
                       # code bug, and this is the reliable fix

make count-games SOURCE=data/downloads/lichess_2024-06.pgn.zst   # see below
make prepare SOURCE=... CHUNK_SIZE=700000                         # see below
make train                                                         # see below
make play CHECKPOINT=training/checkpoints/bucket_1000_1/best.pt   # play a game vs a checkpoint (terminal)
make serve            # FastAPI backend + board UI at http://127.0.0.1:8000
```

### Makefile variables

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `SOURCE` | `count-games`, `prepare` | required | Path to the `.pgn`/`.pgn.zst` source dump |
| `CHUNK_SIZE` | `prepare` | required | Games scanned per chunk, across all buckets together |
| `BUCKETS` | `prepare`, `train` | all buckets | Comma-separated bucket names, e.g. `1000,2000` — see `data/chess_data/brackets.py` |
| `MAX_CHUNKS` | `prepare` | none (process until exhausted) | Stop after this many chunks — useful for a quick sample run |
| `EPOCHS` | `train` | `10` | Epochs trained per chunk |
| `CHECKPOINT` | `play` | required | Path to a `.pt` checkpoint |

## Processing a full Lichess dump (chunked, all buckets in one pass)

A full monthly Lichess dump is 30GB+ compressed, and one giant bucket
`.npz` built from it can easily be several GB in RAM — `.npz` can never
be memory-mapped (not even uncompressed; see `chess_training/dataset.py`'s
docstring), so the whole thing loads into RAM every time you train on it.
So instead of one giant file per bucket, `chess_data.prepare` builds
every bucket in small chunks, and `chess_training.train` trains through
them incrementally, picking up where the last chunk left off.

**One pass processes every bucket together.** The expensive part of
handling a 30GB+ dump is reading and decompressing it at all — once
you're streaming through it, checking a game's headers against three elo
ranges instead of one costs almost nothing extra. So `make prepare` reads
the source file exactly ONCE and builds the 1000/1500/2000 buckets
simultaneously, rather than three separate full passes over the same
data.

```bash
# optional: see how many games are in the dump, to help pick a chunk size
make count-games SOURCE=data/downloads/lichess_2024-06.pgn.zst

# processes the WHOLE file in one command, building every bucket at once,
# in chunks of 700,000 games each
make prepare SOURCE=data/downloads/lichess_2024-06.pgn.zst CHUNK_SIZE=700000

# only build specific buckets (still one pass over the source):
make prepare SOURCE=... CHUNK_SIZE=700000 BUCKETS=1000,2000
```

That produces `data/processed/bucket_1000_1.npz`, `bucket_1000_2.npz`,
..., `bucket_1500_1.npz`, `bucket_2000_1.npz`, etc — one file per
(bucket, chunk) pair, for every chunk until the source is exhausted. A
bucket with zero matching games in a particular chunk simply doesn't get
a file for that chunk (not an error — some chunks may have none of a
narrow elo range by chance).

**Only delete the source file once every bucket you want has actually
been built from it.** Since `make prepare` already builds every bucket in
one run, that's normally just "after this one command finishes" — no
need to keep it around across separate per-bucket runs the way an
earlier, less efficient design would have required.

```bash
make train                          # trains every bucket found under data/processed/
make train BUCKETS=1000,2000        # just these two
make train EPOCHS=5                 # epochs per chunk (default 10)
```

`make train` auto-discovers every bucket's chunk files, sorts them
numerically, and trains through them in order — chunk 2 resumes from
chunk 1's checkpoint automatically (architecture and cumulative epoch
numbering are read from the checkpoint itself, not re-specified), same
for chunk 3 from chunk 2, and so on. It prints each bucket's final
checkpoint path when done; point `make play`/`make serve` at that.

## Running tests

```bash
make test
# or scoped to one package:
uv run --package chess-shared pytest shared/tests/ -v
uv run --package chess-data pytest data/tests/ -v
uv run --package chess-training pytest training/tests/ -v
uv run --package chess-backend pytest backend/tests/ -v
```

## Per-package documentation

Each package has its own README with a full command/flag reference:
[`shared/README.md`](shared/README.md), [`data/README.md`](data/README.md),
[`training/README.md`](training/README.md),
[`backend/README.md`](backend/README.md).