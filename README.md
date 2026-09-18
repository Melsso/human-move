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
`make prepare-1`, etc. self-heal regardless of what ran before them. If
you ever do need a raw command, always add `--all-packages`.

When `training`/`backend` get built, add them to `[tool.uv.workspace]
members` in the root `pyproject.toml` and run `uv sync --all-packages`
again.

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
- [x] `backend` — FastAPI service (`chess_backend`) that discovers every
      `training/checkpoints/<tier>/best.pt`, loads/caches models lazily,
      and exposes `/api/tiers` + `/api/move`. Also serves the static
      frontend at `/` from the same process (no CORS setup needed for a
      local project like this). Fully tested (`backend/tests/`) with
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

The tier dropdown is populated from whatever's actually sitting in
`training/checkpoints/*/best.pt` — no hardcoded tier names, so it reflects
whatever you've actually trained, under whatever directory names you
happened to use.

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

make prepare-1 / prepare-2 / prepare-3   # rebuild the sample rating buckets
make train-1000 / train-1500 / train-2000   # train each rating tier
make play CHECKPOINT=training/checkpoints/bucket_1000/best.pt   # play a game vs a checkpoint (terminal)
make serve            # FastAPI backend + board UI at http://127.0.0.1:8000
```

`train-*` targets assume your real, full-size buckets are at
`data/processed/bucket_1000.npz` etc (not the small `sample_bucket_*.npz`
files `prepare-*` produces) — adjust the paths in the Makefile if yours are
named differently.

## Running tests

```bash
make test
# or scoped to one package:
uv run --package chess-shared pytest shared/tests/ -v
uv run --package chess-data pytest data/tests/ -v
uv run --package chess-training pytest training/tests/ -v
```