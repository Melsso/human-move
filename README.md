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
├── shared/     # board/move encoding — the ONE source of truth, imported by data, training, and backend
├── data/       # downloads Lichess PGNs, filters by rating, converts to training tensors
├── training/   # model definition, training loop, checkpoints, elo calibration (not built yet)
├── backend/    # inference API — loads a checkpoint per tier, serves moves (not built yet)
└── frontend/   # board UI (not built yet)
```

`shared` exists because board/move encoding has to be byte-for-byte
identical between training and inference — if it drifts, the model gets
fed different input at serving time than it was trained on, and fails
silently (still returns *a* move, just usually a bad one, with nothing that
looks like an error).

## Setup

Each package is an installable local package (own `pyproject.toml`). Since
they're workspace-local, install `shared` first, then the others in
editable mode, all into one venv:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e ./shared
pip install -e ./data
# pip install -e ./training   # once it exists
# pip install -e ./backend    # once it exists
```

Editable installs mean changes to `shared/chess_shared/*.py` are picked up
immediately by `data`, `training`, and `backend` without reinstalling
anything.

## Status

- [x] `shared` — board encoding (18x8x8 planes) + move encoding (4096-way
      from/to classification), fully tested (`shared/tests/`)
- [x] `data` — PGN filtering + streaming `.zst` preprocessing into rating
      buckets, fully tested (`data/tests/`). See `data/README.md` for how
      to actually pull a Lichess dump and run it.
- [ ] `training` — model + training loop + elo calibration eval
- [ ] `backend` — FastAPI inference service
- [ ] `frontend` — board UI

## Running tests

```bash
cd shared && python -m pytest tests/ -v
cd ../data && python -m pytest tests/ -v
```
