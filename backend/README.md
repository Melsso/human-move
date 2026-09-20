# chess-backend

FastAPI inference service: discovers trained checkpoints, loads/caches
them, and serves moves against them. Also serves the static frontend
(`static/index.html`, a drag-and-drop board using `chessboard.js` +
`chess.js`) from the same process — deliberately no separate frontend
dev server, so there's no CORS configuration to get wrong for a
local-only project like this.

## What's here

- **`main.py`** — the FastAPI app and routes.
- **`inference.py`** — `ModelRegistry`: discovers checkpoints under
  `training/checkpoints/`, loads/caches models lazily, computes a single
  move for a position. Kept separate from `main.py` so it's directly
  unit-testable without going through HTTP.
- **`schemas.py`** — pydantic request/response models.
- **`static/index.html`** — the board UI.

## Commands

```bash
uv run --package chess-backend uvicorn chess_backend.main:app --reload --port 8000
# or, from the repo root:
make serve

uv run --package chess-backend pytest backend/tests/ -v
# or:
make test
```

Then open `http://127.0.0.1:8000`.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | `{"status": "ok"}` — liveness check |
| `/api/tiers` | GET | List available tiers (`tier`, `epoch`, `val_top1`, `val_top3` per tier) |
| `/api/move` | POST | `{fen, move_uci, tier, temperature?}` → applies your move, computes the model's reply, returns both resulting positions plus its top-5 candidate moves with probabilities |
| `/` | GET | The board UI (`static/index.html`) |

`POST /api/move` expects `fen` to be the position **before** your move
(it applies `move_uci` itself and checks legality from there) — a common
frontend bug is capturing the FEN *after* a client-side move already
mutated local game state; see the frontend's `onDrop` handler for how
that's avoided.

## Tier discovery

`ModelRegistry.discover_tiers()` scans `training/checkpoints/` and groups
directories named `bucket_<name>_<chunk>` (what chunked training via
`make train` produces) by `<name>`, exposing only the **highest** chunk
number's checkpoint per bucket — the most-trained one, since
`chess_training.train`'s `--resume-from` is cumulative across chunks. So
you get one clean `"1000"`/`"1500"`/`"2000"` tier per bucket, not a
separate entry per chunk. A directory that doesn't match that naming
pattern (a one-off custom name) still shows up as its own standalone
tier, unchanged.

Models are loaded lazily (on first request for that tier) and cached —
the first request for a given tier pays the checkpoint-load cost, every
request after that doesn't.

## Design notes worth knowing

- The backend is the single source of truth for game state and legality.
  Every `/api/move` call is stateless and self-contained (you always send
  the full current position) — no server-side session state, matching the
  REST style already used elsewhere in this project.
- A move's legality is double-checked server-side even though the
  frontend's `chess.js` also validates client-side — never trust the
  client for anything that affects what gets written to a response.