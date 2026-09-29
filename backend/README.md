# chess-backend

FastAPI service that serves moves for each rating tier. It also serves the
static board UI (`static/index.html`, using `chessboard.js` and `chess.js`)
from the same process, so there is no CORS setup.

## Modules

- **`main.py`** — the FastAPI app and routes.
- **`inference.py`**
  - `ModelRegistry` finds checkpoints and loads them lazily.
  - `get_player(tier)` returns a `Player`, which combines the tier's model with
    its `TierConfig` from `chess_shared.tiers`.
  - `Player.choose_move(board, top_k, rng)` runs the network and then
    `chess_shared.pick_move`. Search, weight and temperature therefore come
    from the tier config, never from the request.
- **`engine_eval.py`** — optional Stockfish evaluation for the UI's eval bar.
  It degrades to `available: false` when Stockfish isn't installed.
- **`schemas.py`** — pydantic request and response models.
- **`static/index.html`** — the board UI.

## Commands

```bash
make serve
uv run --package chess-backend pytest backend/tests/ -v
```

Then open `http://127.0.0.1:8000`.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | `{"status": "ok"}` |
| `/api/tiers` | GET | Available tiers with the `epoch`, `val_top1` and `val_top3` of the checkpoint each uses |
| `/api/move` | POST | `{fen, move_uci, tier}` applies your move, computes the reply, and returns both positions plus the network's top-5 candidate moves with probabilities |
| `/api/eval` | POST | `{fen}` returns `{available, score_cp, mate}` from Stockfish, from white's perspective |
| `/` | GET | The board UI |

`fen` in `/api/move` is the position **before** your move. Requests are
stateless and the server re-validates legality. The candidate list shows the
network's own probabilities. On searching tiers the chosen move can differ
from the top candidate, because the search re-ranks the moves.

## Tiers

Tiers are defined in `shared/chess_shared/tiers.py`, not discovered from
directory names. `/api/tiers` lists a configured tier only if its checkpoint
exists. Checkpoints live in `training/checkpoints/bucket_<name>_<chunk>/`, and
the highest chunk's `best.pt` is used for each name, since training is
cumulative across chunks. Directories with other names are ignored.

| Tier | Checkpoint | Behaviour |
|---|---|---|
| 1000 | `bucket_1000_*` | policy only, temperature 0.3 |
| 1500 | `bucket_1500_*` | search depth 3, weight 1 |
| 2000 | `bucket_1500_*` | same model, search depth 6, weight 4 |

Models load on first use and are cached. Tiers 1500 and 2000 share one loaded
model. Searching tiers are slower, and depth 6 can take seconds per move.
Requests run in FastAPI's thread pool.

## Stockfish (optional, eval bar only)

Install Stockfish (`brew install stockfish`, `apt install stockfish`) or set
`STOCKFISH_PATH` to the binary. Without it the eval toggle disables itself.