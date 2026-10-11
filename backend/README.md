# chess-backend

FastAPI service that serves moves for each tier. It also serves the static board UI from the same process, so there is no CORS setup and no build step.

## Modules

- **`main.py`**: the FastAPI app and routes.
- **`inference.py`**
  - `ModelRegistry` finds checkpoints and loads them lazily.
  - `get_player(tier)` returns a `Player`, which combines the tier's model with its `TierConfig` from `chess_shared.tiers`.
  - `Player.choose_move(board, top_k, rng)` runs the network and then `chess_shared.pick_move`, so search, weight and temperature come from the tier config and never from the request.
- **`helpers.py`**: `build_board`, which validates the FEN and rebuilds the position from a move history.
- **`engine_eval.py`**: optional Stockfish evaluation for the UI's eval bar. It degrades to `available: false` when Stockfish isn't installed.
- **`schemas.py`**: pydantic request and response models.
- **`static/`**: `index.html`, `styles.css`, `app.js` and `vendor/` (chess.js, chessboard.js, jQuery and the piece images, all served locally).

## Commands

```bash
make serve                                # uses training/checkpoints
CHECKPOINTS_DIR=weights make serve        # uses downloaded weights
uv run --package chess-backend pytest backend/tests/ -v
```

Then open `http://127.0.0.1:8000`.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | `{"status": "ok"}` |
| `/api/tiers` | GET | Available tiers with the `epoch`, `val_top1` and `val_top3` of the checkpoint each uses |
| `/api/move` | POST | Applies the human move (if given), computes the model's reply, and returns both positions plus the network's top-5 candidates |
| `/api/eval` | POST | `{fen}` returns `{available, score_cp, mate}` from Stockfish, from white's perspective |
| `/` | GET | The board UI |

### `POST /api/move`

```json
{
  "fen": "position before the human move",
  "history": ["e2e4", "e7e5"],
  "move_uci": "g1f3",
  "tier": "1500"
}
```

- `fen` and `tier` are required.
- `move_uci` is optional. Omit it to let the model move first, which is how the UI starts a game when you play black.
- `history` is the list of UCI moves from the standard starting position to `fen`. It is optional, but without it the server can't see repetition: the network's repetition input and the search's repetition detection stay blank and threefold draws can't be claimed. If given, the server replays it and returns `400` when it contains an illegal move or doesn't end at `fen`.
- Draws are claimed the same way the evaluation does (`claim_draw=True`), so threefold repetition and the 50-move rule end the game.

The response has `human_move_san` (null when no move was sent), `fen_after_human`, `model_move_uci`, `model_move_san`, `fen_after_model`, `game_over`, `result` and `top_candidates`. The server is stateless and re-validates every move. The candidate list shows the network's own probabilities. On searching tiers the chosen move can differ from the top candidate, because the search re-ranks the moves.

Errors: `400` for a bad FEN, move or history, `404` for an unknown tier or one without a checkpoint.

## Tiers and checkpoints

Tiers are defined in `shared/chess_shared/tiers.py`, not discovered from directory names. `/api/tiers` lists a configured tier only if its checkpoint exists. Checkpoints are looked up in `CHECKPOINTS_DIR` (default `training/checkpoints`) as `bucket_<name>_<chunk>/best.pt`, and the highest chunk wins per name, since training is cumulative. Directories with other names are ignored.

| Tier | Checkpoint | Behaviour |
|---|---|---|
| 1000 | `bucket_1000_*` | policy only, temperature 0.3 |
| 1500 | `bucket_2000_*` | search depth 3, weight 1 |
| 2000 | `bucket_2000_*` | same model, search depth 6, weight 4 |

Models load on first use and are cached. Tiers 1500 and 2000 share one loaded model. Searching tiers are slower, and depth 6 can take seconds per move.
Requests run in FastAPI's thread pool.

## The UI

Play as white, black or random, with drag or click-to-move, legal-move and last-move highlights, a promotion picker, take back, resign, move history with review navigation (arrow keys, Home and End), the model's candidate moves, and an optional eval bar. Late replies from a previous game are discarded, and a rejected move is rolled back locally.

## Stockfish (optional, eval bar only)

Install Stockfish (`brew install stockfish`, `apt install stockfish`) or set `STOCKFISH_PATH` to the binary. Without it the eval toggle disables itself.