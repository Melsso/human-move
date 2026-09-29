# chess-shared

Torch-free core used by every other package: board and move encoding, masked
inference math, the hand-written search, and the tier configuration. If
encoding ever drifted between data preparation and serving, the model would
silently receive different input than it was trained on, so everything that
must agree lives here.

## Modules

- **`board_encoding.py`** — `encode_board(board) -> (20, 8, 8) float32`, always
  from the side to move's perspective (mirrored when black is to move).
  Planes 0-5 own pieces, 6-11 opponent pieces, 12-13 own kingside/queenside
  castling, 14-15 opponent castling, 16 en passant, 17-18 last move from/to,
  19 repetition. `NUM_PLANES = 20`.
- **`move_encoding.py`** — 20480-way move space
  (`from × to × 5 promotion offsets`), `NUM_MOVES = 20480`, mover-perspective
  conversions (`perspective_move_to_index`, `perspective_index_to_move`) and
  `legal_move_mask(board)`.
- **`inference.py`** — `masked_softmax`, `select_move_index` (greedy at
  temperature 0, sampled otherwise), `top_k_moves`.
- **`search.py`** — `rank_candidates(board, moves, log_probs, weight, depth,
  exact, max_nodes)`: alpha-beta search used to re-rank the network's moves.
- **`tiers.py`** — `TierConfig` and `TIER_CONFIGS`, the only place tiers are
  defined.
- **`selection.py`** — `pick_move(board, probs, config, top_k, rng)`: turns
  the network's legal-move probabilities into a move plus the top candidates,
  according to a tier's config.
- **`engine.py`** — `find_stockfish()` (`STOCKFISH_PATH`, then `PATH`).

## Tiers

| Tier | Checkpoint | Temperature | Search |
|---|---|---|---|
| 1000 | 1000 | 0.3 | none |
| 1500 | 1500 | 0 | depth 3, weight 1 |
| 2000 | 1500 | 0 | depth 6, weight 4 |

`TierConfig` fields: `checkpoint` (which trained model), `temperature`,
`search_depth` (0 disables search), `search_weight`, `search_top_k` (default
6) and `search_max_nodes` (default 0, unlimited). Edit `TIER_CONFIGS` to change
a tier; the backend and eval both read it.

## How a move is chosen

Without search (`search_depth == 0`) the move is sampled from the network's
legal-move distribution at the tier's temperature (argmax at 0).

With search, the top `search_top_k` policy moves are re-scored as:

```
score = log(policy_probability) + search_weight * eval_in_pawns
```

`eval_in_pawns` comes from a negamax search of the position after that move,
to `search_depth` plies counting the root move as ply 1, clamped to ±100
pawns. The search uses iterative deepening, a transposition table,
killer/history ordering, null-move pruning, late-move reductions, check
extensions, quiescence with delta pruning, repetition detection, and a static
evaluation (piece-square tables, mobility, pawn structure, rook files, king
safety, bishop pair). Without temperature, root moves after the first are
searched with a bound derived from the best score so far. With temperature
above 0 every score is computed exactly. If `search_max_nodes` is hit, the
scores of the last completed depth are used. The final move is the argmax of
the scores, or a sample from `softmax(score / temperature)`.

## Tests

```bash
uv run --package chess-shared pytest shared/tests/ -v
```