# chess-shared

Board/move encoding and masked-inference logic — the single source of
truth imported by `data`, `training`, and `backend`. If board/move
encoding ever drifted between the code that prepares training data and
the code that serves live moves, the model would get fed different input
at inference time than it was trained on, and fail silently (it would
still return *a* move, just usually a bad one, with nothing that looks
like an error). Everything in this package exists to make that
impossible.

## What's here

- **`board_encoding.py`** — `encode_board(board) -> (18, 8, 8) float32
  array`. 12 planes for piece type × color, 1 side-to-move plane, 4
  castling-rights planes, 1 en-passant plane.
- **`move_encoding.py`** — `move_to_index(move) -> int` /
  `index_to_move(index, board) -> chess.Move`, a 4096-way (`from_square ×
  to_square`) move space, plus `legal_move_mask(board) -> (4096,) float32
  array`. Underpromotions collapse onto the same index as queen
  promotion — see the module docstring for why that's an acceptable
  simplification, not a bug.
- **`inference.py`** — `masked_softmax(logits, legal_mask)`,
  `select_move_index(probs, temperature=0.0)`, `top_k_moves(probs, k=5)`.
  Turns a model's raw (4096,) logits into an actual legal move. Lives here
  rather than in `training` or `backend` specifically because both need
  this exact same logic — `chess_training.play` and the real backend must
  never be able to drift apart on how a move actually gets chosen.

## Commands

Run from anywhere in the workspace (`uv run --package` finds the root
automatically), or via the Makefile from the repo root:

```bash
uv run --package chess-shared pytest shared/tests/ -v
# or, from the repo root:
make test          # runs every package's tests, including this one
```

## Design notes worth knowing

- `NUM_PLANES` (18) and `NUM_MOVES` (4096) are the two constants
  everything else in the project is shaped around — a model's input/output
  layer, a `.npz` bucket's `boards`/`moves` array shapes, a backend
  response's move indices, all trace back to these.
- Board tensors are **not** flipped for black-to-move; the side-to-move
  plane carries that information instead. Row 0 is always rank 8
  (black's back rank), regardless of whose turn it is.