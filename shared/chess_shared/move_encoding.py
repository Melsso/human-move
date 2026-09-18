"""
Maps chess moves to/from a fixed integer index space so the model's output
layer can be an ordinary (NUM_MOVES,) classification head.

DESIGN DECISION (worth knowing, revisit later if it matters):
We encode a move purely as (from_square, to_square) -> 64*64 = 4096 classes.
Underpromotions (promoting to rook/bishop/knight instead of queen) collapse
onto the SAME index as the plain queen-promotion move, since they share the
same from/to squares. We resolve the ambiguity at decode time by always
assuming queen promotion. Underpromotions are rare in human play (mostly a
stalemate/perpetual-check trick) and this keeps the action space small and
easy to reason about for a first version. If you want to represent them
properly later, the standard fix is to add extra planes for underpromotion
directions (this is what AlphaZero's 4672-way action space does) -- but
that's a deliberate upgrade to make later, not a bug to fix now.
"""

import chess
import numpy as np

NUM_MOVES = 64 * 64  # 4096


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Turn a predicted index back into a concrete move on `board`.
    Needs the board to know whether this is a pawn move landing on the back
    rank (and therefore needs promotion=QUEEN set, or python-chess will
    reject it as illegal).
    """
    from_square, to_square = divmod(index, 64)
    piece = board.piece_at(from_square)
    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_square)
        if to_rank in (0, 7):
            promotion = chess.QUEEN
    return chess.Move(from_square, to_square, promotion=promotion)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    (NUM_MOVES,) float32 array: 1.0 at indices that are legal right now,
    0.0 elsewhere. Multiply this into the model's raw logits (as -inf where
    mask==0, or renormalize probabilities) before sampling so the model can
    never output an illegal move, no matter what it thinks.
    """
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index(move)] = 1.0
    return mask
