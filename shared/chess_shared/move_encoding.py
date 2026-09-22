from __future__ import annotations

import chess
import numpy as np

NUM_PROMOTIONS = 5

_PROMOTION_TO_OFFSET = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
_OFFSET_TO_PROMOTION = {v: k for k, v in _PROMOTION_TO_OFFSET.items()}

NUM_MOVES = 64 * 64 * NUM_PROMOTIONS


def _to_mover_perspective(move: chess.Move, turn: chess.Color) -> chess.Move:
    if turn == chess.WHITE:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def move_to_index(move: chess.Move) -> int:
    promo_offset = _PROMOTION_TO_OFFSET[move.promotion]
    return (move.from_square * 64 + move.to_square) * NUM_PROMOTIONS + promo_offset


def index_to_move(index: int) -> chess.Move:
    combo, promo_offset = divmod(index, NUM_PROMOTIONS)
    from_square, to_square = divmod(combo, 64)
    return chess.Move(
        from_square, to_square, promotion=_OFFSET_TO_PROMOTION[promo_offset]
    )


def perspective_move_to_index(move: chess.Move, turn: chess.Color) -> int:
    return move_to_index(_to_mover_perspective(move, turn))


def perspective_index_to_move(index: int, board: chess.Board) -> chess.Move:
    mover_move = index_to_move(index)
    return _to_mover_perspective(mover_move, board.turn)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        mask[perspective_move_to_index(move, board.turn)] = 1.0
    return mask
