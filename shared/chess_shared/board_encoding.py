from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 20

_PIECE_TYPE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

MOVER_KINGSIDE_CASTLE_PLANE = 12
MOVER_QUEENSIDE_CASTLE_PLANE = 13
OPPONENT_KINGSIDE_CASTLE_PLANE = 14
OPPONENT_QUEENSIDE_CASTLE_PLANE = 15
EN_PASSANT_PLANE = 16
LAST_MOVE_FROM_PLANE = 17
LAST_MOVE_TO_PLANE = 18
REPETITION_PLANE = 19


def _square_to_row_col(square: int) -> tuple[int, int]:
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def _perspective_square(square: int, mirror: bool) -> int:
    return chess.square_mirror(square) if mirror else square


def encode_board(board: chess.Board) -> np.ndarray:
    mirror = board.turn == chess.BLACK
    mover = board.turn
    opponent = not mover

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane_idx = _PIECE_TYPE_TO_PLANE[piece.piece_type]
        if piece.color != mover:
            plane_idx += 6
        row, col = _square_to_row_col(_perspective_square(square, mirror))
        planes[plane_idx, row, col] = 1.0

    if board.has_kingside_castling_rights(mover):
        planes[MOVER_KINGSIDE_CASTLE_PLANE, :, :] = 1.0
    if board.has_queenside_castling_rights(mover):
        planes[MOVER_QUEENSIDE_CASTLE_PLANE, :, :] = 1.0
    if board.has_kingside_castling_rights(opponent):
        planes[OPPONENT_KINGSIDE_CASTLE_PLANE, :, :] = 1.0
    if board.has_queenside_castling_rights(opponent):
        planes[OPPONENT_QUEENSIDE_CASTLE_PLANE, :, :] = 1.0

    if board.ep_square is not None:
        row, col = _square_to_row_col(_perspective_square(board.ep_square, mirror))
        planes[EN_PASSANT_PLANE, row, col] = 1.0

    if board.move_stack:
        last_move = board.move_stack[-1]
        from_row, from_col = _square_to_row_col(
            _perspective_square(last_move.from_square, mirror)
        )
        to_row, to_col = _square_to_row_col(
            _perspective_square(last_move.to_square, mirror)
        )
        planes[LAST_MOVE_FROM_PLANE, from_row, from_col] = 1.0
        planes[LAST_MOVE_TO_PLANE, to_row, to_col] = 1.0

    if board.is_repetition(2):
        planes[REPETITION_PLANE, :, :] = 1.0

    return planes
