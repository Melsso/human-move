"""
Converts a python-chess Board into a fixed-size tensor a neural net can consume.

This file is imported by BOTH the data-preparation package (to build training
examples) and the backend (to build the input for a live inference call).
That's the whole point of the `shared` package: if this encoding ever changes,
it changes in exactly one place, and training + serving can never silently
drift out of sync with each other.

Layout: 18 planes of 8x8.
  planes 0-5   : white pawn, knight, bishop, rook, queen, king
  planes 6-11  : black pawn, knight, bishop, rook, queen, king
  plane  12    : all 1s if it's white's turn to move, else all 0s
  plane  13    : white kingside castling right (all 1s or all 0s)
  plane  14    : white queenside castling right
  plane  15    : black kingside castling right
  plane  16    : black queenside castling right
  plane  17    : en-passant target square (1 at that square, else 0)

Row 0 of the array is always rank 8 (black's back rank) and column 0 is
always the a-file, regardless of whose turn it is. We deliberately do NOT
flip the board for black-to-move -- the "side to move" plane (12) tells the
network whose turn it is instead. Simpler to reason about and debug.
"""

import chess
import numpy as np

NUM_PLANES = 18

_PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def _square_to_row_col(square: int) -> tuple[int, int]:
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def encode_board(board: chess.Board) -> np.ndarray:
    """Return a (NUM_PLANES, 8, 8) float32 tensor representing `board`."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane_idx = _PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        row, col = _square_to_row_col(square)
        planes[plane_idx, row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        row, col = _square_to_row_col(board.ep_square)
        planes[17, row, col] = 1.0

    return planes
