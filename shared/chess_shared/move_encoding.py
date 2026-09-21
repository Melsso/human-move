import chess
import numpy as np

NUM_MOVES = 64 * 64


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    from_square, to_square = divmod(index, 64)
    piece = board.piece_at(from_square)
    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_square)
        if to_rank in (0, 7):
            promotion = chess.QUEEN
    return chess.Move(from_square, to_square, promotion=promotion)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index(move)] = 1.0
    return mask
