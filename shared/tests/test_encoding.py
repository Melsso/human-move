import random

import chess
import numpy as np
from chess_shared.board_encoding import NUM_PLANES, encode_board
from chess_shared.move_encoding import (
    NUM_MOVES,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


def test_starting_position_shape_and_side_to_move() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor.shape == (NUM_PLANES, 8, 8)
    assert np.all(tensor[12] == 1.0)
    assert tensor[:12].sum() == 32


def test_side_to_move_flips_after_a_move() -> None:
    board = chess.Board()
    board.push_san("e4")
    tensor = encode_board(board)
    assert np.all(tensor[12] == 0.0)


def test_castling_rights_reflected() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert np.all(tensor[13] == 1.0)
    assert np.all(tensor[14] == 1.0)
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("O-O")
    tensor = encode_board(board)
    assert np.all(tensor[13] == 0.0)
    assert np.all(tensor[14] == 0.0)


def test_move_index_roundtrip_for_random_legal_moves() -> None:
    random.seed(0)
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            board = chess.Board()
        legal = list(board.legal_moves)
        move = random.choice(legal)
        idx = move_to_index(move)
        assert 0 <= idx < NUM_MOVES
        decoded = index_to_move(idx, board)
        assert decoded.from_square == move.from_square
        assert decoded.to_square == move.to_square
        board.push(move)


def test_legal_move_mask_matches_board_legal_moves() -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    mask = legal_move_mask(board)
    assert mask.sum() == len(list(board.legal_moves))
    for move in board.legal_moves:
        assert mask[move_to_index(move)] == 1.0


def test_promotion_move_decodes_as_queen_promotion() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    promo_move = chess.Move(chess.A7, chess.A8, promotion=chess.QUEEN)
    idx = move_to_index(promo_move)
    decoded = index_to_move(idx, board)
    assert decoded.promotion == chess.QUEEN
    assert board.is_legal(decoded)
