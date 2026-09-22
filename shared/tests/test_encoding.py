from __future__ import annotations

import random

import chess
import numpy as np
from chess_shared.board_encoding import (
    LAST_MOVE_FROM_PLANE,
    LAST_MOVE_TO_PLANE,
    MOVER_KINGSIDE_CASTLE_PLANE,
    MOVER_QUEENSIDE_CASTLE_PLANE,
    NUM_PLANES,
    OPPONENT_KINGSIDE_CASTLE_PLANE,
    OPPONENT_QUEENSIDE_CASTLE_PLANE,
    REPETITION_PLANE,
    _square_to_row_col,
    encode_board,
)
from chess_shared.move_encoding import (
    NUM_MOVES,
    index_to_move,
    legal_move_mask,
    move_to_index,
    perspective_index_to_move,
    perspective_move_to_index,
)


def test_starting_position_shape_and_piece_counts() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor.shape == (NUM_PLANES, 8, 8)
    assert tensor[:6].sum() == 16
    assert tensor[6:12].sum() == 16


def test_mover_pieces_occupy_the_same_slot_regardless_of_color() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor[5, 7, 4] == 1.0
    assert tensor[11, 0, 4] == 1.0

    board.push_san("e4")
    tensor = encode_board(board)
    assert tensor[5, 7, 4] == 1.0
    assert tensor[11, 0, 4] == 1.0


def test_castling_rights_full_at_game_start() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert np.all(tensor[MOVER_KINGSIDE_CASTLE_PLANE] == 1.0)
    assert np.all(tensor[MOVER_QUEENSIDE_CASTLE_PLANE] == 1.0)
    assert np.all(tensor[OPPONENT_KINGSIDE_CASTLE_PLANE] == 1.0)
    assert np.all(tensor[OPPONENT_QUEENSIDE_CASTLE_PLANE] == 1.0)


def test_castling_rights_after_one_side_castles() -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("O-O")
    tensor = encode_board(board)
    assert np.all(tensor[OPPONENT_KINGSIDE_CASTLE_PLANE] == 0.0)
    assert np.all(tensor[OPPONENT_QUEENSIDE_CASTLE_PLANE] == 0.0)
    assert np.all(tensor[MOVER_KINGSIDE_CASTLE_PLANE] == 1.0)
    assert np.all(tensor[MOVER_QUEENSIDE_CASTLE_PLANE] == 1.0)


def test_no_last_move_planes_at_game_start() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor[LAST_MOVE_FROM_PLANE].sum() == 0.0
    assert tensor[LAST_MOVE_TO_PLANE].sum() == 0.0


def test_last_move_planes_reflect_opponents_move_mirrored() -> None:
    board = chess.Board()
    board.push_san("e4")
    tensor = encode_board(board)

    mirrored_from_row, mirrored_from_col = _square_to_row_col(
        chess.square_mirror(chess.E2)
    )
    mirrored_to_row, mirrored_to_col = _square_to_row_col(chess.square_mirror(chess.E4))
    assert tensor[LAST_MOVE_FROM_PLANE, mirrored_from_row, mirrored_from_col] == 1.0
    assert tensor[LAST_MOVE_TO_PLANE, mirrored_to_row, mirrored_to_col] == 1.0
    assert tensor[LAST_MOVE_FROM_PLANE].sum() == 1.0
    assert tensor[LAST_MOVE_TO_PLANE].sum() == 1.0


def test_repetition_plane_not_set_on_fresh_position() -> None:
    board = chess.Board()
    tensor = encode_board(board)
    assert np.all(tensor[REPETITION_PLANE] == 0.0)


def test_repetition_plane_set_when_position_recurs() -> None:
    board = chess.Board()
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    tensor = encode_board(board)
    assert np.all(tensor[REPETITION_PLANE] == 1.0)


def test_canonical_move_index_roundtrip_covers_every_promotion() -> None:
    promotions = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    seen_indices = set()
    for from_sq in (chess.A7, chess.E2, chess.H7):
        for to_sq in (chess.A8, chess.E4, chess.H8):
            for promo in promotions:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                idx = move_to_index(move)
                assert 0 <= idx < NUM_MOVES
                assert idx not in seen_indices, f"index collision for {move.uci()}"
                seen_indices.add(idx)
                decoded = index_to_move(idx)
                assert decoded == move


def test_different_promotion_pieces_get_different_indices() -> None:
    base = (chess.A7, chess.A8)
    indices = {
        promo: move_to_index(chess.Move(*base, promotion=promo))
        for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    }
    assert len(set(indices.values())) == 4


def test_perspective_move_index_roundtrip_for_random_legal_moves() -> None:
    random.seed(0)
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            board = chess.Board()
        legal = list(board.legal_moves)
        move = random.choice(legal)
        idx = perspective_move_to_index(move, board.turn)
        assert 0 <= idx < NUM_MOVES
        decoded = perspective_index_to_move(idx, board)
        assert decoded == move
        board.push(move)


def test_legal_move_mask_matches_board_legal_moves_white_to_move() -> None:
    board = chess.Board()
    mask = legal_move_mask(board)
    assert mask.sum() == 20
    for move in board.legal_moves:
        assert mask[perspective_move_to_index(move, board.turn)] == 1.0


def test_legal_move_mask_matches_board_legal_moves_black_to_move() -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    mask = legal_move_mask(board)
    assert mask.sum() == len(list(board.legal_moves))
    for move in board.legal_moves:
        assert mask[perspective_move_to_index(move, board.turn)] == 1.0


def test_promotion_type_is_preserved_not_collapsed_to_queen() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        promo_move = chess.Move(chess.A7, chess.A8, promotion=promo)
        idx = perspective_move_to_index(promo_move, board.turn)
        decoded = perspective_index_to_move(idx, board)
        assert decoded.promotion == promo, (
            f"expected {chess.piece_name(promo)} promotion to round-trip, "
            f"got {chess.piece_name(decoded.promotion)}"
        )
        assert board.is_legal(decoded)


def test_promotion_roundtrips_for_black_too() -> None:
    board = chess.Board("7k/8/8/8/8/8/p7/7K b - - 0 1")
    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        promo_move = chess.Move(chess.A2, chess.A1, promotion=promo)
        idx = perspective_move_to_index(promo_move, board.turn)
        decoded = perspective_index_to_move(idx, board)
        assert decoded == promo_move
        assert board.is_legal(decoded)
