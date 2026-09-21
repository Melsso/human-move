import chess
from chess_eval.common import classify_result, random_opening


def test_random_opening_produces_legal_position() -> None:
    board = random_opening(seed=42, plies=8)
    assert board.is_valid()
    assert board.fullmove_number >= 1


def test_random_opening_is_deterministic_for_same_seed() -> None:
    board1 = random_opening(seed=42, plies=8)
    board2 = random_opening(seed=42, plies=8)
    assert board1.fen() == board2.fen()


def test_random_opening_differs_across_seeds() -> None:
    board1 = random_opening(seed=1, plies=8)
    board2 = random_opening(seed=2, plies=8)
    assert board1.fen() != board2.fen()


def test_random_opening_zero_plies_is_starting_position() -> None:
    board = random_opening(seed=42, plies=0)
    assert board.fen() == chess.Board().fen()


def test_random_opening_stops_early_on_game_over() -> None:
    board = random_opening(seed=7, plies=1000)
    assert board.is_valid()


def test_classify_result_win_as_white() -> None:
    assert classify_result("1-0", subject_is_white=True) == "WIN"


def test_classify_result_loss_as_white() -> None:
    assert classify_result("0-1", subject_is_white=True) == "LOSS"


def test_classify_result_win_as_black() -> None:
    assert classify_result("0-1", subject_is_white=False) == "WIN"


def test_classify_result_loss_as_black() -> None:
    assert classify_result("1-0", subject_is_white=False) == "LOSS"


def test_classify_result_draw_regardless_of_color() -> None:
    assert classify_result("1/2-1/2", subject_is_white=True) == "DRAW"
    assert classify_result("1/2-1/2", subject_is_white=False) == "DRAW"


def test_classify_result_unfinished_game_counts_as_draw() -> None:
    assert classify_result("*", subject_is_white=True) == "DRAW"
