import random

import chess
from chess_shared.search import evaluate, rank_candidates


def _random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    for _ in range(n):
        board = chess.Board()
        for _ in range(rng.randint(4, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def _best(board: chess.Board, depth: int, **kw) -> chess.Move:
    moves = list(board.legal_moves)
    scores = rank_candidates(board, moves, [0.0] * len(moves), 1.0, depth, **kw)
    return moves[max(range(len(moves)), key=lambda i: scores[i])]


def test_start_position_is_symmetric() -> None:
    assert evaluate(chess.Board()) == 10


def test_evaluate_is_color_symmetric_under_mirror() -> None:
    for board in _random_positions(50):
        assert evaluate(board) == evaluate(board.mirror())


def test_finds_mate_in_one() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert _best(board, depth=2).uci() == "a1a8"


def test_captures_free_queen() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert _best(board, depth=2).uci() == "d1d5"


def test_search_does_not_mutate_board() -> None:
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        board.push_san(san)
    fen, stack = board.fen(), list(board.move_stack)
    _best(board, depth=3)
    assert board.fen() == fen
    assert list(board.move_stack) == stack


def test_node_cap_falls_back_to_policy_scores() -> None:
    board = chess.Board()
    moves = list(board.legal_moves)
    log_probs = [-float(i) for i in range(len(moves))]
    scores = rank_candidates(board, moves, log_probs, 1.0, 4, max_nodes=1)
    assert scores == log_probs
