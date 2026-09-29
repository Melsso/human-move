import chess
import numpy as np
from chess_shared import (
    TIER_CONFIGS,
    TierConfig,
    legal_move_mask,
    perspective_move_to_index,
    pick_move,
)


def uniform_probs(board: chess.Board) -> np.ndarray:
    mask = legal_move_mask(board)
    return mask / mask.sum()


def test_greedy_without_search_picks_policy_argmax() -> None:
    board = chess.Board()
    target = chess.Move.from_uci("g1f3")
    probs = np.zeros(legal_move_mask(board).shape)
    probs[perspective_move_to_index(target, board.turn)] = 1.0
    move, _ = pick_move(board, probs, TierConfig(checkpoint="x"))
    assert move == target


def test_candidates_are_sorted_legal_and_limited_to_top_k() -> None:
    board = chess.Board()
    move, candidates = pick_move(
        board, uniform_probs(board), TierConfig(checkpoint="x"), top_k=3
    )
    assert move in board.legal_moves
    assert len(candidates) == 3
    probs = [c["prob"] for c in candidates]
    assert probs == sorted(probs, reverse=True)
    for c in candidates:
        assert chess.Move.from_uci(c["uci"]) in board.legal_moves


def test_temperature_sampling_is_reproducible_with_seeded_rng() -> None:
    board = chess.Board()
    probs = uniform_probs(board)
    config = TierConfig(checkpoint="x", temperature=1.0)
    first = pick_move(board, probs, config, rng=np.random.default_rng(7))[0]
    second = pick_move(board, probs, config, rng=np.random.default_rng(7))[0]
    assert first == second
    assert first in board.legal_moves


def test_search_finds_mate_in_one() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    config = TierConfig(
        checkpoint="x", search_depth=2, search_weight=1.0, search_top_k=100
    )
    move, _ = pick_move(board, uniform_probs(board), config)
    assert move == chess.Move.from_uci("a1a8")


def test_search_captures_hanging_queen() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    config = TierConfig(
        checkpoint="x", search_depth=2, search_weight=1.0, search_top_k=100
    )
    move, _ = pick_move(board, uniform_probs(board), config)
    assert move == chess.Move.from_uci("d1d5")


def test_search_with_temperature_returns_legal_move_and_keeps_board() -> None:
    board = chess.Board()
    board.push_san("e4")
    fen, stack = board.fen(), list(board.move_stack)
    config = TierConfig(
        checkpoint="x", temperature=0.5, search_depth=2, search_weight=1.0
    )
    move, _ = pick_move(
        board, uniform_probs(board), config, rng=np.random.default_rng(0)
    )
    assert move in board.legal_moves
    assert board.fen() == fen
    assert list(board.move_stack) == stack


def test_tier_configs_match_intended_setup() -> None:
    assert TIER_CONFIGS["1000"].search_depth == 0
    assert TIER_CONFIGS["1500"].search_depth > 0
    assert TIER_CONFIGS["2000"].checkpoint == TIER_CONFIGS["1500"].checkpoint
    assert TIER_CONFIGS["2000"].search_depth > TIER_CONFIGS["1500"].search_depth
    assert TIER_CONFIGS["2000"].search_weight > TIER_CONFIGS["1500"].search_weight
