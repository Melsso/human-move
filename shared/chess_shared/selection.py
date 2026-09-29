from __future__ import annotations

import math
from typing import TypedDict

import chess
import numpy as np

from chess_shared.inference import select_move_index, top_k_moves
from chess_shared.move_encoding import perspective_index_to_move
from chess_shared.search import rank_candidates
from chess_shared.tiers import TierConfig

PROB_FLOOR = 1e-12


class Candidate(TypedDict):
    uci: str
    san: str
    prob: float


def _candidates(board: chess.Board, probs: np.ndarray, k: int) -> list[Candidate]:
    result: list[Candidate] = []
    for idx, prob in top_k_moves(probs, k=k):
        move = perspective_index_to_move(idx, board)
        result.append(Candidate(uci=move.uci(), san=board.san(move), prob=prob))
    return result


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def pick_move(
    board: chess.Board,
    probs: np.ndarray,
    config: TierConfig,
    top_k: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[chess.Move, list[Candidate]]:
    candidates = _candidates(board, probs, top_k)

    if config.search_depth <= 0:
        index = select_move_index(probs, config.temperature, rng)
        return perspective_index_to_move(index, board), candidates

    pool = top_k_moves(probs, k=config.search_top_k)
    moves = [perspective_index_to_move(idx, board) for idx, _ in pool]
    log_probs = [math.log(max(prob, PROB_FLOOR)) for _, prob in pool]
    scores = rank_candidates(
        board,
        moves,
        log_probs,
        config.search_weight,
        config.search_depth,
        exact=config.temperature > 0.0,
        max_nodes=config.search_max_nodes,
    )
    index = select_move_index(_softmax(np.array(scores)), config.temperature, rng)
    return moves[index], candidates
