from __future__ import annotations

import random

import chess

DEFAULT_OPENING_PLIES = 8


def random_opening(seed: int, plies: int = DEFAULT_OPENING_PLIES) -> chess.Board:
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        board.push(rng.choice(legal_moves))
    return board


def classify_result(raw_result: str, subject_is_white: bool) -> str:
    if subject_is_white:
        mapping = {"1-0": "WIN", "0-1": "LOSS"}
    else:
        mapping = {"0-1": "WIN", "1-0": "LOSS"}
    return mapping.get(raw_result, "DRAW")
