from __future__ import annotations

import random
from pathlib import Path

import chess
from chess_backend.inference import ModelRegistry, Player
from chess_shared import TIER_CONFIGS

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


def subject_to_move(board: chess.Board, subject_is_white: bool) -> bool:
    return (board.turn == chess.WHITE) == subject_is_white


def push_legal(board: chess.Board, move: chess.Move) -> None:
    if move not in board.legal_moves:
        raise RuntimeError(
            f"model produced illegal move {move} on position {board.fen()}"
        )
    board.push(move)


def require_tiers(tiers: list[str]) -> None:
    unknown = [t for t in tiers if t not in TIER_CONFIGS]
    if unknown:
        raise ValueError(f"unknown tier(s) {unknown}, available: {list(TIER_CONFIGS)}")


def load_player(tier: str, checkpoints_dir: str | None) -> Player:
    registry = (
        ModelRegistry(Path(checkpoints_dir)) if checkpoints_dir else ModelRegistry()
    )
    return registry.get_player(tier)
