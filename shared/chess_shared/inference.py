from __future__ import annotations

import chess
import numpy as np
import torch
from chess_training.model import MaiaPolicyNet

from chess_shared.board_encoding import encode_board
from chess_shared.model_registery import Candidate
from chess_shared.move_encoding import legal_move_mask, perspective_index_to_move
from chess_shared.search import rank_candidates


def masked_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    if not np.any(legal_mask > 0):
        raise ValueError(
            "legal_mask has no legal moves set (all zeros) -- can't be a real "
            "chess position, check the board wasn't already game-over."
        )
    masked_logits = np.where(legal_mask > 0, logits, -np.inf)
    shifted = masked_logits - np.max(masked_logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def select_move_index(
    probs: np.ndarray,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    if temperature == 0.0:
        return int(np.argmax(probs))

    rng = rng or np.random.default_rng()
    logits = np.log(np.clip(probs, 1e-12, None)) / temperature
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    tempered = exp / exp.sum()
    return int(rng.choice(len(tempered), p=tempered))


def top_k_moves(probs: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    k = min(k, int(np.count_nonzero(probs)))
    top_indices = np.argpartition(probs, -k)[-k:]
    top_indices = top_indices[np.argsort(-probs[top_indices])]
    return [(int(i), float(probs[i])) for i in top_indices]


@torch.no_grad()
def compute_model_move(
    model: MaiaPolicyNet,
    board: chess.Board,
    temperature: float = 0.0,
    top_k: int = 5,
    search_depth: int = 0,
    search_weight: float = 1.0,
    search_top_k: int = 6,
    search_max_nodes: int = 0,
) -> tuple[chess.Move, list[Candidate]]:
    board_tensor = torch.from_numpy(encode_board(board)).unsqueeze(0)
    logits = model(board_tensor).squeeze(0).numpy()

    mask = legal_move_mask(board)
    probs = masked_softmax(logits, mask)

    candidates: list[Candidate] = []
    for idx, prob in top_k_moves(probs, k=top_k):
        move = perspective_index_to_move(idx, board)
        candidates.append({"uci": move.uci(), "san": board.san(move), "prob": prob})

    if search_depth <= 0:
        move_idx = select_move_index(probs, temperature=temperature)
        return perspective_index_to_move(move_idx, board), candidates

    pool = top_k_moves(probs, k=search_top_k)
    pool_moves = [perspective_index_to_move(idx, board) for idx, _ in pool]
    log_probs = [float(np.log(max(p, 1e-12))) for _, p in pool]

    combined = np.array(
        rank_candidates(
            board,
            pool_moves,
            log_probs,
            search_weight,
            search_depth,
            exact=temperature > 0.0,
            max_nodes=search_max_nodes,
        )
    )

    if temperature <= 0.0:
        choice = int(np.argmax(combined))
    else:
        z = combined / temperature
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        choice = int(np.random.default_rng().choice(len(p), p=p))

    return pool_moves[choice], candidates
