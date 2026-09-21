from __future__ import annotations

import numpy as np


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
