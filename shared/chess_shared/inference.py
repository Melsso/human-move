"""
Turns raw model logits into an actual legal move index. Lives in `shared`
rather than in `training` or `backend` specifically because both need this
exact same logic -- a quick terminal script for playing against a
checkpoint, and later the real backend, should never be able to drift
apart on how a move actually gets chosen from the model's output.

The model itself has NO idea which moves are legal (see MaiaPolicyNet's
docstring in chess_training/model.py) -- it just outputs a score for all
4096 possible from/to combinations. Masking to the legal ones and turning
that into a move is entirely this module's job.
"""

from __future__ import annotations

import numpy as np


def masked_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """
    Softmax over `logits`, restricted to positions where `legal_mask` is 1.
    Illegal-move positions get probability exactly 0.0, not just "small" --
    we set their logit to -inf before the softmax rather than zeroing the
    probability afterward, so the remaining probabilities still sum to 1.
    """
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
    """
    Pick a single move index from a probability distribution.

    temperature=0.0 (default): greedy -- always the highest-probability
    legal move. This is what you want for a "play at rating X" opponent:
    Maia's own approach is to disable search/sampling entirely and just
    take the policy network's top choice.

    temperature>0.0: sample proportionally (softmax-tempered), useful for
    introducing variety (e.g. self-play data generation, or deliberately
    making the model feel less robotic) but NOT what you want for a
    calibrated "plays like a 1000/2000" opponent, since it makes the model
    play weaker than its training data would suggest.
    """
    if temperature == 0.0:
        return int(np.argmax(probs))

    rng = rng or np.random.default_rng()
    logits = np.log(np.clip(probs, 1e-12, None)) / temperature
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    tempered = exp / exp.sum()
    return int(rng.choice(len(tempered), p=tempered))


def top_k_moves(probs: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Return the k highest-probability (move_index, probability) pairs."""
    k = min(k, int(np.count_nonzero(probs)))
    top_indices = np.argpartition(probs, -k)[-k:]
    top_indices = top_indices[np.argsort(-probs[top_indices])]
    return [(int(i), float(probs[i])) for i in top_indices]
