"""
Discovers trained checkpoints, loads/caches them, and computes a single
model move for a position. Kept separate from `main.py`'s route handlers
so this logic is directly unit-testable without going through HTTP.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import chess
import torch
from chess_shared import (
    NUM_MOVES,
    NUM_PLANES,
    encode_board,
    legal_move_mask,
    masked_softmax,
    select_move_index,
    top_k_moves,
)
from chess_shared.move_encoding import index_to_move
from chess_training.model import MaiaPolicyNet

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "training" / "checkpoints"


class TierNotFoundError(Exception):
    def __init__(self, tier: str, available: list[str]) -> None:
        self.tier = tier
        self.available = available
        super().__init__(f"unknown tier '{tier}', available: {available}")


class Candidate(TypedDict):
    uci: str
    san: str
    prob: float


class CheckpointMeta(TypedDict):
    epoch: int
    val_top1: float
    val_top3: float


class ModelRegistry:
    """
    Scans `checkpoints_dir` for `<tier_name>/best.pt` and lazily loads/caches
    each model on first use. Tier names are whatever you actually named your
    checkpoint directories (e.g. "bucket_1000", "sample_bucket_2000") -- we
    deliberately don't hardcode "1000"/"1500"/"2000" anywhere, since that
    naming has already proven inconsistent in practice.
    """

    def __init__(self, checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR) -> None:
        self.checkpoints_dir = checkpoints_dir
        self._models: dict[str, MaiaPolicyNet] = {}
        self._meta: dict[str, CheckpointMeta] = {}

    def discover_tiers(self) -> dict[str, Path]:
        tiers: dict[str, Path] = {}
        if not self.checkpoints_dir.exists():
            return tiers
        for d in sorted(self.checkpoints_dir.iterdir()):
            best = d / "best.pt"
            if best.exists():
                tiers[d.name] = best
        return tiers

    def get_model(self, tier: str) -> MaiaPolicyNet:
        if tier in self._models:
            return self._models[tier]

        available = self.discover_tiers()
        if tier not in available:
            raise TierNotFoundError(tier, sorted(available))

        checkpoint = torch.load(available[tier], map_location="cpu", weights_only=True)
        model = MaiaPolicyNet(
            in_planes=NUM_PLANES,
            num_moves=NUM_MOVES,
            num_blocks=checkpoint["num_blocks"],
            num_filters=checkpoint["num_filters"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._models[tier] = model
        self._meta[tier] = {
            "epoch": checkpoint["epoch"],
            "val_top1": checkpoint["val_top1"],
            "val_top3": checkpoint["val_top3"],
        }
        return model

    def get_meta(self, tier: str) -> CheckpointMeta:
        self.get_model(tier)
        return self._meta[tier]


@torch.no_grad()
def compute_model_move(
    model: MaiaPolicyNet, board: chess.Board, temperature: float = 0.0, top_k: int = 5
) -> tuple[chess.Move, list[Candidate]]:
    """
    Returns (chosen_move, top_k_candidates). Candidates are included in the
    response mainly so a UI can show "the model considered these other
    moves too" -- useful for judging whether it's playing sensibly, same
    idea as chess_training.play's terminal output.
    """
    board_tensor = torch.from_numpy(encode_board(board)).unsqueeze(0)
    logits = model(board_tensor).squeeze(0).numpy()

    mask = legal_move_mask(board)
    probs = masked_softmax(logits, mask)

    candidates: list[Candidate] = []
    for idx, prob in top_k_moves(probs, k=top_k):
        move = index_to_move(idx, board)
        candidates.append({"uci": move.uci(), "san": board.san(move), "prob": prob})

    move_idx = select_move_index(probs, temperature=temperature)
    chosen_move = index_to_move(move_idx, board)
    return chosen_move, candidates
