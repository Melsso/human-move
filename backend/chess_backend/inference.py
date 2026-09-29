from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import chess
import numpy as np
import torch
from chess_shared import (
    TIER_CONFIGS,
    Candidate,
    TierConfig,
    encode_board,
    legal_move_mask,
    masked_softmax,
    pick_move,
)
from chess_training.model import MaiaPolicyNet, load_checkpoint, model_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "training" / "checkpoints"
_CHUNK_DIR_RE = re.compile(r"^bucket_(.+)_(\d+)$")


class TierNotFoundError(Exception):
    def __init__(self, tier: str, available: list[str]) -> None:
        self.tier = tier
        self.available = available
        super().__init__(f"unknown tier '{tier}', available: {available}")


class CheckpointMeta(TypedDict):
    epoch: int
    val_top1: float
    val_top3: float


@torch.no_grad()
def policy_probs(model: MaiaPolicyNet, board: chess.Board) -> np.ndarray:
    tensor = torch.from_numpy(encode_board(board)).unsqueeze(0)
    logits = model(tensor).squeeze(0).numpy()
    return masked_softmax(logits, legal_move_mask(board))


@dataclass(frozen=True)
class Player:
    tier: str
    config: TierConfig
    model: MaiaPolicyNet

    def choose_move(
        self,
        board: chess.Board,
        top_k: int = 5,
        rng: np.random.Generator | None = None,
    ) -> tuple[chess.Move, list[Candidate]]:
        probs = policy_probs(self.model, board)
        return pick_move(board, probs, self.config, top_k=top_k, rng=rng)


class ModelRegistry:
    def __init__(self, checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR) -> None:
        self.checkpoints_dir = checkpoints_dir
        self._loaded: dict[str, tuple[MaiaPolicyNet, CheckpointMeta]] = {}
        self._players: dict[str, Player] = {}
        self._lock = threading.Lock()

    def discover_checkpoints(self) -> dict[str, Path]:
        if not self.checkpoints_dir.exists():
            return {}
        latest: dict[str, tuple[int, Path]] = {}
        for directory in self.checkpoints_dir.iterdir():
            match = _CHUNK_DIR_RE.match(directory.name)
            best = directory / "best.pt"
            if match is None or not best.exists():
                continue
            name, chunk = match.group(1), int(match.group(2))
            if name not in latest or chunk > latest[name][0]:
                latest[name] = (chunk, best)
        return {name: path for name, (_, path) in latest.items()}

    def available_tiers(self) -> list[str]:
        checkpoints = self.discover_checkpoints()
        return [
            tier
            for tier, config in TIER_CONFIGS.items()
            if config.checkpoint in checkpoints
        ]

    def get_player(self, tier: str) -> Player:
        with self._lock:
            if tier in self._players:
                return self._players[tier]
            available = self.available_tiers()
            if tier not in available:
                raise TierNotFoundError(tier, available)
            config = TIER_CONFIGS[tier]
            model, _ = self._load(config.checkpoint)
            player = Player(tier, config, model)
            self._players[tier] = player
            return player

    def get_meta(self, tier: str) -> CheckpointMeta:
        player = self.get_player(tier)
        return self._loaded[player.config.checkpoint][1]

    def _load(self, checkpoint_name: str) -> tuple[MaiaPolicyNet, CheckpointMeta]:
        if checkpoint_name in self._loaded:
            return self._loaded[checkpoint_name]
        checkpoint = load_checkpoint(self.discover_checkpoints()[checkpoint_name])
        model = model_from_checkpoint(checkpoint)
        model.eval()
        meta = CheckpointMeta(
            epoch=checkpoint["epoch"],
            val_top1=checkpoint["val_top1"],
            val_top3=checkpoint["val_top3"],
        )
        self._loaded[checkpoint_name] = (model, meta)
        return model, meta
