from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from chess_training.model import MaiaPolicyNet


def write_synthetic_checkpoint(path: Path, epoch: int = 3) -> None:
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=2, num_filters=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_top1": 0.31,
            "val_top3": 0.55,
            "num_blocks": 2,
            "num_filters": 8,
            "in_planes": NUM_PLANES,
            "num_moves": NUM_MOVES,
        },
        path,
    )


@pytest.fixture
def make_checkpoints(tmp_path: Path) -> Callable[..., Path]:
    def build(dir_names: list[str], epochs: dict[str, int] | None = None) -> Path:
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        for name in dir_names:
            tier_dir = checkpoints_dir / name
            tier_dir.mkdir()
            write_synthetic_checkpoint(
                tier_dir / "best.pt", (epochs or {}).get(name, 3)
            )
        return checkpoints_dir

    return build
