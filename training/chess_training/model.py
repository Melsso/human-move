from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from torch import nn

CHECKPOINT_ENCODING = {"in_planes": NUM_PLANES, "num_moves": NUM_MOVES}


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class MaiaPolicyNet(nn.Module):
    def __init__(
        self,
        in_planes: int,
        num_moves: int,
        num_blocks: int = 6,
        num_filters: int = 64,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        policy_channels = 32
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_channels * 8 * 8, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.tower(x)
        x = self.policy_head(x)
        x = torch.flatten(x, start_dim=1)
        return self.policy_fc(x)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in CHECKPOINT_ENCODING.items():
        found = checkpoint.get(key)
        if found is not None and found != expected:
            raise ValueError(
                f"checkpoint '{path}' was trained with {key}={found}, but the "
                f"current encoding uses {key}={expected}. Retrain on data built "
                "with the current chess_data.prepare."
            )
    return checkpoint


def model_from_checkpoint(checkpoint: dict[str, Any]) -> MaiaPolicyNet:
    model = MaiaPolicyNet(
        in_planes=NUM_PLANES,
        num_moves=NUM_MOVES,
        num_blocks=checkpoint["num_blocks"],
        num_filters=checkpoint["num_filters"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
