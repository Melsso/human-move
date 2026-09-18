"""
The policy network: given a board tensor, predict a distribution over
moves. Architecture follows the original Maia Chess paper (McIlroy-Young
et al., 2020) -- a 6-block, 64-filter residual CNN, no search, probed
directly for a move distribution. Maia's own network reuses Leela Chess
Zero's 1858-way move encoding (since Maia ships as an actual Lc0 weights
file); we use our own simpler 4096-way from/to encoding from
`chess_shared.move_encoding` instead, since we're not trying to produce
Lc0-loadable weights, just a standalone PyTorch model.

Why a residual CNN and not something fancier: chess positions are
naturally grid-structured (8x8), and convolutions are good at picking up
local tactical patterns (piece attacks, pawn structures) while the
residual tower lets information propagate across the whole board through
depth. This is the same reasoning AlphaZero/Leela/Maia all share -- we're
deliberately not innovating on architecture here, just reproducing a
proven one at a proven size.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Two 3x3 convolutions with batch norm and a skip connection, the
    standard ResNet block used throughout Leela/Maia's residual tower.
    """

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
    """
    6x64 residual CNN policy network (matches the original Maia paper's
    architecture). Input: (batch, in_planes, 8, 8) board tensor. Output:
    (batch, num_moves) raw logits over the move space -- NOT masked to
    legal moves here, that happens at inference time in the backend using
    `chess_shared.move_encoding.legal_move_mask`, since masking during
    training would hide the model's confidence in illegal moves, which is
    a useful diagnostic (see the eval/calibration code).
    """

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
