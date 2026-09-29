from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class InMemoryBatchLoader:
    def __init__(
        self,
        boards: torch.Tensor,
        moves: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
    ) -> None:
        self.boards = boards
        self.moves = moves
        self.indices = indices.to(boards.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._gen = torch.Generator().manual_seed(seed)

    @property
    def num_samples(self) -> int:
        return len(self.indices)

    def __len__(self) -> int:
        n = len(self.indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        idx = self.indices
        if self.shuffle:
            perm = torch.randperm(len(idx), generator=self._gen)
            idx = idx[perm.to(idx.device)]
        for i in range(len(self)):
            b = idx[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.boards[b], self.moves[b]


def make_train_val_split(
    npz_path: str,
    batch_size: int,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[InMemoryBatchLoader, InMemoryBatchLoader]:
    data = np.load(npz_path)
    boards = data["boards"]
    moves = data["moves"]
    game_ids = data["game_ids"]

    boards_t = torch.from_numpy(np.ascontiguousarray(boards))
    moves_t = torch.from_numpy(np.ascontiguousarray(moves).astype(np.int64))

    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    shuffled_games = rng.permutation(unique_games)
    n_val_games = max(1, int(len(unique_games) * val_fraction))
    val_game_ids = shuffled_games[:n_val_games]

    is_val = np.isin(game_ids, val_game_ids)
    val_indices = torch.from_numpy(np.nonzero(is_val)[0])
    train_indices = torch.from_numpy(np.nonzero(~is_val)[0])

    print(
        f"  split: {len(unique_games)} games "
        f"({len(unique_games) - n_val_games} train / {n_val_games} val) -> "
        f"{len(train_indices)} train positions / {len(val_indices)} val positions"
    )

    train_loader = InMemoryBatchLoader(
        boards_t,
        moves_t,
        train_indices,
        batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    val_loader = InMemoryBatchLoader(
        boards_t,
        moves_t,
        val_indices,
        batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    return train_loader, val_loader
