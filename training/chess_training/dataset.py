from __future__ import annotations

import warnings
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
    compact_boards: bool = False,
) -> tuple[InMemoryBatchLoader, InMemoryBatchLoader]:
    data = np.load(npz_path, mmap_mode="r")
    boards = data["boards"]
    moves = data["moves"]

    if not isinstance(boards, np.memmap):
        warnings.warn(
            f"'{npz_path}' is a .npz file, which can never be memory-mapped "
            "regardless of compression -- the full array was loaded into RAM "
            f"instead ({boards.nbytes / 1e9:.2f}GB for boards alone). If this bucket "
            "is uncomfortably large for your machine's RAM, process your source data "
            "in smaller chunks instead (see chess_data.prepare's --skip-games), "
            "rather than trying to fix this via mmap.",
            stacklevel=2,
        )

    if compact_boards and boards.dtype == np.float32:
        boards = boards.astype(np.float16)

    boards_t = torch.from_numpy(np.ascontiguousarray(boards))
    moves_t = torch.from_numpy(np.ascontiguousarray(moves).astype(np.int64))

    n = boards_t.shape[0]
    rng = np.random.default_rng(seed)
    perm = torch.from_numpy(rng.permutation(n))
    n_val = max(1, int(n * val_fraction))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

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
