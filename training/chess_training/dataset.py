"""
Loads the (board, move) tensors produced by `chess_data.prepare` as a
PyTorch Dataset.

MEMORY NOTE, worth reading if you're on a RAM-constrained machine (e.g. an
8/16GB M4): `chess_data.prepare` writes buckets with `np.savez_compressed`.
NumPy's `mmap_mode` is silently ignored for compressed .npz archives --
there is no way to memory-map a compressed file, since it has to be
decompressed into memory to be read at all. So despite requesting
`mmap_mode="r"` below, the array is fully loaded into RAM regardless. We
detect this and warn once rather than pretending it's memory-mapped when
it isn't. If a bucket ever gets large enough that this matters, the real
fix is to have `chess_data.prepare` write with plain `np.savez` (no
compression) instead -- larger on disk, but then `mmap_mode` genuinely
works and positions get paged in from disk on demand instead of sitting
in RAM.

We load the underlying arrays exactly ONCE and share them between the
train and val `Dataset` objects (see `make_train_val_split`) -- loading
separately per split would otherwise double memory use for no reason,
since both splits are just index views into the same array.

IMPORTANT CAVEAT: the split is by *position index*, not by *game*. Since
`chess_data.prepare` writes positions from the same game consecutively,
and doesn't shuffle across games before saving, a naive index-based split
risks leaking near-identical positions (same game, different move number)
across train/val. We shuffle indices with a fixed seed before splitting to
reduce (not eliminate) this -- positions from one game can still land on
both sides. This is a known simplification, not correctness bug: fine for
getting the training loop working and for early-stage eyeballing of
validation accuracy, but if you want a rigorous held-out eval later, the
real fix is a game-level split in `chess_data.prepare` itself (write
held-out games to a separate file from the start).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessPositionDataset(Dataset):
    """
    Thin index-view wrapper around already-loaded `boards`/`moves` arrays
    -- does NOT load anything itself, see `make_train_val_split`.
    """

    def __init__(
        self, boards: np.ndarray, moves: np.ndarray, indices: np.ndarray
    ) -> None:
        self._boards = boards
        self._moves = moves
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._indices[i]
        board = torch.from_numpy(np.array(self._boards[idx]))
        move = torch.tensor(int(self._moves[idx]), dtype=torch.long)
        return board, move


def make_train_val_split(
    npz_path: str,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[ChessPositionDataset, ChessPositionDataset]:
    data = np.load(npz_path, mmap_mode="r")
    boards = data["boards"]
    moves = data["moves"]

    if not isinstance(boards, np.memmap):
        warnings.warn(
            f"'{npz_path}' is a compressed .npz (from savez_compressed), so it could not "
            "actually be memory-mapped -- the full array was loaded into RAM instead "
            f"({boards.nbytes / 1e9:.2f}GB for boards alone). See the module docstring in "
            "chess_training/dataset.py if this matters for your machine's RAM.",
            stacklevel=2,
        )

    n = boards.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_ds = ChessPositionDataset(boards, moves, train_indices)
    val_ds = ChessPositionDataset(boards, moves, val_indices)
    return train_ds, val_ds
