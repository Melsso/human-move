"""
Loads the (board, move) tensors produced by `chess_data.prepare` as a
PyTorch Dataset.

MEMORY NOTE, worth reading if you're on a RAM-constrained machine (e.g. an
8/16GB M4): `.npz` files -- whether written with `np.savez_compressed` or
plain `np.savez` -- can NEVER be memory-mapped. This isn't a compression
thing; per NumPy's own docs, `mmap_mode` has no effect on any zipped file,
since `.npz` is always a zip archive under the hood (an earlier version of
this docstring incorrectly claimed switching to uncompressed .npz would
fix it -- it doesn't). The only way to actually get `mmap_mode` to do
something is a plain, non-zipped `.npy` file written with `np.save`.

Given that, the real fix for a bucket that's too large to comfortably load
into RAM isn't a different `.npz` flag -- it's not making one giant bucket
in the first place. `chess_data.prepare`'s `--skip-games`/`--max-games`
let you process a big PGN dump in smaller chunks instead (see the root
README's chunked-training section), each easily small enough to load
whole, and `chess_training.train`'s `--resume-from` lets you train across
them incrementally. That sidesteps this limitation entirely rather than
working around it.

We still detect and warn here (rather than silently eating the RAM cost)
so a bucket that turns out bigger than expected is visible immediately.

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
            f"'{npz_path}' is a .npz file, which can never be memory-mapped "
            "regardless of compression -- the full array was loaded into RAM "
            f"instead ({boards.nbytes / 1e9:.2f}GB for boards alone). If this bucket "
            "is uncomfortably large for your machine's RAM, process your source data "
            "in smaller chunks instead (see chess_data.prepare's --skip-games and this "
            "module's docstring), rather than trying to fix this via mmap.",
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
