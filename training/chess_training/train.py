"""
Trains one rating-bucket policy network on the .npz produced by
`chess_data.prepare`.

Usage:
    python -m chess_training.train \
        ../data/processed/bucket_1000.npz \
        --out-dir checkpoints/bucket_1000 \
        --epochs 10 --batch-size 256 --lr 1e-3

What this script does NOT do (yet, on purpose): tune hyperparameters for
you, early-stop, or run the elo-calibration eval against Stockfish -- that
comes next, once we've confirmed this loop actually converges on your real
data. Get one bucket training and the loss/accuracy curves looking sane
before running all three.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from torch import nn
from torch.utils.data import DataLoader

from chess_training.dataset import make_train_val_split
from chess_training.model import MaiaPolicyNet


def pick_device() -> torch.device:
    """
    MPS (Metal) is Apple Silicon's GPU backend for PyTorch -- meaningfully
    faster than CPU for a CNN like this, but less mature than CUDA (some
    ops fall back to CPU silently, occasional numerical quirks). We prefer
    it when available, fall back to CUDA (if you're running this on a
    rented GPU instead of locally), then plain CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for boards, moves in loader:
        boards, moves = boards.to(device), moves.to(device)
        logits = model(boards)
        total_loss += criterion(logits, moves).item()

        top3 = logits.topk(3, dim=1).indices
        correct_top1 += (top3[:, 0] == moves).sum().item()
        correct_top3 += (top3 == moves.unsqueeze(1)).any(dim=1).sum().item()
        n += moves.size(0)

    return total_loss / n, correct_top1 / n, correct_top3 / n


def train(
    npz_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    num_blocks: int,
    num_filters: int,
    num_workers: int,
) -> None:
    device = pick_device()
    print(f"device: {device}")

    train_ds, val_ds = make_train_val_split(str(npz_path), val_fraction=val_fraction)
    print(f"train positions: {len(train_ds)}, val positions: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if len(train_loader) == 0:
        raise ValueError(
            f"batch_size={batch_size} is larger than the training split "
            f"({len(train_ds)} positions) with drop_last=True, so every epoch "
            "would silently train on zero batches. Use a smaller batch size, "
            "more data, or a larger val_fraction so more data lands on the train side."
        )

    model = MaiaPolicyNet(
        in_planes=NUM_PLANES,
        num_moves=NUM_MOVES,
        num_blocks=num_blocks,
        num_filters=num_filters,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for boards, moves in train_loader:
            boards, moves = boards.to(device), moves.to(device)

            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, moves)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        val_loss, val_top1, val_top3 = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start

        print(
            f"epoch {epoch:>3}/{epochs} "
            f"train_loss={running_loss / max(n_batches, 1):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.4f} "
            f"val_top3={val_top3:.4f} "
            f"({elapsed:.1f}s)"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_top1": val_top1,
            "val_top3": val_top3,
            "num_blocks": num_blocks,
            "num_filters": num_filters,
        }
        torch.save(checkpoint, out_dir / f"epoch_{epoch}.pt")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(checkpoint, out_dir / "best.pt")
            print(
                f"  -> new best (val_top1={val_top1:.4f}), saved to {out_dir / 'best.pt'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz_path", type=Path, help="Path to a bucket .npz from chess_data.prepare"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Where to save checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--num-blocks", type=int, default=6, help="Residual blocks (Maia uses 6)"
    )
    parser.add_argument(
        "--num-filters", type=int, default=64, help="Conv filters (Maia uses 64)"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    train(
        args.npz_path,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
