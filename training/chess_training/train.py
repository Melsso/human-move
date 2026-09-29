from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from torch import nn

from chess_training.dataset import InMemoryBatchLoader, make_train_val_split
from chess_training.model import MaiaPolicyNet, load_checkpoint, model_from_checkpoint


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: InMemoryBatchLoader, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct_top1 = torch.zeros((), device=device, dtype=torch.long)
    correct_top3 = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for boards, moves in loader:
        boards = boards.to(device, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        logits = model(boards)
        total_loss += criterion(logits, moves)

        top3 = logits.topk(3, dim=1).indices
        correct_top1 += (top3[:, 0] == moves).sum()
        correct_top3 += (top3 == moves.unsqueeze(1)).any(dim=1).sum()
        n += moves.size(0)

    return (
        total_loss.item() / n,
        correct_top1.item() / n,
        correct_top3.item() / n,
    )


def train(
    npz_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    num_blocks: int,
    num_filters: int,
    resume_from: Path | None = None,
    save_every_epoch: bool = True,
) -> None:
    device = pick_device()
    print(f"device: {device}")

    start_epoch = 0
    resumed: dict[str, Any] | None = None
    if resume_from is not None:
        resumed = load_checkpoint(resume_from)
        num_blocks = resumed["num_blocks"]
        num_filters = resumed["num_filters"]
        start_epoch = resumed["epoch"]
        print(
            f"resuming from {resume_from} (was at epoch {start_epoch}, "
            f"val_top1={resumed['val_top1']:.4f}) -- architecture: "
            f"{num_blocks} blocks x {num_filters} filters"
        )

    train_loader, val_loader = make_train_val_split(
        str(npz_path), batch_size=batch_size, val_fraction=val_fraction
    )
    print(
        f"train positions: {train_loader.num_samples}, "
        f"val positions: {val_loader.num_samples}"
    )

    if len(train_loader) == 0:
        raise ValueError(
            f"batch_size={batch_size} is larger than the training split "
            f"({train_loader.num_samples} positions) with drop_last=True, so every "
            "epoch would silently train on zero batches. Use a smaller batch size, "
            "more data, or a larger val_fraction so more data lands on the train side."
        )

    if resumed is None:
        model = MaiaPolicyNet(
            in_planes=NUM_PLANES,
            num_moves=NUM_MOVES,
            num_blocks=num_blocks,
            num_filters=num_filters,
        )
    else:
        model = model_from_checkpoint(resumed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        model.train()
        epoch_start = time.time()
        running_loss = torch.zeros((), device=device)
        n_batches = 0

        for boards, moves in train_loader:
            boards = boards.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(boards), moves)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach()
            n_batches += 1

        train_loss = running_loss.item() / max(n_batches, 1)
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start

        print(
            f"epoch {global_epoch:>4} (chunk epoch {local_epoch}/{epochs}) "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.4f} "
            f"val_top3={val_top3:.4f} "
            f"({elapsed:.1f}s)"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": global_epoch,
            "val_top1": val_top1,
            "val_top3": val_top3,
            "num_blocks": num_blocks,
            "num_filters": num_filters,
            "in_planes": NUM_PLANES,
            "num_moves": NUM_MOVES,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if save_every_epoch:
            torch.save(checkpoint, out_dir / f"epoch_{global_epoch}.pt")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(checkpoint, out_dir / "best.pt")
            print(
                f"  -> new best (val_top1={val_top1:.4f}), saved to {out_dir / 'best.pt'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "npz_path", type=Path, help="Path to a bucket .npz from chess_data.prepare"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Where to save checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint (.pt) to continue training from. Architecture and epoch "
        "numbering are read from the checkpoint.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write epoch_N.pt each epoch. best.pt and last.pt are always written.",
    )
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
        resume_from=args.resume_from,
        save_every_epoch=args.save_every_epoch,
    )


if __name__ == "__main__":
    main()
