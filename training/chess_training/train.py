from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from torch import nn

from chess_training.dataset import InMemoryBatchLoader, make_train_val_split
from chess_training.model import MaiaPolicyNet


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_boards(
    boards: torch.Tensor, device: torch.device, channels_last: bool
) -> torch.Tensor:
    boards = boards.to(device, non_blocking=True).float()
    if channels_last:
        boards = boards.contiguous(memory_format=torch.channels_last)
    return boards


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: InMemoryBatchLoader,
    device: torch.device,
    amp: bool,
    channels_last: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct_top1 = torch.zeros((), device=device, dtype=torch.long)
    correct_top3 = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for boards, moves in loader:
        boards = prepare_boards(boards, device, channels_last)
        moves = moves.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(boards)
        logits = logits.float()
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
    amp: bool = False,
    channels_last: bool = False,
    compact_boards: bool = False,
    save_every_epoch: bool = True,
) -> None:
    device = pick_device()
    print(f"device: {device}")
    if amp and device.type == "cpu":
        print("amp requested but device is cpu; disabling")
        amp = False

    start_epoch = 0
    resumed_state_dict = None
    if resume_from is not None:
        resume_checkpoint = torch.load(
            resume_from, map_location="cpu", weights_only=True
        )
        num_blocks = resume_checkpoint["num_blocks"]
        num_filters = resume_checkpoint["num_filters"]
        resumed_state_dict = resume_checkpoint["model_state_dict"]
        start_epoch = resume_checkpoint["epoch"]
        print(
            f"resuming from {resume_from} (was at epoch {start_epoch}, "
            f"val_top1={resume_checkpoint['val_top1']:.4f}) -- architecture: "
            f"{num_blocks} blocks x {num_filters} filters"
        )

    train_loader, val_loader = make_train_val_split(
        str(npz_path),
        batch_size=batch_size,
        val_fraction=val_fraction,
        compact_boards=compact_boards,
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

    model = MaiaPolicyNet(
        in_planes=NUM_PLANES,
        num_moves=NUM_MOVES,
        num_blocks=num_blocks,
        num_filters=num_filters,
    )
    if resumed_state_dict is not None:
        model.load_state_dict(resumed_state_dict)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        model.train()
        epoch_start = time.time()
        running_loss = torch.zeros((), device=device)
        n_batches = 0

        for boards, moves in train_loader:
            boards = prepare_boards(boards, device, channels_last)
            moves = moves.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits = model(boards)
            loss = criterion(logits.float(), moves)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()
            n_batches += 1

        train_loss = running_loss.item() / max(n_batches, 1)
        val_loss, val_top1, val_top3 = evaluate(
            model, val_loader, device, amp, channels_last
        )
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
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--num-blocks", type=int, default=6, help="Residual blocks (Maia uses 6)"
    )
    parser.add_argument(
        "--num-filters", type=int, default=64, help="Conv filters (Maia uses 64)"
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint (.pt) to continue training from -- for chunked/incremental "
        "training across multiple dataset chunks. Architecture and epoch numbering "
        "are read from the checkpoint automatically.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="fp16 autocast. Compare val_loss against an fp32 run before trusting it.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="channels_last memory format. Benchmark; gains vary on MPS.",
    )
    parser.add_argument(
        "--compact-boards",
        action="store_true",
        help="Store float32 boards as float16 in RAM (halves memory). Only safe if "
        "board planes are binary / small integers.",
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
        amp=args.amp,
        channels_last=args.channels_last,
        compact_boards=args.compact_boards,
        save_every_epoch=args.save_every_epoch,
    )


if __name__ == "__main__":
    main()
