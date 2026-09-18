"""
Play a game against a trained checkpoint, right in the terminal. This is
the fastest way to sanity-check a bucket's weights actually feel like a
chess opponent, before investing in the real backend/frontend.

Usage:
    python -m chess_training.play training/checkpoints/bucket_1000/best.pt
    python -m chess_training.play training/checkpoints/bucket_1000/best.pt --human-color black
    python -m chess_training.play training/checkpoints/bucket_1000/best.pt --temperature 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch
from chess_shared import (
    NUM_MOVES,
    NUM_PLANES,
    encode_board,
    legal_move_mask,
    masked_softmax,
    select_move_index,
    top_k_moves,
)
from chess_shared.move_encoding import index_to_move

from chess_training.model import MaiaPolicyNet


def load_model(checkpoint_path: Path) -> MaiaPolicyNet:
    """
    Rebuilds the model using the architecture (num_blocks/num_filters)
    saved inside the checkpoint itself, so you never have to remember or
    guess what a given checkpoint was trained with.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MaiaPolicyNet(
        in_planes=NUM_PLANES,
        num_moves=NUM_MOVES,
        num_blocks=checkpoint["num_blocks"],
        num_filters=checkpoint["num_filters"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"loaded checkpoint: epoch {checkpoint['epoch']}, "
        f"val_top1={checkpoint['val_top1']:.4f}, val_top3={checkpoint['val_top3']:.4f}"
    )
    return model


@torch.no_grad()
def model_move(
    model: MaiaPolicyNet, board: chess.Board, temperature: float
) -> chess.Move:
    board_tensor = torch.from_numpy(encode_board(board)).unsqueeze(0)
    logits = model(board_tensor).squeeze(0).numpy()

    mask = legal_move_mask(board)
    probs = masked_softmax(logits, mask)

    top5 = top_k_moves(probs, k=5)
    print("  model's top candidates:")
    for idx, p in top5:
        move = index_to_move(idx, board)
        print(f"    {board.san(move):<8} {p:.1%}")

    move_idx = select_move_index(probs, temperature=temperature)
    return index_to_move(move_idx, board)


def play(
    checkpoint_path: Path, human_color: str, temperature: float, fen: str | None
) -> None:
    model = load_model(checkpoint_path)
    board = chess.Board(fen) if fen else chess.Board()
    human_is_white = human_color == "white"

    print(board)
    print()

    while not board.is_game_over():
        human_turn = (
            board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK
        )

        if human_turn:
            move_str = input("your move (SAN or UCI, e.g. 'Nf3' or 'g1f3'): ").strip()
            try:
                try:
                    move = board.parse_san(move_str)
                except ValueError:
                    move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    print("  illegal move, try again")
                    continue
            except ValueError:
                print("  couldn't parse that move, try again (e.g. 'Nf3' or 'g1f3')")
                continue
        else:
            print("model is thinking...")
            move = model_move(model, board, temperature)
            print(f"  model plays: {board.san(move)}")

        board.push(move)
        print()
        print(board)
        print()

    print(f"game over: {board.result()}")
    if board.is_checkmate():
        print("checkmate")
    elif board.is_stalemate():
        print("stalemate")
    elif board.is_insufficient_material():
        print("draw by insufficient material")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint")
    parser.add_argument("--human-color", choices=["white", "black"], default="white")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = always the model's top move (matches how Maia actually plays). "
        ">0.0 = sample instead, for more variety but a weaker/less calibrated opponent.",
    )
    parser.add_argument(
        "--fen", type=str, default=None, help="Start from a custom position"
    )
    args = parser.parse_args()

    play(args.checkpoint, args.human_color, args.temperature, args.fen)


if __name__ == "__main__":
    main()
