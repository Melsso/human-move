from pathlib import Path

import chess
import torch
from chess_training.model import MaiaPolicyNet
from chess_training.play import load_model, model_move


def _write_synthetic_checkpoint(
    path: Path, num_blocks: int = 2, num_filters: int = 8
) -> None:
    from chess_shared import NUM_MOVES, NUM_PLANES

    model = MaiaPolicyNet(
        NUM_PLANES, NUM_MOVES, num_blocks=num_blocks, num_filters=num_filters
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_top1": 0.25,
        "val_top3": 0.5,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
    }
    torch.save(checkpoint, path)


def test_load_model_reconstructs_architecture_from_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "test.pt"
    _write_synthetic_checkpoint(ckpt_path, num_blocks=3, num_filters=16)

    model = load_model(ckpt_path)
    assert isinstance(model, MaiaPolicyNet)
    assert len(model.tower) == 3
    assert not model.training  # load_model should leave it in eval mode


def test_model_move_returns_a_legal_move_from_starting_position(tmp_path: Path):
    ckpt_path = tmp_path / "test.pt"
    _write_synthetic_checkpoint(ckpt_path)
    model = load_model(ckpt_path)

    board = chess.Board()
    move = model_move(model, board, temperature=0.0)
    assert move in board.legal_moves


def test_model_move_returns_a_legal_move_mid_game(tmp_path: Path):
    ckpt_path = tmp_path / "test.pt"
    _write_synthetic_checkpoint(ckpt_path)
    model = load_model(ckpt_path)

    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        board.push_san(san)

    move = model_move(model, board, temperature=0.0)
    assert move in board.legal_moves


def test_model_move_with_sampling_still_returns_legal_moves(tmp_path: Path):
    ckpt_path = tmp_path / "test.pt"
    _write_synthetic_checkpoint(ckpt_path)
    model = load_model(ckpt_path)

    board = chess.Board()
    for _ in range(20):
        move = model_move(model, board, temperature=1.0)
        assert move in board.legal_moves
