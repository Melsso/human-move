from pathlib import Path

import chess
import torch
from chess_backend.inference import ModelRegistry, TierNotFoundError, compute_model_move
from chess_shared import NUM_MOVES, NUM_PLANES
from chess_training.model import MaiaPolicyNet


def _write_synthetic_checkpoint(
    path: Path, num_blocks: int = 2, num_filters: int = 8
) -> None:
    model = MaiaPolicyNet(
        NUM_PLANES, NUM_MOVES, num_blocks=num_blocks, num_filters=num_filters
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 3,
        "val_top1": 0.31,
        "val_top3": 0.55,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
    }
    torch.save(checkpoint, path)


def _make_checkpoints_dir(tmp_path: Path, tier_names: list[str]) -> Path:
    checkpoints_dir = tmp_path / "checkpoints"
    for name in tier_names:
        tier_dir = checkpoints_dir / name
        tier_dir.mkdir(parents=True)
        _write_synthetic_checkpoint(tier_dir / "best.pt")
    return checkpoints_dir


def test_discover_tiers_finds_directories_with_best_pt(tmp_path: Path):
    checkpoints_dir = _make_checkpoints_dir(
        tmp_path, ["bucket_1000", "sample_bucket_2000"]
    )
    registry = ModelRegistry(checkpoints_dir)
    tiers = registry.discover_tiers()
    assert set(tiers.keys()) == {"bucket_1000", "sample_bucket_2000"}


def test_discover_tiers_ignores_directories_without_best_pt(tmp_path: Path):
    checkpoints_dir = tmp_path / "checkpoints"
    (checkpoints_dir / "bucket_1000").mkdir(parents=True)
    (checkpoints_dir / "bucket_1000" / "epoch_1.pt").write_text("not a real checkpoint")

    registry = ModelRegistry(checkpoints_dir)
    assert registry.discover_tiers() == {}


def test_discover_tiers_on_missing_directory_returns_empty(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "does_not_exist")
    assert registry.discover_tiers() == {}


def test_get_model_loads_and_caches(tmp_path: Path):
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["bucket_1000"])
    registry = ModelRegistry(checkpoints_dir)

    model1 = registry.get_model("bucket_1000")
    model2 = registry.get_model("bucket_1000")
    assert model1 is model2


def test_get_model_raises_on_unknown_tier(tmp_path: Path):
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["bucket_1000"])
    registry = ModelRegistry(checkpoints_dir)

    try:
        registry.get_model("bucket_9999")
        raise AssertionError("expected TierNotFoundError")
    except TierNotFoundError as e:
        assert e.tier == "bucket_9999"
        assert e.available == ["bucket_1000"]


def test_get_meta_returns_checkpoint_metadata(tmp_path: Path):
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["bucket_1000"])
    registry = ModelRegistry(checkpoints_dir)

    meta = registry.get_meta("bucket_1000")
    assert meta["epoch"] == 3
    assert meta["val_top1"] == 0.31
    assert meta["val_top3"] == 0.55


def test_compute_model_move_returns_legal_move_and_candidates(tmp_path: Path):
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["bucket_1000"])
    registry = ModelRegistry(checkpoints_dir)
    model = registry.get_model("bucket_1000")

    board = chess.Board()
    move, candidates = compute_model_move(model, board, top_k=3)

    assert move in board.legal_moves
    assert len(candidates) == 3
    for c in candidates:
        assert 0.0 <= c["prob"] <= 1.0
        assert isinstance(c["san"], str)
        assert isinstance(c["uci"], str)
    probs = [c["prob"] for c in candidates]
    assert probs == sorted(probs, reverse=True)
