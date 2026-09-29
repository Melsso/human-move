from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from chess_shared.board_encoding import NUM_PLANES
from chess_shared.inference import (
    compute_model_move,
    masked_softmax,
    select_move_index,
    top_k_moves,
)
from chess_shared.model_registery import ModelRegistry, TierNotFoundError
from chess_shared.move_encoding import NUM_MOVES, legal_move_mask, move_to_index
from chess_training.model import MaiaPolicyNet


def test_masked_softmax_zeroes_illegal_moves_exactly():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=NUM_MOVES)
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    legal_indices = [10, 200, 3000]
    for i in legal_indices:
        mask[i] = 1.0

    probs = masked_softmax(logits, mask)
    assert np.isclose(probs.sum(), 1.0)
    illegal_mask = np.ones(NUM_MOVES, dtype=bool)
    illegal_mask[legal_indices] = False
    assert np.all(probs[illegal_mask] == 0.0)
    assert np.all(probs[legal_indices] > 0.0)


def test_masked_softmax_matches_real_board_legal_moves():
    board = chess.Board()
    mask = legal_move_mask(board)
    rng = np.random.default_rng(1)
    logits = rng.normal(size=NUM_MOVES)

    probs = masked_softmax(logits, mask)
    for idx in range(NUM_MOVES):
        if mask[idx] == 0:
            assert probs[idx] == 0.0
    assert np.isclose(probs.sum(), 1.0)


def test_masked_softmax_raises_on_no_legal_moves():
    logits = np.zeros(NUM_MOVES)
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    with pytest.raises(ValueError, match="no legal moves"):
        masked_softmax(logits, mask)


def test_select_move_index_greedy_picks_argmax():
    probs = np.zeros(NUM_MOVES)
    probs[42] = 0.9
    probs[7] = 0.1
    assert select_move_index(probs, temperature=0.0) == 42


def test_select_move_index_never_picks_a_zero_probability_move():
    rng = np.random.default_rng(2)
    board = chess.Board()
    mask = legal_move_mask(board)

    for _ in range(200):
        logits = rng.normal(size=NUM_MOVES)
        probs = masked_softmax(logits, mask)

        greedy_idx = select_move_index(probs, temperature=0.0)
        assert mask[greedy_idx] == 1.0

        sampled_idx = select_move_index(probs, temperature=1.0, rng=rng)
        assert mask[sampled_idx] == 1.0


def test_top_k_moves_returns_sorted_highest_first():
    probs = np.zeros(NUM_MOVES)
    probs[1] = 0.1
    probs[2] = 0.5
    probs[3] = 0.4

    result = top_k_moves(probs, k=3)
    assert [idx for idx, _ in result] == [2, 3, 1]
    assert result[0][1] == pytest.approx(0.5)


def test_top_k_moves_handles_fewer_nonzero_than_k():
    probs = np.zeros(NUM_MOVES)
    probs[5] = 1.0
    result = top_k_moves(probs, k=5)
    assert len(result) == 1
    assert result[0][0] == 5


def test_full_pipeline_on_real_position_produces_a_legal_move():
    from chess_shared.move_encoding import index_to_move

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    mask = legal_move_mask(board)
    rng = np.random.default_rng(3)
    logits = rng.normal(size=NUM_MOVES)
    probs = masked_softmax(logits, mask)
    move_idx = select_move_index(probs, temperature=0.0)

    move = index_to_move(move_idx)
    assert board.is_legal(move)
    assert move_to_index(move) == move_idx


def _write_synthetic_checkpoint(
    path: Path, num_blocks: int = 2, num_filters: int = 8, epoch: int = 3
) -> None:
    model = MaiaPolicyNet(
        NUM_PLANES, NUM_MOVES, num_blocks=num_blocks, num_filters=num_filters
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
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


def test_discover_tiers_groups_chunked_directories_picking_highest_chunk(
    tmp_path: Path,
):
    checkpoints_dir = tmp_path / "checkpoints"
    for chunk, epoch in [(1, 10), (2, 20), (3, 30)]:
        d = checkpoints_dir / f"bucket_1000_{chunk}"
        d.mkdir(parents=True)
        _write_synthetic_checkpoint(d / "best.pt", epoch=epoch)

    registry = ModelRegistry(checkpoints_dir)
    tiers = registry.discover_tiers()

    assert set(tiers.keys()) == {"1000"}
    assert tiers["1000"] == checkpoints_dir / "bucket_1000_3" / "best.pt"

    meta = registry.get_meta("1000")
    assert meta["epoch"] == 30


def test_discover_tiers_handles_chunk_numbers_out_of_lexicographic_order(
    tmp_path: Path,
):
    checkpoints_dir = tmp_path / "checkpoints"
    for chunk in [2, 9, 10]:
        d = checkpoints_dir / f"bucket_1000_{chunk}"
        d.mkdir(parents=True)
        _write_synthetic_checkpoint(d / "best.pt", epoch=chunk * 10)

    registry = ModelRegistry(checkpoints_dir)
    tiers = registry.discover_tiers()
    assert tiers["1000"] == checkpoints_dir / "bucket_1000_10" / "best.pt"


def test_discover_tiers_mixes_chunked_and_standalone_directories(tmp_path: Path):
    checkpoints_dir = tmp_path / "checkpoints"
    for chunk in [1, 2]:
        d = checkpoints_dir / f"bucket_1000_{chunk}"
        d.mkdir(parents=True)
        _write_synthetic_checkpoint(d / "best.pt", epoch=chunk)

    standalone_dir = checkpoints_dir / "my_custom_experiment"
    standalone_dir.mkdir(parents=True)
    _write_synthetic_checkpoint(standalone_dir / "best.pt", epoch=99)

    registry = ModelRegistry(checkpoints_dir)
    tiers = registry.discover_tiers()

    assert set(tiers.keys()) == {"1000", "my_custom_experiment"}
    assert tiers["1000"] == checkpoints_dir / "bucket_1000_2" / "best.pt"
    assert tiers["my_custom_experiment"] == standalone_dir / "best.pt"


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
