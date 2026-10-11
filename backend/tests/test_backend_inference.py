from collections.abc import Callable
from pathlib import Path

import chess
import pytest
from chess_backend.inference import ModelRegistry, TierNotFoundError


def test_discover_checkpoints_groups_chunks_and_picks_highest_numerically(
    make_checkpoints: Callable[..., Path],
) -> None:
    checkpoints_dir = make_checkpoints(
        ["bucket_1000_2", "bucket_1000_9", "bucket_1000_10", "bucket_2000_1"]
    )
    found = ModelRegistry(checkpoints_dir).discover_checkpoints()
    assert found == {
        "1000": checkpoints_dir / "bucket_1000_10" / "best.pt",
        "2000": checkpoints_dir / "bucket_2000_1" / "best.pt",
    }


def test_discover_checkpoints_ignores_directories_without_best_pt(
    tmp_path: Path,
) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    (checkpoints_dir / "bucket_1000_1").mkdir(parents=True)
    (checkpoints_dir / "bucket_1000_1" / "epoch_1.pt").write_text("x")
    assert ModelRegistry(checkpoints_dir).discover_checkpoints() == {}


def test_discover_checkpoints_ignores_unrecognised_directory_names(
    make_checkpoints: Callable[..., Path],
) -> None:
    checkpoints_dir = make_checkpoints(["my_experiment"])
    assert ModelRegistry(checkpoints_dir).discover_checkpoints() == {}


def test_discover_checkpoints_on_missing_directory_returns_empty(
    tmp_path: Path,
) -> None:
    assert ModelRegistry(tmp_path / "missing").discover_checkpoints() == {}


def test_available_tiers_follow_tier_config(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_1000_1", "bucket_2000_1"]))
    assert registry.available_tiers() == ["1000", "1500", "2000"]


def test_tier_2000_needs_only_the_1500_checkpoint(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_2000_1"]))
    assert registry.available_tiers() == ["1500", "2000"]


def test_get_player_caches_and_shares_one_model_between_1500_and_2000(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_2000_1"]))
    assert registry.get_player("1500") is registry.get_player("1500")
    assert registry.get_player("1500").model is registry.get_player("2000").model
    assert len(registry._loaded) == 1


def test_get_player_raises_for_tier_without_checkpoint(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_1000_1"]))
    with pytest.raises(TierNotFoundError) as exc_info:
        registry.get_player("1500")
    assert exc_info.value.tier == "1500"
    assert exc_info.value.available == ["1000"]


def test_get_player_raises_for_unknown_tier(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_1000_1"]))
    with pytest.raises(TierNotFoundError):
        registry.get_player("9999")


def test_get_meta_uses_highest_chunk_and_is_shared_by_2000(
    make_checkpoints: Callable[..., Path],
) -> None:
    checkpoints_dir = make_checkpoints(
        ["bucket_2000_1", "bucket_2000_2"],
        epochs={"bucket_2000_1": 10, "bucket_2000_2": 20},
    )
    registry = ModelRegistry(checkpoints_dir)
    assert registry.get_meta("1500")["epoch"] == 20
    assert registry.get_meta("2000") == registry.get_meta("1500")


def test_non_searching_player_returns_legal_move_and_sorted_candidates(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_1000_1"]))
    board = chess.Board()
    move, candidates = registry.get_player("1000").choose_move(board, top_k=3)
    assert move in board.legal_moves
    assert len(candidates) == 3
    probs = [c["prob"] for c in candidates]
    assert probs == sorted(probs, reverse=True)


def test_searching_player_returns_legal_move_and_restores_board(
    make_checkpoints: Callable[..., Path],
) -> None:
    registry = ModelRegistry(make_checkpoints(["bucket_2000_1"]))
    board = chess.Board()
    board.push_san("e4")
    fen_before = board.fen()
    move, candidates = registry.get_player("1500").choose_move(board)
    assert move in board.legal_moves
    assert board.fen() == fen_before
    assert len(candidates) > 0
