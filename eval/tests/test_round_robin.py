from pathlib import Path

import pytest
import torch
from chess_eval.round_robin import _stable_seed, play_game, round_robin, run_pairing
from chess_eval.stats import MatchResult
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
        "epoch": 1,
        "val_top1": 0.3,
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


def test_stable_seed_is_deterministic() -> None:
    assert _stable_seed("1000", "2000", 5) == _stable_seed("1000", "2000", 5)


def test_stable_seed_differs_across_game_numbers() -> None:
    assert _stable_seed("1000", "2000", 0) != _stable_seed("1000", "2000", 1)


def test_stable_seed_differs_across_pairings() -> None:
    assert _stable_seed("1000", "2000", 0) != _stable_seed("1000", "1500", 0)


def test_play_game_returns_a_valid_result_string() -> None:
    model_a = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    model_b = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    result = play_game(model_a, model_b, a_is_white=True, seed=1)
    assert result in {"1-0", "0-1", "1/2-1/2"}


def test_run_pairing_games_sum_to_requested_count(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["1000", "2000"])
    result = run_pairing("1000", "2000", games=6, checkpoints_dir=checkpoints_dir)
    assert isinstance(result, MatchResult)
    assert result.games == 6


def test_round_robin_produces_all_pairwise_combinations(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["1000", "1500", "2000"])
    summary = round_robin(
        ["1000", "1500", "2000"], games_per_pairing=2, checkpoints_dir=checkpoints_dir
    )
    pairings = summary["pairings"]
    assert isinstance(pairings, list)
    assert len(pairings) == 3
    seen = {(p["tier_a"], p["tier_b"]) for p in pairings}
    assert seen == {("1000", "1500"), ("1000", "2000"), ("1500", "2000")}


def test_round_robin_writes_json_when_out_path_given(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["1000", "1500"])
    out_path = tmp_path / "results" / "round_robin.json"
    round_robin(
        ["1000", "1500"],
        games_per_pairing=2,
        checkpoints_dir=checkpoints_dir,
        out_path=out_path,
    )
    assert out_path.exists()


def test_round_robin_raises_with_fewer_than_two_tiers(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["1000"])
    with pytest.raises(ValueError, match="at least 2 tiers"):
        round_robin(["1000"], checkpoints_dir=checkpoints_dir)
