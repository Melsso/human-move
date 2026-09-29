from pathlib import Path

import pytest
import torch
from chess_backend.inference import Player
from chess_eval.round_robin import _stable_seed, play_game, round_robin, run_pairing
from chess_eval.stats import MatchResult
from chess_shared import NUM_MOVES, NUM_PLANES, TIER_CONFIGS, TierConfig
from chess_training.model import MaiaPolicyNet


@pytest.fixture(autouse=True)
def _disable_search(monkeypatch: pytest.MonkeyPatch) -> None:
    for tier, config in list(TIER_CONFIGS.items()):
        monkeypatch.setitem(
            TIER_CONFIGS,
            tier,
            TierConfig(checkpoint=config.checkpoint, temperature=config.temperature),
        )


def _write_synthetic_checkpoint(path: Path) -> None:
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=2, num_filters=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "val_top1": 0.3,
            "val_top3": 0.55,
            "num_blocks": 2,
            "num_filters": 8,
            "in_planes": NUM_PLANES,
            "num_moves": NUM_MOVES,
        },
        path,
    )


def _make_checkpoints_dir(tmp_path: Path, dir_names: list[str]) -> Path:
    checkpoints_dir = tmp_path / "checkpoints"
    for name in dir_names:
        tier_dir = checkpoints_dir / name
        tier_dir.mkdir(parents=True)
        _write_synthetic_checkpoint(tier_dir / "best.pt")
    return checkpoints_dir


def _random_player(name: str) -> Player:
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    model.eval()
    return Player(name, TierConfig(checkpoint=name, temperature=0.3), model)


def test_stable_seed_is_deterministic() -> None:
    assert _stable_seed("1000", "2000", 5) == _stable_seed("1000", "2000", 5)


def test_stable_seed_differs_across_game_numbers() -> None:
    assert _stable_seed("1000", "2000", 0) != _stable_seed("1000", "2000", 1)


def test_stable_seed_differs_across_pairings() -> None:
    assert _stable_seed("1000", "2000", 0) != _stable_seed("1000", "1500", 0)


def test_play_game_returns_a_valid_result_string() -> None:
    result = play_game(_random_player("a"), _random_player("b"), True, seed=1)
    assert result in {"1-0", "0-1", "1/2-1/2"}


def test_play_game_is_deterministic_for_same_seed() -> None:
    a, b = _random_player("a"), _random_player("b")
    assert play_game(a, b, True, seed=3) == play_game(a, b, True, seed=3)


def test_run_pairing_games_sum_to_requested_count(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(
        tmp_path, ["bucket_1000_1", "bucket_1500_1"]
    )
    result = run_pairing(
        "1000", "1500", games=4, checkpoints_dir=checkpoints_dir, workers=1
    )
    assert isinstance(result, MatchResult)
    assert result.games == 4


def test_round_robin_produces_all_pairwise_combinations(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(
        tmp_path, ["bucket_1000_1", "bucket_1500_1"]
    )
    summary = round_robin(
        ["1000", "1500", "2000"],
        games_per_pairing=2,
        checkpoints_dir=checkpoints_dir,
        workers=1,
    )
    pairings = summary["pairings"]
    assert isinstance(pairings, list)
    seen = {(p["tier_a"], p["tier_b"]) for p in pairings}
    assert seen == {("1000", "1500"), ("1000", "2000"), ("1500", "2000")}


def test_round_robin_writes_json_when_out_path_given(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(
        tmp_path, ["bucket_1000_1", "bucket_1500_1"]
    )
    out_path = tmp_path / "results" / "round_robin.json"
    round_robin(
        ["1000", "1500"],
        games_per_pairing=2,
        checkpoints_dir=checkpoints_dir,
        out_path=out_path,
        workers=1,
    )
    assert out_path.exists()


def test_round_robin_raises_with_fewer_than_two_tiers(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["bucket_1000_1"])
    with pytest.raises(ValueError, match="at least 2 tiers"):
        round_robin(["1000"], checkpoints_dir=checkpoints_dir)


def test_round_robin_raises_for_unknown_tier(tmp_path: Path) -> None:
    checkpoints_dir = _make_checkpoints_dir(tmp_path, ["bucket_1000_1"])
    with pytest.raises(ValueError, match="unknown tier"):
        round_robin(["1000", "9999"], checkpoints_dir=checkpoints_dir)
