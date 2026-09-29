from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from chess_backend.inference import Player

from chess_eval.common import (
    classify_result,
    load_player,
    push_legal,
    random_opening,
    require_tiers,
    subject_to_move,
)
from chess_eval.stats import MatchResult, elo_estimate

DEFAULT_GAMES_PER_PAIRING = 500

_worker_players: tuple[Player, Player] | None = None


def play_game(player_a: Player, player_b: Player, a_is_white: bool, seed: int) -> str:
    board = random_opening(seed=seed)
    rng = np.random.default_rng(seed)
    while not board.is_game_over(claim_draw=True):
        player = player_a if subject_to_move(board, a_is_white) else player_b
        move, _ = player.choose_move(board, top_k=1, rng=rng)
        push_legal(board, move)
    return board.result(claim_draw=True)


def _stable_seed(tier_a: str, tier_b: str, game_number: int) -> int:
    digest = hashlib.sha256(f"{tier_a}:{tier_b}:{game_number}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _init_worker(tier_a: str, tier_b: str, checkpoints_dir: str | None) -> None:
    global _worker_players
    _worker_players = (
        load_player(tier_a, checkpoints_dir),
        load_player(tier_b, checkpoints_dir),
    )


def _init_pool_worker(tier_a: str, tier_b: str, checkpoints_dir: str | None) -> None:
    torch.set_num_threads(1)
    _init_worker(tier_a, tier_b, checkpoints_dir)


def _play_single_game(job: tuple[int, int]) -> tuple[int, str, str]:
    game_number, seed = job
    assert _worker_players is not None
    a_is_white = game_number % 2 == 0
    raw_result = play_game(_worker_players[0], _worker_players[1], a_is_white, seed)
    return game_number, classify_result(raw_result, a_is_white), raw_result


def _run_games(
    jobs: list[tuple[int, int]],
    workers: int,
    tier_a: str,
    tier_b: str,
    checkpoints_dir: Path | None,
) -> Iterator[tuple[int, str, str]]:
    init_args = (tier_a, tier_b, str(checkpoints_dir) if checkpoints_dir else None)
    if workers == 1:
        _init_worker(*init_args)
        yield from map(_play_single_game, jobs)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers, initializer=_init_pool_worker, initargs=init_args
    ) as pool:
        yield from pool.imap_unordered(_play_single_game, jobs)


def run_pairing(
    tier_a: str,
    tier_b: str,
    games: int,
    checkpoints_dir: Path | None = None,
    workers: int | None = None,
) -> MatchResult:
    workers = workers or max(1, min(mp.cpu_count(), games))
    jobs = [(i, _stable_seed(tier_a, tier_b, i)) for i in range(games)]
    counts = {"WIN": 0, "DRAW": 0, "LOSS": 0}

    for game_number, outcome, raw_result in _run_games(
        jobs, workers, tier_a, tier_b, checkpoints_dir
    ):
        counts[outcome] += 1
        print(
            f"  game {game_number + 1}/{games}: {tier_a} {outcome} ({raw_result})",
            flush=True,
        )

    return MatchResult(counts["WIN"], counts["DRAW"], counts["LOSS"])


def round_robin(
    tiers: list[str],
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    checkpoints_dir: Path | None = None,
    out_path: Path | None = None,
    workers: int | None = None,
) -> dict[str, object]:
    if len(tiers) < 2:
        raise ValueError(f"need at least 2 tiers for a round robin, got {tiers}")
    require_tiers(tiers)

    results = []
    for tier_a, tier_b in itertools.combinations(tiers, 2):
        print(f"=== {tier_a} vs {tier_b} ===")
        match_result = run_pairing(
            tier_a, tier_b, games_per_pairing, checkpoints_dir, workers
        )
        estimate = elo_estimate(match_result)
        print(
            f"  {tier_a}: W{match_result.wins}/D{match_result.draws}/L{match_result.losses} "
            f"score={match_result.score * 100:.1f}% relative_elo={estimate}\n"
        )
        results.append(
            {
                "tier_a": tier_a,
                "tier_b": tier_b,
                "wins_a": match_result.wins,
                "draws": match_result.draws,
                "losses_a": match_result.losses,
                "score_a": match_result.score,
                "relative_elo_a_minus_b": estimate.point,
                "relative_elo_low": estimate.low,
                "relative_elo_high": estimate.high,
            }
        )

    summary: dict[str, object] = {
        "tiers": tiers,
        "games_per_pairing": games_per_pairing,
        "pairings": results,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"saved results to {out_path}")

    print(
        "\nNOTE: relative_elo_a_minus_b is tier_a's Elo relative to tier_b in THIS "
        "matchup only -- not directly comparable across different pairings the way a "
        "shared Stockfish reference point is (see chess_eval.stockfish). Use this to "
        "sanity-check ordering (does 2000 beat 1500 convincingly?), not as a unified "
        "rating scale on its own."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tiers", help="Comma-separated tier names, e.g. '1000,1500,2000'"
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_PAIRING)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="Save results as JSON to this path"
    )
    args = parser.parse_args()

    round_robin(
        args.tiers.split(","), args.games, out_path=args.out, workers=args.workers
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
