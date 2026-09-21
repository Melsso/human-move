from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import chess
from chess_backend.inference import ModelRegistry, compute_model_move
from chess_training.model import MaiaPolicyNet

from chess_eval.common import classify_result, random_opening
from chess_eval.stats import MatchResult, elo_estimate

DEFAULT_GAMES_PER_PAIRING = 500


def play_game(
    model_a: MaiaPolicyNet,
    model_b: MaiaPolicyNet,
    a_is_white: bool,
    seed: int,
) -> str:
    board = random_opening(seed=seed)
    while not board.is_game_over(claim_draw=True):
        a_turn = board.turn == chess.WHITE if a_is_white else board.turn == chess.BLACK
        model = model_a if a_turn else model_b
        move, _ = compute_model_move(model, board, temperature=0.3, top_k=1)
        if move not in board.legal_moves:
            raise RuntimeError(
                f"model produced illegal move {move} on position {board.fen()}"
            )
        board.push(move)
    return board.result(claim_draw=True)


def run_pairing(
    tier_a: str,
    tier_b: str,
    games: int,
    checkpoints_dir: Path | None = None,
) -> MatchResult:
    registry = ModelRegistry(checkpoints_dir) if checkpoints_dir else ModelRegistry()
    model_a = registry.get_model(tier_a)
    model_b = registry.get_model(tier_b)

    wins = draws = losses = 0
    for i in range(games):
        a_is_white = i % 2 == 0
        seed = _stable_seed(tier_a, tier_b, i)
        raw_result = play_game(model_a, model_b, a_is_white, seed)
        outcome = classify_result(raw_result, a_is_white)

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        else:
            draws += 1

        print(f"  game {i + 1}/{games}: {tier_a} {outcome} ({raw_result})", flush=True)

    return MatchResult(wins, draws, losses)


def _stable_seed(tier_a: str, tier_b: str, game_number: int) -> int:
    digest = hashlib.sha256(f"{tier_a}:{tier_b}:{game_number}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def round_robin(
    tiers: list[str],
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    checkpoints_dir: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, object]:
    if len(tiers) < 2:
        raise ValueError(f"need at least 2 tiers for a round robin, got {tiers}")

    pairings = list(itertools.combinations(tiers, 2))
    results = []

    for tier_a, tier_b in pairings:
        print(f"=== {tier_a} vs {tier_b} ===")
        match_result = run_pairing(tier_a, tier_b, games_per_pairing, checkpoints_dir)
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
        "sanity-check ordering (does 2000 beat 1000 convincingly?), not as a unified "
        "rating scale on its own."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "tiers", help="Comma-separated tier names, e.g. '1000,1500,2000'"
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_PAIRING)
    parser.add_argument(
        "--out", type=Path, default=None, help="Save results as JSON to this path"
    )
    args = parser.parse_args()

    tiers = args.tiers.split(",")
    round_robin(tiers, args.games, out_path=args.out)


if __name__ == "__main__":
    main()
