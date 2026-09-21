from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
from chess_backend.inference import ModelRegistry, compute_model_move
from chess_training.model import MaiaPolicyNet

from chess_eval.common import classify_result, random_opening
from chess_eval.stats import MatchResult, elo_estimate

DEFAULT_STOCKFISH_ELOS = [1320, 1500, 1700, 1900, 2100]
DEFAULT_GAMES_PER_LEVEL = 100
DEFAULT_MOVE_TIME_MS = 300


def find_stockfish() -> Path:
    path = shutil.which("stockfish")
    if path is None:
        raise RuntimeError(
            "stockfish not found on PATH. Install it first: "
            "'brew install stockfish' (macOS), 'apt install stockfish' (Debian/Ubuntu), "
            "'dnf install stockfish' (Fedora), or download a build from "
            "https://stockfishchess.org/download/ and make sure it's on PATH."
        )
    return Path(path)


@dataclass
class GameOutcome:
    game_number: int
    model_color: str
    result: str
    raw_result: str
    error: str | None = None


def play_game(
    stockfish_path: Path,
    model: MaiaPolicyNet,
    stockfish_elo: int,
    model_is_white: bool,
    seed: int,
    move_time_ms: int,
    threads: int,
) -> str:
    board = random_opening(seed=seed)
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
    try:
        engine.configure(
            {"Threads": threads, "UCI_LimitStrength": True, "UCI_Elo": stockfish_elo}
        )
        while not board.is_game_over(claim_draw=True):
            model_turn = (
                board.turn == chess.WHITE
                if model_is_white
                else board.turn == chess.BLACK
            )
            if model_turn:
                move, _ = compute_model_move(model, board, temperature=0.0, top_k=1)
                if move not in board.legal_moves:
                    raise RuntimeError(
                        f"model produced illegal move {move} on position {board.fen()}"
                    )
                board.push(move)
            else:
                limit = chess.engine.Limit(time=move_time_ms / 1000)
                result = engine.play(board, limit)
                assert result.move is not None
                board.push(result.move)
    finally:
        engine.quit()

    return board.result(claim_draw=True)


_worker_model: MaiaPolicyNet | None = None
_worker_stockfish_path: Path | None = None


def _init_worker(tier: str, checkpoints_dir: str | None, stockfish_path: str) -> None:
    global _worker_model, _worker_stockfish_path
    registry = (
        ModelRegistry(Path(checkpoints_dir)) if checkpoints_dir else ModelRegistry()
    )
    _worker_model = registry.get_model(tier)
    _worker_stockfish_path = Path(stockfish_path)


def _play_single_game(args: tuple[int, int, int, int]) -> GameOutcome:
    game_number, stockfish_elo, move_time_ms, threads = args
    assert _worker_model is not None and _worker_stockfish_path is not None

    model_is_white = game_number % 2 == 0
    model_color = "White" if model_is_white else "Black"
    seed = stockfish_elo * 10_000 + game_number

    try:
        raw_result = play_game(
            _worker_stockfish_path,
            _worker_model,
            stockfish_elo,
            model_is_white,
            seed,
            move_time_ms,
            threads,
        )
    except chess.engine.EngineError as e:
        return GameOutcome(game_number, model_color, "ERROR", "*", error=str(e))

    outcome = classify_result(raw_result, model_is_white)
    return GameOutcome(game_number, model_color, outcome, raw_result)


def run_match(
    stockfish_path: Path,
    tier: str,
    stockfish_elo: int,
    games: int,
    checkpoints_dir: Path | None = None,
    move_time_ms: int = DEFAULT_MOVE_TIME_MS,
    threads: int = 1,
    workers: int | None = None,
) -> tuple[MatchResult, list[GameOutcome]]:
    workers = workers or max(1, min(mp.cpu_count(), games))
    jobs = [(i, stockfish_elo, move_time_ms, threads) for i in range(games)]

    ctx = mp.get_context("spawn")
    outcomes: list[GameOutcome] = []
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            tier,
            str(checkpoints_dir) if checkpoints_dir else None,
            str(stockfish_path),
        ),
    ) as pool:
        for outcome in pool.imap_unordered(_play_single_game, jobs):
            outcomes.append(outcome)
            status = (
                outcome.result
                if outcome.result != "ERROR"
                else f"ERROR ({outcome.error})"
            )
            print(
                f"  game {outcome.game_number + 1:3d}/{games} | "
                f"model: {outcome.model_color:<5} | {status:<6} | raw: {outcome.raw_result}",
                flush=True,
            )

    errors = [o for o in outcomes if o.result == "ERROR"]
    if errors:
        print(
            f"  WARNING: {len(errors)}/{games} games errored (engine issue, not scored) "
            "and are excluded below"
        )

    wins = sum(1 for o in outcomes if o.result == "WIN")
    draws = sum(1 for o in outcomes if o.result == "DRAW")
    losses = sum(1 for o in outcomes if o.result == "LOSS")
    return MatchResult(wins, draws, losses), outcomes


def calibrate(
    tier: str,
    stockfish_elos: list[int] | None = None,
    games_per_level: int = DEFAULT_GAMES_PER_LEVEL,
    checkpoints_dir: Path | None = None,
    move_time_ms: int = DEFAULT_MOVE_TIME_MS,
    threads: int = 1,
    workers: int | None = None,
    out_path: Path | None = None,
) -> dict[str, object]:
    stockfish_elos = stockfish_elos or DEFAULT_STOCKFISH_ELOS
    stockfish_path = find_stockfish()
    print(f"stockfish: {stockfish_path}")
    print(f"model tier: {tier}")
    print(f"games per level: {games_per_level}, move time: {move_time_ms}ms\n")

    level_results = []
    for elo in stockfish_elos:
        print(f"=== stockfish {elo} ===")
        start = time.time()
        match_result, _outcomes = run_match(
            stockfish_path,
            tier,
            elo,
            games_per_level,
            checkpoints_dir,
            move_time_ms,
            threads,
            workers,
        )
        elapsed = time.time() - start
        estimate = elo_estimate(match_result)
        implied = elo + estimate.point
        implied_low = elo + estimate.low
        implied_high = elo + estimate.high

        print(
            f"  W{match_result.wins}/D{match_result.draws}/L{match_result.losses} "
            f"score={match_result.score * 100:.1f}% "
            f"implied={implied:.0f} [{implied_low:.0f}, {implied_high:.0f}] "
            f"({elapsed:.0f}s)\n"
        )

        level_results.append(
            {
                "stockfish_elo": elo,
                "wins": match_result.wins,
                "draws": match_result.draws,
                "losses": match_result.losses,
                "score": match_result.score,
                "implied_elo": implied,
                "implied_elo_low": implied_low,
                "implied_elo_high": implied_high,
            }
        )

    summary: dict[str, object] = {
        "tier": tier,
        "games_per_level": games_per_level,
        "move_time_ms": move_time_ms,
        "levels": level_results,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"saved results to {out_path}")

    print(
        "\nIMPORTANT: these are relative benchmarks against Stockfish's own UCI_Elo "
        "weakening, NOT calibrated against real human Lichess ratings -- useful for "
        "comparing your own buckets and tracking changes over time, not as an exact "
        "claim about human-equivalent strength. Also: every number above carries a "
        "wide confidence interval at this game count -- read the bracket, not just "
        "the point estimate."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tier", help="Tier name to benchmark, e.g. '1000'")
    parser.add_argument(
        "--stockfish-elos",
        type=str,
        default=",".join(str(e) for e in DEFAULT_STOCKFISH_ELOS),
        help="Comma-separated Stockfish UCI_Elo levels to test against",
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LEVEL)
    parser.add_argument("--move-time-ms", type=int, default=DEFAULT_MOVE_TIME_MS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="Save results as JSON to this path"
    )
    args = parser.parse_args()

    elos = [int(e) for e in args.stockfish_elos.split(",")]
    calibrate(
        args.tier,
        elos,
        args.games,
        move_time_ms=args.move_time_ms,
        threads=args.threads,
        workers=args.workers,
        out_path=args.out,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
