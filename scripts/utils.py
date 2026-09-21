from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import platform
import random
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import chess
import chess.engine
from chess_backend.inference import (
    ModelRegistry,
    compute_model_move,
)

TIER = "1500"
MOVE_TIME_MS = 50
THREADS = 1
WORKERS = 6
OPENING_PLIES = 8

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / ".stockfish"
ENGINE_PATH = ENGINE_DIR / "stockfish"

registry = ModelRegistry()


def download_stockfish() -> Path:
    if ENGINE_PATH.exists():
        return ENGINE_PATH

    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    machine = platform.machine().lower()
    api_url = (
        "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
    )

    request = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "chess-model-benchmark",
        },
    )

    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    assets = release["assets"]
    candidates = [
        asset
        for asset in assets
        if asset["name"].lower() == "stockfish-macos-universal.tar.gz"
    ]
    if not candidates:
        available = "\n".join(asset["name"] for asset in assets)
        raise RuntimeError(
            f"Could not find a Stockfish binary for "
            f"{system}/{machine}.\n\n"
            f"Available assets:\n{available}"
        )

    asset = candidates[0]
    print(f"Downloading {release['name']}: {asset['name']}")
    download_request = urllib.request.Request(
        asset["browser_download_url"],
        headers={
            "User-Agent": "chess-model-benchmark",
        },
    )
    with urllib.request.urlopen(download_request) as response:
        data = response.read()

    if asset["name"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            executable_candidates = [
                name
                for name in archive.namelist()
                if Path(name).name.lower()
                in {
                    "stockfish.exe",
                    "stockfish",
                }
            ]
            if not executable_candidates:
                raise RuntimeError(
                    f"Could not find Stockfish executable in {asset['name']}"
                )
            executable = executable_candidates[0]
            with archive.open(executable) as source:
                ENGINE_PATH.write_bytes(source.read())

    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            executable_candidates = [
                member
                for member in archive.getmembers()
                if member.isfile() and "stockfish" in Path(member.name).name.lower()
            ]
            if not executable_candidates:
                raise RuntimeError(
                    f"Could not find Stockfish executable in {asset['name']}"
                )
            member = executable_candidates[0]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError("Could not extract Stockfish binary")
            ENGINE_PATH.write_bytes(extracted.read())

    mode = ENGINE_PATH.stat().st_mode
    ENGINE_PATH.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Stockfish ready: {ENGINE_PATH}")
    return ENGINE_PATH


def model_move(board: chess.Board) -> chess.Move:
    model = registry.get_model(TIER)
    move, _ = compute_model_move(
        model,
        board,
        temperature=0.0,
        top_k=1,
    )

    return move


def score(wins: int, draws: int, losses: int) -> float:
    total = wins + draws + losses

    if total == 0:
        return 0.0

    return (wins + 0.5 * draws) / total


def elo_difference(score_value: float) -> float:
    score_value = max(0.001, min(0.999, score_value))
    return 400 * math.log10(score_value / (1 - score_value))


def random_opening(seed: int, plies: int = OPENING_PLIES) -> chess.Board:
    rng = random.Random(seed)
    board = chess.Board()

    for _ in range(plies):
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            break

        board.push(rng.choice(legal_moves))

    return board


def play_game(
    engine: chess.engine.SimpleEngine, model_is_white: bool, seed: int
) -> str:
    board = random_opening(seed=seed)

    while not board.is_game_over(claim_draw=True):
        model_turn = (
            board.turn == chess.WHITE if model_is_white else board.turn == chess.BLACK
        )

        if model_turn:
            move = model_move(board)
            if move not in board.legal_moves:
                raise RuntimeError(
                    f"Model produced illegal move {move} on position {board.fen()}"
                )
            board.push(move)

        else:
            result = engine.play(board, chess.engine.Limit(time=MOVE_TIME_MS / 1000))
            board.push(result.move)

    return board.result(claim_draw=True)


def play_single_game(args: tuple[int, int]) -> tuple[int, str, str, str]:
    game_number, stockfish_elo = args
    model_is_white = game_number % 2 == 0
    model_color = "White" if model_is_white else "Black"
    seed = stockfish_elo * 10000 + game_number

    engine = chess.engine.SimpleEngine.popen_uci(str(ENGINE_PATH))
    engine.configure(
        {
            "Threads": THREADS,
            "UCI_LimitStrength": True,
            "UCI_Elo": stockfish_elo,
        }
    )
    try:
        result = play_game(engine, model_is_white=model_is_white, seed=seed)
    finally:
        engine.quit()

    if model_is_white:
        if result == "1-0":
            model_result = "WIN"
        elif result == "0-1":
            model_result = "LOSS"
        else:
            model_result = "DRAW"

    else:
        if result == "0-1":
            model_result = "WIN"
        elif result == "1-0":
            model_result = "LOSS"
        else:
            model_result = "DRAW"

    return (game_number, model_color, model_result, result)


def run_match(stockfish_elo: int, games: int) -> tuple[int, int, int]:
    wins, draws, losses = 0, 0, 0
    worker_count = min(WORKERS, games)
    print(f"Running {games} games with {worker_count} parallel workers...")
    jobs = [(game_number, stockfish_elo) for game_number in range(games)]
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=worker_count) as pool:
        results = pool.imap_unordered(play_single_game, jobs)
        completed = 0

        for completed, (
            game_number,
            model_color,
            model_result,
            raw_result,
        ) in enumerate(results, start=1):
            stockfish_color = "Black" if model_color == "White" else "White"

            if model_result == "WIN":
                wins += 1
            elif model_result == "DRAW":
                draws += 1
            else:
                losses += 1

            print(
                f"game {game_number + 1:3d}/{games} | "
                f"Model: {model_color:<5} | "
                f"Stockfish: {stockfish_color:<5} | "
                f"Model: {model_result:<4} | "
                f"Result: {raw_result} | "
                f"Completed: {completed}/{games}",
                flush=True,
            )

    return (wins, draws, losses)
