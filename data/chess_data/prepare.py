"""
Turns a raw Lichess PGN dump (optionally .zst-compressed) into
a compact .npz file of (board_tensor, move_index) pairs,
filtered down to one rating bucket.

Usage:
    python -m chess_data.prepare \
        data/downloads/lichess_2024-06.pgn.zst \
        data/processed/bucket_1000.npz \
        --min-elo 900 --max-elo 1100 \
        --max-games 200000

Lichess dumps are large (a single month can be 30GB+ compressed). We stream
through the file game-by-game rather than loading it into memory, and read
directly out of the .zst stream so you never need to keep a decompressed
copy on disk.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import zstandard as zstd
from chess_shared import encode_board, move_to_index
from tqdm import tqdm

from chess_data.filters import should_keep_game


def _open_pgn_stream(path: Path) -> TextIO:
    """Return a text-mode file-like object, transparently decompressing .zst."""
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(path.open("rb"))
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def extract_positions(
    game: chess.pgn.Game,
    max_positions_per_game: int = 40,
    skip_first_n_plies: int = 6,
) -> list[tuple[np.ndarray, int]]:
    """
    Walk through a game's moves, yielding (board_before_move, move_played)
    pairs. We skip the first few plies (book openings are memorized, not
    "decided", and are the same across every rating band, so they carry
    little signal about skill level) and cap positions per game so one very
    long game can't dominate the dataset.
    """
    positions = []
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= skip_first_n_plies:
            positions.append((encode_board(board), move_to_index(move)))
            if len(positions) >= max_positions_per_game:
                break
        board.push(move)
    return positions


def process_pgn(
    pgn_path: Path,
    out_path: Path,
    min_elo: int,
    max_elo: int,
    max_games: int | None = None,
    max_positions_per_game: int = 40,
    skip_first_n_plies: int = 6,
    log_every: int = 2000,
) -> None:
    boards: list[np.ndarray] = []
    moves: list[int] = []
    games_seen = 0
    games_kept = 0

    with _open_pgn_stream(pgn_path) as f:
        pbar = tqdm(desc=f"[{min_elo}-{max_elo}]", unit="game")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1
            pbar.update(1)

            if should_keep_game(game, min_elo, max_elo):
                games_kept += 1
                for board_tensor, move_idx in extract_positions(
                    game, max_positions_per_game, skip_first_n_plies
                ):
                    boards.append(board_tensor)
                    moves.append(move_idx)

            if games_seen % log_every == 0:
                pbar.set_postfix(kept=games_kept, positions=len(boards))

            if max_games is not None and games_seen >= max_games:
                break
        pbar.close()

    if not boards:
        raise RuntimeError(
            f"No positions extracted. Scanned {games_seen} games, kept {games_kept}. "
            "Check your elo range and that the PGN actually has rated standard games."
        )

    boards_arr = np.stack(boards).astype(np.float32)
    moves_arr = np.array(moves, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, boards=boards_arr, moves=moves_arr)

    print(
        f"done: kept {games_kept}/{games_seen} games "
        f"-> {len(boards)} positions -> {out_path} "
        f"({boards_arr.nbytes / 1e6:.1f}MB boards, {moves_arr.nbytes / 1e6:.1f}MB moves)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_path", type=Path, help="Path to .pgn or .pgn.zst file")
    parser.add_argument("out_path", type=Path, help="Path to write .npz output")
    parser.add_argument("--min-elo", type=int, required=True)
    parser.add_argument("--max-elo", type=int, required=True)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-positions-per-game", type=int, default=40)
    parser.add_argument("--skip-first-n-plies", type=int, default=6)
    args = parser.parse_args()

    process_pgn(
        args.pgn_path,
        args.out_path,
        args.min_elo,
        args.max_elo,
        max_games=args.max_games,
        max_positions_per_game=args.max_positions_per_game,
        skip_first_n_plies=args.skip_first_n_plies,
    )


if __name__ == "__main__":
    main()
