from __future__ import annotations

import argparse
from pathlib import Path

import chess.pgn
from tqdm import tqdm

from chess_data.pgn_io import open_pgn_stream


def count_games(pgn_path: Path) -> int:
    count = 0
    with (
        open_pgn_stream(pgn_path) as stream,
        tqdm(desc="counting games", unit="game") as pbar,
    ):
        while chess.pgn.read_headers(stream) is not None:
            count += 1
            pbar.update(1)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_path", type=Path, help="Path to .pgn or .pgn.zst file")
    args = parser.parse_args()

    total = count_games(args.pgn_path)
    print(f"{total} games in {args.pgn_path}")


if __name__ == "__main__":
    main()
