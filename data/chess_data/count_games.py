from __future__ import annotations

import argparse
from pathlib import Path

import chess.pgn
from tqdm import tqdm

from chess_data.prepare import _open_pgn_stream


def count_games(pgn_path: Path) -> int:
    count = 0
    with _open_pgn_stream(pgn_path) as f:
        pbar = tqdm(desc="counting games", unit="game")
        while True:
            headers = chess.pgn.read_headers(f)
            if headers is None:
                break
            count += 1
            pbar.update(1)
        pbar.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_path", type=Path, help="Path to .pgn or .pgn.zst file")
    args = parser.parse_args()

    total = count_games(args.pgn_path)
    print(f"{total} games in {args.pgn_path}")


if __name__ == "__main__":
    main()
