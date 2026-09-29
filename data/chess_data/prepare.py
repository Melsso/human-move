from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
from chess_shared import encode_board, perspective_move_to_index
from tqdm import tqdm

from chess_data.array_store import GrowableArrayStore
from chess_data.brackets import DEFAULT_BRACKETS, EloBracket, brackets_by_name
from chess_data.filters import MultiBucketGameBuilder
from chess_data.pgn_io import open_pgn_stream

DEFAULT_MAX_POSITIONS_PER_GAME = 40
DEFAULT_SKIP_FIRST_N_PLIES = 10
LOG_EVERY = 2000
INITIAL_CAPACITY = 50_000


class SourceExhaustedError(RuntimeError):
    pass


class _TrackingVisitorFactory:
    def __init__(self, brackets: list[EloBracket]) -> None:
        self._brackets = brackets
        self.last_instance: MultiBucketGameBuilder | None = None

    def __call__(self) -> MultiBucketGameBuilder:
        self.last_instance = MultiBucketGameBuilder(self._brackets)
        return self.last_instance

    @property
    def last_matched_bracket(self) -> EloBracket | None:
        if self.last_instance is None:
            return None
        return self.last_instance.matched_bracket


class _BucketBuffer:
    def __init__(
        self, board_shape: tuple[int, ...], initial_capacity: int, tmp_dir: Path
    ) -> None:
        self.boards = GrowableArrayStore(
            board_shape, np.float32, initial_capacity, tmp_dir
        )
        self.moves = GrowableArrayStore((), np.int64, initial_capacity, tmp_dir)
        self.game_ids = GrowableArrayStore((), np.int64, initial_capacity, tmp_dir)

    @property
    def count(self) -> int:
        return self.moves.count

    def append(self, board: np.ndarray, move: int, game_id: int) -> None:
        self.boards.append(board)
        self.moves.append(move)
        self.game_ids.append(game_id)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            boards=self.boards.finalized(),
            moves=self.moves.finalized(),
            game_ids=self.game_ids.finalized(),
        )

    def cleanup(self) -> None:
        self.boards.cleanup()
        self.moves.cleanup()
        self.game_ids.cleanup()


def extract_positions(
    game: chess.pgn.Game,
    max_positions_per_game: int = DEFAULT_MAX_POSITIONS_PER_GAME,
    skip_first_n_plies: int = DEFAULT_SKIP_FIRST_N_PLIES,
) -> list[tuple[np.ndarray, int]]:
    board = game.board()
    candidates: list[tuple[np.ndarray, int]] = []
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= skip_first_n_plies:
            board_tensor = encode_board(board)
            move_idx = perspective_move_to_index(move, board.turn)
            candidates.append((board_tensor, move_idx))
        board.push(move)

    if len(candidates) <= max_positions_per_game:
        return candidates

    sample_positions = np.linspace(0, len(candidates) - 1, num=max_positions_per_game)
    sample_indices = np.unique(sample_positions.round().astype(int))
    return [candidates[i] for i in sample_indices]


def _skip_games(stream: TextIO, count: int, chunk: int) -> None:
    with tqdm(
        total=count, desc=f"chunk {chunk}: skipping to start", unit="game"
    ) as bar:
        for skipped in range(count):
            if chess.pgn.read_headers(stream) is None:
                raise SourceExhaustedError(
                    f"skip_games={count} but the file only has {skipped} "
                    f"games -- nothing left to process for chunk {chunk}."
                )
            bar.update(1)


def process_chunk(
    pgn_path: Path,
    out_dir: Path,
    brackets: list[EloBracket],
    chunk: int,
    chunk_size: int,
    max_positions_per_game: int = DEFAULT_MAX_POSITIONS_PER_GAME,
    skip_first_n_plies: int = DEFAULT_SKIP_FIRST_N_PLIES,
) -> dict[str, int]:
    skip_games = (chunk - 1) * chunk_size
    games_seen = 0
    visitor_factory = _TrackingVisitorFactory(brackets)

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".prepare-tmp-"
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        buffers: dict[str, _BucketBuffer] = {}

        with open_pgn_stream(pgn_path) as stream:
            if skip_games > 0:
                _skip_games(stream, skip_games, chunk)

            with tqdm(desc=f"chunk {chunk}", unit="game", total=chunk_size) as pbar:
                while games_seen < chunk_size:
                    game = chess.pgn.read_game(stream, Visitor=visitor_factory)
                    if game is None:
                        break
                    games_seen += 1
                    pbar.update(1)

                    bracket = visitor_factory.last_matched_bracket
                    if bracket is not None and game.mainline_moves():
                        for board_tensor, move_idx in extract_positions(
                            game, max_positions_per_game, skip_first_n_plies
                        ):
                            buffer = buffers.get(bracket.name)
                            if buffer is None:
                                buffer = _BucketBuffer(
                                    board_tensor.shape, INITIAL_CAPACITY, tmp_dir
                                )
                                buffers[bracket.name] = buffer
                            buffer.append(
                                board_tensor, move_idx, skip_games + games_seen
                            )

                    if games_seen % LOG_EVERY == 0:
                        pbar.set_postfix({n: b.count for n, b in buffers.items()})

        counts: dict[str, int] = {}
        for bracket in brackets:
            buffer = buffers.get(bracket.name)
            if buffer is None or buffer.count == 0:
                counts[bracket.name] = 0
                continue
            counts[bracket.name] = buffer.count
            buffer.save(out_dir / f"bucket_{bracket.name}_{chunk}.npz")
            buffer.cleanup()

    return counts


def prepare_all_chunks(
    pgn_path: Path,
    out_dir: Path,
    brackets: list[EloBracket],
    chunk_size: int,
    max_chunks: int | None = None,
    max_positions_per_game: int = DEFAULT_MAX_POSITIONS_PER_GAME,
    skip_first_n_plies: int = DEFAULT_SKIP_FIRST_N_PLIES,
) -> None:
    chunk = 1
    totals: dict[str, int] = {b.name: 0 for b in brackets}

    while True:
        if max_chunks is not None and chunk > max_chunks:
            print(f"\nreached --max-chunks={max_chunks}, stopping")
            break

        print(
            f"\n=== chunk {chunk} "
            f"(games {(chunk - 1) * chunk_size + 1}-{chunk * chunk_size}) ==="
        )
        try:
            counts = process_chunk(
                pgn_path,
                out_dir,
                brackets,
                chunk,
                chunk_size,
                max_positions_per_game=max_positions_per_game,
                skip_first_n_plies=skip_first_n_plies,
            )
        except SourceExhaustedError as e:
            print(f"{e}")
            print(f"source exhausted after {chunk - 1} chunk(s) -- done")
            break

        for name, n in counts.items():
            totals[name] += n
            if n > 0:
                print(f"  bucket_{name}_{chunk}.npz: {n} positions")
            else:
                print(
                    f"  bucket_{name}: no matching games in this chunk, no file written"
                )

        chunk += 1

    print("\n=== summary across all chunks ===")
    for name, total in totals.items():
        if total > 0:
            print(f"  {name}: {total} total positions")
        else:
            print(
                f"  {name}: 0 total positions -- WARNING: this bucket's elo range "
                "never matched anything in the whole file, check its range in "
                "chess_data/brackets.py"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_path", type=Path, help="Path to .pgn or .pgn.zst file")
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Directory to write bucket_<name>_<chunk>.npz files into",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Games per chunk, scanned across all buckets together. Run "
        "chess_data.count_games first to know the total game count.",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default=None,
        help="Comma-separated bucket names to build (default: all -- "
        + ",".join(b.name for b in DEFAULT_BRACKETS)
        + ").",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after this many chunks (default: process until exhausted).",
    )
    parser.add_argument(
        "--max-positions-per-game", type=int, default=DEFAULT_MAX_POSITIONS_PER_GAME
    )
    parser.add_argument(
        "--skip-first-n-plies", type=int, default=DEFAULT_SKIP_FIRST_N_PLIES
    )
    args = parser.parse_args()

    bucket_names = args.buckets.split(",") if args.buckets else None
    brackets = brackets_by_name(bucket_names)

    prepare_all_chunks(
        args.pgn_path,
        args.out_dir,
        brackets,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        max_positions_per_game=args.max_positions_per_game,
        skip_first_n_plies=args.skip_first_n_plies,
    )


if __name__ == "__main__":
    main()
