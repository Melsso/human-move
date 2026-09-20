from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt
import zstandard as zstd
from chess_shared import encode_board, move_to_index
from tqdm import tqdm

from chess_data.brackets import DEFAULT_BRACKETS, EloBracket, brackets_by_name
from chess_data.filters import MultiBucketGameBuilder


class SourceExhaustedError(RuntimeError):
    pass


def _open_pgn_stream(path: Path) -> TextIO:
    """Return a text-mode file-like object, transparently decompressing .zst."""
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(path.open("rb"))
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


class _TrackingVisitorFactory:
    def __init__(self, brackets: list[EloBracket]) -> None:
        self._brackets = brackets
        self.last_instance: MultiBucketGameBuilder | None = None

    def __call__(self) -> MultiBucketGameBuilder:
        self.last_instance = MultiBucketGameBuilder(self._brackets)
        return self.last_instance


class _GrowableArrayStore:
    def __init__(
        self,
        row_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        initial_capacity: int,
        tmp_dir: Path,
    ) -> None:
        self._row_shape = row_shape
        self._dtype = dtype
        self._tmp_dir = tmp_dir
        self._capacity = max(initial_capacity, 1)
        self._count = 0
        self._path = self._new_backing_path()
        self._array = np.memmap(
            self._path, dtype=dtype, mode="w+", shape=(self._capacity, *row_shape)
        )

    def _new_backing_path(self) -> Path:
        fd, name = tempfile.mkstemp(dir=self._tmp_dir, suffix=".dat")
        os.close(fd)
        return Path(name)

    @property
    def count(self) -> int:
        return self._count

    def append(self, row: object) -> None:
        if self._count >= self._capacity:
            self._grow()
        self._array[self._count] = row
        self._count += 1

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        new_path = self._new_backing_path()
        new_array = np.memmap(
            new_path,
            dtype=self._dtype,
            mode="w+",
            shape=(new_capacity, *self._row_shape),
        )
        new_array[: self._count] = self._array[: self._count]
        new_array.flush()

        old_path = self._path
        self._array = new_array
        self._path = new_path
        self._capacity = new_capacity
        old_path.unlink(missing_ok=True)

    def finalized(self) -> np.ndarray:
        self._array.flush()
        return self._array[: self._count]

    def cleanup(self) -> None:
        del self._array
        self._path.unlink(missing_ok=True)


def extract_positions(
    game: chess.pgn.Game,
    max_positions_per_game: int = 40,
    skip_first_n_plies: int = 6,
) -> list[tuple[np.ndarray, int]]:
    positions = []
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= skip_first_n_plies:
            positions.append((encode_board(board), move_to_index(move)))
            if len(positions) >= max_positions_per_game:
                break
        board.push(move)
    return positions


def process_chunk(
    pgn_path: Path,
    out_dir: Path,
    brackets: list[EloBracket],
    chunk: int,
    chunk_size: int,
    max_positions_per_game: int = 40,
    skip_first_n_plies: int = 6,
    log_every: int = 2000,
    initial_capacity: int = 50_000,
) -> dict[str, int]:
    skip_games = (chunk - 1) * chunk_size
    games_seen = 0

    visitor_factory = _TrackingVisitorFactory(brackets)

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=".prepare-tmp-"
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        boards_stores: dict[str, _GrowableArrayStore] = {}
        moves_stores: dict[str, _GrowableArrayStore] = {}

        with _open_pgn_stream(pgn_path) as f:
            if skip_games > 0:
                skip_pbar = tqdm(
                    total=skip_games,
                    desc=f"chunk {chunk}: skipping to start",
                    unit="game",
                )
                skipped = 0
                while skipped < skip_games:
                    headers = chess.pgn.read_headers(f)
                    if headers is None:
                        skip_pbar.close()
                        raise SourceExhaustedError(
                            f"skip_games={skip_games} but the file only has {skipped} "
                            f"games -- nothing left to process for chunk {chunk}."
                        )
                    skipped += 1
                    skip_pbar.update(1)
                skip_pbar.close()

            pbar = tqdm(desc=f"chunk {chunk}", unit="game", total=chunk_size)
            while games_seen < chunk_size:
                game = chess.pgn.read_game(f, Visitor=visitor_factory)
                if game is None:
                    break
                games_seen += 1
                pbar.update(1)

                matched_bracket = visitor_factory.last_instance.matched_bracket  # type: ignore[union-attr]
                if matched_bracket is not None and game.mainline_moves():
                    name = matched_bracket.name
                    for board_tensor, move_idx in extract_positions(
                        game, max_positions_per_game, skip_first_n_plies
                    ):
                        if name not in boards_stores:
                            boards_stores[name] = _GrowableArrayStore(
                                board_tensor.shape,
                                np.float32,
                                initial_capacity,
                                tmp_dir,
                            )
                            moves_stores[name] = _GrowableArrayStore(
                                (), np.int64, initial_capacity, tmp_dir
                            )
                        boards_stores[name].append(board_tensor)
                        moves_stores[name].append(move_idx)

                if games_seen % log_every == 0:
                    pbar.set_postfix(
                        {name: s.count for name, s in moves_stores.items()}
                    )
            pbar.close()

        counts: dict[str, int] = {}
        for bracket in brackets:
            name = bracket.name
            if name not in boards_stores or boards_stores[name].count == 0:
                counts[name] = 0
                continue

            boards_view = boards_stores[name].finalized()
            moves_view = moves_stores[name].finalized()
            counts[name] = boards_stores[name].count

            out_path = out_dir / f"bucket_{name}_{chunk}.npz"
            np.savez_compressed(out_path, boards=boards_view, moves=moves_view)

            boards_stores[name].cleanup()
            moves_stores[name].cleanup()

    return counts


def prepare_all_chunks(
    pgn_path: Path,
    out_dir: Path,
    brackets: list[EloBracket],
    chunk_size: int,
    max_chunks: int | None = None,
    max_positions_per_game: int = 40,
    skip_first_n_plies: int = 6,
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        help="Games per chunk, scanned across ALL buckets together. Run "
        "chess_data.count_games first if you want to know the total game "
        "count before picking a value.",
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
        help="Stop after this many chunks (useful for a quick sample run instead "
        "of processing the whole file; default: process until exhausted).",
    )
    parser.add_argument("--max-positions-per-game", type=int, default=40)
    parser.add_argument("--skip-first-n-plies", type=int, default=6)
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
