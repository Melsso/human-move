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

Filtering happens header-first: a game's headers are enough to decide
whether we want it (see chess_data.filters.FilteringGameBuilder), so games
that don't pass the filter never have their movetext tokenized, SAN-parsed,
or pushed onto a board -- only the (usually small minority of) games we
actually keep pay that cost.

CHUNKED PROCESSING (--skip-games / --max-games together): rather than
processing an entire multi-million-game dump into one giant .npz (which
costs a lot of RAM to train on later, .npz can never be truly memory-
mapped regardless of compression), process it in slices and train
incrementally across them instead -- see the root README's "chunked
training" section. --skip-games N tells this script to fast-forward past
the first N games (using the same cheap header-only read as the real
filter, so skipping is fast even for a large N) before it starts actually
processing the next --max-games games for this chunk.
"""

from __future__ import annotations

import argparse
import functools
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

from chess_data.filters import FilteringGameBuilder


def _open_pgn_stream(path: Path) -> TextIO:
    """Return a text-mode file-like object, transparently decompressing .zst."""
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(path.open("rb"))
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


class _GrowableArrayStore:
    """
    Accumulates fixed-shape rows into a numpy array backed by an on-disk
    memmap, instead of a Python list of arrays.

    A Python list of tens of millions of small numpy arrays is expensive in
    two ways: each array carries its own object overhead on top of its data,
    and everything has to live in RAM at once, with a final np.stack() that
    briefly needs a second, contiguous copy of the whole thing. Backing the
    growing array with a memmap instead means the data mostly lives on disk
    and is paged in by the OS as needed, so resident memory stays low no
    matter how large the final dataset is.

    Capacity grows by doubling (like a dynamic array): when full, a new,
    larger backing file is allocated and the existing rows are copied over
    via a memmap-to-memmap assignment (which streams through the page cache
    rather than materializing in Python), then the old backing file is
    dropped. This keeps the number of copies logarithmic in the final size.
    """

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
        """A memmap view trimmed to the rows actually written (no copy)."""
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
    skip_games: int = 0,
    max_positions_per_game: int = 40,
    skip_first_n_plies: int = 6,
    log_every: int = 2000,
    initial_capacity: int = 1_000_000,
) -> None:
    games_seen = 0
    games_kept = 0

    visitor_factory = functools.partial(
        FilteringGameBuilder, min_elo=min_elo, max_elo=max_elo
    )

    if max_games is not None:
        initial_capacity = max(
            1, min(initial_capacity, max_games * max_positions_per_game)
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=out_path.parent, prefix=".prepare-tmp-"
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        moves_store = _GrowableArrayStore((), np.int64, initial_capacity, tmp_dir)
        boards_store: _GrowableArrayStore | None = None

        with _open_pgn_stream(pgn_path) as f:
            if skip_games > 0:
                skip_pbar = tqdm(
                    total=skip_games, desc="skipping to chunk start", unit="game"
                )
                skipped = 0
                while skipped < skip_games:
                    headers = chess.pgn.read_headers(f)
                    if headers is None:
                        skip_pbar.close()
                        raise RuntimeError(
                            f"skip_games={skip_games} but the file only has {skipped} games "
                            "-- nothing left to process in this chunk."
                        )
                    skipped += 1
                    skip_pbar.update(1)
                skip_pbar.close()

            pbar = tqdm(desc=f"[{min_elo}-{max_elo}]", unit="game")
            while True:
                game = chess.pgn.read_game(f, Visitor=visitor_factory)
                if game is None:
                    break
                games_seen += 1
                pbar.update(1)

                if game.mainline_moves():
                    games_kept += 1
                    for board_tensor, move_idx in extract_positions(
                        game, max_positions_per_game, skip_first_n_plies
                    ):
                        if boards_store is None:
                            boards_store = _GrowableArrayStore(
                                board_tensor.shape,
                                np.float32,
                                initial_capacity,
                                tmp_dir,
                            )
                        boards_store.append(board_tensor)
                        moves_store.append(move_idx)

                if games_seen % log_every == 0:
                    pbar.set_postfix(kept=games_kept, positions=moves_store.count)

                if max_games is not None and games_seen >= max_games:
                    break
            pbar.close()

        if boards_store is None or boards_store.count == 0:
            raise RuntimeError(
                f"No positions extracted. Scanned {games_seen} games, kept {games_kept}. "
                "Check your elo range and that the PGN actually has rated standard games."
            )

        boards_view = boards_store.finalized()
        moves_view = moves_store.finalized()
        n_positions = boards_store.count
        boards_nbytes = boards_view.nbytes
        moves_nbytes = moves_view.nbytes

        np.savez_compressed(out_path, boards=boards_view, moves=moves_view)

        boards_store.cleanup()
        moves_store.cleanup()

    print(
        f"done: kept {games_kept}/{games_seen} games "
        f"-> {n_positions} positions -> {out_path} "
        f"({boards_nbytes / 1e6:.1f}MB boards, {moves_nbytes / 1e6:.1f}MB moves)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_path", type=Path, help="Path to .pgn or .pgn.zst file")
    parser.add_argument("out_path", type=Path, help="Path to write .npz output")
    parser.add_argument("--min-elo", type=int, required=True)
    parser.add_argument("--max-elo", type=int, required=True)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument(
        "--skip-games",
        type=int,
        default=0,
        help="Fast-forward past this many games before processing -- use to pull "
        "successive chunks out of one big dump (e.g. chunk 2 of a 700k-game chunk "
        "size: --skip-games 700000 --max-games 700000).",
    )
    parser.add_argument("--max-positions-per-game", type=int, default=40)
    parser.add_argument("--skip-first-n-plies", type=int, default=6)
    args = parser.parse_args()

    process_pgn(
        args.pgn_path,
        args.out_path,
        args.min_elo,
        args.max_elo,
        max_games=args.max_games,
        skip_games=args.skip_games,
        max_positions_per_game=args.max_positions_per_game,
        skip_first_n_plies=args.skip_first_n_plies,
    )


if __name__ == "__main__":
    main()
