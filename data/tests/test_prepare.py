import io
import sys
from pathlib import Path

import chess.pgn
import numpy as np
import pytest
import zstandard as zstd
from chess_data.brackets import DEFAULT_BRACKETS, brackets_by_name
from chess_data.prepare import (
    SourceExhaustedError,
    _GrowableArrayStore,
    extract_positions,
    main,
    prepare_all_chunks,
    process_chunk,
)
from chess_shared import NUM_MOVES, NUM_PLANES

SAMPLE_PGN = """\
[Event "Rated Rapid game"]
[White "a"]
[Black "b"]
[WhiteElo "1000"]
[BlackElo "1020"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0

[Event "Rated Rapid game"]
[White "c"]
[Black "d"]
[WhiteElo "2250"]
[BlackElo "2290"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 0-1

[Event "Casual Rapid game"]
[White "a"]
[Black "b"]
[WhiteElo "1000"]
[BlackElo "1000"]
[Termination "Normal"]

1. e4 e5 1-0
"""


def _make_larger_pgn(n_repeats: int) -> str:
    return SAMPLE_PGN * n_repeats


def test_extract_positions_shapes() -> None:
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
    assert game is not None
    positions = extract_positions(game, max_positions_per_game=40, skip_first_n_plies=2)
    assert len(positions) > 0
    board_tensor, move_idx = positions[0]
    assert board_tensor.shape == (NUM_PLANES, 8, 8)
    assert 0 <= move_idx < NUM_MOVES


def test_process_chunk_writes_separate_files_per_matched_bracket(
    tmp_path: Path,
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    counts = process_chunk(
        pgn_path,
        out_dir,
        list(DEFAULT_BRACKETS),
        chunk=1,
        chunk_size=10,
        skip_first_n_plies=2,
    )

    assert counts["1000"] > 0
    assert counts["2000"] > 0
    assert counts["1500"] == 0

    assert (out_dir / "bucket_1000_1.npz").exists()
    assert (out_dir / "bucket_2000_1.npz").exists()
    assert not (out_dir / "bucket_1500_1.npz").exists()

    data_1000 = np.load(out_dir / "bucket_1000_1.npz")
    assert data_1000["boards"].shape[1:] == (NUM_PLANES, 8, 8)
    assert data_1000["boards"].shape[0] == counts["1000"]


def test_process_chunk_with_zst_source(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn.zst"
    cctx = zstd.ZstdCompressor()
    with open(pgn_path, "wb") as f:
        f.write(cctx.compress(SAMPLE_PGN.encode("utf-8")))
    out_dir = tmp_path / "out"

    counts = process_chunk(
        pgn_path,
        out_dir,
        list(DEFAULT_BRACKETS),
        chunk=1,
        chunk_size=10,
        skip_first_n_plies=2,
    )
    assert counts["1000"] > 0
    assert counts["2000"] > 0


def test_process_chunk_raises_source_exhausted_when_skip_exceeds_file(
    tmp_path: Path,
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    with pytest.raises(SourceExhaustedError) as exc_info:
        process_chunk(pgn_path, out_dir, list(DEFAULT_BRACKETS), chunk=10, chunk_size=1)
    assert "only has" in str(exc_info.value)


def test_process_chunk_handles_partial_final_chunk_without_error(
    tmp_path: Path,
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    counts = process_chunk(
        pgn_path,
        out_dir,
        list(DEFAULT_BRACKETS),
        chunk=1,
        chunk_size=100,
        skip_first_n_plies=2,
    )
    assert counts["1000"] > 0
    assert counts["2000"] > 0


def test_process_chunk_cleans_up_its_temp_directory(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    process_chunk(
        pgn_path,
        out_dir,
        list(DEFAULT_BRACKETS),
        chunk=1,
        chunk_size=10,
        skip_first_n_plies=2,
    )

    leftover = list(out_dir.glob(".prepare-tmp-*"))
    assert leftover == [], f"leftover temp dirs were not cleaned up: {leftover}"


def test_chunked_processing_matches_single_pass_processing(tmp_path: Path) -> None:
    big_pgn = _make_larger_pgn(20)
    pgn_path = tmp_path / "big.pgn"
    pgn_path.write_text(big_pgn)
    brackets = list(DEFAULT_BRACKETS)

    single_dir = tmp_path / "single"
    process_chunk(
        pgn_path, single_dir, brackets, chunk=1, chunk_size=60, skip_first_n_plies=2
    )

    chunked_dir = tmp_path / "chunked"
    prepare_all_chunks(
        pgn_path, chunked_dir, brackets, chunk_size=20, skip_first_n_plies=2
    )

    for bracket in brackets:
        name = bracket.name
        single_path = single_dir / f"bucket_{name}_1.npz"
        if not single_path.exists():
            assert list(chunked_dir.glob(f"bucket_{name}_*.npz")) == []
            continue

        single_data = np.load(single_path)
        single_pairs = {
            (int(m), row.tobytes())
            for m, row in zip(single_data["moves"], single_data["boards"], strict=True)
        }

        chunked_pairs: set[tuple[int, bytes]] = set()
        for chunk_path in sorted(chunked_dir.glob(f"bucket_{name}_*.npz")):
            chunk_data = np.load(chunk_path)
            chunked_pairs |= {
                (int(m), row.tobytes())
                for m, row in zip(
                    chunk_data["moves"], chunk_data["boards"], strict=True
                )
            }

        assert single_pairs == chunked_pairs, f"mismatch for bracket {name}"


def test_prepare_all_chunks_creates_expected_files_and_stops_at_exhaustion(
    tmp_path: Path,
) -> None:
    big_pgn = _make_larger_pgn(10)
    pgn_path = tmp_path / "big.pgn"
    pgn_path.write_text(big_pgn)
    out_dir = tmp_path / "out"

    prepare_all_chunks(pgn_path, out_dir, list(DEFAULT_BRACKETS), chunk_size=9)

    all_files = list(out_dir.glob("bucket_*.npz"))
    assert len(all_files) > 0
    for f in all_files:
        chunk_num = int(f.stem.rsplit("_", 1)[1])
        assert 1 <= chunk_num <= 4


def test_prepare_all_chunks_respects_max_chunks(tmp_path: Path) -> None:
    big_pgn = _make_larger_pgn(10)
    pgn_path = tmp_path / "big.pgn"
    pgn_path.write_text(big_pgn)
    out_dir = tmp_path / "out"

    prepare_all_chunks(
        pgn_path, out_dir, list(DEFAULT_BRACKETS), chunk_size=5, max_chunks=1
    )

    all_files = list(out_dir.glob("bucket_*.npz"))
    assert len(all_files) > 0
    for f in all_files:
        chunk_num = int(f.stem.rsplit("_", 1)[1])
        assert chunk_num == 1


def test_prepare_all_chunks_with_bucket_subset(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    only_1000 = brackets_by_name(["1000"])
    prepare_all_chunks(
        pgn_path, out_dir, only_1000, chunk_size=10, skip_first_n_plies=2
    )

    assert (out_dir / "bucket_1000_1.npz").exists()
    assert not (out_dir / "bucket_2000_1.npz").exists()
    assert list(out_dir.glob("bucket_2000_*.npz")) == []


def test_growable_array_store_grows_and_preserves_data(tmp_path: Path) -> None:
    store = _GrowableArrayStore(
        row_shape=(3,), dtype=np.float32, initial_capacity=2, tmp_dir=tmp_path
    )
    rows = [np.array([i, i + 1, i + 2], dtype=np.float32) for i in range(10)]
    for row in rows:
        store.append(row)

    assert store.count == 10
    result = store.finalized()
    assert result.shape == (10, 3)
    for i, row in enumerate(rows):
        assert np.array_equal(result[i], row)

    store.cleanup()


def test_growable_array_store_cleanup_removes_backing_file(tmp_path: Path) -> None:
    store = _GrowableArrayStore(
        row_shape=(2,), dtype=np.float32, initial_capacity=2, tmp_dir=tmp_path
    )
    store.append(np.array([1.0, 2.0], dtype=np.float32))
    backing_path = store._path
    assert backing_path.exists()

    store.cleanup()
    assert not backing_path.exists()


def test_cli_runs_end_to_end_and_produces_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys, "argv", ["prepare", str(pgn_path), str(out_dir), "--chunk-size", "10"]
    )
    main()

    assert (out_dir / "bucket_1000_1.npz").exists()
    assert (out_dir / "bucket_2000_1.npz").exists()


def test_cli_respects_buckets_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            str(pgn_path),
            str(out_dir),
            "--chunk-size",
            "10",
            "--buckets",
            "1000",
        ],
    )
    main()

    assert (out_dir / "bucket_1000_1.npz").exists()
    assert not (out_dir / "bucket_2000_1.npz").exists()


def test_cli_rejects_unknown_bucket_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            str(pgn_path),
            str(out_dir),
            "--chunk-size",
            "10",
            "--buckets",
            "9999",
        ],
    )
    with pytest.raises(ValueError, match="unknown bucket"):
        main()
