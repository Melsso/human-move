import io
from pathlib import Path

import chess.pgn
import numpy as np
import zstandard as zstd
from chess_data.prepare import extract_positions, process_pgn
from chess_shared import NUM_MOVES, NUM_PLANES

SAMPLE_PGN = """\
[Event "Rated Blitz game"]
[White "a"]
[Black "b"]
[WhiteElo "1000"]
[BlackElo "1020"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0

[Event "Rated Blitz game"]
[White "c"]
[Black "d"]
[WhiteElo "1900"]
[BlackElo "1950"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 0-1

[Event "Casual Blitz game"]
[White "a"]
[Black "b"]
[WhiteElo "1000"]
[BlackElo "1000"]
[Termination "Normal"]

1. e4 e5 1-0
"""


def test_extract_positions_shapes() -> None:
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
    assert game is not None
    positions = extract_positions(game, max_positions_per_game=40, skip_first_n_plies=2)
    assert len(positions) > 0
    board_tensor, move_idx = positions[0]
    assert board_tensor.shape == (NUM_PLANES, 8, 8)
    assert 0 <= move_idx < NUM_MOVES


def test_process_pgn_end_to_end_plain_file(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_path = tmp_path / "out.npz"

    process_pgn(pgn_path, out_path, min_elo=900, max_elo=1100, skip_first_n_plies=2)

    data = np.load(out_path)
    assert "boards" in data and "moves" in data
    assert data["boards"].shape[1:] == (NUM_PLANES, 8, 8)
    assert data["boards"].shape[0] == data["moves"].shape[0]
    assert data["boards"].shape[0] > 0


def test_process_pgn_end_to_end_zst_file(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn.zst"
    cctx = zstd.ZstdCompressor()
    with open(pgn_path, "wb") as f:
        f.write(cctx.compress(SAMPLE_PGN.encode("utf-8")))
    out_path = tmp_path / "out.npz"

    process_pgn(pgn_path, out_path, min_elo=1800, max_elo=2000, skip_first_n_plies=2)

    data = np.load(out_path)
    assert data["boards"].shape[0] > 0


def test_process_pgn_raises_when_nothing_matches(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    out_path = tmp_path / "out.npz"

    try:
        process_pgn(pgn_path, out_path, min_elo=2900, max_elo=3000)
        assert False, "expected RuntimeError for an elo range that matches nothing"
    except RuntimeError:
        pass
