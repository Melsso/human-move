import sys
from pathlib import Path

import pytest
import zstandard as zstd
from chess_data.count_games import count_games, main

SAMPLE_PGN = """\
[Event "Rated Blitz game"]
[White "a"]
[Black "b"]
[WhiteElo "1000"]
[BlackElo "1020"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Rated Blitz game"]
[White "c"]
[Black "d"]
[WhiteElo "1900"]
[BlackElo "1950"]

1. d4 d5 2. c4 e6 0-1

[Event "Casual Blitz game"]
[White "a"]
[Black "b"]
[WhiteElo "1000"]
[BlackElo "1000"]

1. e4 e5 1-0
"""


def test_count_games_plain_file(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    assert count_games(pgn_path) == 3


def test_count_games_zst_file(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn.zst"
    cctx = zstd.ZstdCompressor()
    with open(pgn_path, "wb") as f:
        f.write(cctx.compress(SAMPLE_PGN.encode("utf-8")))
    assert count_games(pgn_path) == 3


def test_count_games_empty_file(tmp_path: Path) -> None:
    pgn_path = tmp_path / "empty.pgn"
    pgn_path.write_text("")
    assert count_games(pgn_path) == 0


def test_cli_prints_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)

    monkeypatch.setattr(sys, "argv", ["count_games", str(pgn_path)])
    main()

    captured = capsys.readouterr()
    assert "3 games" in captured.out
