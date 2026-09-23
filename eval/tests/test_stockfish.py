import shutil

import pytest
from chess_eval.stockfish import find_stockfish


def test_find_stockfish_is_reexported_from_chess_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/stockfish")
    assert str(find_stockfish()) == "/opt/homebrew/bin/stockfish"
