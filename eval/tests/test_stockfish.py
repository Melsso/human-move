import shutil

import pytest
from chess_eval.stockfish import find_stockfish


def test_find_stockfish_raises_clear_error_when_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="brew install stockfish"):
        find_stockfish()


def test_find_stockfish_returns_path_when_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/stockfish")
    path = find_stockfish()
    assert str(path) == "/opt/homebrew/bin/stockfish"
