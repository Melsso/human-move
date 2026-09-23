import shutil
import stat
from pathlib import Path

import pytest
from chess_shared.engine import find_stockfish


@pytest.fixture(autouse=True)
def _clear_stockfish_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)


def test_find_stockfish_raises_clear_error_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="brew install stockfish"):
        find_stockfish()


def test_find_stockfish_returns_path_from_which(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/stockfish")
    assert str(find_stockfish()) == "/opt/homebrew/bin/stockfish"


def test_find_stockfish_prefers_stockfish_path_env_var_over_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_binary = tmp_path / "stockfish"
    fake_binary.write_text("#!/bin/sh\necho fake\n")
    fake_binary.chmod(fake_binary.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("STOCKFISH_PATH", str(fake_binary))
    monkeypatch.setattr(shutil, "which", lambda _name: "/opt/homebrew/bin/stockfish")

    assert find_stockfish() == fake_binary


def test_find_stockfish_raises_when_stockfish_path_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STOCKFISH_PATH", "/definitely/does/not/exist/stockfish")
    with pytest.raises(RuntimeError, match="not a file"):
        find_stockfish()


def test_find_stockfish_raises_when_stockfish_path_is_not_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    non_executable = tmp_path / "stockfish"
    non_executable.write_text("not actually a binary")
    mode = non_executable.stat().st_mode
    non_executable.chmod(mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)

    monkeypatch.setenv("STOCKFISH_PATH", str(non_executable))
    with pytest.raises(RuntimeError, match=r"chmod \+x"):
        find_stockfish()
