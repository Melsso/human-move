import chess
import chess.engine
import pytest
from chess_backend.engine_eval import EngineManager, EngineUnavailableError


class _FakeUciEngine:
    def __init__(self, score_cp: int = 42) -> None:
        self.analyse_calls = 0
        self.quit_called = False
        self._score_cp = score_cp

    def analyse(
        self, board: chess.Board, limit: chess.engine.Limit
    ) -> dict[str, chess.engine.PovScore]:
        self.analyse_calls += 1
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(self._score_cp), chess.WHITE)
        }

    def quit(self) -> None:
        self.quit_called = True


def test_evaluate_returns_score_from_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUciEngine(score_cp=123)
    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: fake)

    manager = EngineManager()
    score = manager.evaluate(chess.Board())

    assert score.white().score() == 123


def test_engine_started_only_once_across_multiple_evaluate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeUciEngine()
    start_calls = []
    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(
        chess.engine.SimpleEngine,
        "popen_uci",
        lambda _path: (start_calls.append(1), fake)[1],
    )

    manager = EngineManager()
    manager.evaluate(chess.Board())
    manager.evaluate(chess.Board())
    manager.evaluate(chess.Board())

    assert len(start_calls) == 1
    assert fake.analyse_calls == 3


def test_evaluate_raises_engine_unavailable_when_stockfish_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> str:
        raise RuntimeError("stockfish not found")

    monkeypatch.setattr("chess_backend.engine_eval.find_stockfish", _raise)

    manager = EngineManager()
    with pytest.raises(EngineUnavailableError):
        manager.evaluate(chess.Board())


def test_unavailable_is_sticky_does_not_retry_find_stockfish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def _raise() -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("stockfish not found")

    monkeypatch.setattr("chess_backend.engine_eval.find_stockfish", _raise)

    manager = EngineManager()
    for _ in range(3):
        with pytest.raises(EngineUnavailableError):
            manager.evaluate(chess.Board())

    assert call_count == 1


def test_close_calls_quit_on_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUciEngine()
    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: fake)

    manager = EngineManager()
    manager.evaluate(chess.Board())
    manager.close()

    assert fake.quit_called is True


def test_close_is_safe_when_engine_never_started() -> None:
    manager = EngineManager()
    manager.close()
