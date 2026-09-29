from collections.abc import Callable
from pathlib import Path

import chess
import chess.engine
import chess_backend.main as main_module
import pytest
from chess_backend.engine_eval import EngineManager
from chess_backend.inference import ModelRegistry
from fastapi.testclient import TestClient


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


@pytest.fixture
def client_factory(
    monkeypatch: pytest.MonkeyPatch, make_checkpoints: Callable[..., Path]
) -> Callable[[list[str]], TestClient]:
    def build(dir_names: list[str]) -> TestClient:
        monkeypatch.setattr(
            main_module, "registry", ModelRegistry(make_checkpoints(dir_names))
        )
        return TestClient(main_module.app)

    return build


def test_health() -> None:
    client = TestClient(main_module.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tiers_lists_configured_tiers_with_available_checkpoints(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1", "bucket_1500_1"])
    response = client.get("/api/tiers")
    assert response.status_code == 200
    assert [t["tier"] for t in response.json()] == ["1000", "1500", "2000"]


def test_tiers_empty_when_no_checkpoints(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory([])
    response = client.get("/api/tiers")
    assert response.status_code == 200
    assert response.json() == []


def test_move_happy_path_returns_model_reply(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1"])
    response = client.post(
        "/api/move",
        json={"fen": chess.Board().fen(), "move_uci": "e2e4", "tier": "1000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["human_move_san"] == "e4"
    assert body["model_move_uci"] is not None
    assert body["game_over"] is False
    assert len(body["top_candidates"]) > 0
    assert chess.Board(body["fen_after_model"]).fullmove_number >= 1


def test_move_with_searching_tier_returns_model_reply(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1500_1"])
    response = client.post(
        "/api/move",
        json={"fen": chess.Board().fen(), "move_uci": "e2e4", "tier": "1500"},
    )
    assert response.status_code == 200
    assert response.json()["model_move_uci"] is not None


def test_move_rejects_illegal_move(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1"])
    response = client.post(
        "/api/move",
        json={"fen": chess.Board().fen(), "move_uci": "e2e5", "tier": "1000"},
    )
    assert response.status_code == 400


def test_move_rejects_malformed_fen(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1"])
    response = client.post(
        "/api/move",
        json={"fen": "not a real fen", "move_uci": "e2e4", "tier": "1000"},
    )
    assert response.status_code == 400


def test_move_rejects_unknown_tier(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1"])
    response = client.post(
        "/api/move",
        json={"fen": chess.Board().fen(), "move_uci": "e2e4", "tier": "9999"},
    )
    assert response.status_code == 404


def test_move_rejects_tier_whose_checkpoint_is_missing(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1"])
    response = client.post(
        "/api/move",
        json={"fen": chess.Board().fen(), "move_uci": "e2e4", "tier": "2000"},
    )
    assert response.status_code == 404


def test_move_returns_game_over_without_model_reply_on_checkmate(
    client_factory: Callable[[list[str]], TestClient],
) -> None:
    client = client_factory(["bucket_1000_1"])
    board = chess.Board()
    for san in ["f3", "e5", "g4"]:
        board.push_san(san)

    response = client.post(
        "/api/move",
        json={"fen": board.fen(), "move_uci": "d8h4", "tier": "1000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["game_over"] is True
    assert body["model_move_uci"] is None
    assert body["result"] == "0-1"


def test_eval_returns_available_false_when_stockfish_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> str:
        raise RuntimeError("stockfish not found")

    monkeypatch.setattr("chess_backend.engine_eval.find_stockfish", _raise)
    monkeypatch.setattr(main_module, "engine_manager", EngineManager())
    client = TestClient(main_module.app)

    response = client.post("/api/eval", json={"fen": chess.Board().fen()})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["score_cp"] is None
    assert body["mate"] is None


def test_eval_returns_score_when_stockfish_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeUciEngine(score_cp=77)
    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: fake)
    monkeypatch.setattr(main_module, "engine_manager", EngineManager())
    client = TestClient(main_module.app)

    response = client.post("/api/eval", json={"fen": chess.Board().fen()})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["score_cp"] == 77
    assert body["mate"] is None


def test_eval_returns_mate_score(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MateEngine:
        def analyse(
            self, board: chess.Board, limit: chess.engine.Limit
        ) -> dict[str, chess.engine.PovScore]:
            return {"score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)}

        def quit(self) -> None:
            pass

    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda _path: _MateEngine()
    )
    monkeypatch.setattr(main_module, "engine_manager", EngineManager())
    client = TestClient(main_module.app)

    response = client.post("/api/eval", json={"fen": chess.Board().fen()})
    body = response.json()
    assert body["available"] is True
    assert body["mate"] == 3
    assert body["score_cp"] is None


def test_eval_rejects_malformed_fen() -> None:
    client = TestClient(main_module.app)
    response = client.post("/api/eval", json={"fen": "not a real fen"})
    assert response.status_code == 400
