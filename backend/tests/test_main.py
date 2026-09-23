from pathlib import Path

import chess
import chess.engine
import pytest
import torch
from chess_backend.engine_eval import EngineManager
from chess_backend.inference import ModelRegistry
from chess_backend.main import app
from chess_shared import NUM_MOVES, NUM_PLANES
from chess_training.model import MaiaPolicyNet
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


def _write_synthetic_checkpoint(
    path: Path, num_blocks: int = 2, num_filters: int = 8
) -> None:
    model = MaiaPolicyNet(
        NUM_PLANES, NUM_MOVES, num_blocks=num_blocks, num_filters=num_filters
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 2,
        "val_top1": 0.3,
        "val_top3": 0.55,
        "num_blocks": num_blocks,
        "num_filters": num_filters,
    }
    torch.save(checkpoint, path)


def _client_with_synthetic_checkpoints(
    tmp_path: Path, tier_names: list[str]
) -> TestClient:
    checkpoints_dir = tmp_path / "checkpoints"
    for name in tier_names:
        tier_dir = checkpoints_dir / name
        tier_dir.mkdir(parents=True)
        _write_synthetic_checkpoint(tier_dir / "best.pt")

    test_registry = ModelRegistry(checkpoints_dir)
    import chess_backend.main as main_module

    main_module.registry = test_registry
    return TestClient(app)


def test_health():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tiers_lists_available_checkpoints(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(
        tmp_path, ["bucket_1000", "bucket_2000"]
    )
    response = client.get("/api/tiers")
    assert response.status_code == 200
    tiers = {t["tier"] for t in response.json()}
    assert tiers == {"bucket_1000", "bucket_2000"}


def test_tiers_empty_when_no_checkpoints(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(tmp_path, [])
    response = client.get("/api/tiers")
    assert response.status_code == 200
    assert response.json() == []


def test_move_happy_path_returns_model_reply(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(tmp_path, ["bucket_1000"])
    board = chess.Board()

    response = client.post(
        "/api/move",
        json={"fen": board.fen(), "move_uci": "e2e4", "tier": "bucket_1000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["human_move_san"] == "e4"
    assert body["model_move_uci"] is not None
    assert body["game_over"] is False
    assert len(body["top_candidates"]) > 0

    resulting_board = chess.Board(body["fen_after_model"])
    assert resulting_board.fullmove_number >= 1


def test_move_rejects_illegal_move(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(tmp_path, ["bucket_1000"])
    board = chess.Board()

    response = client.post(
        "/api/move",
        json={"fen": board.fen(), "move_uci": "e2e5", "tier": "bucket_1000"},
    )
    assert response.status_code == 400


def test_move_rejects_malformed_fen(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(tmp_path, ["bucket_1000"])
    response = client.post(
        "/api/move",
        json={"fen": "not a real fen", "move_uci": "e2e4", "tier": "bucket_1000"},
    )
    assert response.status_code == 400


def test_move_rejects_unknown_tier(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(tmp_path, ["bucket_1000"])
    board = chess.Board()
    response = client.post(
        "/api/move",
        json={"fen": board.fen(), "move_uci": "e2e4", "tier": "bucket_9999"},
    )
    assert response.status_code == 404


def test_move_returns_game_over_without_model_reply_on_checkmate(tmp_path: Path):
    client = _client_with_synthetic_checkpoints(tmp_path, ["bucket_1000"])
    board = chess.Board()
    for san in ["f3", "e5", "g4"]:
        board.push_san(san)

    response = client.post(
        "/api/move",
        json={"fen": board.fen(), "move_uci": "d8h4", "tier": "bucket_1000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["game_over"] is True
    assert body["model_move_uci"] is None
    assert body["result"] == "0-1"


def test_eval_returns_available_false_when_stockfish_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise() -> str:
        raise RuntimeError("stockfish not found")

    monkeypatch.setattr("chess_backend.engine_eval.find_stockfish", _raise)

    import chess_backend.main as main_module

    main_module.engine_manager = EngineManager()
    client = TestClient(app)

    response = client.post("/api/eval", json={"fen": chess.Board().fen()})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["score_cp"] is None
    assert body["mate"] is None


def test_eval_returns_score_when_stockfish_available(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeUciEngine(score_cp=77)
    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _path: fake)

    import chess_backend.main as main_module

    main_module.engine_manager = EngineManager()
    client = TestClient(app)

    response = client.post("/api/eval", json={"fen": chess.Board().fen()})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["score_cp"] == 77
    assert body["mate"] is None


def test_eval_returns_mate_score(monkeypatch: pytest.MonkeyPatch):
    class _MateEngine:
        def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
            return {"score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)}

        def quit(self) -> None:
            pass

    monkeypatch.setattr(
        "chess_backend.engine_eval.find_stockfish", lambda: "/fake/path"
    )
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda _path: _MateEngine()
    )

    import chess_backend.main as main_module

    main_module.engine_manager = EngineManager()
    client = TestClient(app)

    response = client.post("/api/eval", json={"fen": chess.Board().fen()})
    body = response.json()
    assert body["available"] is True
    assert body["mate"] == 3
    assert body["score_cp"] is None


def test_eval_rejects_malformed_fen():
    client = TestClient(app)
    response = client.post("/api/eval", json={"fen": "not a real fen"})
    assert response.status_code == 400
