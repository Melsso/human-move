from __future__ import annotations

import threading

import chess
import chess.engine
from chess_shared import find_stockfish

ANALYSIS_TIME_S = 0.15


class EngineUnavailableError(Exception):
    pass


class EngineManager:
    def __init__(self) -> None:
        self._engine: chess.engine.SimpleEngine | None = None
        self._lock = threading.Lock()
        self._unavailable = False

    def _ensure_started(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine
        if self._unavailable:
            raise EngineUnavailableError("stockfish is not available")
        try:
            path = find_stockfish()
            self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        except Exception as e:
            self._unavailable = True
            raise EngineUnavailableError(str(e)) from e
        return self._engine

    def evaluate(self, board: chess.Board) -> chess.engine.PovScore:
        with self._lock:
            engine = self._ensure_started()
            info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_S))
            score = info["score"]
            assert isinstance(score, chess.engine.PovScore)
            return score

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.quit()
                self._engine = None
